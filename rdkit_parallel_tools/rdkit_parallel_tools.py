import gzip
import io
import logging
import multiprocessing
from collections.abc import Iterable
from typing import Callable, ContextManager
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rich.progress import track

chem_file_extensions = (".sdf", ".sd", ".SDF", ".SD", ".smi", ".txt", ".smiles")
logger = logging.getLogger('rdkit_parallel_tools')

# make duckdb optional
try:
    import duckdb
except ImportError:
    logger.warning("duckdb is not installed, can't use DuckDbWriter.")


def chem_input_to_file(file) -> io.IOBase:
    """
    Convenience function to create file-like object from different forms of input (str or Path, sdf, sdf.gz, smi, smi.gz)
    for a molecule containing file (smiles or sdf)

    Input must be a string pointing to a valid file with a valid extension or gzipped file (.gz) or already be a
    file-like object (io.TextIOBase) that is returned as-is.

    :param file: input to convert to a file-like object
    :return: file-like object
    """
    if isinstance(file, str):
        if file.endswith(chem_file_extensions):
            logger.debug(f"Found a {file.endswith()}. Return file-like object")
            return open(file)
        elif file.endswith(".gz"):
            logger.debug("Found gzipped file. Return file-like object")
            return gzip.open(file, "rt")
        else:
            raise ValueError(f"Found file with unsupported file extension {file.endswith()}.")
    elif isinstance(file, Path):
        if file.suffix in chem_file_extensions:
            logger.debug(f"Found file with extension {file.suffix}. Return file-like object")
            return file.open()
        elif file.suffix == ".gz":
            logger.debug("Found gzipped file. Return file-like object")
            return gzip.open(file, "rt")
        else:
            raise ValueError(f"Found file with invalid file extension {file.suffix}.")
    elif isinstance(file, io.TextIOBase):
        # file like object, return as-is
        logger.debug("Found existing TextIOBase. Return as-is")
        return file
    else:
        error_message = f"Found input of type {type(file)} which is not a 'str' or valid file-like object."
        logger.error(error_message)
        raise ValueError(error_message)


def raw_sd_reader(file: io.TextIOBase) -> Iterable[str]:
    """
    Read a sd-file but return only raw sd-data as string and not a rdkit molecule instance.

    Code adapted from below blog post from Noel O'Boyle:
    https://baoilleach.blogspot.com/2020/05/python-patterns-for-processing-large.html

    :param file: a file-like object
    :return: iterator that returns raw sd-block for each entry
    """
    data = []
    for line in file:
        data.append(line)
        if line.startswith("$$$$"):
            yield "".join(data)
            data = []


def chunked_raw_sd_reader(file, num_mols: int = 100) -> Iterable[str]:
    """
    Read a sd-file but return only raw sd-data as string in chunks of "num_mols".

    When using multiprocessing, it's preferably to send the raw molecule data in chunks to each worker and have the
    workers generate the molecules from the "sd-blocks". This is easy with SMILES (one molecule per line) and this is
    simply a helper function to also make this easy when the source is a sd-file.

    Code taken from below blog post from Noel O'Boyle:
    https://baoilleach.blogspot.com/2020/05/python-patterns-for-processing-large.html

    :param file: a file-like object
    :param num_mols: number of raw sd-blocks to return per iteration
    :return: list of raw sd-blocks of size num_mols
    """
    reader = raw_sd_reader(file)
    tmp = []
    i = 0
    for sdf in reader:
        if i == num_mols:
            yield "".join(tmp)
            tmp = []
            i = 0
        i += 1
        tmp.append(sdf)
    # yield final chunk
    yield "".join(tmp)


def mol_to_sd(mol: Chem.Mol, additional_properties: dict = {}) -> str:
    """
    Converts a molecule to an sd-string, eg molblock + all properties + optional additional properties to add not
    stored in the mol object.

    :param mol: the molecule
    :param additional_properties:  optional additional properties to add to the sd file not already part of the mol
    :return: molecule as sdf string
    """
    sdf = [Chem.MolToMolBlock(mol)]
    # append existing properties
    for prop_name in mol.GetPropNames():
        value = mol.GetProp(prop_name)
        sdf.append(f">  <{prop_name}>\n{value}\n")
    if additional_properties is not None and len(additional_properties) > 1:
        # Append new properties to sd-file
        for key, value in additional_properties.items():
            sdf.append(f">  <{key}>\n{value}\n")
    sdf.append('$$$$')
    return '\n'.join(sdf)


def parallel_calculation(data_generator: Iterable[str], calc_func: Callable, data_writer: ContextManager,
                         data_writer_args: tuple = None, data_writer_kwargs: dict = None, num_workers: int = -1):
    """
    Calculate in a streaming and multiprocessing fashion "calc_func" for all molecules in the passed-in data.

    It is up to the caller to ensure the input functions are compatible.

    data_reader must generate a string for each molecule which the calc_function can convert to a molecule.
    calc_function must return the data in the way data_writer expects it (say as sdf or csv).
    data_writer is a ContextManager that must return an object which has a "write" method.
    The write method could also write to a database. It's entirely up to the caller to define how the results should
    be written to disk.

    Note that the generated output might have the molecules in different order than in input.

    :param data_generator: function that reads the molecules (one at a time or in blocks)
    :param calc_func: the function to perform parallel calculation on
    :param data_writer: function that generates output file
    :param data_writer_args: positional arguments to pass to the data_writer
    :param data_writer_kwargs: keyword arguments to pass to the data_writer
    :param num_workers: how many processes to use, by default all available logical processors
    """
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()
    if data_writer_kwargs is None:
        data_writer_kwargs = {}
    if data_writer_args is None:
        data_writer_args = ()
    with multiprocessing.Pool(num_workers) as pool:
        with data_writer(*data_writer_args, **data_writer_kwargs) as writer:
            for data in track(pool.imap_unordered(calc_func, data_generator), description="Calculating..."):
                writer.write(data)


def sd_to_sd_parallel_calculation(sd_file: io.TextIOBase, sd_output: str, calc_func, num_workers: int = -1):
    """
    Calculate in a streaming and multiprocessing fashion "calc_func" for all molecules in the passed-in sd-file.

    `calc_func` must accept a single str argument where str is a sdf as string. It can contain one or more molecules.
    The function can then iterate over the molecules by using a MolSupplier:

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdf)
    for mol in suppl:
        # do stuff here

    The function must then return a sdf as a string containing all the molecules. It can contain additional properties
    or changed coordinates like 3D coords + minimization. etc. whatever your calculation of interest is.

    See `calc_descriptors_for_sd` as an example how such a function must look.

    Note that the generated output file might have the molecules in different order.

    :param sd_file: sdf containing the molecules
    :param sd_output: output sd-file. gzipped if this ends with ".gz"
    :param calc_func: the function to perform parallel calculation on
    :param num_workers: how many processes to use, by default all available logical processors
    """
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()
    # taken from https://stackoverflow.com/questions/56504405/just-one-with-open-file-as-f-based-on-a-conditional
    opener = gzip.open if sd_output.endswith(".gz") else open
    with multiprocessing.Pool(num_workers) as pool:
        with opener(sd_output, "wt") as out:
            miter = chunked_raw_sd_reader(sd_file)
            for data in track(pool.imap_unordered(calc_func, miter), description="Calculating..."):
                out.write(data)


def calc_descriptors_for_sd(sdf: str):
    """
    Calculate all descriptors for passed-in molecules inside the sd-string and appends them as new properties to
    the sd-string for each molecule
    :param sdf: a sdf string containing one or more molecules
    :return: sd-string with calculates properties for each molecule added
    """
    ms = Chem.SDMolSupplier()
    ms.SetData(sdf)
    res = []
    for m in ms:
        desc = Descriptors.CalcMolDescriptors(m)
        res.append(mol_to_sd(m, desc))
    return '\n'.join(res)


def calculate_mol_descriptors(sd_file: io.TextIOBase, sd_output, num_workers=-1):
    """
    Calculate in a streaming and multiprocessing fashion all descriptors for all molecules in the passed-in sd-file.

    Note that the generated output file might have the molecules in different order.

    This function is meant as an example for processing very large sd-files.

    :param sd_file: sdf containing the molecules
    :param sd_output: output sdf with all the calculated properties added to each molecule
    :param num_workers: how many processes to use, by default all available logical processors
    """
    sd_to_sd_parallel_calculation(sd_file, sd_output, calc_func=calc_descriptors_for_sd, num_workers=num_workers)


class SmilesWriterWrapper:
    """
    A Context Manager Wrapper around rdkit.Chem.SmilesWriter to be used in parallel_calculation function to write the
    molecules as SMILES. Same as with SmilesWriter, to avoid output of molecule name you need to set nameHeader="".

    This example just reads and then writes out the SMILES again but explains the intended usage.

    smi_file = "simple_test.smi"
    out_smi = "out.smi"
    with open(smi_file, "r") as f:
        parallel_calculation(f, to_rdkit, SmilesWriterWrapper, data_writer_args=(out_smi,),
                             data_writer_kwargs={"includeHeader": False}, num_workers=1)
    """

    def __init__(self, file, delimiter=" ", nameHeader="Name", includeHeader=True, isomericSmiles=True,
                 kekuleSmiles=False):
        self.writer = Chem.SmilesWriter(file, delimiter, nameHeader, includeHeader, isomericSmiles, kekuleSmiles)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def write(self, mol):
        """
        Write the molecule or list of molecules to a SMILES file.
        :param mol: rdkit molecule to write
        """
        if isinstance(mol, Chem.Mol):
            self.writer.write(mol)
        elif isinstance(mol, list):
            for mol in mol:
                self.writer.write(mol)
        else:
            raise ValueError(f"Expected rdkit mol or list of mols, but got {type(mol)}.")


class DuckDBWriter:
    """
    Proof-of-Concept how to write results directly to a database

    Here great care need to be taken that the calculation function returns the columns in the correct order, the
    same order as the column order of the table and all columns must be present.
    """

    def __init__(self, database, prepared_statement):
        self.con = duckdb.connect(database = database, read_only = False)
        self.prepared_statement = prepared_statement

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()
        return False

    def write(self, data):
        if not isinstance(data, list):
            raise ValueError(f"Data must be a list of values to insert but got {data} instead")
        if isinstance(data[0], list):
            # assumption: list of list, multiple inserts
            self.con.executemany(self.prepared_statement, data)
        else:
            self.con.execute(self.prepared_statement, data)
