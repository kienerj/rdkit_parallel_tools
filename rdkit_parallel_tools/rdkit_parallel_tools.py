import gzip
import io
import logging
import multiprocessing
from collections.abc import Iterable
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

sdf_extensions = (".sdf", ".sd", ".SDF", ".SD")
logger = logging.getLogger('rdkit_parallel_tools')


def input_to_file(file) -> io.IOBase:
    """
    Convenience function to create file-like object from different forms of input (str or Path, sdf for sdf.gz)

    Handles different input for "file" and returns a file-like object.

    Input must be a string pointing to a valid sd-file with a valid extension or gzipped sd-file (.gz) or already be
    a file-like object (io.TextIOBase) that is returned as-is.

    :param file: input to convert to a file-like object
    :return: file-like object
    """
    if isinstance(file, str):
        if file.endswith(sdf_extensions):
            logger.debug("Found sd-file. Return file-like object")
            return open(file)
        elif file.endswith(".gz"):
            logger.debug("Found gzipped sd-file. Return file-like object")
            return gzip.open(file, "rt")
        else:
            raise ValueError("Found file with invalid file extension.")
    elif isinstance(file, Path):
        if file.suffix in sdf_extensions:
            logger.debug("Found sd-file. Return file-like object")
            return file.open()
        elif file.suffix == ".gz":
            logger.debug("Found gzipped sd-file. Return file-like object")
            return gzip.open(file, "rt")
        else:
            raise ValueError(f"Found file with invalid file extension {file.suffix}.")
    elif isinstance(file, io.TextIOBase):
        # file like object, return as-is
        logger.debug("Found existing BufferedIOBase. Return as-is")
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


def calc_descriptors_for_sd(sdf: str):
    """
    Calculate all descriptors for passed in molecules inside the sd-string and appends them as new properties to
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

    This function is meant for processing very large sd-files.

    :param sd_file: sdf containing the molecules
    :param sd_output: output sdf with all the calculated properties added to each molecule
    :param num_workers: how many processes to use, by default all available logical processors
    """

    if num_workers >= 0:
        num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
        with open(sd_output, "w") as out:
            miter = chunked_raw_sd_reader(sd_file)
            for data in pool.imap_unordered(calc_descriptors_for_sd, miter):
                out.write(data)
