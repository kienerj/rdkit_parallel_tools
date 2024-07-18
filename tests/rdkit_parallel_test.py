import os
import logging
import unittest
from rdkit.Chem import Descriptors
from rdkit_parallel_tools import *

logger = logging.getLogger("rdkit_parallel_tools")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class RDKitParallelTest(unittest.TestCase):

    NUM_MOLS_CDK2 = 47

    def setUp(self):
        self.data_dir = os.environ["CONDA_PREFIX"] + "/Library/share/RDKit/Docs/Book/data"
        self.cdk2_path = self.data_dir + "/cdk2.sdf"

    def test_raw_reader(self):
        with open(self.cdk2_path) as f:
            it = raw_sd_reader(f)
            self.assertIsNotNone(it, "Iterator was 'None'.")
            rslt = list(it)
        self.assertEqual(len(rslt), self.NUM_MOLS_CDK2,
                         f"Found unexpected number of {len(rslt)} molecules in test sd-file. Expected {self.NUM_MOLS_CDK2}.")

    def test_calc_descriptors_for_sd(self):
        with open(self.cdk2_path) as f:
            it = raw_sd_reader(f)
            sdf = next(it)
            rslt = calc_descriptors_for_sd(sdf)
        self.assertIsNotNone(rslt)
        self.assertNotEqual(sdf, rslt, "Output same as input. Descriptors not added to sd-file.")
        self.assertTrue("MolWt" in rslt, "Descriptor MolWt not found in result.")

    def test_calc_descriptors(self):
        with open(self.cdk2_path) as f:
            calculate_mol_descriptors(f, "files/out.sdf", num_workers=1)
        ms = Chem.SDMolSupplier("files/out.sdf")
        mols = [m for m in ms]
        self.assertEqual(len(mols), self.NUM_MOLS_CDK2, f"Output file contains unexpected amount of molecules. "
                                                        f"Found: {len(mols)}, Expected:{self.NUM_MOLS_CDK2}")
        m = mols[0]
        mw = m.GetProp("MolWt")
        self.assertIsNotNone(mw, "MolWt property in first molecule not found.")

    def test_smiles_processing(self):
        # Doesn't really test anything but keeping it here as proof-of-concept
        smi_file = r"files/simple_test.smi"
        num_workers = 1
        with open(smi_file, "r") as f:
            with multiprocessing.Pool(num_workers) as pool:
                for data in pool.imap(smiles_to_rdkit, f, 1):
                    self.assertIsNotNone(data, "Expected RDKit molecules but was 'None'.")
                    self.assertTrue(isinstance(data, Chem.rdchem.Mol), f"Expected RDKit molecules but was {type(data)}.")

    def test_parallel_calculation_smiles(self):
        smi_file = r"files/simple_test.smi"
        out_smi = r"files/out.smi"
        with open(smi_file, "r") as f:
            parallel_calculation(f, smiles_to_rdkit, SmilesWriterWrapper, data_writer_args=(out_smi,),
                                 data_writer_kwargs={"includeHeader": False, "nameHeader": ""}, num_workers=1)
        mols = []
        with open(out_smi, "r") as o:
            for l in o:
                mols.append(l)
        self.assertTrue(len(mols) == 4, f"Expected 4 lines in output but got {len(mols)}.")

    def test_duckdbwriter(self):
        db_file = "files/test.db"
        con = duckdb.connect(database=db_file)
        try:
            con.execute("CREATE TABLE test (smi TEXT)")
            with DuckDBWriter(db_file, "INSERT INTO test VALUES(?)") as writer:
                writer.write(["c1ccccc1"])
                writer.write([["CCC"], ["CCC=O"]])
            data = con.sql("SELECT * from test").fetchall()
            self.assertTrue(len(data) == 3, f"Expected 3 rows in database but got {len(data)}.")
            self.assertEqual("c1ccccc1", data[0][0])
        finally:
            con.close()
            os.remove(db_file)

    def test_duckdb_parallel_simple(self):
        smi_file = r"files/simple_test.smi"
        db_file = "files/test.db"
        ps = "INSERT INTO test VALUES(?)"
        con = duckdb.connect(database=db_file)
        try:
            con.execute("CREATE TABLE test (smi TEXT)")
            with open(smi_file, "r") as f:
                parallel_calculation(f, smiles_to_list, DuckDBWriter, data_writer_args=(db_file, ps,), num_workers=1)
            data = con.sql("SELECT * from test").fetchall()
            self.assertTrue(len(data) == 4, f"Expected 4 rows in database but got {len(data)}.")
            self.assertEqual("CCC", data[0][0])
        finally:
            con.close()
            os.remove(db_file)

    def test_duckdb_parallel_descriptors(self):
        smi_file = r"files/simple_test.smi"
        db_file = "files/test.db"
        ps = build_prepared_stmnt()
        con = duckdb.connect(database=db_file)
        try:
            con.execute(build_create_table_stmt())
            with open(smi_file, "r") as f:
                parallel_calculation(f, calc_descriptors_smiles, DuckDBWriter, data_writer_args=(db_file, ps,), num_workers=1)
            data = con.sql("SELECT * from descriptors").fetchall()
            self.assertTrue(len(data) == 4, f"Expected 4 rows in database but got {len(data)}.")
            self.assertEqual("CCC", data[0][0])
            self.assertAlmostEqual(2.125, data[0][1])
            self.assertEqual(0, data[0][210])
        finally:
            con.close()
            os.remove(db_file)


def smiles_to_rdkit(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def smiles_to_list(smiles: str):
    return [smiles.strip()]


def build_create_table_stmt():
    stmnt = "CREATE TABLE descriptors (smi TEXT, "
    desc_names = [desc[0] for desc in Descriptors._descList]
    r = ", ".join([name + " DOUBLE" for name in desc_names])
    stmnt += (r + ")")
    return stmnt


def build_prepared_stmnt():
    stmnt = "INSERT INTO descriptors VALUES (?, "
    desc_names = [desc[0] for desc in Descriptors._descList]
    r = ", ".join(["?" for n in desc_names])
    stmnt += (r + ")")
    return stmnt


def calc_descriptors_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    desc = Descriptors.CalcMolDescriptors(mol)
    result = [smiles.strip()]
    result.extend([x for x in desc.values()])
    return result