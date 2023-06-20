import os
import logging
import unittest
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

