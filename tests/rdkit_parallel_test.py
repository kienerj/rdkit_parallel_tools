import os
import logging
import unittest
from rdkit_parallel_tools import *

logger = logging.getLogger("rdkit_parallel_tools")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class RDKitParallelTest(unittest.TestCase):

    NUM_MOLS_CDK2 = 47

    def test_raw_reader(self):
        data_dir = os.environ["CONDA_PREFIX"] + "/Library/share/RDKit/Docs/Book/data"
        cdk2_path = data_dir + "/cdk2.sdf"
        it = raw_sd_reader(cdk2_path)
        self.assertIsNotNone(it, "Iterator was 'None'.")
        rslt = list(it)
        self.assertEqual(len(rslt), self.NUM_MOLS_CDK2,
                         f"Found unexpected number of {len(rslt)} molecules in test sd-file. Expected {self.NUM_MOLS_CDK2}.")
