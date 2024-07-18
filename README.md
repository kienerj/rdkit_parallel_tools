# RDKit Parallel Tools

Functions to simplify usage of RDKit with larger number of molecules using all the cores on multicore computers. 

A work in progress.

### Example

Calculate all descriptors for all molecules in the sd-file in a parallel and streaming fashion.

Write results to new sd-file with descriptors added as new properties.

```python
from rdkit_parallel_tools import *


def your_custom_fn(sdf: str) -> str:
    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdf)
    res = []
    for mol in suppl:
        # replace below line with any calculation of choice that either
        # calculates properties or modifies the molecule like 3D coords
        desc = Descriptors.CalcMolDescriptors(mol)
        res.append(mol_to_sd(mol, desc))  # convert molecule to a "sdf-string"
    return '\n'.join(res)


# convert different types of input to proper file-like object
sd_file = chem_input_to_file("path/to/in.sdf.gz")

# Reads sdf in streaming fashion, performs desired calculation on all cpu cores 
# and writes results to gzipped sdf.
sd_to_sd_parallel_calculation(sd_file, "out.sdf.gz", calc_func=your_custom_fn)
```

To make this work, the sd-file must be read chunk-wise as text and only inside the workers (e.g. custom function) converted to a rdkit molecule.

## Installation

The suggested approach to try it out is to create a new conda environment from an environment.yml:

```yaml
name: rdkit_parallel
channels:  
  - conda-forge 
dependencies:
  - python>=3.11
  - rdkit>=2023.03.1
  - pip

```

```bash
conda env create -f environment.yml
```

Then clone this repository and install it in [development mode](https://packaging.python.org/tutorials/installing-packages/#installing-from-a-local-src-tree) into this new environment using pip:

```bash
python -m pip install -e c:\path\to\rdkit_parallel_tools
```

If you clone the repo with git (vs downloading), this has the advantage that you can use git to pull new commits. 
You can then immediately use the new version without any further changes.