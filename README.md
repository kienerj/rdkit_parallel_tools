# RDKit Parallel Tools

Functions to simplify usage of all cores on multicore computers. 

A work in progress.

**Example:**

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


# convert different types of input to proper file-like object (io.IOBase)
sd_file = sd_input_to_file("path/to/in.sdf.gz")

sd_to_sd_parallel_calculation(sd_file, "out.sdf.gz", calc_func=your_custom_fn)
# Reads sdf in streaming fashion, performs desired calculation on all cpu cores 
# and writes results to gzipped sdf.
```

To make this work, sd-file must be read chunk-wise as text and only inside the workers (e.g. custom function) converted to a rdkit molecule.