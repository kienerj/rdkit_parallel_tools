import gzip
import io
import logging
from collections.abc import Iterable

sdf_extensions = ("sdf", "sd", "SDF", "SD")
logger = logging.getLogger('rdkit_parallel_tools')


def input_to_file(file) -> io.IOBase:
    """
    Handles different input for "file" and returns a file-like object
    Input must be a string pointing to a valid sd-file with a valid extension or gzipped sd-file (.gz) or already be
    an open file-like object (returned as-is)
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
    elif isinstance(file, io.BufferedIOBase):
        # file like object, return as-is
        logger.debug("Found existing BufferedIOBase. Return as-is")
        return file
    else:
        error_message = "Found input which is not a 'str' or valid file-like object."
        logger.error(error_message)
        raise ValueError(error_message)


def raw_sd_reader(file) -> Iterable[str]:
    """
    Read an sd-file but return only raw sd-data as string and not a rdkit molecule instance.

    :param file: a filepath (str) or file-like object
    :return: iterator that returns raw sd-block for each entry
    """
    file = input_to_file(file)
    data = []
    for line in file:
        data.append(line)
        if line.startswith("$$$$"):
            yield "".join(data)
            data = []
    file.close()

def chunked_raw_sd_reader(file, num_mols: int = 100) -> Iterable[str]:
    """
    Read a sd-file but return only raw sd-data as string in chunks of "num_mols"

    :param file: a filepath (str) or file-like object
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
