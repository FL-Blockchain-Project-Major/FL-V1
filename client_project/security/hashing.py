import hashlib
import numpy as np


def hash_parameters(parameters):
    """
    Generate a deterministic SHA-256 hash for the
    complete set of model parameters.
    The hash covers: shape, dtype, and raw bytes of every tensor.
    """
    sha256 = hashlib.sha256()
    for parameter in parameters:
        parameter = np.ascontiguousarray(parameter)
        sha256.update(str(parameter.shape).encode("utf-8"))
        sha256.update(str(parameter.dtype).encode("utf-8"))
        sha256.update(parameter.tobytes())
    return sha256.hexdigest()
