import hashlib
import numpy as np


def hash_parameters(parameters):
    """
    Generate a deterministic SHA-256 hash for the
    complete set of model parameters.
    """

    sha256 = hashlib.sha256()

    for parameter in parameters:

        # Ensure consistent memory layout
        parameter = np.ascontiguousarray(parameter)

        # Include metadata
        sha256.update(
            str(parameter.shape).encode("utf-8")
        )

        sha256.update(
            str(parameter.dtype).encode("utf-8")
        )

        # Add parameter bytes
        sha256.update(
            parameter.tobytes()
        )

    return sha256.hexdigest()