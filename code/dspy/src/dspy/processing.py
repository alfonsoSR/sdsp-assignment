import numpy as np
from .constants import CLIGHT


def normalize_antenna_coordinates(
    f_k: float, antenna_coordinates: np.ndarray
) -> np.ndarray:
    """Normalize antenna coordinates

    Calculates normalized version of antenna coordinates as defined in section 3.3 of van der Veen and Wijnholds 2013.

    :param f_k: Center frequency of the RF subband [Hz]
    :param antenna_coordinates: Unnormalized coordinates of the antenna (WHICH FRAME??) [m]
    :return: Normalized coordinates of the station [-]
    """
    normalization_factor = 2.0 * np.pi * f_k / CLIGHT
    return normalization_factor * antenna_coordinates


def calculate_pointing_vector(l: float, m: float) -> np.ndarray | None:

    argument = 1.0 - l**2 - m**2
    if argument < 0:
        return None

    return np.array([l, m, np.sqrt(argument)])


def calculate_array_response_vector(
    normalized_antenna_coordinates: np.ndarray, pointing_vector: np.ndarray
) -> np.ndarray:

    # Calculate argument of complex exponential
    array_response_argument = np.sum(
        normalized_antenna_coordinates * pointing_vector[:, None], axis=0
    )

    return np.exp(1j * array_response_argument)
