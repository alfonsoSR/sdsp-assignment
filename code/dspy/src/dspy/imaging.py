import numpy as np
from . import processing as dproc, io as dio
from typing import Literal


class ImageGenerator:

    CALCULATOR_BASE: str = "calculate_pixel_value_"

    def __init__(
        self,
        normalized_antenna_coordinates: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> None:

        self.antenna_coords = normalized_antenna_coordinates
        self.correlation = correlation_matrix
        self.inv_correlation = np.linalg.inv(self.correlation)
        self.n_antennas = self.antenna_coords.shape[1]

        return None

    def calculate_pixel_value_dirty(self, l: float, m: float) -> complex:

        # Calculate pointing (direction
        pointing_vector = dproc.calculate_pointing_vector(l, m)
        if pointing_vector is None:
            return np.nan

        # Calculate array response and conjugate
        array_response = dproc.calculate_array_response_vector(
            self.antenna_coords, pointing_vector
        )
        array_response_conj = np.conjugate(array_response)

        # Compute intensity for current pixel
        return (
            array_response_conj[None, :]
            @ (self.correlation @ array_response[:, None])
        )[0][0]

    def calculate_pixel_value_mvdr(self, l: float, m: float) -> complex:

        # Calculate pointing (direction
        pointing_vector = dproc.calculate_pointing_vector(l, m)
        if pointing_vector is None:
            return np.nan

        # Calculate array response and conjugate
        array_response = dproc.calculate_array_response_vector(
            self.antenna_coords, pointing_vector
        )
        array_response_conj = np.conjugate(array_response)

        # Compute intensity for current pixel
        return (
            1.0
            / (
                array_response_conj[None, :]
                @ (self.inv_correlation @ array_response[:, None])
            )[0][0]
        )

    def calculate_pixel_value_aar(self, l: float, m: float) -> complex:

        # Calculate pointing (direction
        pointing_vector = dproc.calculate_pointing_vector(l, m)
        if pointing_vector is None:
            return np.nan

        # Calculate array response and conjugate
        array_response = dproc.calculate_array_response_vector(
            self.antenna_coords, pointing_vector
        )
        array_response_conj = np.conjugate(array_response)

        # Compute intensity for current pixel
        denominator_sqrt = (
            array_response_conj[None, :]
            @ self.inv_correlation
            @ self.inv_correlation
            @ array_response[:, None]
        )[0][0]
        numerator = (
            array_response_conj[None, :]
            @ self.inv_correlation
            @ array_response[:, None]
        )[0][0]

        return numerator / (denominator_sqrt * denominator_sqrt)

    def generate_image(
        self,
        method: Literal["dirty", "mvdr", "aar"],
        l_grid_vector: np.ndarray,
        m_grid_vector: np.ndarray,
    ) -> "dio.RadioImage":

        # Sanity check on method name
        if not hasattr(self, f"{self.CALCULATOR_BASE}{method}"):
            raise ValueError(f"Invalid method: {method}")

        # Initialize container for image
        image = np.zeros(
            (len(l_grid_vector), len(m_grid_vector)), dtype=complex
        )

        # Loop over grid and compute image
        for i, li in enumerate(l_grid_vector):
            for j, mi in enumerate(m_grid_vector):
                image[i][j] = getattr(self, f"{self.CALCULATOR_BASE}{method}")(
                    li, mi
                )

        return dio.RadioImage(l_grid_vector, m_grid_vector, image)
