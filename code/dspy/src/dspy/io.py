from dataclasses import dataclass
from .constants import CLIGHT
from astropy import time as atime
import numpy as np
from pathlib import Path
from scipy import io as sio
import pickle as pkl


@dataclass
class LofarData:
    """Data structure for LOFAR data

    :param central_channel_frequency: MISSING DOCUMENTATION
    :param lofar_core_lon: MISSING DOCUMENTATION
    :param lofar_core_lat: MISSING DOCUMENTATION
    :param observation_epoch: MISSING DOCUMENTATION
    :param antenna_coordinates: MISSING DOCUMENTATION
    :param sample_covariance_matrix: MISSING DOCUMENTATION
    """

    central_channel_frequency: float
    lofar_core_lon: float
    lofar_core_lat: float
    observation_epoch: atime.Time
    antenna_coordinates: np.ndarray
    sample_covariance_matrix: np.ndarray

    @property
    def number_antennas(self) -> int:
        """Number of antennas involved in observations"""
        return self.antenna_coordinates.shape[0]

    @property
    def central_channel_wavelength(self) -> float:
        """Wavelength representation of the central channel frequency"""
        return CLIGHT / self.central_channel_frequency

    @classmethod
    def from_mat_file(cls, source: Path) -> "LofarData":
        """Initialize from .mat file

        :param source: Path to .mat file
        """

        # Load raw data into dictionary of np.ndarray
        raw_data = sio.loadmat(source)

        # Cast raw data to appropriate types
        central_channel_frequency = float(raw_data["freq"][0][0])
        lofar_core_lon = float(raw_data["lon"][0][0])
        lofar_core_lat = float(raw_data["lat"][0][0])
        antenna_coordinates = np.array(raw_data["poslocal"], dtype=float)
        sample_covariance_matrix = np.array(raw_data["Rh"], dtype=complex)

        # Matlab has this weird format for the dates and I don't want to learn
        # how to work with it. I just check the date is the one I know and
        # hardcode its value
        __observation_epoch = raw_data["t_obs_matlab"][0][0]
        if int(__observation_epoch) != 735061:
            raise ValueError("Unexpected epoch")
        observation_epoch = atime.Time("2012-07-11T22:22:20", scale="utc")

        return cls(
            central_channel_frequency,
            lofar_core_lon,
            lofar_core_lat,
            observation_epoch,
            antenna_coordinates,
            sample_covariance_matrix,
        )


class RadioImage:

    ATTRIBUTES: list[str] = ["l_grid_vector", "m_grid_vector", "image"]

    def __init__(
        self,
        l_grid_vector: np.ndarray,
        m_grid_vector: np.ndarray,
        image: np.ndarray,
    ) -> None:

        self.l_grid_vector = l_grid_vector
        self.m_grid_vector = m_grid_vector
        self.image = image

        return None

    def to_file(self, path: Path) -> None:

        if path.suffix != ".pkl":
            raise ValueError(f"Invalid path suffix: {path.suffix}")

        data_dict = {
            attr: self.__getattribute__(attr) for attr in self.ATTRIBUTES
        }

        with path.open("wb") as buffer:
            pkl.dump(data_dict, buffer)

        return None

    @classmethod
    def from_file(cls, path: Path) -> "RadioImage":

        if path.suffix != ".pkl":
            raise ValueError(f"Invalid path suffix: {path.suffix}")

        with path.open("rb") as buffer:
            data_dict = pkl.load(buffer)

        return cls(**data_dict)
