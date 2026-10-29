from pathlib import Path
import numpy as np
from dspy import io as dio, constants as dconst

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "lofar_data"


def normalize_station_coordinates(
    f_k: float, station_coordinates: np.ndarray
) -> np.ndarray:
    """Normalize station coordinates

    Calculates normalized version of station coordinates as defined in section 3.3 of van der Veen and Wijnholds 2013.

    :param f_k: Center frequency of the RF subband [Hz]
    :param station_coordinates: Unnormalized coordinates of the station (WHICH FRAME??) [m]
    :return: Normalized coordinates of the station [-]
    """
    normalization_factor = 2.0 * np.pi * f_k / dconst.CLIGHT
    return normalization_factor * station_coordinates


if __name__ == "__main__":

    # Load data from source file
    source_file = DATA_DIR / "lofar_DSP_data_1.mat"
    data = dio.LofarData.from_mat_file(source_file)

    # Normalize station coordinates
    z_antennas = normalize_station_coordinates(
        data.central_channel_frequency, data.antenna_coordinates
    )

    # Calculate all baselines (in normalized coordinates)
    normalized_baselines = z_antennas[None:, :] - z_antennas[:, None, :]

    # Calculate maximum baseline length
    approx_baseline_lengths = np.linalg.norm(
        data.antenna_coordinates[:, None, :2]
        - data.antenna_coordinates[None, :, :2],
        axis=-1,
    )
    max_baseline_length = np.max(approx_baseline_lengths)

    # Define a grid of image coordinates
    frac_angle = 0.25
    dl = frac_angle * data.central_channel_wavelength / max_baseline_length
    _l = np.arange(0, 1, dl)
    l = np.concatenate((-np.flip(_l[1:]), _l))
    m = l.copy()

    # Calculate pointing vectors
    lv, mv = np.meshgrid(l, m)
    nv = np.sqrt(-(lv * lv) - (mv * mv) + 1.0)
    pv = np.array([lv, mv, nv])

    with ng.Mosaic("ab") as canvas:

        with canvas.subplot() as left:
            left.imshow(lv)

        with canvas.subplot() as right:
            right.imshow(nv)

    exit(0)

    print(data.central_channel_frequency * 1e-6)
    print(data.number_antennas / 6)

    visibility = data.sample_covariance_matrix

    print(visibility.shape)
    print(data.number_antennas)

    setup = ng.PlotSetup(aspect="auto")
    with ng.SingleAxis(setup) as fig:

        fig.imshow(np.imag(visibility))

    #     fig.line(
    #         data.antenna_coordinates.T[0],
    #         data.antenna_coordinates.T[1],
    #         fmt=".",
    #     )

    # exit(0)

    # # What type of data do we have?
    # fig = plt.figure()
    # ax = fig.add_subplot(
    #     1, 1, 1, projection=ccrs.Mercator(min_latitude=-60, max_latitude=80)
    # )
    # ax.set_global()
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="k")
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # geodetic = ccrs.Geodetic()

    # ax.plot(
    #     data.lofar_core_lon,
    #     data.lofar_core_lat,
    #     transform=geodetic,
    #     zorder=5,
    #     marker="o",
    # )

    # plt.show()
