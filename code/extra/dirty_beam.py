from pathlib import Path
import numpy as np
from dspy import io as dio, constants as dconst

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "lofar_data"


def calculate_dirty_beam_image_component(
    function_input: tuple[float, float, np.ndarray, np.ndarray],
) -> list[complex]:

    li, mi, _baselines, _correlation_matrix = function_input

    argument = 1.0 - li**2 - mi**2
    if argument < 0:
        return [np.nan, np.nan]

    pq_vec = np.array([li, mi, np.sqrt(argument)])
    prod = np.sum(_baselines * pq_vec[:, None, None], axis=0)
    exponential = np.cos(prod) + (1j * np.sin(prod))

    return [np.sum(exponential), np.sum(_correlation_matrix * exponential)]


def calculate_mvdr_image_component(
    function_input: tuple[float, float, np.ndarray, np.ndarray, np.ndarray],
) -> list[complex]:

    # Unpack input
    li, mi, antenna_coords, correlation, inv_correlation = function_input

    # Calculate pointing vector for current direction
    argument = 1.0 - li**2 - mi**2
    if argument < 0:
        return [np.nan, np.nan]
    pq_vec = np.array([li, mi, np.sqrt(argument)])

    # Calculate array response for current direction
    __argument = np.sum(antenna_coords * pq_vec[:, None], axis=0)
    a_vec = np.exp(1j * __argument)
    a_vec_conj = np.exp(-1j * __argument)

    # Compute denominator
    # other = a_vec_conj[None, :] @ (inv_correlation @ a_vec[:, None])
    prueba = a_vec_conj[None, :] @ (correlation @ a_vec[:, None])

    return [np.nan, prueba[0][0]]


def normalize_antenna_coordinates(
    f_k: float, antenna_coordinates: np.ndarray
) -> np.ndarray:
    """Normalize antenna coordinates

    Calculates normalized version of antenna coordinates as defined in section 3.3 of van der Veen and Wijnholds 2013.

    :param f_k: Center frequency of the RF subband [Hz]
    :param antenna_coordinates: Unnormalized coordinates of the antenna (WHICH FRAME??) [m]
    :return: Normalized coordinates of the station [-]
    """
    normalization_factor = 2.0 * np.pi * f_k / dconst.CLIGHT
    return normalization_factor * antenna_coordinates


def normalize_data(data):

    ur = 1
    lr = 0
    _min = np.min(data)
    _max = np.max(data)

    m = (ur - lr) / (_max - _min)
    n = ur - _max * m

    return (m * data) + n


if __name__ == "__main__":

    # Load data from source file
    source_file = DATA_DIR / "lofar_DSP_data_1.mat"
    data = dio.LofarData.from_mat_file(source_file)

    # Select phase center
    phase_center_idx = 0

    # Calculate antenna coordinates
    antenna_coordinates = normalize_antenna_coordinates(
        data.central_channel_frequency, data.antenna_coordinates.T
    )

    # Calculate all baselines
    baselines = (
        antenna_coordinates[:, :, None] - antenna_coordinates[:, None, :]
    )

    # # Normalize baselines
    # # (normalized_baselines)_{i,j} = (z_i - z_j)
    # normalized_baselines = normalize_antenna_coordinates(
    #     data.central_channel_frequency, baselines
    # )

    # Calculate maximum baseline length
    approx_baseline_lengths = np.linalg.norm(baselines, axis=0)
    max_baseline_length = np.max(approx_baseline_lengths)

    # Define a grid of image coordinates
    frac_angle = 0.25
    dl = frac_angle * data.central_channel_wavelength / max_baseline_length
    _l = np.arange(0, 1, dl)
    l_vec = np.concatenate((-np.flip(_l[1:]), _l))

    # Define correlation matrix and compute inverse
    corr = data.sample_covariance_matrix
    inv_corr = np.linalg.inv(corr)

    # Initialize containers for dirty beam and image
    # dirty_beam = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    # dirty_image = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    # image = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    image_list = []

    # Loop over pointing directions and compute intensity
    for i, l in enumerate(l_vec):

        print(f"Iteration {i} out of {len(l_vec)}")

        current_image_part = []

        for mi in l_vec:

            current_image_part.append(
                calculate_mvdr_image_component(
                    (l, mi, antenna_coordinates, corr, inv_corr)
                )[1]
            )

        # with mp.Pool() as pool:

        #     # # Dirty image
        #     # output = np.array(
        #     #     pool.map(
        #     #         calculate_dirty_beam_image_component,
        #     #         iter([(l, mi, baselines, corr) for mi in l_vec]),
        #     #     )
        #     # ).T

        #     # MVDR
        #     output = np.array(
        #         pool.map(
        #             calculate_mvdr_image_component,
        #             iter(
        #                 [
        #                     (l, mi, antenna_coordinates, corr, inv_corr)
        #                     for mi in l_vec
        #                 ]
        #             ),
        #         )
        #     ).T

        image_list.append(current_image_part)
        # dirty_image[i] = np.array(output[1])

    # Save image to output file
    # np.save("dirty_image_test.npy", dirty_image)
    np.save("dirty_image_mvdr.npy", np.array(image_list))

    # # dirty_image = np.load("dirty_image.npy")
    # # dirty_beam = np.load("dirty_beam.npy")

    # print("Finished computing image")
    # l_mesh, m_mesh = np.meshgrid(l_vec, l_vec)
    # dirty_beam_real = np.real(dirty_image)

    # fig, ax = plt.subplots()
    # pcm = ax.pcolor(l_mesh, m_mesh, dirty_beam_real)
    # fig.colorbar(pcm, ax=ax)
    # plt.show()
