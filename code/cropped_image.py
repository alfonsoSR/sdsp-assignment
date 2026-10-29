from pathlib import Path
import numpy as np
from dspy import io as dio, processing as dproc, imaging as dimag
from typing import Literal

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "lofar_data"
OUTDIR = BASE_DIR / "images"


if __name__ == "__main__":

    # Select method
    image_generation_method: Literal["dirty", "mvdr", "aar"] = "dirty"

    # Log method
    print(f"Generating image using: {image_generation_method.upper()}")

    # Load data from source file
    source_file = DATA_DIR / "lofar_DSP_data_1.mat"
    data = dio.LofarData.from_mat_file(source_file)

    # Select phase center
    phase_center_idx = 0

    # Calculate normalized antenna coordinates
    antenna_coordinates = data.antenna_coordinates.T
    normalized_antenna_coordinates = dproc.normalize_antenna_coordinates(
        data.central_channel_frequency, data.antenna_coordinates.T
    )

    # Calculate maximum baseline length
    baselines = (
        antenna_coordinates[:, :, None] - antenna_coordinates[:, None, :]
    )
    approx_baseline_lengths = np.linalg.norm(baselines, axis=0)
    max_baseline_length = np.max(approx_baseline_lengths)

    # Specify ranges for l and m
    l_range = (-0.4, -0.3)
    m_range = (0.1, 0.2)

    # Define a grid of image coordinates
    frac_angle = 0.01
    dl = frac_angle * data.central_channel_wavelength / max_baseline_length
    _l = np.arange(0, 1, dl)
    l_vec = np.concatenate((-np.flip(_l[1:]), _l))

    l_mask = (l_vec >= l_range[0]) * (l_vec <= l_range[1])
    m_mask = (l_vec >= m_range[0]) * (l_vec <= m_range[1])

    l_vec_masked = l_vec[l_mask]
    m_vec_masked = l_vec[m_mask]

    # Define correlation matrix and compute inverse
    corr = data.sample_covariance_matrix
    inv_corr = np.linalg.inv(corr)

    # Initialize image generator
    generator = dimag.ImageGenerator(normalized_antenna_coordinates, corr)

    # Generate dirty image
    image = generator.generate_image(
        image_generation_method, l_vec_masked, m_vec_masked
    )

    # Save output`
    image.to_file(OUTDIR / f"{image_generation_method}_cropped_image.pkl")

    # # Initialize containers for dirty beam and image
    # # dirty_beam = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    # # dirty_image = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    # # image = np.zeros((len(l_vec), len(l_vec)), dtype=complex)
    # image_list = []

    # # Loop over pointing directions and compute intensity
    # for i, l in enumerate(l_vec):

    #     print(f"Iteration {i} out of {len(l_vec)}")

    #     current_image_part = []

    #     for mi in l_vec:

    #         current_image_part.append(
    #             calculate_mvdr_image_component(
    #                 (l, mi, antenna_coordinates, corr, inv_corr)
    #             )[1]
    #         )

    #     # with mp.Pool() as pool:

    #     #     # # Dirty image
    #     #     # output = np.array(
    #     #     #     pool.map(
    #     #     #         calculate_dirty_beam_image_component,
    #     #     #         iter([(l, mi, baselines, corr) for mi in l_vec]),
    #     #     #     )
    #     #     # ).T

    #     #     # MVDR
    #     #     output = np.array(
    #     #         pool.map(
    #     #             calculate_mvdr_image_component,
    #     #             iter(
    #     #                 [
    #     #                     (l, mi, antenna_coordinates, corr, inv_corr)
    #     #                     for mi in l_vec
    #     #                 ]
    #     #             ),
    #     #         )
    #     #     ).T

    #     image_list.append(current_image_part)
    #     # dirty_image[i] = np.array(output[1])

    # # Save image to output file
    # # np.save("dirty_image_test.npy", dirty_image)
    # np.save("dirty_image_mvdr.npy", np.array(image_list))

    # # dirty_image = np.load("dirty_image.npy")
    # # dirty_beam = np.load("dirty_beam.npy")

    # print("Finished computing image")
    # l_mesh, m_mesh = np.meshgrid(l_vec, l_vec)
    # dirty_beam_real = np.real(dirty_image)

    # fig, ax = plt.subplots()
    # pcm = ax.pcolor(l_mesh, m_mesh, dirty_beam_real)
    # fig.colorbar(pcm, ax=ax)
    # plt.show()
