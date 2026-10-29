from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from dspy import io as dio
import sys

# Paths
SOURCE_DIR = Path(__file__).parent / "images"
OUTDIR = Path(__file__).parent.parent / "report/figures"
OUTDIR.mkdir(exist_ok=True, parents=True)


def estimate_background_noise(
    l_grid: np.ndarray,
    m_grid: np.ndarray,
    image: np.ndarray,
    l_range: tuple[float, float],
    m_range: tuple[float, float],
) -> float:
    """Estimates background noise"""

    # Define mask for image
    l_mask = (l_grid >= l_range[0]) * (l_grid <= l_range[1])
    m_mask = (m_grid >= m_range[0]) * (m_grid <= m_range[1])
    complete_mask = l_mask * m_mask

    return np.std(image[complete_mask])


def estimate_fwhm_and_sidelobe_ratio(source_path: Path) -> tuple[float, float]:

    # Load cropped image
    cropped_data = dio.RadioImage.from_file(source_path)
    l_grid, m_grid = np.meshgrid(
        cropped_data.l_grid_vector, cropped_data.m_grid_vector
    )
    cropped_image_real = np.real(cropped_data.image)

    # Compute peak intensity and indices
    peak_intensity = np.nanmax(cropped_image_real)
    peak_indices = np.where(cropped_image_real == peak_intensity)
    peak_idx = peak_indices[0][0]
    peak_jdx = peak_indices[1][0]

    # Calculate distance from each pixel to peak
    peak_coords = np.array(
        [l_grid[peak_idx][peak_jdx], m_grid[peak_idx][peak_jdx]]
    )[:, None, None]
    grid_coords = np.array([l_grid, m_grid])
    distance_grid = np.linalg.norm(grid_coords - peak_coords, axis=0)

    # Find FWHM
    above_half_mask = np.array(
        cropped_image_real >= 0.5 * peak_intensity, dtype=int
    )
    distance_grid_masked = distance_grid * above_half_mask
    fwhm = 2.0 * np.max(distance_grid_masked)

    # Compute relative intensity of the sidelobe
    sidelobe_pixel: list[int] = [186, 536]
    sidelobe_ratio = (
        cropped_image_real[sidelobe_pixel[0]][sidelobe_pixel[1]]
        / peak_intensity
    )

    # setup = ng.PlotSetup(aspect="equal")
    # with ng.SingleAxis(setup) as fig:

    #     fig.imshow(image_external_masked)

    # # Find peak in cropped image
    # side_lobe_peak = np.nanmax(image_external_masked)
    # side_lobe_peak_indices = np.where(image_external_masked == side_lobe_peak)
    # peak_idx = side_lobe_peak_indices[0][0]
    # peak_jdx = side_lobe_peak_indices[1][0]

    # print(side_lobe_peak_indices)

    # fig, ax = plt.subplots()
    # pcm = ax.pcolor(l_grid, m_grid, image_external_masked)
    # ax.set_title(f"Dirty image: {method.upper()}")
    # ax.set_xlabel(r"East $\longleftarrow$ l $\longrightarrow$ West")
    # ax.set_ylabel(r"South $\longleftarrow$ m $\longrightarrow$ North")
    # cbar = fig.colorbar(pcm, ax=ax)
    # cbar.set_label(
    #     r"$10\cdot \log_{10}\left(\text{I}/\text{I}_{max}\right)$ [dB]"
    # )
    # plt.show()

    # setup = ng.PlotSetup(aspect="equal")
    # with ng.SingleAxis(setup) as fig:
    #     fig.imshow(image_external_masked, cmap="viridis")

    return fwhm, sidelobe_ratio


if __name__ == "__main__":

    # Select method
    method: str = sys.argv[-1]

    # Load figure data
    data = dio.RadioImage.from_file(SOURCE_DIR / f"{method}_image.pkl")

    # Define grid and get real component of intensity
    l_mesh, m_mesh = np.meshgrid(data.l_grid_vector, data.m_grid_vector)
    image_real = np.real(data.image)

    # Filter out nan values
    mask = ~np.isnan(image_real)
    image_filtered = image_real[mask]

    # Find peak intensity in image
    peak_intensity = np.max(image_filtered)
    image_normalized = 10.0 * np.log10(image_real / peak_intensity)

    # Load cropped data
    cropped_data = dio.RadioImage.from_file(
        SOURCE_DIR / f"{method}_cropped_image.pkl"
    )
    l_cropped, m_cropped = np.meshgrid(
        cropped_data.l_grid_vector, cropped_data.m_grid_vector
    )
    image_cropped = np.real(cropped_data.image)
    log_image_cropped = 10.0 * np.log10(image_cropped / peak_intensity)

    # Estimate background noise
    background_noise = estimate_background_noise(
        l_mesh, m_mesh, image_real, (-0.25, 0.25), (0, 0.5)
    )

    # Calculate estimated SNR
    snr = peak_intensity / background_noise

    # Estimate sharpness of peaks
    fwhm, sidelobe_ratio = estimate_fwhm_and_sidelobe_ratio(
        SOURCE_DIR / f"{method}_cropped_image.pkl"
    )

    print(f"SNR: {snr:.2f}")
    print(f"FWHM: {fwhm:.2e}")
    print(r"$I_{side}/I_{max}$" + f": {sidelobe_ratio:.4f}")

    # Define figure title
    title = (
        f"{method.upper()} dirty image: "
        + f"SNR = {snr:.2f}, FWHM = {fwhm:.2e}"
    )

    fig, axes = plt.subplots(1, 2, layout="tight", figsize=(7.5, 3))
    # fig.suptitle(title)

    # Plot image on left side
    left = axes[0]
    left.set_aspect("equal")
    pcm = left.pcolor(
        l_mesh,
        m_mesh,
        image_normalized,
    )
    left.set_title("Complete field")
    left.set_xlabel(r"East $\longleftarrow$ l $\longrightarrow$ West")
    left.set_ylabel(r"South $\longleftarrow$ m $\longrightarrow$ North")
    cbar = fig.colorbar(pcm, ax=left)
    cbar.set_label(
        r"$10\cdot \log_{10}\left(\text{I}/\text{I}_{max}\right)$ [dB]"
    )

    # Plot cropped image on right side
    right = axes[1]
    right.set_aspect("equal")
    rpcm = right.pcolor(
        l_cropped,
        m_cropped,
        log_image_cropped,
    )
    right.set_title("Brightest source")
    right.set_xlabel(r"East $\longleftarrow$ l $\longrightarrow$ West")
    right.set_ylabel(r"South $\longleftarrow$ m $\longrightarrow$ North")
    rcbar = fig.colorbar(rpcm, ax=right)
    rcbar.set_label(
        r"$10\cdot \log_{10}\left(\text{I}/\text{I}_{max}\right)$ [dB]"
    )

    # # Make table on right side
    # table_data = np.array(
    #     [["-", f"{snr:.2f}"], ["-", f"{fwhm:.2f}"]],
    # )
    # right = axes[1]
    # right.axis("off")
    # right.table(
    #     cellText=table_data,
    #     rowLabels=["SNR", "FWHM"],
    #     colLabels=["Unit", "Value"],
    #     loc="center",
    #     cellLoc="center",
    #     rowLoc="center",
    #     colColours=["tab:blue", "tab:blue"],
    #     colWidths=[0.2, 0.8],
    # )

    fig.savefig(OUTDIR / f"{method}-dual.png", dpi=300)
    plt.show()
    # print(snr)

    # fig, ax = plt.subplots()
    # pcm = ax.pcolor(l_mesh, m_mesh, foo)
    # ax.set_title(f"Dirty image: {method.upper()}")
    # ax.set_xlabel(r"East $\longleftarrow$ l $\longrightarrow$ West")
    # ax.set_ylabel(r"South $\longleftarrow$ m $\longrightarrow$ North")
    # cbar = fig.colorbar(pcm, ax=ax)
    # cbar.set_label(
    #     r"$10\cdot \log_{10}\left(\text{I}/\text{I}_{max}\right)$ [dB]"
    # )
    # plt.show()
