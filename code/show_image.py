from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from dspy import io as dio
import sys

# Paths
SOURCE_DIR = Path(__file__).parent / "images"
OUTDIR = Path(__file__).parent.parent / "report/figures"
OUTDIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":

    # Select method
    method: str = sys.argv[-1]
    print(f"Selected method: {method}")

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

    fig, ax = plt.subplots()
    pcm = ax.pcolor(l_mesh, m_mesh, image_normalized)
    ax.set_title(f"Dirty image: {method.upper()}")
    ax.set_xlabel(r"East $\longleftarrow$ l $\longrightarrow$ West")
    ax.set_ylabel(r"South $\longleftarrow$ m $\longrightarrow$ North")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(
        r"$10\cdot \log_{10}\left(\text{I}/\text{I}_{max}\right)$ [dB]"
    )
    fig.savefig(OUTDIR / f"{method}.png", dpi=300)
    plt.show()
