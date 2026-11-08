"""Total-variation denoising against scikit-image's denoise_tv_chambolle (grey and per channel)."""

from importlib.metadata import version

import numpy as np
import skimage.restoration as R
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.signal_helpers import blocks_image

REFERENCE: dict[str, str] = {"scikit-image": version("scikit-image")}


def cases() -> list[Case]:
    g = rng(4480)
    out: list[Case] = []
    noisy = blocks_image(16, 18) + 0.1 * g.normal(size=(16, 18))
    for weight, eps, iters in [(0.1, 2e-4, 200), (0.3, 1e-5, 50), (0.05, 2e-4, 3)]:
        res = R.denoise_tv_chambolle(noisy, weight=weight, eps=eps, max_num_iter=iters)
        out.append(
            case(
                f"denoiseTvChambolle/{weight}/{iters}",
                "denoiseTvChambolle",
                [noisy],
                res,
                rtol=1e-10,
                atol=1e-12,
                weight=weight,
                eps=eps,
                maxIter=iters,
            )
        )
    rgb = np.stack([noisy, noisy[::-1], noisy[:, ::-1]], -1)
    res = R.denoise_tv_chambolle(rgb, weight=0.15, channel_axis=-1)
    out.append(
        case(
            "denoiseTvChambolle/rgb",
            "denoiseTvChambolle",
            [rgb],
            res,
            rtol=1e-10,
            atol=1e-12,
            weight=0.15,
            eps=2e-4,
            maxIter=200,
        )
    )
    return out
