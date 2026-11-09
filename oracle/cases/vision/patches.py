"""Image patches and pixel graphs against sklearn.feature_extraction.image: all the windows of an image (grey and
colour), their reconstruction by averaging (from perturbed patches too), the patch extractor over a batch, and the
weighted pixel-grid graph (as a dense matrix) of an image, of a volume and under a mask."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.feature_extraction.image import (
    PatchExtractor,
    extract_patches_2d,
    img_to_graph,
    reconstruct_from_patches_2d,
)

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(8)
    out: list[Case] = []
    grey = g.uniform(size=(7, 9))
    colour = g.uniform(size=(6, 8, 3))
    for name, image, size in [
        ("grey", grey, (3, 4)),
        ("colour", colour, (2, 3)),
        ("whole", grey, (7, 9)),
        ("column", grey, (7, 1)),
    ]:
        patches = extract_patches_2d(image, size)
        noisy = patches + 0.1 * g.normal(size=patches.shape)
        # scikit-learn divides by min(i + 1, p, size - i), which counts too many windows when a patch is more than
        # half the image; the rebuilt images are compared only where its formula is right.
        exact = size[0] * 2 <= image.shape[0] and size[1] * 2 <= image.shape[1]
        out.append(
            case(
                f"extract/{name}",
                "extract",
                [image],
                {
                    "patches": patches,
                    **({"rebuilt": reconstruct_from_patches_2d(patches, image.shape)} if exact else {}),
                },
                rtol=1e-12,
                atol=1e-12,
                patchHeight=size[0],
                patchWidth=size[1],
            )
        )
        if not exact:
            continue
        out.append(
            case(
                f"reconstruct/{name}",
                "reconstruct",
                [noisy],
                {"image": reconstruct_from_patches_2d(noisy, image.shape)},
                rtol=1e-12,
                atol=1e-12,
                height=image.shape[0],
                width=image.shape[1],
            )
        )
    images = g.uniform(size=(3, 6, 7))
    out.append(
        case(
            "extractor/grey",
            "extractor",
            [images],
            {"patches": PatchExtractor(patch_size=(3, 2)).transform(images)},
            patchHeight=3,
            patchWidth=2,
        )
    )
    colour_batch = g.uniform(size=(2, 5, 5, 3))
    out.append(
        case(
            "extractor/colour",
            "extractor",
            [colour_batch],
            {"patches": PatchExtractor(patch_size=(2, 2)).transform(colour_batch)},
            patchHeight=2,
            patchWidth=2,
        )
    )
    default = g.uniform(size=(2, 20, 30))
    out.append(
        case(
            "extractor/default-size",
            "extractor",
            [default],
            {"patches": PatchExtractor().transform(default)},
            patchHeight=0,
            patchWidth=0,
        )
    )
    volume = g.uniform(size=(3, 4, 2))
    mask = g.uniform(size=(5, 6)) > 0.35
    image = g.uniform(size=(5, 6))
    for name, img, m in [
        ("image", image, None),
        ("volume", volume, None),
        ("masked", image, mask),
        ("tiny", np.array([[1.0, 3.0], [2.0, 7.0]]), None),
    ]:
        dense = img_to_graph(img, mask=m, return_as=np.ndarray)  # pyright: ignore[reportArgumentType]
        inputs = [img] if m is None else [img, m.astype(np.float64)]
        out.append(
            case(f"graph/{name}", "graph", inputs, {"matrix": dense}, rtol=1e-12, atol=1e-12, masked=m is not None)
        )
    return out
