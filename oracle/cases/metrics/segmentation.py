"""Segmentation metrics against MONAI (Dice, Hausdorff distance and HD95 of mask boundaries), scikit-learn (macro
Jaccard over pixels) and torchmetrics (panoptic quality); the boundary F1 from its definition over scipy's distance
transform.
"""

from importlib.metadata import version

import monai
import numpy as np
import scipy.ndimage as ndi
import sklearn.metrics as M
import torch
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.metrics.meandice import compute_dice
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from torchmetrics.functional.detection import panoptic_quality

REFERENCE: dict[str, str] = {"monai": monai.__version__, "torchmetrics": version("torchmetrics")}

# MONAI computes in float32.
F32 = 1e-6


def as_batch(m: FloatArray) -> torch.Tensor:
    return torch.tensor(m[None, None], dtype=torch.float64)


def blob(g: np.random.Generator, shape: tuple[int, int], centre: tuple[float, float], radius: float) -> FloatArray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    r = np.hypot(yy - centre[0], xx - centre[1]) + g.uniform(-0.8, 0.8, size=shape)
    return (r < radius).astype(np.float64)


def boundary(m: FloatArray) -> FloatArray:
    return np.logical_xor(m > 0, ndi.binary_erosion(m > 0)).astype(np.float64)


def boundary_f1(p: FloatArray, t: FloatArray, tol: float) -> float:
    bp, bt = boundary(p), boundary(t)
    dt_t = ndi.distance_transform_edt(1 - bt)
    dt_p = ndi.distance_transform_edt(1 - bp)
    precision = float((np.asarray(dt_t)[bp > 0] <= tol).mean())
    recall = float((np.asarray(dt_p)[bt > 0] <= tol).mean())
    return 2 * precision * recall / (precision + recall)


def cases() -> list[Case]:
    g = rng(68)
    p = blob(g, (24, 28), (11.0, 13.0), 7.5)
    t = blob(g, (24, 28), (12.0, 14.5), 7.0)
    labels_t = g.integers(0, 4, size=(10, 12)).astype(np.float64)
    labels_p = np.where(g.uniform(size=(10, 12)) < 0.7, labels_t, g.integers(0, 4, size=(10, 12))).astype(np.float64)
    out: list[Case] = [
        case(
            "diceCoefficient",
            "diceCoefficient",
            [p, t],
            np.asarray(float(compute_dice(as_batch(p), as_batch(t), include_background=True)[0, 0])),
            rtol=F32,
        ),
        case("maskIou", "maskIou", [p, t], np.asarray(float(((p > 0) & (t > 0)).sum() / ((p > 0) | (t > 0)).sum()))),
        case(
            "maskHausdorffDistance",
            "maskHausdorffDistance",
            [p, t],
            np.asarray(float(compute_hausdorff_distance(as_batch(p), as_batch(t), include_background=True)[0, 0])),
            rtol=F32,
        ),
        case(
            "maskHausdorffDistance/hd95",
            "maskHausdorffDistance",
            [p, t],
            np.asarray(
                float(
                    compute_hausdorff_distance(as_batch(p), as_batch(t), include_background=True, percentile=95)[0, 0]
                )
            ),
            percentile=95,
            rtol=F32,
        ),
        case(
            "maskHausdorffDistance/directed",
            "maskHausdorffDistance",
            [p, t],
            np.asarray(
                float(
                    compute_hausdorff_distance(as_batch(p), as_batch(t), include_background=True, directed=True)[0, 0]
                )
            ),
            directed=True,
            rtol=F32,
        ),
        case(
            "meanIou",
            "meanIou",
            [labels_p, labels_t],
            np.asarray(M.jaccard_score(labels_t.ravel(), labels_p.ravel(), average="macro")),
        ),
        case("boundaryF1", "boundaryF1", [p, t], np.asarray(boundary_f1(p, t, 2.0))),
        case("boundaryF1/tolerance1", "boundaryF1", [p, t], np.asarray(boundary_f1(p, t, 1.0)), tolerance=1.0),
    ]
    # Panoptic maps: things 0 and 1 with instances, stuff 6, and an unknown category 9 that becomes void.
    cat_t = np.where(t > 0, 0, np.where(p > 0, 1, 6))
    inst_t = np.where(cat_t == 0, 1, np.where(cat_t == 1, (np.arange(28)[None, :] // 14) + 1, 0))
    cat_p = cat_t.copy()
    cat_p[:3, :5] = 9
    inst_p = inst_t.copy()
    inst_p[cat_p == 1] = 1
    pred = np.stack([cat_p, inst_p], axis=-1).astype(np.float64)
    target = np.stack([cat_t, inst_t], axis=-1).astype(np.float64)
    pq = panoptic_quality(
        torch.tensor(pred[None]).long(),
        torch.tensor(target[None]).long(),
        things={0, 1},
        stuffs={6},
        allow_unknown_preds_category=True,
    )
    out.append(
        case(
            "panopticQuality",
            "panopticQuality",
            [pred, target],
            np.asarray(float(pq)),
            things=[0, 1],
            stuffs=[6],
            rtol=1e-7,
        )
    )
    return out
