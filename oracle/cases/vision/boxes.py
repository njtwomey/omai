"""Boxes against torchvision (box_convert, box_area, clip_boxes_to_image, remove_small_boxes, nms, batched_nms, the
anchor generator, BoxCoder and Matcher, all in float64 except the float32 anchor generator) and OpenCV's soft NMS
(cv2.dnn.softNMSBoxes on integer xywh rectangles)."""

from importlib.metadata import version

import cv2
import numpy as np
import torch
import torchvision
import torchvision.ops as ops
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from torchvision.models.detection._utils import BoxCoder, Matcher
from torchvision.models.detection.anchor_utils import AnchorGenerator

REFERENCE: dict[str, str] = {
    "torchvision": torchvision.__version__,
    "opencv-python-headless": version("opencv-python-headless"),
}

EXACT = {"rtol": 1e-12, "atol": 1e-12}


def random_boxes(g: np.random.Generator, n: int, size: float = 50.0) -> FloatArray:
    xy = g.uniform(0, size, (n, 2))
    wh = g.uniform(2, size / 3, (n, 2))
    return np.concatenate([xy, xy + wh], 1)


def t64(x: FloatArray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))


def cases() -> list[Case]:
    g = rng(4390)
    out: list[Case] = []
    boxes = random_boxes(g, 12)
    for a, b in [("xyxy", "xywh"), ("xyxy", "cxcywh"), ("xywh", "cxcywh"), ("cxcywh", "xyxy"), ("xywh", "xyxy")]:
        out.append(
            case(
                f"boxConvert/{a}/{b}",
                "boxConvert",
                [boxes],
                ops.box_convert(t64(boxes), a, b).numpy(),
                **EXACT,
                source=a,
                target=b,
            )
        )
    out.append(case("boxArea", "boxArea", [boxes], ops.box_area(t64(boxes)).numpy(), **EXACT))
    out.append(
        case(
            "clipBoxes",
            "clipBoxes",
            [boxes - 10],
            ops.clip_boxes_to_image(t64(boxes - 10), (30, 40)).numpy(),
            **EXACT,
            size=[30, 40],
        )
    )
    kept = ops.remove_small_boxes(t64(boxes), 8.0).numpy()
    out.append(case("removeSmallBoxes", "removeSmallBoxes", [boxes], kept.astype(np.float64), **EXACT, minSize=8.0))
    crowd = random_boxes(g, 40, 30.0)
    scores = g.permutation(40) / 40 + 0.01
    classes = g.integers(0, 3, 40)
    for thr in [0.3, 0.5, 0.7]:
        keep = ops.nms(t64(crowd), t64(scores), thr).numpy()
        out.append(case(f"nms/{thr}", "nms", [crowd, scores], keep.astype(np.float64), **EXACT, iouThreshold=thr))
        bkeep = ops.batched_nms(t64(crowd), t64(scores), torch.from_numpy(classes), thr).numpy()
        out.append(
            case(
                f"batchedNms/{thr}",
                "batchedNms",
                [crowd, scores, classes.astype(np.float64)],
                bkeep.astype(np.float64),
                **EXACT,
                iouThreshold=thr,
            )
        )
    ints = np.rint(crowd).astype(np.int64)
    rects = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in ints]
    s32 = scores.astype(np.float32)
    for method, code, sigma, thr in [
        ("gaussian", cv2.dnn.SoftNMSMethod_SOFTNMS_GAUSSIAN, 0.5, 0.3),
        ("linear", cv2.dnn.SoftNMSMethod_SOFTNMS_LINEAR, 0.5, 0.3),
        ("gaussian", cv2.dnn.SoftNMSMethod_SOFTNMS_GAUSSIAN, 0.2, 0.3),
    ]:
        updated, idx = cv2.dnn.softNMSBoxes(rects, s32.tolist(), 0.05, thr, 0, sigma, code)
        enc = np.concatenate([np.asarray(idx, np.float64).ravel(), np.asarray(updated, np.float64).ravel()])
        out.append(
            case(
                f"softNms/{method}/{sigma}",
                "softNms",
                [ints.astype(np.float64), s32.astype(np.float64)],
                enc,
                rtol=1e-6,
                atol=1e-7,
                method=method,
                sigma=sigma,
                iouThreshold=thr,
                scoreThreshold=0.05,
            )
        )
    sizes, ratios = (16, 32, 64), (0.5, 1.0, 2.0)
    gen = AnchorGenerator(sizes=(sizes,), aspect_ratios=(ratios,))
    base = gen.generate_anchors(list(sizes), list(ratios)).numpy().astype(np.float64)
    out.append(case("generateAnchors", "generateAnchors", [], base, **EXACT, sizes=list(sizes), ratios=list(ratios)))
    grid = gen.grid_anchors([[3, 4]], [[torch.tensor(8), torch.tensor(16)]])[0].numpy().astype(np.float64)
    out.append(
        case("gridAnchors", "gridAnchors", [base], grid, **EXACT, gridSize=[3, 4], stride=[8, 16]),
    )
    refs = random_boxes(g, 10)
    gts = refs + g.normal(0, 2, refs.shape)
    gts[:, 2:] = np.maximum(gts[:, 2:], gts[:, :2] + 1)
    for weights in [(1.0, 1.0, 1.0, 1.0), (10.0, 10.0, 5.0, 5.0)]:
        coder = BoxCoder(weights)
        codes = coder.encode_single(t64(gts), t64(refs)).numpy()
        out.append(
            case(
                f"encodeBoxes/{weights[0]}",
                "encodeBoxes",
                [gts, refs],
                codes,
                rtol=1e-10,
                atol=1e-12,
                weights=list(weights),
            )
        )
        noisy = codes + g.normal(0, 0.1, codes.shape)
        noisy[0, 2] = 9.0
        dec = coder.decode_single(t64(noisy), t64(refs)).numpy()
        out.append(
            case(
                f"decodeBoxes/{weights[0]}",
                "decodeBoxes",
                [noisy, refs],
                dec,
                rtol=1e-10,
                atol=1e-10,
                weights=list(weights),
            )
        )
    targets = random_boxes(g, 4)
    anchors = np.concatenate([targets + g.normal(0, 1.5, targets.shape), random_boxes(g, 16)])
    iou = ops.box_iou(t64(targets), t64(anchors))
    for allow in [False, True]:
        m = Matcher(0.7, 0.3, allow_low_quality_matches=allow)(iou.clone()).numpy()
        out.append(
            case(
                f"matchAnchors/{allow}",
                "matchAnchors",
                [targets, anchors],
                m.astype(np.float64),
                **EXACT,
                allowLowQuality=allow,
            )
        )
    return out
