"""Box overlaps against torchvision.ops, COCO mean average precision against pycocotools' COCOeval, and the PASCAL VOC
average precision of py-faster-rcnn's voc_eval (the standard reference implementation, without the +1 pixel
convention).
"""

import contextlib
import io
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import torchvision
import torchvision.ops as ops
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REFERENCE: dict[str, str] = {"torchvision": torchvision.__version__, "pycocotools": version("pycocotools")}

STATS = [
    "map",
    "map50",
    "map75",
    "mapSmall",
    "mapMedium",
    "mapLarge",
    "mar1",
    "mar10",
    "mar100",
    "marSmall",
    "marMedium",
    "marLarge",
]


def random_boxes(g: np.random.Generator, n: int, scale: float) -> FloatArray:
    xy = g.uniform(0, 200, size=(n, 2))
    wh = g.uniform(4, scale, size=(n, 2))
    return np.concatenate([xy, xy + wh], axis=1)


def coco_stats(images: list[dict[str, Any]]) -> FloatArray:
    gt: dict[str, Any] = {"images": [], "annotations": [], "categories": [{"id": c} for c in [1, 2, 3]]}
    dt: list[dict[str, Any]] = []
    ann = 1
    for i, im in enumerate(images):
        gt["images"].append({"id": i + 1})
        for box, label, crowd in zip(im["gt_boxes"], im["gt_labels"], im["crowd"], strict=True):
            w, h = box[2] - box[0], box[3] - box[1]
            gt["annotations"].append(
                {
                    "id": ann,
                    "image_id": i + 1,
                    "category_id": int(label),
                    "bbox": [box[0], box[1], w, h],
                    "area": w * h,
                    "iscrowd": int(crowd),
                }
            )
            ann += 1
        for box, label, score in zip(im["dt_boxes"], im["dt_labels"], im["scores"], strict=True):
            dt.append(
                {
                    "image_id": i + 1,
                    "category_id": int(label),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": float(score),
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        setattr(coco, "dataset", gt)  # noqa: B010 - pycocotools types the attribute too narrowly
        coco.createIndex()
        res = coco.loadRes(cast(Any, dt))  # loadRes takes a list of results or a file name
        ev = COCOeval(coco, res, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return np.asarray(ev.stats, dtype=np.float64)


def voc_ap(images: list[dict[str, Any]], cls: int, eleven: bool) -> float:
    """py-faster-rcnn's voc_eval for one class, with continuous box coordinates."""
    recs = []
    npos = 0
    for im in images:
        keep = im["gt_labels"] == cls
        diff = im["difficult"][keep].astype(bool)
        recs.append({"boxes": im["gt_boxes"][keep], "difficult": diff, "det": np.zeros(keep.sum(), dtype=bool)})
        npos += int((~diff).sum())
    dets = [
        (i, s, b)
        for i, im in enumerate(images)
        for b, s, lab in zip(im["dt_boxes"], im["scores"], im["dt_labels"], strict=True)
        if lab == cls
    ]
    order = np.argsort([-s for _, s, _ in dets], kind="stable")
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for d, k in enumerate(order):
        i, _, bb = dets[k]
        r = recs[i]
        ovmax, jmax = -np.inf, -1
        if len(r["boxes"]):
            gb = r["boxes"]
            iw = np.maximum(np.minimum(gb[:, 2], bb[2]) - np.maximum(gb[:, 0], bb[0]), 0)
            ih = np.maximum(np.minimum(gb[:, 3], bb[3]) - np.maximum(gb[:, 1], bb[1]), 0)
            inter = iw * ih
            uni = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) - inter
            ov = inter / uni
            ovmax, jmax = float(np.max(ov)), int(np.argmax(ov))
        if ovmax > 0.5:
            if not r["difficult"][jmax]:
                if not r["det"][jmax]:
                    tp[d] = 1
                    r["det"][jmax] = True
                else:
                    fp[d] = 1
        else:
            fp[d] = 1
    fp, tp = np.cumsum(fp), np.cumsum(tp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    if eleven:
        return float(sum((np.max(prec[rec >= t]) if np.sum(rec >= t) else 0) / 11 for t in np.arange(0.0, 1.1, 0.1)))
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def cases() -> list[Case]:
    g = rng(67)
    a, b = random_boxes(g, 5, 60), random_boxes(g, 4, 60)
    out: list[Case] = [
        case("boxIou", "boxIou", [a, b], to_numpy(ops.box_iou(to_torch(a), to_torch(b)))),
        case(
            "generalizedBoxIou",
            "generalizedBoxIou",
            [a, b],
            to_numpy(ops.generalized_box_iou(to_torch(a), to_torch(b))),
        ),
        case(
            "distanceBoxIou",
            "distanceBoxIou",
            [a, b],
            to_numpy(ops.distance_box_iou(to_torch(a), to_torch(b))),
            rtol=1e-10,
        ),
        case(
            "completeBoxIou",
            "completeBoxIou",
            [a, b],
            to_numpy(ops.complete_box_iou(to_torch(a), to_torch(b))),
            rtol=1e-10,
        ),
    ]
    images: list[dict[str, Any]] = []
    for _ in range(4):
        ng = int(g.integers(3, 7))
        gt_boxes = random_boxes(g, ng, 120)
        gt_labels = g.integers(1, 4, size=ng)
        crowd = (g.uniform(size=ng) < 0.1).astype(np.int64)
        difficult = (g.uniform(size=ng) < 0.15).astype(np.int64)
        keep = g.uniform(size=ng) < 0.8
        jitter = gt_boxes[keep] + g.normal(scale=4, size=(int(keep.sum()), 4))
        jitter[:, 2:] = np.maximum(jitter[:, 2:], jitter[:, :2] + 1)
        extra = random_boxes(g, 3, 80)
        dt_boxes = np.concatenate([jitter, extra])
        dt_labels = np.concatenate([gt_labels[keep], g.integers(1, 4, size=3)])
        scores = g.uniform(size=len(dt_boxes))
        images.append(
            {
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "crowd": crowd,
                "difficult": difficult,
                "dt_boxes": dt_boxes,
                "dt_labels": dt_labels,
                "scores": scores,
            }
        )
    params: dict[str, Any] = {"images": [{k: v.tolist() for k, v in im.items()} for im in images]}
    stats = coco_stats(images)
    for name, value in zip(STATS, stats, strict=True):
        out.append(
            case(f"detectionMeanAveragePrecision/coco/{name}", "cocoStat", [], np.asarray(value), stat=name, **params)
        )
    for protocol, eleven in [("voc", False), ("voc2007", True)]:
        classes = sorted({int(c) for im in images for c in im["gt_labels"]})
        aps = [voc_ap(images, c, eleven) for c in classes]
        out.append(
            case(
                f"detectionMeanAveragePrecision/{protocol}",
                "vocMap",
                [],
                np.asarray(np.mean(aps)),
                protocol=protocol,
                **params,
            )
        )
    return out
