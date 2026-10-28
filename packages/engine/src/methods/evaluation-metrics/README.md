# Evaluation metrics (part 4, principles of learning)

Metrics understood by moving them: classification metrics and their curves (ROC, precision-recall), probabilistic metrics, regression, correlation and agreement, distances, forecasting, ranking, clustering, detection and segmentation, image quality, generative and text metrics, as sessions over `metrics` with a score or prediction slot.

## Capabilities

- Threshold metrics: every classification metric as a threshold sweeps over scores (`thresholdSweep` exists), with class imbalance as the knob.
- Curves: ROC and precision-recall traced point by point, AUC accumulating, the effect of prevalence.
- Probabilistic metrics: log loss, Brier and their decompositions; proper scoring rules shown by the forecaster's incentive.
- Regression and agreement: residual-based metrics, correlation measures on a scatter being perturbed.
- Ranking and retrieval: precision at k, NDCG, MAP computed as a ranking is shuffled.
- Clustering: internal and external indices as a clustering is perturbed.
- Detection and segmentation: IoU, mAP as boxes move; image quality (PSNR, SSIM) as noise grows.

## Examples to make

- ROC drawn point by point as the threshold descends, with the confusion matrix updating (`ContingencyTable` exists).
- Prevalence as the knob: precision-recall shifts while ROC stays.
- Brier decomposition: calibration against resolution on a forecaster that is sharpened or flattened.
- NDCG as one relevant item is dragged up the ranking.

## Rendering

- Threshold explorer (exists), curves, confusion tables, scatter plots with draggable points (`interaction`).
- Missing: a draggable ranking strip; a box-overlap picture for IoU.

## Shared

- `shared/`: the score slot (a model's scores, or synthetic scores from two overlapping distributions with separation and prevalence as knobs).

## Problems

- New chapter; `trust/` calibration pieces split between here and `decision-theory-and-calibration`. Most pages are interactive rather than iterative: the session is one step and the knobs do the work.
