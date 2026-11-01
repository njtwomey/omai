# Data preparation (part 3, data)

Preparation watched: the catalogue of datasets with their truths and the modifiers that plant a problem, missing values diagnosed and imputed, outliers flagged and treated, scaling and power transforms, encoding of categories and targets, resampling and weighting for imbalance, splitting without leakage, pipelines fitted on the training fold only, and batching and loading, as sessions over `learn` (preprocessing, composition, model selection), `data` (modifiers, splits, batching, loading), `datasets` (the registry) and `stats`. Nearly every step is a one-shot transform, so the sessions are studies with a model downstream.

## Structure

The chapter lives under `data/` (section 3 of `methods/README.md`): `catalogue/` holds the pages over the registries (the dataset page of section 8's exemplars) and `preparation/` the studies, one folder per concern, with the imputer, the scaler, the encoder, the sampler and the splitter each a mode of its row. The planted-problem catalogue that `shift-and-drift` and `data-centric-learning` share lives one level up in `data/shared/`.

```
data/
  catalogue/          the registry browsed and queried, a generator with its truth and knobs, modifiers composed into a recipe
  preparation/
    missing/          missingness planted (MCAR, MAR, MNAR) and imputed (the imputer as a mode), the mechanism diagnosed
    outliers/         contamination planted, flagged by robust rules, treated, the downstream effect
    scaling/          the scaler as a mode and its effect on a fit and on gradient descent's path; power transforms
    encoding/         categorical encoders, target encoding with and without cross-fitting, discretisation
    resampling/       class imbalance met by samplers (the sampler as a mode), class weights, cost-sensitive thresholds
    splitting/        splitters, leakage planted and caught, the pipeline fitted inside each fold
    loading/          batches and epochs, bucketing and padding, remote assets fetched and cached
    shared/           the model slot downstream, the table record (below)
  shared/             the planted-problem catalogue shared by the three data chapters (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Where a transform is one shot, the step is the knob swept (the missing rate, the imbalance ratio, the fold) or the mode compared, and `eval/downstream` is the held-out score of the model slot fitted after it; where a transform iterates (MICE, SMOTE's batches, a profile likelihood over λ) the step is its iteration.

### catalogue/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dataset-browser` | The registry (121 tagged entries today) queried by kind, modality, real or synthetic, form and free text, as the selector does; an entry chosen is generated or fetched, its rows drawn with its truth where it has one, its licence and source shown; the collections as saved queries. The "dataset page" of `methods/README.md` section 8. | `dataset/rows` t!, `dataset/meta` m, `dataset/truth` p (the boundary, the function, the Bayes rate), `query/matches` s | `datasets/registry.ts`: `whereDatasets`, `findDataset`, `loadDataset`, `fetchDataset`, `matchesDatasetQuery`; `datasets/catalogue.ts`: `datasetRegistry`; `datasets/collections.ts`: `datasetCollections`, `loadEntry`; `data/truth.ts`: `completeTruth`, `bayesErrorEstimate` | |
| `generator-and-truth` | A generator's knobs turned (noise, separation, imbalance, n): the data redrawn under the same key, the exact truth beside it (the Bayes posterior over the plane, the regression function, the Bayes error); what a model can at best achieve, the yardstick every downstream page uses. | `data/rows` t!, `truth/bayesError` s, `truth/posterior` p, `data/knobs` m | `data/generator.ts`: `generate`, `describeGenerator`; `data/truth.ts`: `bayesPredict`, `posteriorFromJoint`, `excessRisk`; `datasets/classification.ts`: `moons`, `blobs`, `spirals`; `datasets/make-classification.ts`: `makeClassification` | |
| `recipes` | Modifiers composed on a generator (label noise, missingness, outliers, shift, nuisance and correlated features, imbalance) as a recipe that serialises: each modifier's truth kept (which cells are missing, which labels flipped, the shifted posterior), the recipe encoded and parsed back; the data slot every chapter's planted problem is built from. | `recipe/steps` m, `data/rows` t!, `truth/modified` p, `recipe/encoded` m | `data/recipes.ts`: `makeFromRecipe`, `builtinModifiers`, `encodeRecipe`, `parseRecipe`, `describeRecipe`; `data/modifiers/labels.ts`, `data/modifiers/features.ts`, `data/modifiers/shift.ts` | |

### preparation/missing/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `imputation` | Missingness planted under a mechanism (MCAR, MAR, MNAR as a mode of the data, the rate as the knob), then the imputer as a mode: mean, median, most frequent and constant (one shot), k-nearest neighbours (k as the knob), iterative imputation (MICE) one round per step with the imputed values moving and the change falling; the imputed values against the hidden truth (RMSE on the missing cells), the distribution of imputations against the true one (mean imputation's spike), the missing indicator as an extra feature; the downstream score of each. | `imputer/round` s, `imputer/change` s, `imputer/values` t!, `imputer/rmse` s, `data/mask` t!, `eval/downstream` s, `cells/picture` p (the table with missing and imputed cells coloured) | `data/modifiers/features.ts`: `withMissing` (MCAR, MAR, MNAR), `missingModifier`; `learn/preprocessing/imputation.ts`: `simpleImputer` (mean, median, most_frequent, constant ready), `knnImputer` ready, `iterativeImputer` (an `Algorithm` inside; expose it), `missingIndicator` | |
| `missingness-diagnosis` | Which mechanism is it: the missingness map, missing rates by column, the correlation of the indicators, the distribution of the other columns where a value is missing against where it is not (MAR shows, MCAR does not), Little's test as the formal check. | `missing/rates` t!, `missing/map` t!, `missing/test` s, `missing/byColumn` p | `learn/preprocessing/imputation.ts`: `missingIndicator`; `stats/tests/means.ts`: `ttestInd`; `stats/tests/two-sample.ts`: `ks2samp`; `stats/correlation.ts`: `corrcoef`; TODO `stats/tests/missing.ts`: `littleMcarTest` | |

### preparation/outliers/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `outlier-handling` | Contamination planted (the rate and the distance as knobs); the rule as a mode (z-scores, the IQR rule, the median absolute deviation, Mahalanobis distance under a shrunk covariance), the flags against the planted set (precision and recall as the threshold sweeps); then the treatment as a mode (removed, winsorised, down-weighted) with a model's fit before and after, and the robust estimators (Huber, the biweight) beside the treatments. | `outliers/scores` t!, `outliers/flagged` t!, `outliers/precision` s, `outliers/recall` s, `treatment/fit` p, `eval/downstream` s, `study/threshold` proj | `data/modifiers/features.ts`: `withOutliers`; `datasets/outliers.ts`: `blobsWithOutliers`, `linearOutliers`; `stats/descriptive.ts`: `zscore`; `stats/quantiles.ts`: `iqr`, `medianAbsoluteDeviation`; `distances/distances.ts`: `mahalanobisDistance`; `stats/covariance.ts`: `ledoitWolf`; `stats/robust.ts`: `winsorize`, `trimboth`, `huber`, `tukeyBiweight`, `mEstimateLocation`; `exposition/anomaly.ts`: `anomalyScores` | |
| `see anomaly-detection` | Detectors that learn (isolation forests, LOF, one-class models) are that chapter's; here outliers are a data problem met by rules and robust fits. | | | |

### preparation/scaling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `scalers` | The scaler as a mode (standard, min-max, max-abs, robust, the normaliser, quantile to uniform or normal, the power transform): the columns before and after, fitted on the training fold only; gradient descent's path on unscaled against standardised features with the condition number as the quantity; a distance-based model's neighbours changing with the scale (the scaler is the metric). | `solver/objective` s, `solver/iterate` t!, `solver/path` p (on the contours), `data/condition` s, `data/columns` p (before and after), `eval/downstream` s | `learn/preprocessing/scalers.ts`: `standardScaler`, `minMaxScaler`, `maxAbsScaler`, `robustScaler`, `normalizer`, `quantileTransformer`, `powerTransformer`; `optim/minimize/gradient.ts`: `gradientDescentAlgorithm`; `numerics/linalg/basics.ts`: `cond` | |
| `power-transforms` | Box-Cox and Yeo-Johnson with λ as the step over a grid: the profile likelihood choosing it, the skewness falling, the QQ plot straightening; a model's residuals before and after the target transform. | `lambda/value` s, `lambda/logLikelihood` s, `lambda/skewness` s, `data/transformed` t!, `data/qq` p | `stats/transforms.ts`: `boxcox`, `boxcoxNormmax`, `boxcoxLlf`, `yeojohnson`, `yeojohnsonNormmax`; `learn/compose.ts`: `transformedTargetRegressor`; `stats/descriptive.ts`: `skewness` | |

### preparation/encoding/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `categorical-encoders` | The encoder as a mode (one-hot with its dropped level, ordinal, label, the multi-label binariser, hashing with its bucket count and the collisions it causes): the design matrix's width and sparsity, a linear model's coefficients per level; unseen levels at test time handled or not. | `encoder/width` s, `encoder/collisions` s, `encoder/matrix` p, `model/coefficients` t!, `eval/downstream` s | `learn/preprocessing/encoders.ts`: `oneHotEncoder`, `ordinalEncoder`, `labelEncoder`, `labelBinarizer`, `multiLabelBinarizer`, `fitCategories`; `text/features.ts`: `hashingVectorize`; `text/hashing.ts`: `murmurhash3` | |
| `target-encoding` | A high-cardinality column encoded by the target's mean per level: naive (fitted on all rows) leaks the label, the training score soars and the test score does not; cross-fitted (k folds as the knob) closes the gap; the smoothing prior's strength against each level's count; the gap on the tape as the rows per level shrink. | `encoder/levels` t!, `encoder/smoothing` s, `eval/train` s, `eval/test` s, `eval/gap` s, `study/byFolds` proj | `learn/preprocessing/encoders.ts`: `targetEncoder` (`cv` cross-fitting and `smooth` ready); `learn/model-selection/splitters.ts`: `stratifiedKFold` | |
| `discretisation` | Binning as a mode (uniform, quantile, k-means bins) with the bin count as the knob: a linear model on bins fitting a nonlinear curve as a step function, the spline features as the smooth alternative; binarisation by a threshold. | `bins/edges` t!, `bins/count` s, `model/fit` p, `eval/downstream` s | `learn/preprocessing/features.ts`: `kBinsDiscretizer`, `binarizer`, `splineFeatures`; `stats/histograms.ts`: `histogramBinEdges` | |

### preparation/resampling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `imbalance` | An imbalanced problem (the ratio as the knob): the sampler as a mode (random under- and over-sampling, SMOTE with its neighbours drawn, borderline SMOTE, ADASYN, Tomek links removed, NearMiss) with the synthetic points appearing on the plane one batch per step and the decision surface moving; precision, recall and the PR curve against the untouched fit; class weights in the objective as the alternative that adds no points; the threshold moved under a cost matrix. | `sampler/points` t!, `sampler/count` s, `model/surface` p, `eval/precision` s, `eval/recall` s, `eval/prCurve` proj, `threshold/cost` s | `data/modifiers/labels.ts`: `withClassImbalance`, `withPrevalence`; `learn/preprocessing/resampling.ts`: `randomUnderSampler`, `randomOverSampler`, `smote`, `borderlineSmote`, `adasyn`, `tomekLinks`, `nearMiss`; `data/samplers.ts`: `balancedWeights`, `weightedRandomSampler`; `learn/validation.ts`: `sampleWeightsOf`; `metrics/operating-points.ts`: `minimumExpectedCost`, `costCurve` | |

### preparation/splitting/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `splitters` | The splitter as a mode (hold-out, k-fold, stratified, grouped, repeated, shuffle splits, time-series and blocked, purged k-fold, spatial and cluster blocks): the folds drawn over the rows (which rows train, test, or are purged), class balance per fold, groups kept whole; the score's variance across folds against the fold count. | `fold/index` s, `fold/trainSize` s, `fold/testSize` s, `fold/balance` t, `fold/picture` p (rows coloured by fold), `study/variance` proj | `learn/model-selection/splitters.ts`: `kFold`, `stratifiedKFold`, `groupKFold`, `stratifiedGroupKFold`, `repeatedKFold`, `shuffleSplit`, `holdout`, `trainValidationTestSplit`, `foldsAreDisjoint`; `learn/model-selection/blocked-splitters.ts`: `timeSeriesSplit`, `blockedTimeSeriesSplit`, `purgedKFold`, `spatialBlockSplit`, `clusterBlockSplit`; `data/rows.ts`: `trainTestSplit`, `stratifiedSplit` | |
| `leakage` | A leak planted as a mode (a feature derived from the target, duplicates across the split, a scaler or encoder fitted on all rows, a temporal split that looks ahead): the validation score too good; caught by adversarial validation (a classifier telling train from test, its AUC the alarm), near-duplicate detection across folds, and the pipeline refitted inside each fold with the score falling to the truth. | `leak/kind` m, `eval/validation` s, `eval/test` s, `adversarial/auc` s, `duplicates/count` s, `pipeline/stages` p | `stats/tests/two-sample.ts`: `classifierTwoSampleTest`; `text/fingerprinting.ts`: `nearDuplicates`, `minhash`; `numerics/spatial/hashing.ts`: `lshIndex`; `learn/compose.ts`: `pipeline`; `learn/model-selection/cross-validation.ts`: `crossValidate`; TODO `data/modifiers/leaks.ts`: `withLeak` (a target-derived feature), `withDuplicates` | |
| `pipeline` | A preprocessing pipeline (imputer, scaler, encoder, model) fitted on the training fold and applied to the test fold, its stages drawn with the data at each stage; the column transformer routing columns by type; the pipeline's space as the dict of its steps' spaces, so a grid over it is one grid. | `pipeline/stages` p, `stage/output` t!, `eval/test` s, `space/grid` m | `learn/compose.ts`: `pipeline`, `makePipeline`, `columnTransformer`, `featureUnion`; `exposition/pipelines.ts`: `pipelineStages`; `learn/model-selection/search.ts`: `gridSearchCV` | |

### preparation/loading/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `batching` | Rows into batches and epochs (shuffled under a key, the last batch dropped or not), sequences bucketed by length and padded with the padding wasted as the quantity, weighted and balanced samplers drawing the minority more often; the loader as what the data slot hands the trainers of `train/`. | `epoch/index` s, `batch/index` s, `batch/sizes` t, `batch/padding` s, `batch/classShares` t | `data/batching.ts`: `batches`, `epochBatches`, `epochs`, `dataLoader`, `collate`, `padSequences`, `bucketBy`; `data/samplers.ts`: `randomSampler`, `weightedRandomSampler`, `batchSampler`, `balancedWeights` | |
| `remote-assets` | A remote dataset pinned by hash: fetched, verified, parsed (CSV, npy, gzip, tar, zip, IDX) and cached (memory, IndexedDB), its licence shown; a wrong hash refused. | `fetch/bytes` s, `fetch/verified` e, `fetch/cached` e, `dataset/meta` m | `data/loader.ts`: `load`, `sha256Hex`, `memoryCache`, `indexedDbCache`; `data/archives.ts`: `parseCsv`, `parseNpy`, `gunzip`, `untar`, `unzip`; `datasets/images.ts`: `parseIdxImages` | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `data/shared/problems.ts` | The planted-problem catalogue: a generator with modifiers and the truth each leaves (which cells are missing and their values, which rows are outliers, which labels are wrong, the shifted posterior, the leaked feature, the duplicates), as the data slot of every row here and of `shift-and-drift` and `data-centric-learning` | every group |
| `downstream.ts` | The model slot downstream: a small classifier or regressor from `supervised-learning`'s registry fitted after the preparation, its held-out score as `eval/downstream`, its surface or fit as a probe; the same model and key across the modes so the preparation is the only difference | every preparation group |
| `table.ts` | The table record: a small dataset as cells with a status per cell (missing, imputed, outlier, leaked, synthetic, its fold) and a legend, drawn by the table panel the Rendering section asks for | `missing/`, `outliers/`, `splitting/`, `resampling/` |

## Examples to make

- A planted leak: a feature derived from the target, the validation score too good, adversarial validation catching it.
- Target encoding with and without cross-fitting, the train and test gap.
- Scaling as the knob: gradient descent's path on unscaled against standardised features.
- MICE converging, the imputations' distribution against the truth.
- SMOTE's synthetic points appearing on the plane and the decision surface moving, against class weights.

## Rendering

- Rows with an overlay (missing cells, imputed values, folds), the learning-curve and score panels, pipeline stages (`pipelineStages`), decision surfaces.
- Missing: a table panel for small datasets with cells coloured by status (missing, imputed, outlier, leaked, fold); the comparison of two modes under one cursor.

## Problems

- New chapter; nothing built. Most steps are one-shot transforms, so the sessions are studies with a model downstream: `shared/downstream.ts` fills the model slot from `supervised-learning`'s registry, and the comparison of modes (section 5's missing panel) is what every row needs.
- Section 4 of `methods/README.md` places the registries in `methods/shared/options/`; the pages over them (`catalogue/`) live here so the exemplar dataset page has a home. If `data/catalogue/` reads as a duplicate of the options, fold it into `data/shared/`.
- Library gaps are small: a leak modifier, Little's MCAR test, `iterativeImputer`'s inner algorithm exposed. Everything else named here exists.
- `catalogue/` and `loading/` are data infrastructure rather than preparation; both are `data/`, and the chapter could split along that line.
