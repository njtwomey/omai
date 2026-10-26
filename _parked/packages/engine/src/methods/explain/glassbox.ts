/**
 * Glass-box and rule-based models as explanations: an explainable boosting machine fitted to the data (its shape
 * functions and pairwise terms as a `dependence` record, its term importances as `importances`), a points scorecard
 * of a fitted logistic model, RuleFit's rules, and an anchor around one prediction (all three as `rules` records).
 */

import { MdmlError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import {
  type DependenceCurve,
  type DependenceSurface,
  dependence,
  type Explained,
  importances,
  type RuleEntry,
  ruleSet,
} from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  targetVector,
} from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { linearScoresOf } from './_models.ts'

/** Options of {@link explainableBoostingMachine}. */
export interface EbmOptions extends explain.ExplainableBoostingOptions {
  /** Positive class for classification (default the last class). */
  readonly positive?: number
}

/** A fitted explainable boosting machine and its predictions on the data. */
export interface EbmResult {
  readonly model: explain.GlassBoxModel
  /** The model's predictions (probabilities of the positive class for classification). */
  readonly predictions: readonly number[]
}

function minMax(col: ArrayLike<number>): [number, number] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let i = 0; i < col.length; i++) {
    const v = col[i] ?? 0
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return [lo, hi]
}

/**
 * Fit an explainable boosting machine (InterpretML's EBM: cyclic boosting of one shape function per feature, then a
 * few pairwise terms) and explain it by its own parts: each shape function as a step curve and each pairwise term as
 * a surface (`dependence`, method `shape function`), and the terms ranked by their mean absolute contribution
 * (`importances`). For classification data the terms are on the log-odds scale of the positive class.
 *
 * @param data - The data (a classification dataset, or regression targets).
 * @param options - The boosting options and the positive class.
 * @returns The model and its predictions with the two records.
 */
export function explainableBoostingMachine(data: LearnData, options: EbmOptions = {}): Explained<EbmResult> {
  const x = matrixView(data.x, 'explainableBoostingMachine')
  if (data.y === undefined) throw new MdmlError('value', 'explainableBoostingMachine: needs targets')
  const yRaw = Array.from(targetVector(data.y, 'explainableBoostingMachine', x.n))
  const classes = Array.from(new Set(yRaw)).sort((a, b) => a - b)
  const classify = options.task === 'classification' || (options.task === undefined && classes.length === 2)
  const positive = options.positive ?? classes[classes.length - 1] ?? 1
  const y = classify ? yRaw.map((v) => (v === positive ? 1 : 0)) : yRaw
  const model = explain.explainableBoosting(
    { x: data.x, y },
    { ...options, task: classify ? 'classification' : 'regression' },
  )
  const features = featureNamesOf(data, x.d)
  const names = classNamesOf(data, classes)
  const output = classify ? `log odds of ${names[classes.indexOf(positive)] ?? String(positive)}` : 'prediction'
  const curves: DependenceCurve[] = []
  const surfaces: DependenceSurface[] = []
  const rug = (j: number): number[] => {
    const out: number[] = []
    const step = Math.max(1, x.n / 200)
    for (let t = 0; t < x.n; t += step) out.push(x.data[Math.floor(t) * x.d + j] ?? 0)
    return out
  }
  for (const t of model.terms) {
    if (t.features.length === 1) {
      const j = t.features[0]
      const [lo] = minMax(Array.from({ length: x.n }, (_, i) => x.data[i * x.d + j] ?? 0))
      const edges = t.edges as readonly number[]
      curves.push({ feature: j, grid: [lo, ...edges], average: [...(t.values as readonly number[])], rug: rug(j) })
    } else {
      const [a, b] = t.features as readonly [number, number]
      const [ea, eb] = t.edges as readonly [readonly number[], readonly number[]]
      const [la] = minMax(Array.from({ length: x.n }, (_, i) => x.data[i * x.d + a] ?? 0))
      const [lb] = minMax(Array.from({ length: x.n }, (_, i) => x.data[i * x.d + b] ?? 0))
      surfaces.push({
        features: [a, b] as const,
        gridX: [la, ...ea],
        gridY: [lb, ...eb],
        values: (t.values as readonly (readonly number[])[]).map((r) => [...r]),
      })
    }
  }
  const imp = explain.termImportances(model, data.x)
  const termNames = model.terms.map((t) => t.features.map((j) => features[j] ?? `x${j}`).join(' × '))
  return {
    model,
    predictions: explain.glassBoxPredict(model, data.x),
    exposition: [
      dependence({
        method: 'shape function',
        features,
        output,
        curves,
        ...(surfaces.length === 0 ? {} : { surfaces }),
      }),
      importances({
        method: 'EBM term importance',
        features: termNames,
        mean: imp,
        metric: `mean |term| (${output})`,
        direction: 'higher',
        on: 'training',
      }),
    ],
  }
}

/**
 * A points scorecard of a fitted logistic model (a model linear in its inputs, read off its decision function): each
 * feature's quantile bins with the points they add, as a `rules` record (a bin per rule, its points as the weight),
 * and the base points every applicant starts from as the intercept.
 *
 * @param fitted - A fitted binary classifier linear in its inputs (logistic regression).
 * @param data - The data the bins are drawn from.
 * @param options - The scorecard options (bins, pdo, baseScore, baseOdds).
 * @returns The scorecard with a `rules` record.
 */
export function logisticScorecard(
  fitted: Fitted,
  data: LearnData,
  options: explain.ScorecardOptions = {},
): Explained<{ readonly scorecard: explain.Scorecard }> {
  const x = matrixView(data.x, 'logisticScorecard')
  const lin = linearScoresOf(fitted, x)
  if (lin === undefined || lin.b.length !== 1)
    throw new MdmlError('value', 'logisticScorecard: needs a binary classifier linear in its inputs')
  const card = explain.scorecard(
    lin.w.map((r) => r[0] ?? 0),
    lin.b[0] ?? 0,
    fromStorage(x.data, [x.n, x.d], 'float64'),
    options,
  )
  const features = featureNamesOf(data, x.d)
  const rules: RuleEntry[] = card.features.flatMap((f) =>
    f.points.map((p, b) => ({
      clauses: [
        {
          feature: f.feature,
          op: 'in' as const,
          lower: b === 0 ? Number.NEGATIVE_INFINITY : (f.edges[b - 1] ?? 0),
          upper: b === f.edges.length ? Number.POSITIVE_INFINITY : (f.edges[b] ?? 0),
        },
      ],
      weight: p,
      importance: Math.abs(p),
    })),
  )
  return {
    scorecard: card,
    exposition: [ruleSet({ method: 'scorecard', features, output: 'points', rules, intercept: card.basePoints })],
  }
}

function clausesOf(rule: explain.Rule): RuleEntry['clauses'] {
  return rule.map((c) => ({ feature: c.feature, op: c.op, threshold: c.threshold }))
}

/**
 * RuleFit (Friedman and Popescu 2008) on the data: rules from shallow trees on boosting residuals, and linear terms,
 * selected by the lasso; the surviving terms by importance, with their coefficients and support, as a `rules` record.
 *
 * @param data - Rows and (numeric) targets.
 * @param options - The RuleFit options.
 * @returns The model with a `rules` record.
 */
export function ruleFitExplanation(
  data: LearnData,
  options: explain.RuleFitOptions = {},
): Explained<{ readonly model: explain.RuleFitModel }> {
  const x = matrixView(data.x, 'ruleFitExplanation')
  const model = explain.ruleFit(data, options)
  const features = featureNamesOf(data, x.d)
  return {
    model,
    exposition: [
      ruleSet({
        method: 'RuleFit',
        features,
        output: 'prediction',
        intercept: model.intercept,
        rules: model.terms.map((t) => ({
          clauses: t.rule === undefined ? [] : clausesOf(t.rule),
          ...(t.feature === undefined ? {} : { linear: t.feature }),
          weight: t.coefficient,
          support: t.support,
          importance: t.importance,
        })),
      }),
    ],
  }
}

/**
 * An anchor around one prediction (Ribeiro, Singh and Guestrin 2018): the conjunction of feature intervals that keeps
 * the model's prediction with high precision on perturbed rows, with its precision and coverage, as a `rules` record.
 *
 * @param fitted - A fitted classifier (or a regressor, its prediction rounded).
 * @param data - The data (the explained row and the perturbation distribution).
 * @param row - The row explained.
 * @param options - threshold (default 0.95) and key.
 * @returns The anchor with a `rules` record.
 */
export function anchorExplanation(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: { readonly threshold?: number; readonly key?: Key } = {},
): Explained<{ readonly anchor: explain.Anchor }> {
  const x = matrixView(data.x, 'anchorExplanation')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n))
    throw new MdmlError('value', `anchorExplanation: no row ${row}`)
  const predictFn = fitted.predict
  if (predictFn === undefined) throw new MdmlError('value', `anchorExplanation: ${fitted.estimator} cannot predict`)
  const predict = (t: Tensor): Tensor => predictFn(t)
  const point = Array.from(x.data.subarray(row * x.d, (row + 1) * x.d))
  const anchor = explain.anchors(predict, point, fromStorage(x.data, [x.n, x.d], 'float64'), {
    threshold: options.threshold ?? 0.95,
    key: options.key ?? makeKey(0),
  })
  const label = toFlat(predictFn([point]))[0] ?? 0
  const classes = (fitted as { readonly classes?: readonly number[] }).classes ?? []
  const output =
    fitted.kind === 'classifier'
      ? `predicted class ${classNamesOf(data, classes)[classes.indexOf(label)] ?? String(label)}`
      : 'prediction'
  return {
    anchor,
    exposition: [
      ruleSet({
        method: 'anchors',
        features: featureNamesOf(data, x.d),
        output,
        row: point,
        rules: [
          {
            clauses: anchor.predicates.map((p) => ({
              feature: p.feature,
              op: 'in' as const,
              lower: p.lower,
              upper: p.upper,
            })),
            precision: anchor.precision,
            coverage: anchor.coverage,
          },
        ],
      }),
    ],
  }
}
