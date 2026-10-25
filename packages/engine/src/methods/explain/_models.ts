// What an explainer needs from a fitted estimator (private to methods/explain): the output to explain as a
// prediction function, its name, and the model's trees when it is a tree or an ensemble of trees.
import { WhiteboxError } from '../../lib/base/index.ts'
import type * as explain from '../../lib/explain/index.ts'
import { classNamesOf, type Fitted, type LearnData } from '../../lib/learn/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import type { ForestParams, GradientBoostingParams } from '../ensemble/index.ts'
import type { TreeParams } from '../trees/index.ts'

/** Which output of a model an explanation is of. */
export type OutputKind = 'auto' | 'prediction' | 'probability' | 'raw'

/** The output an explainer explains: a function of rows `[m, d]` to `[m]`, and its name. */
export interface ModelOutput {
  readonly kind: Exclude<OutputKind, 'auto'>
  readonly predict: explain.PredictFunction
  /** What the output is (`prediction`, `P(virginica)`, `log odds of malignant`). */
  readonly name: string
  /** The class explained, for classifiers. */
  readonly classIndex?: number
}

function isTree(p: unknown): p is TreeParams {
  const t = p as Partial<TreeParams>
  return Array.isArray(t.left) && Array.isArray(t.feature) && Array.isArray(t.weightedNSamples)
}

function isForest(p: unknown): p is ForestParams {
  const f = p as Partial<ForestParams>
  return Array.isArray(f.trees) && f.trees.every(isTree)
}

function isBoosting(p: unknown): p is GradientBoostingParams {
  const g = p as Partial<GradientBoostingParams>
  return Array.isArray(g.stages) && Array.isArray(g.init) && typeof g.learningRate === 'number'
}

function column(out: Tensor, k: number): Tensor {
  if (out.shape.length < 2) return out
  const width = out.shape[1] ?? 1
  const v = toFlat(out)
  const m = out.shape[0] ?? 0
  return fromStorage(
    Float64Array.from({ length: m }, (_, i) => v[i * width + Math.min(k, width - 1)] ?? 0),
    [m],
    'float64',
  )
}

/**
 * The output of a fitted model to explain. Regressors explain their prediction; classifiers the probability of a class
 * (by default the positive class of a binary problem, else the last), except boosted ones, which explain their raw
 * score (log odds) as their trees add it up.
 */
export function modelOutput(
  fitted: Fitted,
  data: LearnData,
  kind: OutputKind = 'auto',
  classIndex?: number,
): ModelOutput {
  if (fitted.kind !== 'classifier') {
    if (fitted.predict === undefined) throw new WhiteboxError('value', `explain: ${fitted.estimator} cannot predict`)
    if (kind === 'probability' || kind === 'raw')
      throw new WhiteboxError('value', `explain: a ${fitted.kind} has no ${kind} output`)
    const p = fitted.predict
    return { kind: 'prediction', predict: (x) => column(p(x), 0), name: 'prediction' }
  }
  const classes = (fitted as { readonly classes?: readonly number[] }).classes ?? []
  const names = classNamesOf(data, classes)
  const k = classIndex ?? Math.max(0, classes.length - 1)
  if (!(Number.isInteger(k) && k >= 0 && (classes.length === 0 || k < classes.length)))
    throw new WhiteboxError('value', `explain: class index ${k} out of range`)
  const name = names[k] ?? String(k)
  const resolved = kind === 'auto' ? (isBoosting(fitted.params) ? 'raw' : 'probability') : kind
  if (resolved === 'prediction') {
    if (fitted.predict === undefined) throw new WhiteboxError('value', `explain: ${fitted.estimator} cannot predict`)
    const p = fitted.predict
    return { kind: 'prediction', predict: (x) => column(p(x), 0), name: 'predicted class', classIndex: k }
  }
  if (resolved === 'raw') {
    const f = fitted.decisionFunction
    if (f === undefined) throw new WhiteboxError('value', `explain: ${fitted.estimator} has no raw score`)
    const binary = classes.length <= 2
    return {
      kind: 'raw',
      // A binary model's one score is the positive class's log odds.
      predict: (x) => {
        const out = f(x)
        return binary ? (k === 0 ? negate(column(out, 0)) : column(out, 0)) : column(out, k)
      },
      name: binary ? `log odds of ${name}` : `score of ${name}`,
      classIndex: k,
    }
  }
  const proba = fitted.predictProba
  if (proba === undefined) throw new WhiteboxError('value', `explain: ${fitted.estimator} has no probabilities`)
  return { kind: 'probability', predict: (x) => column(proba(x), k), name: `P(${name})`, classIndex: k }
}

function negate(t: Tensor): Tensor {
  return fromStorage(
    Float64Array.from(toFlat(t), (v) => -v),
    [...t.shape],
    'float64',
  )
}

function treeModel(tree: TreeParams, output: 'value' | number): explain.TreeModel {
  return {
    childrenLeft: tree.left,
    childrenRight: tree.right,
    feature: tree.feature.map((f) => Math.max(0, f)),
    threshold: tree.threshold.map((t) => (Number.isFinite(t) ? t : 0)),
    value: tree.value.map((v) => {
      if (output === 'value') return v[0] ?? 0
      const total = v.reduce((a, b) => a + b, 0)
      return total === 0 ? 0 : (v[output] ?? 0) / total
    }),
    cover: tree.weightedNSamples,
  }
}

/** A model's trees, and how its output combines them: offset + scale times their mean or sum. */
export interface TreeEnsemble {
  readonly trees: readonly explain.TreeModel[]
  readonly combine: 'mean' | 'sum'
  readonly scale: number
  readonly offset: number
}

/**
 * The trees of a decision tree, a forest or a gradient-boosting model, for the output given, or `undefined` when the
 * model is not made of trees or its trees do not add up to that output (a boosted model's probabilities).
 */
export function treeEnsembleOf(fitted: Fitted, output: ModelOutput): TreeEnsemble | undefined {
  const p = fitted.params
  const leaf = output.classIndex === undefined ? 'value' : output.classIndex
  if (isTree(p)) {
    if (output.kind !== (p.task === 'regression' ? 'prediction' : 'probability')) return undefined
    return { trees: [treeModel(p, leaf)], combine: 'mean', scale: 1, offset: 0 }
  }
  if (isForest(p)) {
    const task = p.trees[0]?.task
    if (output.kind !== (task === 'regression' ? 'prediction' : 'probability')) return undefined
    return { trees: p.trees.map((t) => treeModel(t, leaf)), combine: 'mean', scale: 1, offset: 0 }
  }
  if (isBoosting(p)) {
    const classifier = p.classes.length > 0
    if (output.kind !== (classifier ? 'raw' : 'prediction')) return undefined
    const binary = p.classes.length <= 2
    const k = binary ? 0 : (output.classIndex ?? 0)
    const sign = classifier && binary && output.classIndex === 0 ? -1 : 1
    return {
      trees: p.stages.map((stage) => treeModel(stage[k] as TreeParams, 'value')),
      combine: 'sum',
      scale: sign * p.learningRate,
      offset: sign * (p.init[k] ?? 0),
    }
  }
  return undefined
}
