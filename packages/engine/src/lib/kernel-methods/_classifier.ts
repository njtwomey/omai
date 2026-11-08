// Private: what the kernel classifiers of this folder share: the kernel options as the estimators take them, the
// training setup (rows, Gram matrix, encoded labels), cross-kernels against the stored rows, and the classifier verbs
// built from per-class decision values.
import { argmaxOf, OmaiError, sigmoidOf } from '../base/index.ts'
import {
  type Kernel,
  type KernelChoice,
  kernelMatrix,
  type PairwiseKernelMetric,
  type ResolvedKernel,
  resolveKernel,
} from '../kernels/index.ts'
import { encodeLabels, type LearnData, type MatrixView, matrixView } from '../learn/index.ts'
import { accuracyScore } from '../metrics/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'

/** The kernel options every kernel estimator takes, scikit-learn's. */
export interface KernelOptions {
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
}

/** The default kernel: RBF with `gamma = 1 / d`. */
export const KERNEL_DEFAULTS: KernelOptions = { kernel: 'rbf', gamma: 'auto', degree: 3, coef0: 1 }

/** The training rows, Gram matrix and labels of a kernel classifier. */
export interface ClassificationSetup {
  readonly n: number
  readonly d: number
  /** The rows, `[n][d]`. */
  readonly rows: number[][]
  readonly resolved: ResolvedKernel
  /** The Gram matrix, row-major `n * n`. */
  readonly gram: Float64Array
  readonly classes: number[]
  readonly codes: Int32Array
}

export const rowsOfView = (m: MatrixView): number[][] =>
  Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))

/** Read labelled data for a kernel classifier: the Gram matrix of the training rows and the encoded labels. */
export function classificationSetup(data: LearnData, h: KernelOptions, who: string): ClassificationSetup {
  const m = matrixView(data.x, who)
  if (data.y === undefined) throw new OmaiError('value', `${who}: labels are required`)
  if (data.weights !== undefined) throw new OmaiError('value', `${who}: sample weights are not supported`)
  const enc = encodeLabels(data.y, who, m.n)
  const resolved = resolveKernel(h as KernelChoice, data.x)
  return {
    n: m.n,
    d: m.d,
    rows: rowsOfView(m),
    resolved,
    gram: kernelMatrix(h, resolved, data.x),
    classes: [...enc.classes],
    codes: enc.codes,
  }
}

/** The kernel values `[m * n]` between query rows and stored rows. */
export function crossKernel(
  h: Pick<KernelChoice, 'kernel'>,
  r: ResolvedKernel,
  x: TensorLike,
  stored: number[][],
): Float64Array {
  return kernelMatrix(h, r, x, stored)
}

/** The sigmoid, stable in both tails. */
export const sigmoid: (f: number) => number = sigmoidOf

/** `log(1 + exp(f))`, stable. */
export const softplus = (f: number): number => (f > 0 ? f + Math.log1p(Math.exp(-f)) : Math.log1p(Math.exp(f)))

/** The verbs every kernel classifier offers. */
export interface KernelClassifierVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

/** The verbs of a classifier whose decision value maps to a probability. */
export interface KernelProbabilityVerbs extends KernelClassifierVerbs {
  /** Class probabilities `[m, 2]` in the order of `classes`. */
  readonly predictProba: (x: TensorLike) => Tensor
}

type Decision = (x: TensorLike) => { readonly m: number; readonly k: number; readonly values: Float64Array }

/**
 * Verbs from per-class decision values `[m * k]` (`k = 1` for a binary problem, whose value is the positive class's):
 * labels by the sign or the arg max, and the decision function as scikit-learn shapes it.
 */
export function classifierVerbs(
  classes: readonly number[],
  d: number,
  _who: string,
  decision: Decision,
): KernelClassifierVerbs {
  const predict = (x: TensorLike): Tensor => {
    const { m, k, values } = decision(x)
    const out = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      out[i] =
        k === 1
          ? (classes[(values[i] ?? 0) > 0 ? 1 : 0] ?? Number.NaN)
          : (classes[argmaxOf(values.subarray(i * k, (i + 1) * k))] ?? Number.NaN)
    }
    return fromStorage(out, [m], 'float64')
  }
  return {
    classes,
    nFeaturesIn: d,
    predict,
    decisionFunction: (x) => {
      const { m, k, values } = decision(x)
      return fromStorage(Float64Array.from(values), k === 1 ? [m] : [m, k], 'float64')
    },
    score: (data) => accuracyScore(data.y ?? [], predict(data.x)).estimate.value,
  }
}

/** {@link classifierVerbs} for a binary problem, with probabilities `proba(f)` of the positive class. */
export function probabilityVerbs(
  classes: readonly number[],
  d: number,
  who: string,
  decision: Decision,
  proba: (f: number) => number,
): KernelProbabilityVerbs {
  return {
    ...classifierVerbs(classes, d, who, decision),
    predictProba: (x) => {
      const { m, values } = decision(x)
      const out = new Float64Array(m * 2)
      for (let i = 0; i < m; i++) {
        const p = proba(values[i] ?? 0)
        out[2 * i] = 1 - p
        out[2 * i + 1] = p
      }
      return fromStorage(out, [m, 2], 'float64')
    },
  }
}

/** Check the reader's view of the query rows. */
export function queryView(x: TensorLike, d: number, who: string): MatrixView {
  return matrixView(x, who, { nFeatures: d })
}
