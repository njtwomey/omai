import { MdmlError } from '../../base/index.ts'
import {
  asTensor,
  concat,
  div,
  matmul,
  mul,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  type Value,
} from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { ClassificationData, TabularData } from '../schema/tabular.ts'
import { bayesErrorEstimate, type Truth } from '../truth.ts'

/** The number of classes and the count of each. */
export function classCounts(data: ClassificationData): number[] {
  const counts = new Array<number>(data.classes.length).fill(0)
  for (const c of data.y.data) if (c >= 0) counts[c] = (counts[c] ?? 0) + 1
  return counts
}

/** Class priors: the truth's when it records them, else the empirical class frequencies. */
export function classPriors(input: Generated<ClassificationData>): number[] {
  const known = input.truth.extra?.['priors']
  if (Array.isArray(known) && known.length === input.data.classes.length) return known as number[]
  const counts = classCounts(input.data)
  const n = counts.reduce((a, b) => a + b, 0)
  return counts.map((c) => c / Math.max(1, n))
}

/** A posterior multiplied on the right by a matrix (clean-to-noisy label transitions). */
export function posteriorTimes(posterior: (x: Value) => Value, m: TensorLike): (x: Value) => Value {
  const t = asTensor(m)
  return (x) => matmul(posterior(x), t)
}

/** The posterior after a change of class priors: `p'(k | x) ∝ p(k | x) ratio_k`. */
export function priorShifted(posterior: (x: Value) => Value, ratios: readonly number[]): (x: Value) => Value {
  const r = tensor(ratios as number[])
  return (x) => {
    const w = mul(posterior(x), r)
    return div(w, sum(w, 1, true))
  }
}

/**
 * A truth without its generating program: a modifier that changes how the data arise makes the program describe the
 * unmodified data, so it is dropped (the modifier's own density and posterior remain).
 */
export function withoutProgram(truth: Truth): Truth {
  const { program: _program, ...rest } = truth
  return rest
}

/** Replace a classification truth's posterior and re-estimate its Bayes error on the data's own rows. */
export function withPosterior(
  truth: Truth,
  posterior: ((x: Value) => Value) | undefined,
  x: Value,
  extra: Readonly<Record<string, unknown>>,
): Truth {
  const { bayesError: _old, posterior: _p, program: _program, ...rest } = truth
  const base: Truth = { ...rest, extra: { ...(truth.extra ?? {}), ...extra } }
  if (posterior === undefined) return base
  return { ...base, posterior, bayesError: bayesErrorEstimate(posterior, x).value }
}

/** The truth's functions of the features composed with a map of the features (`f(inner(x))`). */
export function composedFunctions(
  truth: Truth,
  inner: (x: Value) => Value,
): Pick<Truth, 'regressionFunction' | 'posterior' | 'quantile'> {
  const out: { -readonly [K in 'regressionFunction' | 'posterior' | 'quantile']?: Truth[K] } = {}
  const f = truth.regressionFunction
  if (f !== undefined) out.regressionFunction = (x) => f(inner(x))
  const p = truth.posterior
  if (p !== undefined) out.posterior = (x) => p(inner(x))
  const q = truth.quantile
  if (q !== undefined) out.quantile = (x, level) => q(inner(x), level)
  return out
}

/** Require classification data. */
export function requireClassification(input: Generated, name: string): Generated<ClassificationData> {
  if (input.data.kind !== 'classification')
    throw new MdmlError('value', `${name}: needs classification data, got ${input.data.kind}`)
  return input as Generated<ClassificationData>
}

/** Require data with a feature matrix. */
export function requireTabular(input: Generated, name: string): Generated<TabularData> {
  const k = input.data.kind
  if (k !== 'classification' && k !== 'regression' && k !== 'multilabel' && k !== 'multioutput')
    throw new MdmlError('value', `${name}: needs data with a feature matrix, got ${k}`)
  return input as Generated<TabularData>
}

/** The column indices to act on (default all), checked against the width. */
export function columnsOf(columns: readonly number[] | undefined, d: number, name: string): number[] {
  const cols = columns === undefined ? Array.from({ length: d }, (_, j) => j) : [...columns]
  for (const j of cols) {
    if (!(Number.isInteger(j) && j >= 0 && j < d)) throw new MdmlError('value', `${name}: no column ${j} in ${d}`)
  }
  return cols
}

/** Split a value `[n, d]` into its columns `[n, 1]`. */
export function columnsOfValue(x: Value): Value[] {
  const d = x.shape[1] ?? 1
  return Array.from({ length: d }, (_, j) => sliceAxis(x, 1, j, j + 1))
}

/** The chosen columns of `x` as an `[n, |cols|]` block. */
export function selectColumns(x: Value, cols: readonly number[]): Value {
  return concat(
    cols.map((j) => sliceAxis(x, 1, j, j + 1)),
    1,
  )
}

/** Map the chosen columns of `x` with `f` (applied to the `[n, |cols|]` block), keeping the others. */
export function mapColumns(x: Value, cols: readonly number[], f: (block: Value) => Value): Value {
  const parts = columnsOfValue(x)
  const block = f(
    concat(
      cols.map((j) => parts[j] as Value),
      1,
    ),
  )
  const mapped = columnsOfValue(block)
  for (const [i, j] of cols.entries()) parts[j] = mapped[i] as Value
  return concat(parts, 1)
}

/** Per-column means and standard deviations (ddof 0) of a concrete matrix, ignoring NaN. */
export function columnStats(x: Tensor): { mean: number[]; std: number[] } {
  const [n = 0, d = 1] = x.shape
  const mean = new Array<number>(d).fill(0)
  const std = new Array<number>(d).fill(0)
  for (let j = 0; j < d; j++) {
    let s = 0
    let c = 0
    for (let i = 0; i < n; i++) {
      const v = x.data[i * d + j] ?? Number.NaN
      if (!Number.isNaN(v)) {
        s += v
        c++
      }
    }
    const m = c === 0 ? 0 : s / c
    let ss = 0
    for (let i = 0; i < n; i++) {
      const v = x.data[i * d + j] ?? Number.NaN
      if (!Number.isNaN(v)) ss += (v - m) ** 2
    }
    mean[j] = m
    std[j] = c === 0 ? 0 : Math.sqrt(ss / c)
  }
  return { mean, std }
}

/** The logistic function on a number. */
export function logistic(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
}

/** The intercept `a` with mean(logistic(a + b z_i)) = rate, by bisection (the mean is increasing in `a`). */
export function calibrateIntercept(z: ArrayLike<number>, slope: number, rate: number): number {
  let lo = -60
  let hi = 60
  const meanAt = (a: number) => {
    let s = 0
    for (let i = 0; i < z.length; i++) s += logistic(a + slope * (z[i] ?? 0))
    return s / Math.max(1, z.length)
  }
  for (let it = 0; it < 200; it++) {
    const mid = (lo + hi) / 2
    if (meanAt(mid) < rate) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}
