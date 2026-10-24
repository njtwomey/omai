// Private plumbing of the generative methods (methods/generative): the data a model learns (standardised rows, kept
// with their original coordinates, labels and names for the records), small MLPs, minibatches, and the training loop
// every model runs through `runAlgorithm`: a step function returning named losses, with frames captured at a few dozen
// steps spaced geometrically (dense early, where training changes fastest), so a page can play the training back.
import { WhiteboxError } from '../../lib/base/index.ts'
import type { Exposition } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  encodeLabels,
  type FitOptions,
  featureNamesOf,
  fitRunOptions,
  geometricSteps,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import {
  type ActivationName,
  activationModule,
  Linear,
  type Module,
  type Params,
  Sequential,
  zerosInit,
} from '../../lib/nn/index.ts'
import { applyUpdates, type GradientTransformation } from '../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, subsample } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, type TensorLike, take, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'

/** The rows a generative model learns: standardised for training, with what the records need to draw them. */
export interface Prepared {
  /** Standardised rows `[n, d]` (zero mean, unit variance per feature). */
  readonly x: Tensor
  readonly n: number
  readonly d: number
  /** The per-feature means and standard deviations undone by {@link toData}. */
  readonly mean: readonly number[]
  readonly scale: readonly number[]
  /** The rows in their own coordinates. */
  readonly rows: readonly (readonly number[])[]
  readonly featureNames: readonly string[]
  /** Class indices per row and class names, when the data is labelled (used only to colour the records). */
  readonly labels?: readonly number[]
  readonly classes?: readonly string[]
}

/**
 * The rows of learn data, standardised (or only centred and scaled by one common factor when `isotropic`, so a
 * picture keeps its aspect), with labels kept for colouring.
 */
export function prepare(
  data: LearnData,
  who: string,
  options: { readonly standardise?: boolean; readonly isotropic?: boolean } = {},
): Prepared {
  const m = matrixView(data.x, who)
  if (m.n < 2) throw new WhiteboxError('value', `${who}: needs at least two rows`)
  const { n, d } = m
  const mean = new Array<number>(d).fill(0)
  const scale = new Array<number>(d).fill(1)
  if (options.standardise !== false) {
    for (let j = 0; j < d; j++) {
      let s = 0
      for (let i = 0; i < n; i++) s += m.data[i * d + j] ?? 0
      mean[j] = s / n
      let v = 0
      for (let i = 0; i < n; i++) v += ((m.data[i * d + j] ?? 0) - (mean[j] ?? 0)) ** 2
      scale[j] = Math.sqrt(v / n) || 1
    }
    if (options.isotropic === true) {
      const common = Math.sqrt(scale.reduce((s, v) => s + v * v, 0) / d)
      scale.fill(common)
    }
  }
  const z = new Float64Array(n * d)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) z[i * d + j] = ((m.data[i * d + j] ?? 0) - (mean[j] ?? 0)) / (scale[j] ?? 1)
  const rows = Array.from({ length: n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d)))
  let labels: number[] | undefined
  let classes: string[] | undefined
  if (data.y !== undefined) {
    try {
      const enc = encodeLabels(data.y, who, n)
      if (enc.classes.length >= 2 && enc.classes.length <= 20) {
        labels = Array.from(enc.codes)
        classes = classNamesOf(data, enc.classes)
      }
    } catch {
      // Unlabelled (or continuous targets): nothing to colour by.
    }
  }
  return {
    x: fromStorage(z, [n, d], 'float64') as Tensor,
    n,
    d,
    mean,
    scale,
    rows,
    featureNames: featureNamesOf(data, d),
    ...(labels === undefined ? {} : { labels }),
    ...(classes === undefined ? {} : { classes }),
  }
}

/** Standardised rows (a tensor `[m, d]` or arrays) back in the data's own coordinates. */
export function toData(p: Prepared, z: TensorLike): number[][] {
  const flat = toFlat(z as Tensor)
  const m = Math.floor(flat.length / p.d)
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: p.d }, (_, j) => (flat[i * p.d + j] ?? 0) * (p.scale[j] ?? 1) + (p.mean[j] ?? 0)),
  )
}

/** Rows in the data's own coordinates standardised as the model sees them, as a tensor `[m, d]`. */
export function toModel(p: Prepared, rows: readonly (readonly number[])[]): Tensor {
  const z = new Float64Array(rows.length * p.d)
  for (const [i, r] of rows.entries())
    for (let j = 0; j < p.d; j++) z[i * p.d + j] = ((r[j] ?? 0) - (p.mean[j] ?? 0)) / (p.scale[j] ?? 1)
  return fromStorage(z, [rows.length, p.d], 'float64') as Tensor
}

/** The log-Jacobian of standardising: densities in standardised coordinates minus this are densities of the data. */
export function logScale(p: Prepared): number {
  return p.scale.reduce((s, v) => s + Math.log(v), 0)
}

/**
 * An MLP as a `Sequential`: linear layers with the activation between them (none after the last). With `zeroLast` the
 * last layer starts at zero (a flow's conditioner then starts as the identity map, a residual net as no change).
 */
export function mlp(
  sizes: readonly number[],
  activation: ActivationName,
  key: Key,
  options: { readonly zeroLast?: boolean } = {},
): Sequential {
  const layers: Module[] = []
  for (let l = 0; l + 1 < sizes.length; l++) {
    const lin = new Linear(sizes[l] ?? 1, sizes[l + 1] ?? 1, { key: foldIn(key, l) })
    const last = l + 2 >= sizes.length
    if (last && options.zeroLast === true)
      lin.setParams({ weight: zerosInit([lin.outFeatures, lin.inFeatures]), bias: zerosInit([lin.outFeatures]) })
    layers.push(lin)
    if (!last) layers.push(activationModule(activation))
  }
  return new Sequential(...layers)
}

/** A minibatch of `size` distinct rows (all of them when there are fewer). */
export function minibatch(x: Tensor, key: Key, size: number): Tensor {
  const n = x.shape[0] ?? 0
  if (size >= n) return x
  return take(x, subsample(key, n, size)) as Tensor
}

/** Standard normal draws `[m, d]` (the fixed noise whose images a page follows through training). */
export function noise(key: Key, m: number, d: number): Tensor {
  return normal(key, [m, d]) as Tensor
}

/** One recorded step of training: the step (0 is the initial model) and what was captured there. */
export interface Frame<F> {
  readonly step: number
  readonly frame: F
}

/** The losses of every step, by name, in step order. */
export type LossTrace = Readonly<Record<string, readonly number[]>>

/** How far a training run has got, with the records so far (structurally a hub progress report). */
export interface GenerativeProgress {
  /** The fraction of the steps done. */
  readonly fraction: number
  readonly label: string
  readonly exposition: readonly Exposition[]
}

/** Options of {@link trainLoop}. */
export interface LoopOptions<S, F> {
  /** The run's name (the key is derived under it). */
  readonly name: string
  /** Optimisation steps. */
  readonly steps: number
  /** The model's state at step 0 (from the run's key). */
  readonly init: (key: Key) => S
  /** One step: the next state and the losses measured on the step's minibatch. */
  readonly step: (
    state: S,
    t: number,
    key: Key,
  ) => { readonly state: S; readonly losses: Readonly<Record<string, number>> }
  /** What to keep of a state at a recorded step (samples, a density on a grid, a latent space). */
  readonly capture: (state: S, step: number) => F
  /** The records of the frames and losses so far (called for progress and at the end). */
  readonly records: (frames: readonly Frame<F>[], losses: LossTrace, steps: number) => readonly Exposition[]
  /** The most frames captured (default 40). */
  readonly frames?: number
  readonly fit: FitOptions
  readonly onProgress?: (progress: GenerativeProgress) => void
}

/** What {@link trainLoop} returns. */
export interface LoopResult<S, F> {
  readonly state: S
  readonly frames: readonly Frame<F>[]
  readonly losses: LossTrace
  readonly steps: number
  readonly status: string
  readonly exposition: readonly Exposition[]
}

/**
 * Train a generative model: `steps` calls of its step function through `runAlgorithm` (each step's key derived from the
 * run's), every loss kept, and a frame captured at the geometric schedule of steps (with the first and last); progress
 * is reported with the records so far at every frame.
 */
export function trainLoop<S, F>(options: LoopOptions<S, F>): LoopResult<S, F> {
  const total = Math.max(1, Math.floor(options.steps))
  const schedule = new Set(geometricSteps(total, options.frames ?? 40))
  const frames: Frame<F>[] = []
  const losses: Record<string, number[]> = {}
  const run = fitRunOptions(options.fit, options.name)
  let stopped: string | undefined
  const algorithm: Algorithm<S> = {
    name: options.name,
    config: { steps: total },
    init: (k) => {
      const s = options.init(k)
      if (schedule.has(0)) frames.push({ step: 0, frame: options.capture(s, 0) })
      return s
    },
    step: (s, ctx) => {
      const r = options.step(s, ctx.t, ctx.key)
      for (const [name, v] of Object.entries(r.losses)) {
        const list = losses[name] ?? []
        list.push(v)
        losses[name] = list
        if (!Number.isFinite(v)) stopped = `${name} is not finite at step ${ctx.t + 1}`
      }
      const t = ctx.t + 1
      if (schedule.has(t) || stopped !== undefined) {
        frames.push({ step: t, frame: options.capture(r.state, t) })
        options.onProgress?.({
          fraction: t / total,
          label: `step ${t} of ${total}`,
          exposition: options.records(frames, losses, t),
        })
      }
      return r.state
    },
    observe: () => ({
      metrics: Object.fromEntries(Object.entries(losses).map(([k, v]) => [k, v.at(-1) ?? Number.NaN])),
    }),
    done: () => (stopped === undefined ? undefined : 'diverged'),
  }
  const result = runAlgorithm(algorithm, { ...run, key: run.key ?? makeKey(0), maxSteps: total })
  return {
    state: result.state,
    frames,
    losses,
    steps: result.steps,
    status: stopped ?? result.status,
    exposition: options.records(frames, losses, result.steps),
  }
}

/**
 * A loss trace averaged over buckets of consecutive steps, at most `limit` points: the steps (each bucket's last) and
 * the means.
 */
export function bucketed(values: readonly number[], limit = 150): { steps: number[]; means: number[] } {
  const n = values.length
  const width = Math.max(1, Math.ceil(n / limit))
  const steps: number[] = []
  const means: number[] = []
  for (let a = 0; a < n; a += width) {
    const b = Math.min(n, a + width)
    let s = 0
    for (let i = a; i < b; i++) s += values[i] ?? 0
    steps.push(b)
    means.push(s / (b - a))
  }
  return { steps, means }
}

/** A debiased exponential moving average of a trace (beta 0.9 per point). */
export function smoothed(values: readonly number[], beta = 0.9): number[] {
  let avg = 0
  return values.map((v, t) => {
    avg = beta * avg + (1 - beta) * v
    return avg / (1 - beta ** (t + 1))
  })
}

/** At most `limit` rows, spaced evenly (for records that draw the data beside the samples). */
export function thin<T>(rows: readonly T[], limit: number): T[] {
  if (rows.length <= limit) return [...rows]
  return Array.from({ length: limit }, (_, i) => rows[Math.floor((i * rows.length) / limit)] as T)
}

/** The run's key: the caller's, or key(0). */
export function runKey(fit: FitOptions): Key {
  return fit.key ?? makeKey(0)
}

/** Parameters with their optimiser's state. */
export interface Optimised {
  readonly params: Params
  readonly opt: unknown
}

/** Start optimising some parameters. */
export function optimising(params: Params, optimizer: GradientTransformation): Optimised {
  return { params, opt: optimizer.init(params) }
}

/** One optimiser update of some parameters from their gradients. */
export function descend(optimizer: GradientTransformation, s: Optimised, grads: Params): Optimised {
  const r = optimizer.update(grads, s.opt, s.params)
  return { params: applyUpdates(s.params, r.updates) as Params, opt: r.state }
}
