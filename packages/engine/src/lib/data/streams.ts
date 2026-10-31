/**
 * Data streams with planted drift: points or batches arriving one per step from a classification or a regression
 * generator whose concept changes where the stream says it does. The drift is abrupt (the concept switches at a
 * point), gradual (around the point, each point comes from the new concept with a probability that ramps up),
 * incremental (the concept itself moves smoothly into the new one) or recurring (concepts return on a period); what
 * changes is the concept (P(y|x)), the covariates (P(x)) or the labels (P(y)). Every stream carries its truth, the
 * drift points and the active concept at each step, and is a pure function of its key: `next(t)` gives step `t`
 * without the steps before it, and `streamAlgorithm` runs the same stream as an `Algorithm` that serialises.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, normal, split, uniform } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'

/** Whether the stream's target is a class or a number. */
export type StreamTask = 'classification' | 'regression'

/** How the concept changes: not at all, at a point, around a point point by point, smoothly, or back and forth. */
export type StreamDrift = 'none' | 'abrupt' | 'gradual' | 'incremental' | 'recurring'

/** What changes: the rule P(y|x), the inputs P(x), or the targets P(y) with P(x|y) fixed. */
export type StreamShift = 'concept' | 'covariate' | 'label'

/** What a stream is told. */
export interface StreamOptions {
  readonly task?: StreamTask
  readonly shift?: StreamShift
  readonly drift?: StreamDrift
  /** Steps in the stream (default 1000). */
  readonly length?: number
  /** Points per step (default 1). */
  readonly batch?: number
  /** Features (default 2). */
  readonly dim?: number
  /** Classes of a classification stream (default 2). */
  readonly classes?: number
  /** Drifts, spaced evenly, when `points` is not given (default 1; a recurring stream's follow from its period). */
  readonly count?: number
  /** The steps at which the concept changes, in increasing order; they replace `count`. */
  readonly points?: readonly number[]
  /** Steps a gradual or incremental transition takes, centred on its point (default a tenth of the stream). */
  readonly width?: number
  /** Steps a concept lasts in a recurring stream (default a quarter of the stream). */
  readonly period?: number
  /** Distinct concepts: a drift moves to the next, wrapping round (default: one more than the drifts, 2 if recurring). */
  readonly concepts?: number
  /** How far a new concept lies from the first, from 0 (the same) to 1 (independent, or three standard deviations). */
  readonly severity?: number
  /** Classification: the chance a label is replaced by a random class; regression: the noise on the target. */
  readonly noise?: number
}

/** The stream's knobs as a space, with the defaults of {@link stream}, for controls, sweeps and a later registry. */
export const streamKnobs: DictSpace = dict({
  task: choice(['classification', 'regression'], { default: 'classification', label: 'task' }),
  shift: choice(['concept', 'covariate', 'label'], { default: 'concept', label: 'what changes' }),
  drift: choice(['none', 'abrupt', 'gradual', 'incremental', 'recurring'], { default: 'abrupt', label: 'drift' }),
  length: int(10, 1_000_000, { log: true, default: 1000, label: 'steps' }),
  batch: int(1, 10_000, { log: true, default: 1, label: 'points per step' }),
  dim: int(1, 100, { default: 2, label: 'features' }),
  classes: int(2, 20, { default: 2, label: 'classes', when: { key: 'task', in: ['classification'] } }),
  count: int(1, 20, { default: 1, label: 'drifts' }),
  width: int(1, 100_000, { log: true, default: 100, label: 'transition width (steps)' }),
  period: int(2, 1_000_000, { log: true, default: 250, label: 'period (steps)' }),
  severity: real(0, 1, { default: 1, label: 'severity' }),
  noise: real(0, 1, { default: 0, label: 'noise' }),
})

/** A drift as planted: where it is centred, the steps it spans, and the concepts either side. */
export interface PlantedDrift {
  readonly index: number
  /** The step it is centred on (where an abrupt drift happens). */
  readonly point: number
  /** The first step touched and the last (equal to `point` for an abrupt drift). */
  readonly start: number
  readonly end: number
  readonly from: number
  readonly to: number
  /** Whether `to` is a concept the stream has already been in. */
  readonly recurring: boolean
}

/** The concept in force at a step: the one before the transition under way, the one after, and how far it has gone. */
export interface Transition {
  readonly from: number
  readonly to: number
  /** 0 outside a transition; inside, the share of the way to `to`, in (0, 1). */
  readonly mix: number
}

/** The parameters of one concept, as plain numbers. */
export interface ConceptParameters {
  /** Classification, concept shift: the class scores' weights `[classes, dim]`, flat; the class is the largest score. */
  readonly weights: readonly number[]
  /** Regression: the coefficients of the target on the features. */
  readonly coefficients: readonly number[]
  /** Covariate shift: the mean of the features. */
  readonly mean: readonly number[]
  /** Label shift: the class priors (classification) or the mean of the target (regression, as a single number). */
  readonly priors: readonly number[]
}

/** What the stream knows about itself: where it drifts, and what rule holds at every step. */
export interface StreamTruth {
  readonly task: StreamTask
  readonly shift: StreamShift
  readonly drift: StreamDrift
  readonly length: number
  readonly batch: number
  readonly dim: number
  readonly classes: number
  /** The number of distinct concepts. */
  readonly concepts: number
  /** The drift points, in the sense of `Truth.changepoints`. */
  readonly changepoints: readonly number[]
  readonly drifts: readonly PlantedDrift[]
  /** The transition in force at a step. */
  transitionAt(t: number): Transition
  /** The dominant concept at a step (the new one from the middle of a transition). */
  conceptAt(t: number): number
  /** The parameters of a concept. */
  parameters(concept: number): ConceptParameters
  /**
   * What a concept gives a point, noise aside: the class of largest score (concept shift), the most probable class
   * given the priors (label shift), the coefficients' prediction (regression). For covariate shift the rule is the same
   * in every concept.
   */
  predict(concept: number, x: ArrayLike<number>): number
}

/** One step of a stream: its points and what generated them. */
export interface StreamBatch {
  readonly t: number
  /** The features `[batch, dim]`. */
  readonly x: Tensor<'float64'>
  /** The targets `[batch]`: class indices (int32) for classification, numbers (float64) for regression. */
  readonly y: Tensor<'float64'> | Tensor<'int32'>
  /** The concept each point was drawn from (they differ within a gradual transition). */
  readonly concept: readonly number[]
  /** How far the transition under way has gone (0 outside one). */
  readonly mix: number
}

/** The options of a stream with every default filled in; `points` are the drift points actually planted. */
export type ResolvedStreamOptions = Required<StreamOptions>

/** A stream: its options as resolved, its truth, and random access to its steps. */
export interface DataStream {
  readonly options: ResolvedStreamOptions
  readonly truth: StreamTruth
  /** Step `t` (from 0), a pure function of the key and `t`: asking for it twice, or out of order, gives the same points. */
  next(t: number): StreamBatch
}

interface Parameters {
  readonly weights: Float64Array
  readonly coefficients: Float64Array
  readonly mean: Float64Array
  readonly priors: Float64Array
}

const blend = (base: Float64Array, fresh: Float64Array, s: number): Float64Array => {
  const keep = Math.sqrt(1 - s * s)
  return Float64Array.from(base, (b, i) => keep * b + s * (fresh[i] ?? 0))
}

const lerp = (a: Float64Array, b: Float64Array, m: number): Float64Array =>
  Float64Array.from(a, (x, i) => (1 - m) * x + m * (b[i] ?? 0))

const argmax = (xs: ArrayLike<number>): number => {
  let best = 0
  for (let i = 1; i < xs.length; i++)
    if ((xs[i] ?? Number.NEGATIVE_INFINITY) > (xs[best] ?? Number.NEGATIVE_INFINITY)) best = i
  return best
}

/** The class means of a label-shift classification stream: fixed, well apart, the same in every concept. */
const SEPARATION = 2

function positiveInt(name: string, v: number): number {
  if (!Number.isInteger(v) || v < 1)
    throw new MdmlError('value', `stream: ${name} must be a positive integer, got ${v}`)
  return v
}

/**
 * A stream with planted drift. The same key gives the same stream, step by step, and any step can be asked for on its
 * own. Its truth says where the drifts are and which concept holds at every step, so detectors and adaptive learners
 * can be scored against what was planted.
 *
 * @param k - The stream's key: the concepts' parameters and every point come from it.
 * @param options - Task, what shifts and how, length, batch, and the drift's points, width, period and severity.
 * @returns The stream.
 * @throws MdmlError with code `value` for non-positive sizes, drift points outside the stream or out of order,
 *   transitions that overlap, or a recurring stream with no period to recur on.
 * @concept datasets
 * @example
 * ```ts
 * import { stream } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const s = stream(key(0), { drift: 'abrupt', points: [100, 200], length: 300 })
 * s.truth.changepoints // => [100, 200]
 * [s.truth.conceptAt(99), s.truth.conceptAt(100), s.truth.conceptAt(200)] // => [0, 1, 2]
 * s.next(5).x.shape // => [1, 2]
 * ```
 */
export function stream(k: Key, options: StreamOptions = {}): DataStream {
  const task = options.task ?? 'classification'
  const shift = options.shift ?? 'concept'
  const drift = options.drift ?? 'abrupt'
  const length = positiveInt('length', options.length ?? 1000)
  const batch = positiveInt('batch', options.batch ?? 1)
  const dim = positiveInt('dim', options.dim ?? 2)
  const classes = task === 'classification' ? positiveInt('classes', options.classes ?? 2) : 1
  if (task === 'classification' && classes < 2)
    throw new MdmlError('value', 'stream: a classification stream needs two classes')
  const severity = options.severity ?? 1
  const noise = options.noise ?? 0
  if (!(severity >= 0 && severity <= 1))
    throw new MdmlError('value', `stream: severity must be in [0, 1], got ${severity}`)
  if (!(noise >= 0 && noise <= 1)) throw new MdmlError('value', `stream: noise must be in [0, 1], got ${noise}`)
  const windowed = drift === 'gradual' || drift === 'incremental'
  const width = windowed ? positiveInt('width', options.width ?? Math.max(1, Math.round(length / 10))) : 0
  const period = options.period ?? Math.max(2, Math.round(length / 4))
  if (drift === 'recurring') positiveInt('period', period)
  const points: number[] = [...(options.points ?? [])]
  if (drift === 'none') points.length = 0
  else if (options.points === undefined) {
    if (drift === 'recurring') for (let p = period; p < length; p += period) points.push(p)
    else {
      const count = options.count ?? 1
      for (let i = 0; i < count; i++) points.push(Math.round(((i + 1) * length) / (count + 1)))
    }
  }
  for (const [i, p] of points.entries()) {
    if (!(p > 0 && p < length)) throw new MdmlError('value', `stream: drift point ${p} is outside (0, ${length})`)
    if (i > 0 && !(p - (points[i - 1] as number) >= Math.max(width, 1)))
      throw new MdmlError(
        'value',
        `stream: drift points ${points[i - 1]} and ${p} are closer than a transition (${width})`,
      )
  }
  const concepts = positiveInt('concepts', options.concepts ?? (drift === 'recurring' ? 2 : points.length + 1))
  const conceptOfSegment = (i: number): number => i % concepts

  // The drifts as planted.
  const half = width / 2
  const seen = new Set<number>([conceptOfSegment(0)])
  const drifts: PlantedDrift[] = points.map((p, i) => {
    const to = conceptOfSegment(i + 1)
    const record: PlantedDrift = {
      index: i,
      point: p,
      start: p - half,
      end: p + half,
      from: conceptOfSegment(i),
      to,
      recurring: seen.has(to),
    }
    seen.add(to)
    return record
  })

  const transitionAt = (t: number): Transition => {
    let segment = 0
    for (const [i, p] of points.entries()) {
      if (width > 0 && t > p - half && t < p + half) {
        return { from: conceptOfSegment(i), to: conceptOfSegment(i + 1), mix: (t - (p - half)) / width }
      }
      if (t >= p + half) segment = i + 1
    }
    const c = conceptOfSegment(segment)
    return { from: c, to: c, mix: 0 }
  }
  const conceptAt = (t: number): number => {
    const tr = transitionAt(t)
    return tr.mix >= 0.5 ? tr.to : tr.from
  }

  // The concepts' parameters: the first is the base; the others move away from it by the severity in what shifts.
  const [kBase, kConcepts, kData] = split(foldIn(k, 1), 3) as [Key, Key, Key]
  const [kW, kc, kM, kA] = split(kBase, 4) as [Key, Key, Key, Key]
  const drawn = (key: Key, n: number): Float64Array => Float64Array.from(normal(key, [n]).data)
  const baseWeights = drawn(kW, classes * dim)
  const baseCoefficients = drawn(kc, dim)
  const classMeans = Float64Array.from(drawn(kM, classes * dim), (v) => SEPARATION * v)
  const loadings = drawn(kA, dim)
  const uniformPriors = new Float64Array(classes).fill(1 / classes)
  const parameters: Parameters[] = Array.from({ length: concepts }, (_, c) => {
    const base: Parameters = {
      weights: baseWeights,
      coefficients: baseCoefficients,
      mean: new Float64Array(dim),
      priors: task === 'classification' ? uniformPriors : new Float64Array(1),
    }
    if (c === 0) return base
    const a = foldIn(kConcepts, c)
    if (shift === 'concept')
      return task === 'classification'
        ? { ...base, weights: blend(baseWeights, drawn(a, classes * dim), severity) }
        : { ...base, coefficients: blend(baseCoefficients, drawn(a, dim), severity) }
    if (shift === 'covariate') {
      const direction = drawn(a, dim)
      const norm = Math.hypot(...direction) || 1
      return { ...base, mean: Float64Array.from(direction, (v) => (3 * severity * v) / norm) }
    }
    if (task === 'regression') return { ...base, priors: Float64Array.of(3 * severity * c * (c % 2 === 0 ? -1 : 1)) }
    const onto = c % classes
    const skew = 0.9 * severity
    return {
      ...base,
      priors: Float64Array.from({ length: classes }, (_, j) => (1 - skew) / classes + (j === onto ? skew : 0)),
    }
  })

  const rule = (p: Parameters, x: ArrayLike<number>): number => {
    if (task === 'regression') {
      if (shift !== 'label') return p.coefficients.reduce((s, w, j) => s + w * (x[j] ?? 0), 0)
      // E[y | x] when y ~ N(m, 1) and x = a y + z: the posterior mean of a one-factor model.
      const m = p.priors[0] ?? 0
      let num = 0
      let aa = 0
      for (let j = 0; j < dim; j++) {
        num += (loadings[j] ?? 0) * ((x[j] ?? 0) - (loadings[j] ?? 0) * m)
        aa += (loadings[j] ?? 0) ** 2
      }
      return m + num / (1 + aa)
    }
    if (shift === 'label') {
      // The Bayes rule for x | y ~ N(mean_y, I): the prior and the squared distance to each class mean.
      const scores = Array.from({ length: classes }, (_, c) => {
        let d = 0
        for (let j = 0; j < dim; j++) d += ((x[j] ?? 0) - (classMeans[c * dim + j] ?? 0)) ** 2
        return Math.log(p.priors[c] ?? 1e-300) - d / 2
      })
      return argmax(scores)
    }
    return argmax(
      Array.from({ length: classes }, (_, c) => {
        let s = 0
        for (let j = 0; j < dim; j++) s += (p.weights[c * dim + j] ?? 0) * (x[j] ?? 0)
        return s
      }),
    )
  }

  const truth: StreamTruth = {
    task,
    shift,
    drift,
    length,
    batch,
    dim,
    classes,
    concepts,
    changepoints: points,
    drifts,
    transitionAt,
    conceptAt,
    parameters: (c) => {
      const p = parameters[c]
      if (p === undefined) throw new MdmlError('value', `stream: no concept ${c} among ${concepts}`)
      return {
        weights: Array.from(p.weights),
        coefficients: Array.from(p.coefficients),
        mean: Array.from(p.mean),
        priors: Array.from(p.priors),
      }
    },
    predict: (c, x) => rule(parameters[c] as Parameters, x),
  }

  const next = (t: number): StreamBatch => {
    if (!Number.isInteger(t) || t < 0 || t >= length)
      throw new MdmlError('value', `stream: step ${t} is outside [0, ${length})`)
    const { from, to, mix } = transitionAt(t)
    const [ku, kx, ky, kn] = split(foldIn(kData, t), 4) as [Key, Key, Key, Key]
    const pick = uniform(ku, [batch]).data
    const z = normal(kx, [batch, dim]).data
    const u = uniform(ky, [batch, 3]).data
    const e = normal(kn, [batch]).data
    const concept: number[] = []
    const xs = new Float64Array(batch * dim)
    const ys = new Float64Array(batch)
    const a = parameters[from] as Parameters
    const b = parameters[to] as Parameters
    const incremental = drift === 'incremental' && mix > 0
    const shared: Parameters | undefined = incremental
      ? {
          weights: lerp(a.weights, b.weights, mix),
          coefficients: lerp(a.coefficients, b.coefficients, mix),
          mean: lerp(a.mean, b.mean, mix),
          priors: lerp(a.priors, b.priors, mix),
        }
      : undefined
    for (let i = 0; i < batch; i++) {
      const c = mix === 0 ? from : incremental ? (mix >= 0.5 ? to : from) : (pick[i] ?? 0) < mix ? to : from
      concept.push(c)
      const p = shared ?? (parameters[c] as Parameters)
      const row = xs.subarray(i * dim, (i + 1) * dim)
      if (task === 'classification') {
        let y: number
        if (shift === 'label') {
          y = classes - 1
          let acc = 0
          const total = p.priors.reduce((s, v) => s + v, 0)
          for (let j = 0; j < classes; j++) {
            acc += (p.priors[j] ?? 0) / total
            if ((u[i * 3] ?? 0) < acc) {
              y = j
              break
            }
          }
          for (let j = 0; j < dim; j++) row[j] = (classMeans[y * dim + j] ?? 0) + (z[i * dim + j] ?? 0)
        } else {
          for (let j = 0; j < dim; j++) row[j] = (p.mean[j] ?? 0) + (z[i * dim + j] ?? 0)
          y = rule(p, row)
        }
        ys[i] = (u[i * 3 + 1] ?? 1) < noise ? Math.min(classes - 1, Math.floor((u[i * 3 + 2] ?? 0) * classes)) : y
      } else if (shift === 'label') {
        const y = (p.priors[0] ?? 0) + (e[i] ?? 0)
        for (let j = 0; j < dim; j++) row[j] = (loadings[j] ?? 0) * y + (z[i * dim + j] ?? 0)
        ys[i] = y
      } else {
        for (let j = 0; j < dim; j++) row[j] = (p.mean[j] ?? 0) + (z[i * dim + j] ?? 0)
        ys[i] = rule(p, row) + noise * (e[i] ?? 0)
      }
    }
    return {
      t,
      x: fromStorage(xs, [batch, dim], 'float64'),
      y:
        task === 'classification'
          ? fromStorage(Int32Array.from(ys), [batch], 'int32')
          : fromStorage(ys, [batch], 'float64'),
      concept,
      mix,
    }
  }

  return {
    options: {
      task,
      shift,
      drift,
      length,
      batch,
      dim,
      classes,
      count: points.length,
      points,
      width,
      period,
      concepts,
      severity,
      noise,
    },
    truth,
    next,
  }
}

/** The state of a stream run as an algorithm: its key, the steps delivered, and the last step's points. */
export interface StreamState {
  readonly key: Key
  /** The steps delivered so far. */
  readonly t: number
  /** The points of step `t - 1` (null before the first). */
  readonly batch: StreamBatch | null
}

/** The paths a stream run logs. */
export type StreamPath = 'stream/concept' | 'stream/mix' | 'stream/drift' | 'stream/x' | 'stream/y'

/** The paths a stream run logs: the truth beside the points. */
export const streamContract: Contract<StreamState, StreamPath> = defineContract({
  'stream/concept': scalar({ description: 'the dominant concept at the step' }),
  'stream/mix': scalar({ description: 'how far the transition under way has gone (0 outside one)' }),
  'stream/drift': scalar({ description: '1 at the step a drift is centred on, else 0' }),
  'stream/x': tensor({ role: 'output', shape: ['batch', 'dim'], description: 'the features of the step' }),
  'stream/y': tensor({ role: 'output', shape: ['batch'], description: 'the targets of the step' }),
})

/**
 * A stream as an `Algorithm`: one step delivers one batch, so a learner or a detector can be driven by the stepping of
 * the run, and the truth is logged beside the data (`stream/concept`, `stream/mix`, `stream/drift`, `stream/x`,
 * `stream/y`). The state serialises (the stream itself is rebuilt from the key), so a run can be sought.
 *
 * @param options - As for {@link stream}.
 * @returns The algorithm; it ends after `length` steps.
 * @concept datasets
 * @example
 * ```ts
 * import { streamAlgorithm } from 'mdml-engine/data'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = streamAlgorithm({ length: 20, drift: 'abrupt', points: [10], batch: 4 })
 * const { state, steps } = runAlgorithm(alg, { seed: 1, maxSteps: 100 })
 * [steps, state.batch?.x.shape] // => [20, [4, 2]]
 * ```
 */
export function streamAlgorithm(options: StreamOptions = {}): Algorithm<StreamState, StreamPath> {
  const cache = new Map<string, DataStream>()
  const streamOf = (k: Key): DataStream => {
    const id = k.words.join(',')
    let s = cache.get(id)
    if (s === undefined) {
      s = stream(k, options)
      cache.set(id, s)
    }
    return s
  }
  const length = positiveInt('length', options.length ?? 1000)
  return {
    name: 'stream',
    config: { ...options, points: options.points === undefined ? null : [...options.points] } as unknown as Json,
    init: (k) => {
      streamOf(k)
      return { key: k, t: 0, batch: null }
    },
    step: (s, ctx) => {
      const ds = streamOf(s.key)
      const b = ds.next(s.t)
      ctx.log('stream/concept', ds.truth.conceptAt(s.t))
      ctx.log('stream/mix', b.mix)
      ctx.log('stream/drift', ds.truth.changepoints.some((p) => Math.floor(p) === s.t) ? 1 : 0)
      ctx.log('stream/x', b.x)
      ctx.log('stream/y', b.y)
      return { key: s.key, t: s.t + 1, batch: b }
    },
    done: (s) => (s.t >= length ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamState,
  }
}
