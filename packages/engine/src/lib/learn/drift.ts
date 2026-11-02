/**
 * Stream concept-drift detection and distribution-monitoring statistics: Gama's Drift Detection Method (`ddmAlgorithm`,
 * `ddm`), Baena-Garcia's Early Drift Detection Method (`eddmAlgorithm`, `eddm`), Bifet and Gavalda's Adaptive Windowing
 * (`adwinAlgorithm`, `adwin`), Raab's Kolmogorov-Smirnov Windowing (`kswinAlgorithm`, `kswin`), and the Population
 * Stability Index (`populationStabilityIndex`).
 *
 * @index function Concept-drift detectors: DDM, EDDM, ADWIN, KSWIN.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, permutation } from '../random/index.ts'
import { ks2samp } from '../stats/tests/two-sample.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { keyFor, runOptionsFor } from './_shared.ts'
import type { FitOptions } from './estimator.ts'
import { fractionOption, integerOption, targetVector } from './validation.ts'

/** State of a stream drift detector at a single step. */
export type DriftStatus = 'normal' | 'warning' | 'drift'

/** Summary returned by the batch stream drift runners ({@link ddm}, {@link eddm}, {@link adwin}, {@link kswin}). */
export interface DriftResult {
  /** Detector status at each step `t = 0, ..., T - 1`. */
  readonly statuses: readonly DriftStatus[]
  /** Monitored test statistic at each step. */
  readonly statistics: readonly number[]
  /** Active drift threshold at each step. */
  readonly thresholds: readonly number[]
  /** Step indices at which a warning was raised. */
  readonly warnings: readonly number[]
  /** Step indices at which concept drift was detected. */
  readonly drifts: readonly number[]
  /** Active window or sample count at each step. */
  readonly windowSizes: readonly number[]
}

const detectorSchema = channels({
  metrics: {
    'detector/statistic': { description: 'current drift test statistic' },
    'detector/threshold': { description: 'active drift alarm threshold' },
    'detector/window': { description: 'active window size or sample count' },
  },
  messages: {
    'detector/alarm': {
      description: 'warning or concept-drift alarm emitted on the stream',
      payload: { t: 'number', status: 'string', statistic: 'number', threshold: 'number' },
    },
  },
})

/** Options of {@link ddmAlgorithm} and {@link ddm}. */
export interface DdmOptions extends FitOptions {
  /** Minimum observations in the current concept before testing for drift (default 30). */
  readonly minInstances?: number
  /** Warning multiplier `w_alpha` on `s_min` (default 2.0). */
  readonly warningLevel?: number
  /** Drift multiplier `d_alpha` on `s_min` (default 3.0). */
  readonly driftLevel?: number
  /** Reset statistics upon detecting drift (default true). */
  readonly resetOnDrift?: boolean
}

/** Serialisable state of {@link ddmAlgorithm}. */
export interface DdmState {
  readonly t: number
  readonly n: number
  readonly p: number
  readonly s: number
  readonly pMin: number
  readonly sMin: number
  readonly status: DriftStatus
  readonly statistic: number
  readonly threshold: number
  readonly statuses: readonly DriftStatus[]
  readonly statistics: readonly number[]
  readonly thresholds: readonly number[]
  readonly warnings: readonly number[]
  readonly drifts: readonly number[]
  readonly windowSizes: readonly number[]
}

/**
 * Gama's Drift Detection Method (DDM; Gama, Medas, Castillo and Rodrigues 2004) as a stepped {@link Algorithm} over
 * binary error indicators `e_t in {0, 1}`: track the online error rate `p_n` and its binomial standard deviation
 * `s_n = sqrt(p_n (1 - p_n) / n)`, signalling `'warning'` when `p_n + s_n >= p_min + warningLevel * s_min` and
 * `'drift'` when `p_n + s_n >= p_min + driftLevel * s_min`.
 *
 * @param errors - Stream of error indicators `[T]` (0 for correct, 1 for error).
 * @param options - `minInstances` (default 30), `warningLevel` (default 2), `driftLevel` (default 3) and `resetOnDrift`.
 * @returns Stepped {@link Algorithm} over {@link DdmState}.
 * @concept drift-detection
 * @example
 * ```ts
 * import { ddmAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const stream = [...Array.from({ length: 40 }, (_, i) => (i % 20 === 0 ? 1 : 0)), ...Array.from({ length: 30 }, () => 1)]
 * runAlgorithm(ddmAlgorithm(stream), { maxSteps: stream.length }).state.drifts.length > 0 // => true
 * ```
 */
export function ddmAlgorithm(errors: TensorLike, options: DdmOptions = {}): Algorithm<DdmState> {
  const vec = targetVector(errors, 'ddm')
  const minInstances = integerOption(options.minInstances ?? 30, 'minInstances', 'ddm', 1)
  const warningLevel = options.warningLevel ?? 2.0
  const driftLevel = options.driftLevel ?? 3.0
  if (!(warningLevel > 0 && driftLevel >= warningLevel)) {
    throw new MdmlError('value', 'ddm: need 0 < warningLevel <= driftLevel')
  }
  const resetOnDrift = options.resetOnDrift ?? true

  return {
    name: 'ddm',
    config: { T: vec.length, minInstances, warningLevel, driftLevel, resetOnDrift },
    schema: detectorSchema,
    init: () => ({
      t: 0,
      n: 0,
      p: 0,
      s: 0,
      pMin: Number.POSITIVE_INFINITY,
      sMin: Number.POSITIVE_INFINITY,
      status: 'normal',
      statistic: 0,
      threshold: 1,
      statuses: [],
      statistics: [],
      thresholds: [],
      warnings: [],
      drifts: [],
      windowSizes: [],
    }),
    step: (st) => {
      const err = (vec[st.t] ?? 0) !== 0 ? 1 : 0
      const baseN = st.status === 'drift' && resetOnDrift ? 0 : st.n
      const baseP = st.status === 'drift' && resetOnDrift ? 0 : st.p
      let pMin = st.status === 'drift' && resetOnDrift ? Number.POSITIVE_INFINITY : st.pMin
      let sMin = st.status === 'drift' && resetOnDrift ? Number.POSITIVE_INFINITY : st.sMin

      const n = baseN + 1
      const p = baseP + (err - baseP) / n
      const s = Math.sqrt(Math.max(0, p * (1 - p)) / n)
      const stat = p + s
      let status: DriftStatus = 'normal'
      let threshold = Number.isFinite(pMin) ? pMin + driftLevel * sMin : 1

      if (n >= minInstances) {
        if (stat < pMin + sMin) {
          pMin = p
          sMin = s
        }
        threshold = pMin + driftLevel * sMin
        if (stat >= pMin + driftLevel * sMin) {
          status = 'drift'
        } else if (stat >= pMin + warningLevel * sMin) {
          status = 'warning'
        }
      }

      return {
        t: st.t + 1,
        n,
        p,
        s,
        pMin,
        sMin,
        status,
        statistic: stat,
        threshold,
        statuses: [...st.statuses, status],
        statistics: [...st.statistics, stat],
        thresholds: [...st.thresholds, threshold],
        warnings: status === 'warning' ? [...st.warnings, st.t] : st.warnings,
        drifts: status === 'drift' ? [...st.drifts, st.t] : st.drifts,
        windowSizes: [...st.windowSizes, n],
      }
    },
    done: (st) => (st.t >= vec.length ? 'completed' : undefined),
    observe: (st) => ({
      metrics: {
        'detector/statistic': st.statistic,
        'detector/threshold': st.threshold,
        'detector/window': st.n,
      },
      messages:
        st.status === 'normal'
          ? []
          : [
              {
                kind: 'detector/alarm',
                payload: { t: st.t - 1, status: st.status, statistic: st.statistic, threshold: st.threshold },
              },
            ],
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as DdmState,
  }
}

/**
 * Gama's Drift Detection Method (DDM; Gama et al. 2004) over a stream of error bits.
 *
 * @param errors - Binary error sequence `[T]`.
 * @param options - `minInstances`, `warningLevel`, `driftLevel`, `resetOnDrift` and trace options.
 * @returns The {@link DriftResult} trajectory.
 * @concept drift-detection
 * @example
 * ```ts
 * import { ddm } from 'mdml-engine/learn'
 * const s = [...Array.from({ length: 40 }, (_, i) => (i % 20 === 0 ? 1 : 0)), ...Array.from({ length: 30 }, () => 1)]
 * ddm(s).drifts.length > 0 // => true
 * ```
 */
export function ddm(errors: TensorLike, options: DdmOptions = {}): DriftResult {
  const vec = targetVector(errors, 'ddm')
  const alg = ddmAlgorithm(vec, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'ddm'), maxSteps: vec.length })
  return {
    statuses: state.statuses,
    statistics: state.statistics,
    thresholds: state.thresholds,
    warnings: state.warnings,
    drifts: state.drifts,
    windowSizes: state.windowSizes,
  }
}

/** Options of {@link eddmAlgorithm} and {@link eddm}. */
export interface EddmOptions extends FitOptions {
  /** Minimum number of errors before testing for warning or drift (default 30). */
  readonly minErrors?: number
  /** Warning ratio threshold `alpha` in `(0, 1)` (default 0.95). */
  readonly warningLevel?: number
  /** Drift ratio threshold `beta` in `(0, alpha]` (default 0.90). */
  readonly driftLevel?: number
  /** Reset statistics after a drift alarm (default true). */
  readonly resetOnDrift?: boolean
}

/** Serialisable state of {@link eddmAlgorithm}. */
export interface EddmState {
  readonly t: number
  readonly windowStart: number
  readonly lastErrorStep: number
  readonly numErrors: number
  readonly meanDist: number
  readonly m2Dist: number
  readonly maxMetric: number
  readonly status: DriftStatus
  readonly statistic: number
  readonly statuses: readonly DriftStatus[]
  readonly statistics: readonly number[]
  readonly thresholds: readonly number[]
  readonly warnings: readonly number[]
  readonly drifts: readonly number[]
  readonly windowSizes: readonly number[]
}

/**
 * Early Drift Detection Method (EDDM; Baena-Garcia et al. 2006) as a stepped {@link Algorithm}: track the mean `p'`
 * and standard deviation `s'` of the distance (number of steps) between consecutive errors, comparing
 * `(p' + 2 s') / max(p' + 2 s')` against `warningLevel` (0.95) and `driftLevel` (0.90).
 *
 * @param errors - Stream of error indicators `[T]` (0 for correct, 1 for error).
 * @param options - `minErrors` (default 30), `warningLevel` (default 0.95), `driftLevel` (default 0.90), `resetOnDrift`.
 * @returns Stepped {@link Algorithm} over {@link EddmState}.
 * @concept drift-detection
 * @example
 * ```ts
 * import { eddmAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const slow = Array.from({ length: 120 }, (_, i) => (i % 10 === 0 ? 1 : 0))
 * const fast = Array.from({ length: 40 }, () => 1)
 * const stream = [...slow, ...fast]
 * runAlgorithm(eddmAlgorithm(stream, { minErrors: 10 }), { maxSteps: stream.length }).state.drifts.length > 0 // => true
 * ```
 */
export function eddmAlgorithm(errors: TensorLike, options: EddmOptions = {}): Algorithm<EddmState> {
  const vec = targetVector(errors, 'eddm')
  const minErrors = integerOption(options.minErrors ?? 30, 'minErrors', 'eddm', 2)
  const warningLevel = fractionOption(options.warningLevel ?? 0.95, 'warningLevel', 'eddm')
  const driftLevel = fractionOption(options.driftLevel ?? 0.9, 'driftLevel', 'eddm')
  if (driftLevel > warningLevel) throw new MdmlError('value', 'eddm: driftLevel must be <= warningLevel')
  const resetOnDrift = options.resetOnDrift ?? true

  return {
    name: 'eddm',
    config: { T: vec.length, minErrors, warningLevel, driftLevel, resetOnDrift },
    schema: detectorSchema,
    init: () => ({
      t: 0,
      windowStart: 0,
      lastErrorStep: -1,
      numErrors: 0,
      meanDist: 0,
      m2Dist: 0,
      maxMetric: 0,
      status: 'normal',
      statistic: 1,
      statuses: [],
      statistics: [],
      thresholds: [],
      warnings: [],
      drifts: [],
      windowSizes: [],
    }),
    step: (st) => {
      const reset = st.status === 'drift' && resetOnDrift
      const windowStart = reset ? st.t : st.windowStart
      let lastErrorStep = reset ? -1 : st.lastErrorStep
      let numErrors = reset ? 0 : st.numErrors
      let meanDist = reset ? 0 : st.meanDist
      let m2Dist = reset ? 0 : st.m2Dist
      let maxMetric = reset ? 0 : st.maxMetric
      let status: DriftStatus = 'normal'
      let statistic = reset ? 1 : st.statistic

      const isErr = (vec[st.t] ?? 0) !== 0
      if (isErr) {
        numErrors += 1
        if (lastErrorStep >= 0) {
          const dist = st.t - lastErrorStep
          const k = numErrors - 1
          const delta = dist - meanDist
          meanDist += delta / k
          m2Dist += delta * (dist - meanDist)
          const std = Math.sqrt(m2Dist / k)
          const metric = meanDist + 2 * std
          if (metric > maxMetric) maxMetric = metric
          if (numErrors >= minErrors && maxMetric > 0) {
            statistic = metric / maxMetric
            if (statistic < driftLevel) status = 'drift'
            else if (statistic < warningLevel) status = 'warning'
          }
        }
        lastErrorStep = st.t
      }

      const winSize = st.t - windowStart + 1
      return {
        t: st.t + 1,
        windowStart,
        lastErrorStep,
        numErrors,
        meanDist,
        m2Dist,
        maxMetric,
        status,
        statistic,
        statuses: [...st.statuses, status],
        statistics: [...st.statistics, statistic],
        thresholds: [...st.thresholds, driftLevel],
        warnings: status === 'warning' ? [...st.warnings, st.t] : st.warnings,
        drifts: status === 'drift' ? [...st.drifts, st.t] : st.drifts,
        windowSizes: [...st.windowSizes, winSize],
      }
    },
    done: (st) => (st.t >= vec.length ? 'completed' : undefined),
    observe: (st) => ({
      metrics: {
        'detector/statistic': st.statistic,
        'detector/threshold': driftLevel,
        'detector/window': st.t - st.windowStart,
      },
      messages:
        st.status === 'normal'
          ? []
          : [
              {
                kind: 'detector/alarm',
                payload: { t: st.t - 1, status: st.status, statistic: st.statistic, threshold: driftLevel },
              },
            ],
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as EddmState,
  }
}

/**
 * Early Drift Detection Method (EDDM; Baena-Garcia et al. 2006) over a stream of error bits.
 *
 * @param errors - Binary error sequence `[T]`.
 * @param options - `minErrors`, `warningLevel`, `driftLevel`, `resetOnDrift` and trace options.
 * @returns The {@link DriftResult} trajectory.
 * @concept drift-detection
 * @example
 * ```ts
 * import { eddm } from 'mdml-engine/learn'
 * const slow = Array.from({ length: 120 }, (_, i) => (i % 10 === 0 ? 1 : 0))
 * const fast = Array.from({ length: 40 }, () => 1)
 * eddm([...slow, ...fast], { minErrors: 10 }).drifts.length > 0 // => true
 * ```
 */
export function eddm(errors: TensorLike, options: EddmOptions = {}): DriftResult {
  const vec = targetVector(errors, 'eddm')
  const alg = eddmAlgorithm(vec, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'eddm'), maxSteps: vec.length })
  return {
    statuses: state.statuses,
    statistics: state.statistics,
    thresholds: state.thresholds,
    warnings: state.warnings,
    drifts: state.drifts,
    windowSizes: state.windowSizes,
  }
}

/** A bucket in ADWIN's exponential histogram. */
export interface AdwinBucket {
  readonly count: number
  readonly sum: number
  readonly variance: number
}

/** Options of {@link adwinAlgorithm} and {@link adwin}. */
export interface AdwinOptions extends FitOptions {
  /** Confidence parameter `delta` in `(0, 1)` for the Hoeffding/Bernstein cut test (default 0.002). */
  readonly delta?: number
  /** Minimum sub-window size on each side of a candidate cut (default 5). */
  readonly minSubwindow?: number
  /** Maximum buckets per power-of-two level in the exponential histogram (default 5). */
  readonly maxBuckets?: number
}

/** Serialisable state of {@link adwinAlgorithm}. */
export interface AdwinState {
  readonly t: number
  readonly buckets: readonly AdwinBucket[]
  readonly totalCount: number
  readonly totalSum: number
  readonly totalVar: number
  readonly status: DriftStatus
  readonly statistic: number
  readonly threshold: number
  readonly statuses: readonly DriftStatus[]
  readonly statistics: readonly number[]
  readonly thresholds: readonly number[]
  readonly warnings: readonly number[]
  readonly drifts: readonly number[]
  readonly windowSizes: readonly number[]
}

function compressBuckets(buckets: readonly AdwinBucket[], maxBuckets: number): AdwinBucket[] {
  const out = [...buckets]
  // buckets are ordered from oldest (index 0) to newest (last index).
  let size = 1
  while (size <= (out[0]?.count ?? 0)) {
    const indices: number[] = []
    for (let i = 0; i < out.length; i++) {
      if (out[i]?.count === size) indices.push(i)
    }
    if (indices.length > maxBuckets) {
      // Merge the two oldest buckets of this size.
      const i0 = indices[0] as number
      const i1 = indices[1] as number
      const b0 = out[i0] as AdwinBucket
      const b1 = out[i1] as AdwinBucket
      const count = b0.count + b1.count
      const sum = b0.sum + b1.sum
      const diffMean = b0.sum / b0.count - b1.sum / b1.count
      const variance = b0.variance + b1.variance + (b0.count * b1.count * diffMean * diffMean) / count
      out.splice(i1, 1)
      out[i0] = { count, sum, variance }
    } else {
      size *= 2
    }
  }
  return out
}

function summarizeBuckets(buckets: readonly AdwinBucket[]): {
  readonly count: number
  readonly sum: number
  readonly variance: number
} {
  let count = 0
  let sum = 0
  let variance = 0
  for (const b of buckets) {
    if (count === 0) {
      count = b.count
      sum = b.sum
      variance = b.variance
    } else {
      const nextCount = count + b.count
      const diff = sum / count - b.sum / b.count
      variance += b.variance + (count * b.count * diff * diff) / nextCount
      count = nextCount
      sum += b.sum
    }
  }
  return { count, sum, variance }
}

/**
 * Adaptive Windowing (ADWIN; Bifet and Gavalda 2007) as a stepped {@link Algorithm}: maintain an exponential histogram
 * of stream observations and test bucket boundaries for a statistically significant difference between sub-window means
 * `|mu_0 - mu_1| >= eps_cut`, dropping the older prefix whenever drift is detected.
 *
 * @param stream - Bounded real-valued stream `[T]`.
 * @param options - `delta` (default 0.002), `minSubwindow` (default 5), `maxBuckets` (default 5) and trace options.
 * @returns Stepped {@link Algorithm} over {@link AdwinState}.
 * @concept drift-detection
 * @example
 * ```ts
 * import { adwinAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const stream = [...Array.from({ length: 40 }, () => 0), ...Array.from({ length: 40 }, () => 1)]
 * const st = runAlgorithm(adwinAlgorithm(stream, { delta: 0.01 }), { maxSteps: stream.length }).state
 * st.drifts.length > 0 && st.totalCount < stream.length // => true
 * ```
 */
export function adwinAlgorithm(stream: TensorLike, options: AdwinOptions = {}): Algorithm<AdwinState> {
  const vec = targetVector(stream, 'adwin')
  const delta = fractionOption(options.delta ?? 0.002, 'delta', 'adwin')
  const minSubwindow = integerOption(options.minSubwindow ?? 5, 'minSubwindow', 'adwin', 1)
  const maxBuckets = integerOption(options.maxBuckets ?? 5, 'maxBuckets', 'adwin', 2)

  return {
    name: 'adwin',
    config: { T: vec.length, delta, minSubwindow, maxBuckets },
    schema: detectorSchema,
    init: () => ({
      t: 0,
      buckets: [],
      totalCount: 0,
      totalSum: 0,
      totalVar: 0,
      status: 'normal',
      statistic: 0,
      threshold: 1,
      statuses: [],
      statistics: [],
      thresholds: [],
      warnings: [],
      drifts: [],
      windowSizes: [],
    }),
    step: (st) => {
      const x = vec[st.t] ?? 0
      let buckets = compressBuckets([...st.buckets, { count: 1, sum: x, variance: 0 }], maxBuckets)
      let summary = summarizeBuckets(buckets)
      let status: DriftStatus = 'normal'
      let bestStat = 0
      let bestThresh = 1

      // Repeatedly test cuts from oldest prefix until no sub-window violates the Hoeffding/Bernstein bound.
      let cutFound = true
      while (cutFound && summary.count >= 2 * minSubwindow && buckets.length >= 2) {
        cutFound = false
        const n = summary.count
        const winVar = summary.variance / n
        const deltaPrime = delta / n
        const logTerm = Math.log(2 / deltaPrime)
        let n0 = 0
        let sum0 = 0
        for (let i = 0; i < buckets.length - 1; i++) {
          const b = buckets[i] as AdwinBucket
          n0 += b.count
          sum0 += b.sum
          const n1 = n - n0
          if (n0 < minSubwindow || n1 < minSubwindow) continue
          const sum1 = summary.sum - sum0
          const diff = Math.abs(sum0 / n0 - sum1 / n1)
          const m = 1 / (1 / n0 + 1 / n1)
          const eps = Math.sqrt((2 / m) * winVar * logTerm) + (2 / (3 * m)) * logTerm
          if (diff - bestStat > bestThresh - eps || bestStat === 0) {
            bestStat = diff
            bestThresh = eps
          }
          if (diff >= eps) {
            // Drop the oldest bucket and re-evaluate.
            buckets = buckets.slice(1)
            summary = summarizeBuckets(buckets)
            status = 'drift'
            cutFound = true
            break
          }
        }
      }

      return {
        t: st.t + 1,
        buckets,
        totalCount: summary.count,
        totalSum: summary.sum,
        totalVar: summary.variance,
        status,
        statistic: bestStat,
        threshold: bestThresh,
        statuses: [...st.statuses, status],
        statistics: [...st.statistics, bestStat],
        thresholds: [...st.thresholds, bestThresh],
        warnings: st.warnings,
        drifts: status === 'drift' ? [...st.drifts, st.t] : st.drifts,
        windowSizes: [...st.windowSizes, summary.count],
      }
    },
    done: (st) => (st.t >= vec.length ? 'completed' : undefined),
    observe: (st) => ({
      metrics: {
        'detector/statistic': st.statistic,
        'detector/threshold': st.threshold,
        'detector/window': st.totalCount,
      },
      messages:
        st.status === 'normal'
          ? []
          : [
              {
                kind: 'detector/alarm',
                payload: { t: st.t - 1, status: st.status, statistic: st.statistic, threshold: st.threshold },
              },
            ],
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as AdwinState,
  }
}

/**
 * Adaptive Windowing (ADWIN; Bifet and Gavalda 2007) over a stream.
 *
 * @param stream - Real-valued stream `[T]`.
 * @param options - `delta`, `minSubwindow`, `maxBuckets` and trace options.
 * @returns The {@link DriftResult} trajectory.
 * @concept drift-detection
 * @example
 * ```ts
 * import { adwin } from 'mdml-engine/learn'
 * const r = adwin([...Array.from({ length: 40 }, () => 0), ...Array.from({ length: 40 }, () => 1)], { delta: 0.01 })
 * r.drifts.length > 0 // => true
 * ```
 */
export function adwin(stream: TensorLike, options: AdwinOptions = {}): DriftResult {
  const vec = targetVector(stream, 'adwin')
  const alg = adwinAlgorithm(vec, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'adwin'), maxSteps: vec.length })
  return {
    statuses: state.statuses,
    statistics: state.statistics,
    thresholds: state.thresholds,
    warnings: state.warnings,
    drifts: state.drifts,
    windowSizes: state.windowSizes,
  }
}

/** Options of {@link kswinAlgorithm} and {@link kswin}. */
export interface KswinOptions extends FitOptions {
  /** Significance level `alpha` in `(0, 1)` for the two-sample Kolmogorov-Smirnov test (default 0.005). */
  readonly alpha?: number
  /** Full sliding window capacity `windowSize` (default 100). */
  readonly windowSize?: number
  /** Size `r` of the recent sub-window compared against the older reference sub-window (default 30). */
  readonly statSize?: number
  /** Subsample `statSize` points from the older window (default true; false uses all `windowSize - statSize` points). */
  readonly subsample?: boolean
}

/** Serialisable state of {@link kswinAlgorithm}. */
export interface KswinState {
  readonly t: number
  readonly window: readonly number[]
  readonly status: DriftStatus
  readonly statistic: number
  readonly threshold: number
  readonly pValue: number
  readonly statuses: readonly DriftStatus[]
  readonly statistics: readonly number[]
  readonly thresholds: readonly number[]
  readonly warnings: readonly number[]
  readonly drifts: readonly number[]
  readonly windowSizes: readonly number[]
}

/**
 * Kolmogorov-Smirnov Windowing (KSWIN; Raab, Heusinger and Schleif 2020) as a stepped {@link Algorithm}: maintain a
 * sliding window of length `windowSize`, compare the `statSize` most recent observations against the older observations
 * by the two-sample Kolmogorov-Smirnov test (`ks2samp`), and shrink the window to the recent observations when drift is
 * detected.
 *
 * @param stream - Real-valued stream `[T]`.
 * @param options - `alpha` (default 0.005), `windowSize` (default 100), `statSize` (default 30), `subsample` and `key`.
 * @returns Stepped {@link Algorithm} over {@link KswinState}.
 * @concept drift-detection
 * @example
 * ```ts
 * import { kswinAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const s = [...Array.from({ length: 40 }, (_, i) => (i % 5) * 0.1), ...Array.from({ length: 30 }, (_, i) => 10 + (i % 5) * 0.1)]
 * const alg = kswinAlgorithm(s, { windowSize: 40, statSize: 15, alpha: 0.01 })
 * runAlgorithm(alg, { maxSteps: s.length }).state.drifts.length > 0 // => true
 * ```
 */
export function kswinAlgorithm(stream: TensorLike, options: KswinOptions = {}): Algorithm<KswinState> {
  const vec = targetVector(stream, 'kswin')
  const alpha = fractionOption(options.alpha ?? 0.005, 'alpha', 'kswin')
  const windowSize = integerOption(options.windowSize ?? 100, 'windowSize', 'kswin', 4)
  const statSize = integerOption(options.statSize ?? Math.min(30, Math.floor(windowSize / 2)), 'statSize', 'kswin', 2)
  if (statSize >= windowSize) throw new MdmlError('value', 'kswin: statSize must be < windowSize')
  const subsample = options.subsample ?? true
  const baseKey = keyFor(options, 'kswin')

  return {
    name: 'kswin',
    config: { T: vec.length, alpha, windowSize, statSize, subsample },
    schema: detectorSchema,
    init: () => ({
      t: 0,
      window: [],
      status: 'normal',
      statistic: 0,
      threshold: Math.sqrt(-Math.log(alpha * 0.5) * (1 / statSize)),
      pValue: 1,
      statuses: [],
      statistics: [],
      thresholds: [],
      warnings: [],
      drifts: [],
      windowSizes: [],
    }),
    step: (st) => {
      const x = vec[st.t] ?? 0
      let win = [...st.window, x]
      if (win.length > windowSize) win = win.slice(win.length - windowSize)

      let status: DriftStatus = 'normal'
      let statistic = 0
      let pValue = 1
      const nOld = win.length - statSize
      const nRef = subsample ? Math.min(statSize, Math.max(1, nOld)) : Math.max(1, nOld)
      const threshold = Math.sqrt(-Math.log(alpha * 0.5) * ((nRef + statSize) / (2 * nRef * statSize)))

      if (win.length >= windowSize) {
        const older = win.slice(0, win.length - statSize)
        const recent = win.slice(win.length - statSize)
        const ref =
          subsample && older.length > statSize
            ? toFlat(permutation(foldIn(baseKey, st.t), older.length))
                .slice(0, statSize)
                .map((idx) => older[idx] ?? 0)
            : older
        const res = ks2samp(ref, recent)
        statistic = res.statistic
        pValue = res.pValue
        if (statistic >= threshold && pValue <= alpha) {
          status = 'drift'
          win = recent
        }
      }

      return {
        t: st.t + 1,
        window: win,
        status,
        statistic,
        threshold,
        pValue,
        statuses: [...st.statuses, status],
        statistics: [...st.statistics, statistic],
        thresholds: [...st.thresholds, threshold],
        warnings: st.warnings,
        drifts: status === 'drift' ? [...st.drifts, st.t] : st.drifts,
        windowSizes: [...st.windowSizes, win.length],
      }
    },
    done: (st) => (st.t >= vec.length ? 'completed' : undefined),
    observe: (st) => ({
      metrics: {
        'detector/statistic': st.statistic,
        'detector/threshold': st.threshold,
        'detector/window': st.window.length,
      },
      messages:
        st.status === 'normal'
          ? []
          : [
              {
                kind: 'detector/alarm',
                payload: { t: st.t - 1, status: st.status, statistic: st.statistic, threshold: st.threshold },
              },
            ],
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as KswinState,
  }
}

/**
 * Kolmogorov-Smirnov Windowing (KSWIN; Raab et al. 2020) over a stream.
 *
 * @param stream - Real-valued stream `[T]`.
 * @param options - `alpha`, `windowSize`, `statSize`, `subsample`, `key` and trace options.
 * @returns The {@link DriftResult} trajectory.
 * @concept drift-detection
 * @example
 * ```ts
 * import { kswin } from 'mdml-engine/learn'
 * const s = [...Array.from({ length: 40 }, (_, i) => (i % 5) * 0.1), ...Array.from({ length: 30 }, (_, i) => 10 + (i % 5) * 0.1)]
 * kswin(s, { windowSize: 40, statSize: 15, alpha: 0.01 }).drifts.length > 0 // => true
 * ```
 */
export function kswin(stream: TensorLike, options: KswinOptions = {}): DriftResult {
  const vec = targetVector(stream, 'kswin')
  const alg = kswinAlgorithm(vec, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'kswin'), maxSteps: vec.length })
  return {
    statuses: state.statuses,
    statistics: state.statistics,
    thresholds: state.thresholds,
    warnings: state.warnings,
    drifts: state.drifts,
    windowSizes: state.windowSizes,
  }
}

/** Options of {@link populationStabilityIndex}. */
export interface PsiOptions {
  /** Number of histogram bins (default 10). */
  readonly nBins?: number
  /** Binning strategy: reference quantiles (`'quantile'`, default) or equal-width (`'uniform'`). */
  readonly strategy?: 'quantile' | 'uniform'
  /** Zero-proportion smoothing floor (default 1e-4). */
  readonly eps?: number
}

/** Result of {@link populationStabilityIndex}. */
export interface PsiResult {
  /** Total Population Stability Index `sum_b (q_b - p_b) ln(q_b / p_b)`. */
  readonly psi: number
  /** Per-bin PSI contributions `[nBins]`. */
  readonly bins: readonly number[]
  /** Internal bin edges `[nBins + 1]`. */
  readonly edges: readonly number[]
  /** Reference bin proportions `[nBins]`. */
  readonly referenceProportions: readonly number[]
  /** Current bin proportions `[nBins]`. */
  readonly currentProportions: readonly number[]
  /** Rule-of-thumb verdict: `'normal'` (< 0.1), `'warning'` ([0.1, 0.25)) or `'drift'` (>= 0.25). */
  readonly status: DriftStatus
}

/**
 * Population Stability Index (PSI) between a reference distribution and a current distribution:
 * `PSI = sum_{b=1}^B (q_b - p_b) ln(q_b / p_b)`, where `p_b` and `q_b` are the reference and current bin proportions.
 *
 * @param reference - Reference sample `[n_ref]`.
 * @param current - Current sample `[n_cur]`.
 * @param options - `nBins` (default 10), `strategy` (`'quantile'` or `'uniform'`) and `eps` smoothing (default 1e-4).
 * @returns Total `psi`, per-bin `bins`, `edges`, bin proportions and `status`.
 * @concept drift-detection
 * @example
 * ```ts
 * import { populationStabilityIndex } from 'mdml-engine/learn'
 * const ref = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 * populationStabilityIndex(ref, ref, { nBins: 5 }).psi // => 0
 * populationStabilityIndex(ref, ref.map((v) => v + 10), { nBins: 5 }).status // => 'drift'
 * ```
 */
export function populationStabilityIndex(
  reference: TensorLike,
  current: TensorLike,
  options: PsiOptions = {},
): PsiResult {
  const ref = targetVector(reference, 'populationStabilityIndex')
  const cur = targetVector(current, 'populationStabilityIndex')
  if (ref.length === 0 || cur.length === 0) {
    throw new MdmlError('value', 'populationStabilityIndex: reference and current must be non-empty')
  }
  const k = integerOption(options.nBins ?? 10, 'nBins', 'populationStabilityIndex', 2)
  const eps = options.eps ?? 1e-4
  if (!(eps > 0 && eps < 1)) throw new MdmlError('value', 'populationStabilityIndex: eps must lie in (0, 1)')

  const sortedRef = Float64Array.from(ref).sort()
  const minVal = Math.min(
    sortedRef[0] ?? 0,
    cur.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY),
  )
  const maxVal = Math.max(
    sortedRef[sortedRef.length - 1] ?? 0,
    cur.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY),
  )

  const edges: number[] =
    options.strategy === 'uniform'
      ? Array.from({ length: k + 1 }, (_, i) => minVal + ((maxVal - minVal) * i) / k)
      : Array.from({ length: k + 1 }, (_, i) => {
          const pos = ((sortedRef.length - 1) * i) / k
          const lo = Math.floor(pos)
          const hi = Math.min(sortedRef.length - 1, lo + 1)
          return (sortedRef[lo] ?? 0) + (pos - lo) * ((sortedRef[hi] ?? 0) - (sortedRef[lo] ?? 0))
        })

  const binOf = (v: number): number => {
    let b = 0
    for (let i = 1; i < k; i++) {
      if (v >= (edges[i] ?? 0)) b = i
    }
    return b
  }

  const refCounts = new Float64Array(k)
  const curCounts = new Float64Array(k)
  for (const v of ref) refCounts[binOf(v)] = (refCounts[binOf(v)] ?? 0) + 1
  for (const v of cur) curCounts[binOf(v)] = (curCounts[binOf(v)] ?? 0) + 1

  const refProps: number[] = []
  const curProps: number[] = []
  const bins: number[] = []
  let totalPsi = 0
  for (let b = 0; b < k; b++) {
    const pRaw = (refCounts[b] ?? 0) / ref.length
    const qRaw = (curCounts[b] ?? 0) / cur.length
    refProps.push(pRaw)
    curProps.push(qRaw)
    const p = Math.max(pRaw, eps)
    const q = Math.max(qRaw, eps)
    const contrib = (q - p) * Math.log(q / p)
    bins.push(contrib)
    totalPsi += contrib
  }
  const status: DriftStatus = totalPsi >= 0.25 ? 'drift' : totalPsi >= 0.1 ? 'warning' : 'normal'
  return {
    psi: totalPsi,
    bins,
    edges,
    referenceProportions: refProps,
    currentProportions: curProps,
    status,
  }
}
