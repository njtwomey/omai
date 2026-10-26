/**
 * Hidden Markov models (as in hmmlearn): a sequence explained by hidden states that follow a Markov chain, each
 * emitting observations. Categorical and Gaussian emissions fitted by Baum-Welch (EM over forward-backward), traced
 * iteration by iteration, with state posteriors and Viterbi paths as exposition, plus the occasionally dishonest casino
 * of Durbin et al. as a worked study.
 */

// TODO: Gaussian-mixture emissions (hmmlearn's GMMHMM), tied and spherical covariances, Baum-Welch through the
// probprog EM engine on the HMM as a program, Bayesian (Dirichlet-prior) and variational HMMs.
import { MdmlError } from '../../lib/base/index.ts'
import {
  coefficientTable,
  decisionSurface,
  type Exposition,
  mixtureSteps,
  posteriorGrid,
  stateSequence,
} from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  fitRunOptions,
  type LearnData,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import {
  forwardBackward,
  type HiddenMarkovModel,
  hiddenMarkovModel,
  hmmChain,
  hmmSample,
  viterbi,
} from '../../lib/pgm/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../lib/random/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { stationaryDistribution } from '../../lib/stochastic/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { kmeans } from '../cluster/index.ts'

/** The parameters of a hidden Markov model as plain data (hmmlearn's attribute names in camel case). */
export type HmmParams = {
  readonly emission: 'categorical' | 'gaussian'
  readonly startprob: number[]
  /** transmat[i][j] = P(z_{t+1} = j | z_t = i). */
  readonly transmat: number[][]
  /** Categorical emissions: emissionprob[k][s] = P(x = s | z = k). */
  readonly emissionprob?: number[][]
  /** Gaussian emissions: a mean and a full covariance per state. */
  readonly means?: number[][]
  readonly covars?: number[][][]
  /** The log likelihood of the training sequences under the parameters before the last M-step (as hmmlearn). */
  readonly logLikelihood: number
  readonly nIter: number
  readonly converged: boolean
  readonly stateNames: string[]
}

/** Starting parameters for Baum-Welch (any omitted part is initialised as hmmlearn does). */
export interface HmmStart {
  readonly startprob?: readonly number[]
  readonly transmat?: readonly (readonly number[])[]
  readonly emissionprob?: readonly (readonly number[])[]
  readonly means?: readonly (readonly number[])[]
  readonly covars?: readonly (readonly (readonly number[])[])[]
}

/** Hyperparameters of the HMM estimators. */
export interface HmmHyper {
  /** Hidden states (default 2). */
  readonly nComponents: number
  readonly emission: 'categorical' | 'gaussian'
  /** Gaussian emissions: diagonal (hmmlearn's default) or full covariances. */
  readonly covarianceType: 'diag' | 'full'
  /** Symbols of categorical emissions (default the largest observed symbol plus one). */
  readonly nFeatures?: number
  /** Baum-Welch iterations (default 100). */
  readonly nIter: number
  /** Stop when the log likelihood gains less than this (default 1e-2, as hmmlearn). */
  readonly tol: number
  /** hmmlearn's covariance prior, added to the scatter before dividing by the state occupancy (default 1e-2). */
  readonly covarsPrior: number
  /** Added to the data variance for the initial covariances (default 1e-3). */
  readonly minCovar: number
  readonly init?: HmmStart
  /** State names (default `s0, s1, ...`). */
  readonly stateNames?: readonly string[]
  /** Most Baum-Welch snapshots kept in the exposition (default 50). */
  readonly maxFrames: number
}

/** Verbs of a fitted HMM. */
export interface HmmVerbs {
  /** The Viterbi path of a sequence (rows are time steps). */
  readonly predict: (x: TensorLike) => Tensor
  /** Posterior state probabilities `[T, K]` (forward-backward). */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The log likelihood of a sequence. */
  readonly score: (data: LearnData) => number
  /** The fitted model as the core's `HiddenMarkovModel`. */
  readonly model: HiddenMarkovModel
  readonly nFeaturesIn: number
}

/** A fitted HMM. */
export type FittedHmm = ExplainedFitted<HmmParams, HmmVerbs>

/** The core model of HMM parameters. */
export function hmmOf(
  p: Pick<HmmParams, 'emission' | 'startprob' | 'transmat' | 'emissionprob' | 'means' | 'covars' | 'stateNames'>,
): HiddenMarkovModel {
  const normal = (row: readonly number[]): number[] => {
    const s = row.reduce((a, v) => a + v, 0)
    return row.map((v) => v / s)
  }
  return hiddenMarkovModel({
    states: p.stateNames,
    initial: normal(p.startprob),
    transition: p.transmat.map(normal),
    emission:
      p.emission === 'categorical'
        ? {
            kind: 'categorical',
            symbols: (p.emissionprob?.[0] ?? []).map((_, s) => String(s)),
            probabilities: (p.emissionprob ?? []).map(normal),
          }
        : { kind: 'gaussian', means: p.means ?? [], covariances: p.covars ?? [] },
  })
}

/** A sequence as a matrix (rows are time steps); a vector of symbols is one column. */
function sequenceView(x: TensorLike, who: string, nFeatures?: number): MatrixView {
  const t = asTensor(x)
  const m = t.shape.length === 1 ? fromStorage(Float64Array.from(toFlat(t)), [t.shape[0] ?? 0, 1], 'float64') : t
  return matrixView(m, who, nFeatures === undefined ? {} : { nFeatures })
}

/** Observations of a sequence as the core's chain expects them: symbol indices or vectors. */
function observationsOf(m: MatrixView, emission: 'categorical' | 'gaussian'): (number | number[])[] {
  return Array.from({ length: m.n }, (_, t) =>
    emission === 'categorical' ? (m.data[t * m.d] ?? 0) : Array.from(m.data.subarray(t * m.d, (t + 1) * m.d)),
  )
}

/** The sequences of learn data: the rows split by its `lengths` (one sequence when absent). */
function sequencesOf(data: LearnData, m: MatrixView): MatrixView[] {
  const lengths = (data as { readonly lengths?: unknown }).lengths
  if (!Array.isArray(lengths)) return [m]
  const out: MatrixView[] = []
  let at = 0
  for (const len of lengths as number[]) {
    out.push({ n: len, d: m.d, data: m.data.subarray(at * m.d, (at + len) * m.d) })
    at += len
  }
  if (at !== m.n) throw new MdmlError('shape', `hmm: lengths sum to ${at}, not ${m.n} rows`)
  return out
}

/** The state of Baum-Welch: the parameters, the last log likelihood and the history of every iteration. */
export interface BaumWelchState {
  readonly params: Omit<HmmParams, 'logLikelihood' | 'nIter' | 'converged'>
  readonly logLikelihood: number
  readonly gain: number
  readonly history: {
    readonly params: Omit<HmmParams, 'logLikelihood' | 'nIter' | 'converged'>[]
    readonly logLikelihood: number[]
  }
}

/**
 * Baum-Welch (EM for hidden Markov models; Baum et al. 1970, hmmlearn's loop) as a traced algorithm: each step runs
 * the core's forward-backward on every sequence under the current parameters (E: state posteriors, pairwise
 * posteriors and the log likelihood), then re-estimates the start, transition and emission parameters from the
 * expected counts (M; hmmlearn's covariance prior for Gaussian emissions). Converged when the log likelihood gains
 * less than tol.
 *
 * @param sequences - The training sequences (rows are time steps).
 * @param start - The initial parameters.
 * @param options - Covariance type, prior and tolerance.
 * @returns The algorithm.
 */
export function baumWelchAlgorithm(
  sequences: readonly MatrixView[],
  start: Omit<HmmParams, 'logLikelihood' | 'nIter' | 'converged'>,
  options: { readonly covarianceType?: 'diag' | 'full'; readonly covarsPrior?: number; readonly tol?: number } = {},
): Algorithm<BaumWelchState> {
  const K = start.startprob.length
  const type = options.covarianceType ?? 'diag'
  const prior = options.covarsPrior ?? 1e-2
  const tol = options.tol ?? 1e-2
  const emission = start.emission
  const d = sequences[0]?.d ?? 1
  const S = start.emissionprob?.[0]?.length ?? 0
  const observations = sequences.map((s) => observationsOf(s, emission))
  return {
    name: 'baum-welch',
    config: { states: K, emission, covarianceType: type, tol },
    schema: channels({
      metrics: { logLikelihood: { description: 'log likelihood of the sequences', direction: 'higher' } },
    }),
    init: () => ({
      params: start,
      logLikelihood: Number.NEGATIVE_INFINITY,
      gain: Number.POSITIVE_INFINITY,
      history: { params: [], logLikelihood: [] },
    }),
    step: (s) => {
      const model = hmmOf(s.params)
      const startCounts = new Array<number>(K).fill(0)
      const trans = Array.from({ length: K }, () => new Array<number>(K).fill(0))
      const post = new Array<number>(K).fill(0)
      const obsCounts = Array.from({ length: K }, () => new Array<number>(S).fill(0))
      const obs = Array.from({ length: K }, () => new Array<number>(d).fill(0))
      const obs2 = Array.from({ length: K }, () => Array.from({ length: d }, () => new Array<number>(d).fill(0)))
      let ll = 0
      for (const [q, seq] of observations.entries()) {
        const fb = forwardBackward(hmmChain(model, seq))
        ll += fb.logLikelihood
        const m = sequences[q] as MatrixView
        for (let k = 0; k < K; k++) startCounts[k] = (startCounts[k] ?? 0) + (fb.posteriors[0]?.[k] ?? 0)
        for (const xi of fb.pairwise)
          for (let i = 0; i < K; i++)
            for (let j = 0; j < K; j++) (trans[i] as number[])[j] = (trans[i]?.[j] ?? 0) + (xi[i]?.[j] ?? 0)
        for (const [t, g] of fb.posteriors.entries()) {
          for (let k = 0; k < K; k++) {
            const gk = g[k] ?? 0
            post[k] = (post[k] ?? 0) + gk
            if (emission === 'categorical') {
              const sym = seq[t] as number
              ;(obsCounts[k] as number[])[sym] = (obsCounts[k]?.[sym] ?? 0) + gk
            } else {
              const row = m.data.subarray(t * d, (t + 1) * d)
              for (let a = 0; a < d; a++) {
                ;(obs[k] as number[])[a] = (obs[k]?.[a] ?? 0) + gk * (row[a] ?? 0)
                for (let b = 0; b < d; b++)
                  ((obs2[k] as number[][])[a] as number[])[b] =
                    (obs2[k]?.[a]?.[b] ?? 0) + gk * (row[a] ?? 0) * (row[b] ?? 0)
              }
            }
          }
        }
      }
      const normal = (row: readonly number[]): number[] => {
        const z = row.reduce((a, v) => a + Math.max(v, 0), 0)
        return row.map((v) => (z > 0 ? Math.max(v, 0) / z : 1 / row.length))
      }
      let params: BaumWelchState['params'] = {
        ...s.params,
        startprob: normal(startCounts),
        transmat: trans.map(normal),
      }
      if (emission === 'categorical') params = { ...params, emissionprob: obsCounts.map(normal) }
      else {
        const means = obs.map((o, k) => o.map((v) => v / (post[k] ?? 1)))
        const covars = means.map((mu, k) => {
          const n = post[k] ?? 0
          const scatter = Array.from({ length: d }, (_, a) =>
            Array.from(
              { length: d },
              (_, b) =>
                (obs2[k]?.[a]?.[b] ?? 0) -
                (obs[k]?.[a] ?? 0) * (mu[b] ?? 0) -
                (mu[a] ?? 0) * (obs[k]?.[b] ?? 0) +
                (mu[a] ?? 0) * (mu[b] ?? 0) * n,
            ),
          )
          // hmmlearn: diagonal (prior + scatter) / max(n, 1e-5); full (prior + scatter) / n, the prior on every entry.
          if (type === 'diag')
            return scatter.map((row, a) => row.map((v, b) => (a === b ? (prior + v) / Math.max(n, 1e-5) : 0)))
          return scatter.map((row) => row.map((v) => (prior + v) / n))
        })
        params = { ...params, means, covars }
      }
      return {
        params,
        logLikelihood: ll,
        gain: ll - s.logLikelihood,
        history: { params: [...s.history.params, params], logLikelihood: [...s.history.logLikelihood, ll] },
      }
    },
    done: (s) => (s.history.logLikelihood.length >= 2 && s.gain < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.logLikelihood) ? { logLikelihood: s.logLikelihood } : {} }),
  }
}

/** Initial parameters as hmmlearn makes them: uniform start and transitions, random emission rows (categorical) or
 * k-means means and the data's variance plus minCovar (Gaussian); any given part is kept. */
function initialHmm(
  sequences: readonly MatrixView[],
  h: HmmHyper,
  key: Key,
): Omit<HmmParams, 'logLikelihood' | 'nIter' | 'converged'> {
  const K = h.nComponents
  const init = h.init ?? {}
  const stateNames = h.stateNames === undefined ? Array.from({ length: K }, (_, k) => `s${k}`) : [...h.stateNames]
  const base = {
    emission: h.emission,
    startprob: init.startprob === undefined ? new Array<number>(K).fill(1 / K) : [...init.startprob],
    transmat:
      init.transmat === undefined
        ? Array.from({ length: K }, () => new Array<number>(K).fill(1 / K))
        : init.transmat.map((r) => [...r]),
    stateNames,
  }
  const all: MatrixView = {
    n: sequences.reduce((a, s) => a + s.n, 0),
    d: sequences[0]?.d ?? 1,
    data: Float64Array.from(sequences.flatMap((s) => Array.from(s.data))),
  }
  if (h.emission === 'categorical') {
    let maxSym = 0
    for (const v of all.data) {
      if (!(Number.isInteger(v) && v >= 0))
        throw new MdmlError('value', 'categoricalHMM: symbols must be integers >= 0')
      maxSym = Math.max(maxSym, v)
    }
    const S = h.nFeatures ?? maxSym + 1
    if (init.emissionprob !== undefined) return { ...base, emissionprob: init.emissionprob.map((r) => [...r]) }
    const u = toFlat(uniform(key, [K, S]))
    const emissionprob = Array.from({ length: K }, (_, k) => {
      const row = Array.from({ length: S }, (_, s) => u[k * S + s] ?? 0)
      const z = row.reduce((a, v) => a + v, 0)
      return row.map((v) => v / z)
    })
    return { ...base, emissionprob }
  }
  const d = all.d
  const means =
    init.means?.map((r) => [...r]) ??
    kmeans({ nClusters: K }).fit({ x: fromStorage(all.data, [all.n, d], 'float64') }, { key }).params.centres
  const mu = Array.from({ length: d }, (_, a) => {
    let s = 0
    for (let i = 0; i < all.n; i++) s += all.data[i * d + a] ?? 0
    return s / all.n
  })
  const cov = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) => {
      let s = 0
      for (let i = 0; i < all.n; i++)
        s += ((all.data[i * d + a] ?? 0) - (mu[a] ?? 0)) * ((all.data[i * d + b] ?? 0) - (mu[b] ?? 0))
      const c = s / Math.max(all.n - 1, 1)
      return a === b ? c + h.minCovar : h.covarianceType === 'diag' ? 0 : c
    }),
  )
  return { ...base, means, covars: init.covars?.map((c) => c.map((r) => [...r])) ?? means.map(() => cov) }
}

/** Evenly thinned indices 0..last, at most max of them, always keeping both ends. */
function thinned(last: number, max: number): number[] {
  if (last + 1 <= max) return Array.from({ length: last + 1 }, (_, t) => t)
  const out = new Set<number>()
  for (let j = 0; j < max; j++) out.add(Math.round((j * last) / (max - 1)))
  return [...out].sort((a, b) => a - b)
}

/** The long-run share of each state (the stationary distribution of the transitions), or the start if none. */
function stationaryOf(transmat: readonly (readonly number[])[], startprob: readonly number[]): number[] {
  const pi = toFlat(stationaryDistribution(transmat as number[][]))
  return pi.length === startprob.length && pi.every((v) => Number.isFinite(v)) ? Array.from(pi) : [...startprob]
}

/**
 * The exposition of a hidden Markov model on a sequence: the state posteriors over time as a heat map (a
 * `decisionSurface` over time and state, values P(z_t = k | x)) and the start and transition probabilities as a
 * coefficient table.
 *
 * @param p - The parameters.
 * @param x - A sequence (rows are time steps).
 * @returns The records.
 */
export function hmmExposition(p: HmmParams, x: TensorLike): Exposition[] {
  const m = sequenceView(x, 'hmm')
  const chain = hmmChain(hmmOf(p), observationsOf(m, p.emission))
  const fb = forwardBackward(chain)
  const K = p.startprob.length
  const names: string[] = []
  const estimates: number[] = []
  for (const [k, v] of p.startprob.entries()) {
    names.push(`start → ${p.stateNames[k] ?? k}`)
    estimates.push(v)
  }
  for (const [i, row] of p.transmat.entries())
    for (const [j, v] of row.entries()) {
      names.push(`${p.stateNames[i] ?? i} → ${p.stateNames[j] ?? j}`)
      estimates.push(v)
    }
  return [
    stateSequence({
      states: p.stateNames,
      posteriors: fb.posteriors.map((r) => [...r]),
      decoded: [...viterbi(chain).path],
      // One observed number per step (a die's face, a single measurement): drawn above the paths.
      ...(m.d === 1 ? { observations: { name: 'observation', values: Array.from(m.data) } } : {}),
    }),
    decisionSurface({
      features: ['time', 'state'],
      x: Array.from({ length: m.n }, (_, t) => t),
      y: Array.from({ length: K }, (_, k) => k),
      values: Array.from({ length: K }, (_, k) => fb.posteriors.map((r) => r[k] ?? 0)),
      classes: p.stateNames,
    }),
    coefficientTable({ names, estimates }),
  ]
}

function hmmVerbs(p: HmmParams): HmmVerbs {
  const model = hmmOf(p)
  const d = p.emission === 'gaussian' ? (p.means?.[0]?.length ?? 1) : 1
  const chainOf = (x: TensorLike) => hmmChain(model, observationsOf(sequenceView(x, 'hmm', d), p.emission))
  return {
    model,
    predict: (x) => {
      const path = viterbi(chainOf(x)).path
      return fromStorage(Int32Array.from(path), [path.length], 'int32')
    },
    predictProba: (x) => {
      const post = forwardBackward(chainOf(x)).posteriors
      return fromStorage(Float64Array.from(post.flat()), [post.length, p.startprob.length], 'float64')
    },
    score: (data) => forwardBackward(chainOf(data.x)).logLikelihood,
    nFeaturesIn: d,
  }
}

function fitHmm(data: LearnData, h: HmmHyper, options: FitOptions): { params: HmmParams; exposition: Exposition[] } {
  const name = h.emission === 'categorical' ? 'categoricalHMM' : 'gaussianHMM'
  if (!(Number.isInteger(h.nComponents) && h.nComponents >= 1))
    throw new MdmlError('value', `${name}: nComponents must be a positive integer`)
  const m = sequenceView(data.x, name)
  const sequences = sequencesOf(data, m)
  const run = fitRunOptions(options, name)
  const key = run.key as Key
  const start = initialHmm(sequences, h, foldIn(key, 0))
  const out = runAlgorithm(
    baumWelchAlgorithm(sequences, start, { covarianceType: h.covarianceType, covarsPrior: h.covarsPrior, tol: h.tol }),
    { ...run, key: foldIn(key, 1), maxSteps: h.nIter },
  )
  const s = out.state
  const params: HmmParams = {
    ...s.params,
    logLikelihood: s.logLikelihood,
    nIter: out.steps,
    converged: out.status === 'converged',
  }
  const keep = thinned(s.history.params.length - 1, h.maxFrames)
  const steps = keep.map((t) => s.history.params[t] as BaumWelchState['params'])
  const firstSeq = sequences[0] as MatrixView
  const records: Exposition[] = [
    mixtureSteps({
      family: `hmm-${h.emission}`,
      weights: steps.map((q) => stationaryOf(q.transmat, q.startprob)),
      means: steps.map((q) => (h.emission === 'categorical' ? (q.emissionprob ?? []) : (q.means ?? []))),
      ...(h.emission === 'gaussian' ? { covariances: steps.map((q) => (q.covars ?? []).map((c) => c.flat())) } : {}),
      responsibilities: [forwardBackward(hmmChain(hmmOf(params), observationsOf(firstSeq, h.emission))).posteriors],
      objective: keep.map((t) => s.history.logLikelihood[t] ?? 0),
    }),
    ...hmmExposition(params, fromStorage(Float64Array.from(firstSeq.data), [firstSeq.n, firstSeq.d], 'float64')),
  ]
  return { params, exposition: records }
}

function hmmEstimator(
  hyper: Partial<HmmHyper>,
  emission: 'categorical' | 'gaussian',
): ExplainedEstimator<HmmHyper, FittedHmm> {
  return defineExplainedEstimator<HmmHyper, HmmParams, HmmVerbs>(
    {
      name: emission === 'categorical' ? 'categorical-hmm' : 'gaussian-hmm',
      kind: 'other',
      hyperSpace: (h: HmmHyper): Space =>
        dict({
          nComponents: int(1, 10, { default: h.nComponents, label: 'hidden states' }),
          nIter: int(1, 500, { default: h.nIter, label: 'Baum-Welch iterations' }),
          ...(emission === 'gaussian'
            ? {
                covarianceType: choice(['diag', 'full'], {
                  label: 'covariance type',
                  description: 'Each state’s emission covariance: diagonal or full.',
                  default: h.covarianceType,
                }),
              }
            : {}),
          tol: real(1e-8, 1, {
            label: 'tolerance',
            description: 'Stop when the log-likelihood gains less than this.',
            log: true,
            default: h.tol,
          }),
        }),
      fit: fitHmm,
      methods: hmmVerbs,
    },
    {
      nComponents: hyper.nComponents ?? 2,
      emission,
      covarianceType: hyper.covarianceType ?? 'diag',
      ...(hyper.nFeatures === undefined ? {} : { nFeatures: hyper.nFeatures }),
      nIter: hyper.nIter ?? 100,
      tol: hyper.tol ?? 1e-2,
      covarsPrior: hyper.covarsPrior ?? 1e-2,
      minCovar: hyper.minCovar ?? 1e-3,
      ...(hyper.init === undefined ? {} : { init: hyper.init }),
      ...(hyper.stateNames === undefined ? {} : { stateNames: hyper.stateNames }),
      maxFrames: hyper.maxFrames ?? 50,
    },
  )
}

/**
 * A hidden Markov model with categorical emissions (hmmlearn.hmm.CategoricalHMM), fitted by traced Baum-Welch over
 * the core's forward-backward. Rows of `x` are time steps holding a symbol index; several sequences are passed as a
 * `lengths` field on the data. `predict` is the Viterbi path, `predictProba` the state posteriors. Exposition: a
 * `mixtureSteps` record per Baum-Welch iteration (each state's long-run share as its weight, its emission
 * distribution as its mean, the log likelihood as the objective, the final posteriors as responsibilities), the
 * posterior heat map over time and the transition table (see {@link hmmExposition}).
 *
 * @param hyper - States, symbols, iterations, tolerance, starting parameters and state names.
 * @returns The estimator.
 */
export function categoricalHMM(hyper: Partial<HmmHyper> = {}): ExplainedEstimator<HmmHyper, FittedHmm> {
  return hmmEstimator(hyper, 'categorical')
}

/**
 * A hidden Markov model with Gaussian emissions (hmmlearn.hmm.GaussianHMM, diagonal or full covariances with
 * hmmlearn's covariance prior), fitted by traced Baum-Welch; rows of `x` are time steps. Exposition as
 * {@link categoricalHMM}, with the states' means and covariances per iteration.
 *
 * @param hyper - States, covariance type, iterations, tolerance, priors, starting parameters and state names.
 * @returns The estimator.
 */
export function gaussianHMM(hyper: Partial<HmmHyper> = {}): ExplainedEstimator<HmmHyper, FittedHmm> {
  return hmmEstimator(hyper, 'gaussian')
}

/** {@link categoricalHMM} as a class. */
export class CategoricalHMM extends ExplainedEstimatorClass<HmmHyper, FittedHmm> {
  /** @param hyper - As {@link categoricalHMM}. */
  constructor(hyper: Partial<HmmHyper> = {}) {
    super(categoricalHMM(hyper))
  }
}

/** {@link gaussianHMM} as a class. */
export class GaussianHMM extends ExplainedEstimatorClass<HmmHyper, FittedHmm> {
  /** @param hyper - As {@link gaussianHMM}. */
  constructor(hyper: Partial<HmmHyper> = {}) {
    super(gaussianHMM(hyper))
  }
}

/**
 * The occasionally dishonest casino (Durbin, Eddy, Krogh and Mitchison 1998): a fair die and a loaded one that shows
 * a six half the time; the casino switches to the loaded die with probability 0.05 and back with 0.1. Symbols 0 to 5
 * are the faces 1 to 6.
 *
 * @returns The model.
 */
export function dishonestCasino(): HiddenMarkovModel {
  return hiddenMarkovModel({
    states: ['fair', 'loaded'],
    initial: [0.5, 0.5],
    transition: [
      [0.95, 0.05],
      [0.1, 0.9],
    ],
    emission: {
      kind: 'categorical',
      symbols: ['1', '2', '3', '4', '5', '6'],
      probabilities: [
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
      ],
    },
  })
}

/** Options of {@link casinoStudy}. */
export interface CasinoStudyOptions {
  /** Rolls watched (default 300). */
  readonly rolls?: number
  /** Learn the model from the rolls by Baum-Welch (default true) or decode with the true model. */
  readonly learn?: boolean
  /** Baum-Welch iterations (default 100). */
  readonly nIter?: number
  readonly key?: Key
}

/** The result of {@link casinoStudy}. */
export interface CasinoStudy {
  /** The dice used (0 fair, 1 loaded) and the faces rolled (0 to 5). */
  readonly states: readonly number[]
  readonly rolls: readonly number[]
  /** The Viterbi path and the posterior probability of the loaded die at each roll. */
  readonly decoded: readonly number[]
  readonly loaded: readonly number[]
  /** The share of rolls whose die the Viterbi path names correctly. */
  readonly accuracy: number
  /** The model decoded with (learned, states ordered fair then loaded, or the true one). */
  readonly params: HmmParams
}

/**
 * The dishonest casino as a study: roll the casino's dice, then recover which die was used from the rolls alone,
 * with the true model or one learned by Baum-Welch from random starting emissions (its states ordered so that the
 * one favouring sixes is `loaded`). Exposition: a `posteriorGrid` of P(loaded | rolls) over time with the true dice as
 * the observations, then the records of {@link categoricalHMM} (when learned) or {@link hmmExposition}.
 *
 * @param options - Rolls, whether to learn, iterations and the key.
 * @returns The rolls, the dice, the decoding and its accuracy, with the exposition records.
 */
export function casinoStudy(
  options: CasinoStudyOptions = {},
): CasinoStudy & { readonly exposition: readonly Exposition[] } {
  const key = options.key ?? makeKey(0)
  const truth = dishonestCasino()
  const T = options.rolls ?? 300
  const sample = hmmSample(truth, foldIn(key, 0), T)
  const rolls = sample.observations as number[]
  const x = fromStorage(Float64Array.from(rolls), [T, 1], 'float64')
  let params: HmmParams
  let records: readonly Exposition[]
  if (options.learn ?? true) {
    const f = categoricalHMM({ nComponents: 2, nFeatures: 6, nIter: options.nIter ?? 100, tol: 1e-4 }).fit(
      { x },
      { key: foldIn(key, 1) },
    )
    // Order the states fair, loaded: the loaded die shows more sixes.
    const e = f.params.emissionprob ?? []
    const swap = (e[0]?.[5] ?? 0) > (e[1]?.[5] ?? 0)
    const order = swap ? [1, 0] : [0, 1]
    const p = f.params
    params = {
      ...p,
      startprob: order.map((k) => p.startprob[k] ?? 0),
      transmat: order.map((i) => order.map((j) => p.transmat[i]?.[j] ?? 0)),
      emissionprob: order.map((k) => e[k] ?? []),
      stateNames: ['fair', 'loaded'],
    }
    records = swap ? hmmExposition(params, x) : f.exposition
    if (swap) records = [...f.exposition.filter((r) => r.kind === 'mixtureSteps'), ...records]
  } else {
    params = {
      emission: 'categorical',
      startprob: [...truth.initial],
      transmat: truth.transition.map((r) => [...r]),
      emissionprob: truth.emission.kind === 'categorical' ? truth.emission.probabilities.map((r) => [...r]) : [],
      logLikelihood: forwardBackward(hmmChain(truth, rolls)).logLikelihood,
      nIter: 0,
      converged: true,
      stateNames: ['fair', 'loaded'],
    }
    records = hmmExposition(params, x)
  }
  const chain = hmmChain(hmmOf(params), rolls)
  const decoded = viterbi(chain).path
  const loaded = forwardBackward(chain).posteriors.map((r) => r[1] ?? 0)
  const accuracy = decoded.filter((s, t) => s === sample.states[t]).length / T
  const grid = posteriorGrid({
    inputs: loaded.map((_, t) => [t]),
    mean: loaded,
    variance: loaded.map(() => 0),
    observed: { inputs: sample.states.map((_, t) => [t]), targets: sample.states },
  })
  // The state strip with the true dice beside the decoded ones (the model's own strip, without the truth, is dropped).
  const strip = stateSequence({
    states: params.stateNames,
    posteriors: forwardBackward(chain).posteriors.map((r) => [...r]),
    decoded: [...decoded],
    truth: [...sample.states],
    observations: { name: 'roll', values: [...rolls] },
  })
  return {
    states: sample.states,
    rolls,
    decoded,
    loaded,
    accuracy,
    params,
    exposition: [strip, grid, ...records.filter((r) => r.kind !== 'stateSequence')],
  }
}
