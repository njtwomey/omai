/**
 * The input-output hidden Markov model (Bengio and Frasconi 1995): a hidden Markov model whose transitions and emissions
 * are conditioned on an input observed at every step. The transition out of state i at step t is a softmax over the next
 * states of linear functions of the features [1, u_t]; the emission is either a softmax over symbols (categorical) or a
 * linear regression with a per-state variance (Gaussian). Inference is the ordinary forward-backward pass on a chain
 * with one transition matrix per step. The EM loop's M-step for the softmax parts is a smooth problem
 * (`softmaxRegressionProblem`, a weighted multinomial logistic regression on soft targets) minimised by any solver of
 * the smooth slot (Newton by default); the Gaussian regressions are closed-form weighted least squares. Every
 * coefficient has a small ridge penalty, so the EM objective is the penalised log likelihood and it never decreases.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { analyticObjective, newtonAt, type SmoothProblem, smoothSolvers } from '../../optim/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import type { SpaceValue } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../../trace/index.ts'
import { solveVector } from '../_dense.ts'
import { type ChainModel, forwardBackward } from '../inference/chain.ts'
import type { HmmObservation } from './baum-welch.ts'

const NEG = Number.NEGATIVE_INFINITY

/**
 * The parameters of an input-output hidden Markov model. The coefficient arrays end in the feature axis of length
 * `1 + m` for inputs of length m (the intercept first): `transition[i][j]` are the coefficients of the logit of moving
 * from state i to state j, `emission[k][s]` those of the logit of symbol s in state k (categorical), and
 * `coefficients[k][d]` the regression of dimension d of the observation in state k, with `variances[k][d]` (Gaussian).
 *
 * @concept hidden-markov-models
 */
export type IoHmmParameters =
  | { kind: 'categorical'; start: number[]; transition: number[][][]; emission: number[][][] }
  | { kind: 'gaussian'; start: number[]; transition: number[][][]; coefficients: number[][][]; variances: number[][] }

/** A sequence with its inputs: one input vector and one observation per step. */
export interface IoHmmSequence {
  readonly inputs: readonly (readonly number[])[]
  readonly observations: readonly HmmObservation[]
}

const features = (u: readonly number[]): number[] => [1, ...u]
const vec = (o: HmmObservation): readonly number[] => (typeof o === 'number' ? [o] : o)

/** Softmax of logits. */
function softmax(z: readonly number[]): number[] {
  const m = maxOf(z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, v) => a + v, 0)
  return e.map((v) => v / s)
}

const dotf = (w: readonly number[], f: readonly number[]): number => w.reduce((a, v, p) => a + v * (f[p] ?? 0), 0)

/**
 * The transition matrix out of every state at an input: row i is the softmax over the next states of the linear logits
 * of the features [1, input].
 *
 * @param params - The parameters.
 * @param input - The input vector.
 * @returns The transition matrix, rows summing to one.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const p = { kind: 'categorical', start: [1, 0], transition: [[[0, 1], [0, 0]], [[0, 0], [0, 0]]], emission: [[[0], [0]], [[0], [0]]] } as const
 * learning.ioTransition(p, [0]) // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function ioTransition(params: IoHmmParameters, input: readonly number[]): number[][] {
  const f = features(input)
  return params.transition.map((row) => softmax(row.map((w) => dotf(w, f))))
}

/**
 * The log probability of each state's emission of one observation at an input: the softmax of the symbol logits
 * (categorical) or the Gaussian density of the regression (Gaussian), per state.
 */
function logEmission(params: IoHmmParameters, input: readonly number[], x: HmmObservation): number[] {
  const f = features(input)
  if (params.kind === 'categorical') {
    return params.emission.map((rows) => {
      const z = rows.map((w) => dotf(w, f))
      const m = maxOf(z)
      return (z[x as number] ?? NEG) - (m + Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0)))
    })
  }
  const o = vec(x)
  return params.coefficients.map((rows, k) =>
    rows.reduce((acc, w, d) => {
      const v = params.variances[k]?.[d] ?? 1
      return acc - 0.5 * Math.log(2 * Math.PI * v) - ((o[d] ?? 0) - dotf(w, f)) ** 2 / (2 * v)
    }, 0),
  )
}

/**
 * The chain an input-output model defines on a sequence: the log initial distribution, one log transition matrix per
 * step (the one into step t uses the input at t) and the log emissions, ready for `forwardBackward` and `viterbi`.
 *
 * @param params - The parameters.
 * @param sequence - Inputs and observations.
 * @returns The chain.
 * @throws MdmlError with code `shape` when the inputs and observations differ in length.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[[0], [0]], [[0], [0]]], emission: [[[0], [0]], [[0], [0]]] } as const
 * learning.ioHmmChain(p, { inputs: [[], []], observations: [0, 1] }).logEmission.length // => 2
 * ```
 */
export function ioHmmChain(params: IoHmmParameters, sequence: IoHmmSequence): ChainModel {
  const T = sequence.observations.length
  if (sequence.inputs.length !== T) throw new MdmlError('shape', 'ioHmm: one input per observation')
  return {
    logInitial: params.start.map(Math.log),
    logTransition: Array.from({ length: Math.max(0, T - 1) }, (_, t) =>
      ioTransition(params, sequence.inputs[t + 1] ?? []).map((r) => r.map(Math.log)),
    ),
    logEmission: sequence.observations.map((x, t) => logEmission(params, sequence.inputs[t] ?? [], x)),
  }
}

/**
 * The log likelihood of sequences under an input-output hidden Markov model.
 *
 * @param params - The parameters.
 * @param sequences - The sequences with their inputs.
 * @returns The summed log likelihood.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[[0], [0]], [[0], [0]]], emission: [[[0], [0]], [[0], [0]]] } as const
 * Math.round(learning.ioHmmLogLikelihood(p, [{ inputs: [[], []], observations: [0, 1] }]) * 1e6) / 1e6 // => -1.386294
 * ```
 */
export function ioHmmLogLikelihood(params: IoHmmParameters, sequences: readonly IoHmmSequence[]): number {
  return sequences.reduce((a, s) => a + forwardBackward(ioHmmChain(params, s)).logLikelihood, 0)
}

/**
 * The M-step objective of a softmax regression on soft targets as a smooth problem: coefficients `W[c][p]` (flattened
 * class by class) minimise `sum_n [ s_n logsumexp_c(z_nc) - sum_c R_nc z_nc ] + ridge/2 |W|^2` with logits
 * `z_nc = W_c . f_n` and `s_n = sum_c R_nc` the row's total weight. With R the expected counts of an EM step this is the
 * negative expected complete-data log likelihood of the transition or emission distribution. The gradient and the
 * Hessian are analytic, so any solver of `smoothSolvers` can minimise it (Newton gives the optimum in a few steps).
 *
 * @param f - The features of each row, `[N][p]`.
 * @param targets - The soft targets (expected counts) of each row over the classes, `[N][C]`.
 * @param ridge - The ridge penalty on every coefficient.
 * @param start - The starting coefficients `[C][p]` (default zeros).
 * @returns The problem.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * // the intercept-only problem is minimised by log class proportions: (3, 1) of four
 * const problem = learning.softmaxRegressionProblem([[1]], [[3, 1]], 0)
 * problem.x0 // => [0, 0]
 * ```
 */
export function softmaxRegressionProblem(
  f: readonly (readonly number[])[],
  targets: readonly (readonly number[])[],
  ridge: number,
  start?: readonly (readonly number[])[],
): SmoothProblem {
  const N = f.length
  const C = targets[0]?.length ?? 0
  const P = f[0]?.length ?? 0
  const F = Float64Array.from(f.flat())
  const R = Float64Array.from(targets.flat())
  const weight = Float64Array.from(targets, (r) => r.reduce((a, v) => a + v, 0))
  // the softmax probabilities of every row at the coefficients x, and the log-sum-exp of its logits
  const forward = (x: ArrayLike<number>): { prob: Float64Array; lse: Float64Array; logits: Float64Array } => {
    const prob = new Float64Array(N * C)
    const lse = new Float64Array(N)
    const logits = new Float64Array(N * C)
    for (let n = 0; n < N; n++) {
      let m = Number.NEGATIVE_INFINITY
      for (let c = 0; c < C; c++) {
        let z = 0
        for (let p = 0; p < P; p++) z += (x[c * P + p] ?? 0) * (F[n * P + p] ?? 0)
        logits[n * C + c] = z
        if (z > m) m = z
      }
      let sum = 0
      for (let c = 0; c < C; c++) sum += Math.exp((logits[n * C + c] ?? 0) - m)
      lse[n] = m + Math.log(sum)
      for (let c = 0; c < C; c++) prob[n * C + c] = Math.exp((logits[n * C + c] ?? 0) - (lse[n] ?? 0))
    }
    return { prob, lse, logits }
  }
  return {
    objective: analyticObjective({
      value: (x) => {
        const { lse, logits } = forward(x)
        let v = 0
        for (let n = 0; n < N; n++) {
          v += (weight[n] ?? 0) * (lse[n] ?? 0)
          for (let c = 0; c < C; c++) v -= (R[n * C + c] ?? 0) * (logits[n * C + c] ?? 0)
        }
        let pen = 0
        for (let i = 0; i < C * P; i++) pen += (x[i] ?? 0) ** 2
        return v + 0.5 * ridge * pen
      },
      gradient: (x) => {
        const g = new Float64Array(C * P)
        const { prob } = forward(x)
        for (let n = 0; n < N; n++)
          for (let c = 0; c < C; c++) {
            const r = (weight[n] ?? 0) * (prob[n * C + c] ?? 0) - (R[n * C + c] ?? 0)
            for (let p = 0; p < P; p++) g[c * P + p] = (g[c * P + p] ?? 0) + r * (F[n * P + p] ?? 0)
          }
        for (let i = 0; i < C * P; i++) g[i] = (g[i] ?? 0) + ridge * (x[i] ?? 0)
        return g
      },
      hessian: (x) => {
        const h = new Float64Array(C * P * C * P)
        const { prob } = forward(x)
        for (let n = 0; n < N; n++)
          for (let c = 0; c < C; c++)
            for (let d = 0; d < C; d++) {
              const pc = prob[n * C + c] ?? 0
              const q = (weight[n] ?? 0) * ((c === d ? pc : 0) - pc * (prob[n * C + d] ?? 0))
              for (let p = 0; p < P; p++)
                for (let r = 0; r < P; r++)
                  h[(c * P + p) * C * P + d * P + r] =
                    (h[(c * P + p) * C * P + d * P + r] ?? 0) + q * (F[n * P + p] ?? 0) * (F[n * P + r] ?? 0)
            }
        for (let i = 0; i < C * P; i++) h[i * C * P + i] = (h[i * C * P + i] ?? 0) + ridge
        return h
      },
    }),
    x0: start === undefined ? new Array<number>(C * P).fill(0) : start.flat(),
  }
}

/** Options of {@link ioHmmAlgorithm}. */
export interface IoHmmOptions {
  /** The solver of the softmax M-steps, a value of `smoothSolvers.space` (default Newton at gradient tolerance 1e-10). */
  readonly solver?: SpaceValue
  /** Ridge penalty on every coefficient (default 1e-3: it keeps separable data finite and fixes the softmax's free shift). */
  readonly ridge?: number
  /** Gaussian emissions: the smallest variance (default 1e-4). */
  readonly minVariance?: number
  /** Converged when the penalised log likelihood gains less than this in a step (default 1e-6). */
  readonly tol?: number
}

/** The state of {@link ioHmmAlgorithm}. */
export type IoHmmState = {
  params: IoHmmParameters
  /** The penalised log likelihood under the parameters the last step started from (the quantity EM improved). */
  objective: number | null
  /** The log likelihood alone. */
  logLikelihood: number | null
  gain: number | null
}

const penalty = (p: IoHmmParameters, ridge: number): number => {
  const sq = (a: readonly (readonly (readonly number[])[])[]): number => a.flat(2).reduce((s, v) => s + v * v, 0)
  return 0.5 * ridge * (sq(p.transition) + sq(p.kind === 'categorical' ? p.emission : p.coefficients))
}

/**
 * Expectation-maximisation for an input-output hidden Markov model as a traced {@link Algorithm}. Each step runs
 * forward-backward on every sequence with its per-step transitions (the E-step), then re-estimates the start
 * distribution from the expected first state, the transition coefficients of every source state and the categorical
 * emission coefficients of every state each by minimising a {@link softmaxRegressionProblem} with the `solver` of the
 * slot, and the Gaussian regressions and variances by weighted least squares. The penalised log likelihood (the
 * log likelihood minus the ridge on the coefficients) of the parameters each step started from never decreases. It
 * observes the penalised objective, the log likelihood and the `start` tensor.
 *
 * @param sequences - The training sequences with their inputs.
 * @param start - The starting parameters.
 * @param options - The solver, ridge, variance floor and tolerance.
 * @returns The algorithm.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[[0, 0.1], [0, -0.1]], [[0, -0.1], [0, 0.1]]], emission: [[[1, 0], [0, 0]], [[0, 0], [1, 0]]] } as const
 * const seq = { inputs: [[1], [1], [-1], [-1]], observations: [0, 0, 1, 1] }
 * runAlgorithm(learning.ioHmmAlgorithm([seq], p), { maxSteps: 3 }).steps // => 3
 * ```
 */
export function ioHmmAlgorithm(
  sequences: readonly IoHmmSequence[],
  start: IoHmmParameters,
  options: IoHmmOptions = {},
): Algorithm<IoHmmState> {
  const K = start.start.length
  const ridge = options.ridge ?? 1e-3
  const solver = options.solver ?? newtonAt(1e-10)
  const minVariance = options.minVariance ?? 1e-4
  const tol = options.tol ?? 1e-6
  const minimise = (problem: SmoothProblem, key: Key): number[] =>
    Array.from(
      (runAlgorithm(smoothSolvers.make(problem, solver, key), { maxSteps: 200, key }).state as { x: Float64Array }).x,
    )
  const unflatten = (x: readonly number[], C: number, P: number): number[][] =>
    Array.from({ length: C }, (_, c) => x.slice(c * P, (c + 1) * P))
  return {
    name: 'io-hmm-em',
    config: { states: K, emission: start.kind, ridge, tol, solver },
    schema: channels({
      metrics: {
        objective: {
          description: 'penalised log likelihood under the parameters the step started from',
          direction: 'higher',
        },
        logLikelihood: {
          description: 'log likelihood under the parameters the step started from',
          direction: 'higher',
        },
      },
      tensors: { start: { role: 'param', description: 'the start distribution', shape: [K] } },
    }),
    init: () => ({ params: start, objective: null, logLikelihood: null, gain: null }),
    step: (s, ctx) => {
      const P = s.params.transition[0]?.[0]?.length ?? 1
      const post = sequences.map((seq) => forwardBackward(ioHmmChain(s.params, seq)))
      const logLikelihood = post.reduce((a, p) => a + p.logLikelihood, 0)
      const first = new Array<number>(K).fill(0)
      for (const p of post) for (let k = 0; k < K; k++) first[k] = (first[k] ?? 0) + (p.posteriors[0]?.[k] ?? 0)
      const total = first.reduce((a, v) => a + v, 0)
      const newStart = first.map((v) => v / total)
      // transitions: source state i, rows (sequence, step t >= 1), features of the input at t, soft targets the pair posteriors
      const transition = s.params.transition.map((row, i) => {
        const f: number[][] = []
        const r: number[][] = []
        for (const [n, seq] of sequences.entries())
          for (let t = 1; t < seq.observations.length; t++) {
            f.push(features(seq.inputs[t] ?? []))
            r.push((post[n]?.pairwise[t - 1]?.[i] ?? []).slice())
          }
        const x = minimise(softmaxRegressionProblem(f, r, ridge, row), foldIn(ctx.key, i))
        return unflatten(x, K, P)
      })
      let params: IoHmmParameters
      if (s.params.kind === 'categorical') {
        const emission = s.params.emission.map((rows, k) => {
          const S = rows.length
          const f: number[][] = []
          const r: number[][] = []
          for (const [n, seq] of sequences.entries())
            seq.observations.forEach((x, t) => {
              f.push(features(seq.inputs[t] ?? []))
              const target = new Array<number>(S).fill(0)
              target[x as number] = post[n]?.posteriors[t]?.[k] ?? 0
              r.push(target)
            })
          return unflatten(minimise(softmaxRegressionProblem(f, r, ridge, rows), foldIn(ctx.key, K + k)), S, P)
        })
        params = { kind: 'categorical', start: newStart, transition, emission }
      } else {
        const coefficients: number[][][] = []
        const variances: number[][] = []
        for (let k = 0; k < K; k++) {
          const D = s.params.coefficients[k]?.length ?? 0
          const gram = Array.from({ length: P }, () => new Array<number>(P).fill(0))
          const rhs = Array.from({ length: D }, () => new Array<number>(P).fill(0))
          let mass = 0
          for (const [n, seq] of sequences.entries())
            seq.observations.forEach((x, t) => {
              const g = post[n]?.posteriors[t]?.[k] ?? 0
              const f = features(seq.inputs[t] ?? [])
              const o = vec(x)
              mass += g
              for (let a = 0; a < P; a++) {
                for (let b = 0; b < P; b++)
                  (gram[a] as number[])[b] = (gram[a]?.[b] ?? 0) + g * (f[a] ?? 0) * (f[b] ?? 0)
                for (let d = 0; d < D; d++) (rhs[d] as number[])[a] = (rhs[d]?.[a] ?? 0) + g * (f[a] ?? 0) * (o[d] ?? 0)
              }
            })
          // the ridge of the log likelihood is on the coefficients, so in the least squares it is scaled by the variance
          const beta = rhs.map((r, d) =>
            solveVector(
              gram.map((row, a) =>
                row.map(
                  (v, b) =>
                    v + (a === b ? ridge * (s.params.kind === 'gaussian' ? (s.params.variances[k]?.[d] ?? 1) : 1) : 0),
                ),
              ),
              r,
            ),
          )
          coefficients.push(beta)
          variances.push(
            beta.map((w, d) => {
              let sse = 0
              for (const [n, seq] of sequences.entries())
                seq.observations.forEach((x, t) => {
                  sse +=
                    (post[n]?.posteriors[t]?.[k] ?? 0) *
                    ((vec(x)[d] ?? 0) - dotf(w, features(seq.inputs[t] ?? []))) ** 2
                })
              return Math.max(sse / Math.max(mass, 1e-12), minVariance)
            }),
          )
        }
        params = { kind: 'gaussian', start: newStart, transition, coefficients, variances }
      }
      const objective = logLikelihood - penalty(s.params, ridge)
      return { params, objective, logLikelihood, gain: s.objective === null ? null : objective - s.objective }
    },
    done: (s) => (s.gain !== null && s.gain < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics:
        s.objective === null || s.logLikelihood === null
          ? {}
          : { objective: s.objective, logLikelihood: s.logLikelihood },
      tensors: { start: { role: 'param', value: s.params.start, shape: [K] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as IoHmmState,
  }
}

/**
 * A draw from an input-output hidden Markov model given the inputs: the state path and the observations.
 *
 * @param params - The parameters.
 * @param inputs - One input vector per step.
 * @param key - The random key.
 * @returns The states and the observations.
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const p = { kind: 'categorical', start: [1, 0], transition: [[[0], [-30]], [[-30], [0]]], emission: [[[30], [-30]], [[-30], [30]]] } as const
 * learning.ioHmmSample(p, [[], [], []], key(0)).observations // => [0, 0, 0]
 * ```
 */
export function ioHmmSample(
  params: IoHmmParameters,
  inputs: readonly (readonly number[])[],
  key: Key,
): { readonly states: number[]; readonly observations: HmmObservation[] } {
  const T = inputs.length
  const u = Array.from(toFlat(uniform(foldIn(key, 0), [2 * T + 1])))
  const noise =
    params.kind === 'gaussian'
      ? Array.from(toFlat(normal(foldIn(key, 1), [T, params.coefficients[0]?.length ?? 1])))
      : []
  let next = 0
  const draw = (p: readonly number[]): number => {
    const r = u[next++] ?? 0
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i] ?? 0
      if (r < acc) return i
    }
    return p.length - 1
  }
  const states: number[] = []
  const observations: HmmObservation[] = []
  for (let t = 0; t < T; t++) {
    const z = t === 0 ? draw(params.start) : draw(ioTransition(params, inputs[t] ?? [])[states[t - 1] ?? 0] ?? [])
    states.push(z)
    const f = features(inputs[t] ?? [])
    if (params.kind === 'categorical')
      observations.push(draw(softmax((params.emission[z] ?? []).map((w) => dotf(w, f)))))
    else {
      const D = params.coefficients[z]?.length ?? 0
      observations.push(
        Array.from(
          { length: D },
          (_, d) =>
            dotf(params.coefficients[z]?.[d] ?? [], f) +
            Math.sqrt(params.variances[z]?.[d] ?? 1) * (noise[t * D + d] ?? 0),
        ),
      )
    }
  }
  return { states, observations }
}
