/**
 * Inference on chains, hidden Markov models and linear-chain CRFs: the forward-backward algorithm (filtered and
 * smoothed marginals, pairwise posteriors and the likelihood, scaled or in log space), forward filtering, Viterbi
 * decoding of the most probable path, posterior decoding and forward filtering backward sampling for exact path draws.
 */

import { argmaxOf, logSumExpOf, MdmlError, maxOf } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { nullTracer, type RunHandle, startRun, type TracePolicy, type Tracer } from '../../trace/index.ts'
import type { Rows } from '../_dense.ts'

/**
 * A chain-structured model in log space: the potentials of a hidden Markov model (log initial, transition and emission
 * probabilities) or of a linear-chain CRF (log potentials). The transition is one K x K matrix (homogeneous) or one per
 * step (T - 1 of them); entry [i][j] is the log potential of moving from state i to state j.
 *
 * @concept hidden-markov-models
 */
export interface ChainModel {
  readonly logInitial: readonly number[]
  readonly logTransition: Rows | readonly Rows[]
  /** The log emission (or unary) potential of each state at each step, `[T][K]`. */
  readonly logEmission: Rows
}

/** Options of the chain algorithms. */
export interface ChainOptions {
  /** `scaled` (Rabiner's normalised recursions in probability space, default) or `log` (log-sum-exp recursions). */
  readonly method?: 'scaled' | 'log'
  /** Where to send `forward` and `backward` message events (one per step). */
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** Result of {@link forwardBackward}. */
export interface ForwardBackwardResult {
  /** log of the total over all paths: log p(x) for an HMM, log Z for a CRF. */
  readonly logLikelihood: number
  /** Posterior marginals p(z_t | x), `[T][K]`. */
  readonly posteriors: number[][]
  /** Pairwise posteriors p(z_t = i, z_{t+1} = j | x), `[T - 1][K][K]`. */
  readonly pairwise: number[][][]
  /** Filtered marginals p(z_t | x_{1..t}) (the normalised forward messages), `[T][K]`. */
  readonly forward: number[][]
  /** Backward messages, scaled so that forward * backward is the posterior, `[T][K]`. */
  readonly backward: number[][]
}

interface Checked {
  readonly T: number
  readonly K: number
  readonly transitionAt: (t: number) => Rows
}

function check(chain: ChainModel, name: string): Checked {
  const K = chain.logInitial.length
  const T = chain.logEmission.length
  if (K === 0) throw new MdmlError('value', `${name}: no states`)
  for (const row of chain.logEmission)
    if (row.length !== K) throw new MdmlError('shape', `${name}: every emission row needs ${K} entries`)
  const tr = chain.logTransition
  const perStep = Array.isArray(tr[0]?.[0])
  const mats = (perStep ? tr : [tr]) as readonly Rows[]
  for (const m of mats) {
    if (m.length !== K || m.some((r) => r.length !== K))
      throw new MdmlError('shape', `${name}: transitions must be ${K} x ${K}`)
  }
  if (perStep && mats.length < T - 1) throw new MdmlError('shape', `${name}: ${T - 1} transition matrices needed`)
  return { T, K, transitionAt: (t) => (perStep ? (mats[t] ?? []) : (mats[0] ?? [])) }
}

function trace(options: ChainOptions, name: string, config: Record<string, number | string>): RunHandle {
  return startRun(options.tracer ?? nullTracer, {
    name,
    seed: null,
    config,
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
}

/** exp of a log vector shifted by its maximum; returns the values and the shift. */
function shifted(logs: readonly number[]): { values: number[]; shift: number } {
  const m = maxOf(logs)
  if (m === Number.NEGATIVE_INFINITY) throw new MdmlError('value', 'chain: a step has zero potential for every state')
  return { values: logs.map((l) => Math.exp(l - m)), shift: m }
}

function normaliseRow(v: number[]): number {
  let s = 0
  for (const x of v) s += x
  if (!(s > 0)) throw new MdmlError('value', 'chain: the observations have zero probability')
  for (let i = 0; i < v.length; i++) v[i] = (v[i] ?? 0) / s
  return s
}

/**
 * The forward-backward algorithm on a chain: filtered marginals, smoothed posteriors, pairwise posteriors and the log
 * likelihood, either with Rabiner's scaled recursions or in log space (the two agree to rounding). Emits one `forward`
 * and one `backward` message per step.
 *
 * @param chain - The chain potentials.
 * @param options - Method and tracer.
 * @returns Log-likelihood, posteriors, pairwise posteriors and the forward and backward messages.
 * @throws MdmlError with code `value` when the observations are impossible.
 * @concept forward-backward
 * @index function Forward-backward smoothing on chains; viterbi finds the best path.
 * @example
 * ```ts
 * import { forwardBackward } from 'mdml-engine/pgm'
 * const chain = {
 *   logInitial: [Math.log(0.5), Math.log(0.5)],
 *   logTransition: [[Math.log(0.9), Math.log(0.1)], [Math.log(0.1), Math.log(0.9)]],
 *   logEmission: [[Math.log(0.8), Math.log(0.2)], [Math.log(0.8), Math.log(0.2)]],
 * }
 * forwardBackward(chain).posteriors[1]?.map((p) => Math.round(p * 1000) / 1000) // => [0.919, 0.081]
 * ```
 */
export function forwardBackward(chain: ChainModel, options: ChainOptions = {}): ForwardBackwardResult {
  const { T, K, transitionAt } = check(chain, 'forwardBackward')
  const method = options.method ?? 'scaled'
  const run = trace(options, 'forwardBackward', { method, T, K })
  if (T === 0) {
    run.end('completed', 0, {})
    return { logLikelihood: 0, posteriors: [], pairwise: [], forward: [], backward: [] }
  }
  const result =
    method === 'log' ? logRecursions(chain, T, K, transitionAt) : scaledRecursions(chain, T, K, transitionAt)
  for (let t = 0; t < T; t++) run.message(t, 'forward', { t, values: result.forward[t] ?? [] }, { to: `z${t}` })
  for (let t = T - 1; t >= 0; t--)
    run.message(T - 1 - t, 'backward', { t, values: result.backward[t] ?? [] }, { to: `z${t}` })
  run.end('completed', T, { logLikelihood: result.logLikelihood })
  return result
}

function scaledRecursions(
  chain: ChainModel,
  T: number,
  K: number,
  transitionAt: (t: number) => Rows,
): ForwardBackwardResult {
  const emission = chain.logEmission.map((r) => shifted(r))
  const init = shifted(chain.logInitial)
  const trans = Array.from({ length: Math.max(T - 1, 0) }, (_, t) => {
    const m = transitionAt(t)
    const s = maxOf(m.flat())
    return { values: m.map((r) => r.map((l) => Math.exp(l - s))), shift: s }
  })
  const forward: number[][] = []
  const scales: number[] = []
  let logL = init.shift
  const a0 = init.values.map((p, k) => p * (emission[0]?.values[k] ?? 0))
  scales.push(normaliseRow(a0))
  logL += Math.log(scales[0] ?? 1) + (emission[0]?.shift ?? 0)
  forward.push(a0)
  for (let t = 1; t < T; t++) {
    const prev = forward[t - 1] ?? []
    const A = trans[t - 1]
    const e = emission[t]
    const a = Array.from({ length: K }, (_, j) => {
      let s = 0
      for (let i = 0; i < K; i++) s += (prev[i] ?? 0) * (A?.values[i]?.[j] ?? 0)
      return s * (e?.values[j] ?? 0)
    })
    const c = normaliseRow(a)
    scales.push(c)
    logL += Math.log(c) + (e?.shift ?? 0) + (A?.shift ?? 0)
    forward.push(a)
  }
  const backward: number[][] = new Array(T)
  backward[T - 1] = new Array<number>(K).fill(1)
  for (let t = T - 2; t >= 0; t--) {
    const next = backward[t + 1] ?? []
    const A = trans[t]
    const e = emission[t + 1]
    const c = scales[t + 1] ?? 1
    backward[t] = Array.from({ length: K }, (_, i) => {
      let s = 0
      for (let j = 0; j < K; j++) s += (A?.values[i]?.[j] ?? 0) * (e?.values[j] ?? 0) * (next[j] ?? 0)
      return s / c
    })
  }
  const posteriors = forward.map((a, t) => a.map((v, k) => v * (backward[t]?.[k] ?? 0)))
  const pairwise = Array.from({ length: Math.max(T - 1, 0) }, (_, t) => {
    const A = trans[t]
    const e = emission[t + 1]
    const c = scales[t + 1] ?? 1
    return Array.from({ length: K }, (_, i) =>
      Array.from(
        { length: K },
        (_, j) =>
          ((forward[t]?.[i] ?? 0) * (A?.values[i]?.[j] ?? 0) * (e?.values[j] ?? 0) * (backward[t + 1]?.[j] ?? 0)) / c,
      ),
    )
  })
  return { logLikelihood: logL, posteriors, pairwise, forward, backward }
}

function logRecursions(
  chain: ChainModel,
  T: number,
  K: number,
  transitionAt: (t: number) => Rows,
): ForwardBackwardResult {
  const E = chain.logEmission
  const logAlpha: number[][] = [chain.logInitial.map((l, k) => l + (E[0]?.[k] ?? 0))]
  for (let t = 1; t < T; t++) {
    const A = transitionAt(t - 1)
    const prev = logAlpha[t - 1] ?? []
    logAlpha.push(
      Array.from({ length: K }, (_, j) => logSumExpOf(prev.map((a, i) => a + (A[i]?.[j] ?? 0))) + (E[t]?.[j] ?? 0)),
    )
  }
  const logBeta: number[][] = new Array(T)
  logBeta[T - 1] = new Array<number>(K).fill(0)
  for (let t = T - 2; t >= 0; t--) {
    const A = transitionAt(t)
    const next = logBeta[t + 1] ?? []
    logBeta[t] = Array.from({ length: K }, (_, i) =>
      logSumExpOf(next.map((b, j) => (A[i]?.[j] ?? 0) + (E[t + 1]?.[j] ?? 0) + b)),
    )
  }
  const logL = logSumExpOf(logAlpha[T - 1] ?? [])
  if (logL === Number.NEGATIVE_INFINITY) throw new MdmlError('value', 'chain: the observations have zero probability')
  const forward = logAlpha.map((a) => {
    const z = logSumExpOf(a)
    return a.map((v) => Math.exp(v - z))
  })
  // Scale the backward messages so that forward * backward is the posterior.
  const backward = logBeta.map((b, t) => {
    const z = logSumExpOf(logAlpha[t] ?? [])
    return b.map((v) => Math.exp(v + z - logL))
  })
  const posteriors = logAlpha.map((a, t) => a.map((v, k) => Math.exp(v + (logBeta[t]?.[k] ?? 0) - logL)))
  const pairwise = Array.from({ length: Math.max(T - 1, 0) }, (_, t) => {
    const A = transitionAt(t)
    return Array.from({ length: K }, (_, i) =>
      Array.from({ length: K }, (_, j) =>
        Math.exp((logAlpha[t]?.[i] ?? 0) + (A[i]?.[j] ?? 0) + (E[t + 1]?.[j] ?? 0) + (logBeta[t + 1]?.[j] ?? 0) - logL),
      ),
    )
  })
  return { logLikelihood: logL, posteriors, pairwise, forward, backward }
}

/**
 * Forward filtering: the filtered marginals p(z_t | x_{1..t}), the one-step predictions p(z_t | x_{1..t-1}) and the log
 * likelihood, in one pass.
 *
 * @param chain - The chain potentials.
 * @param options - Tracer (one `forward` message per step).
 * @returns Filtered and predicted marginals and the log-likelihood.
 * @concept forward-backward
 * @example
 * ```ts
 * import { forwardFilter } from 'mdml-engine/pgm'
 * const chain = { logInitial: [0, 0].map(() => Math.log(0.5)), logTransition: [[0, -Infinity], [-Infinity, 0]], logEmission: [[Math.log(0.9), Math.log(0.1)]] }
 * forwardFilter(chain).filtered[0]?.map((p) => Math.round(p * 10) / 10) // => [0.9, 0.1]
 * ```
 */
export function forwardFilter(
  chain: ChainModel,
  options: Omit<ChainOptions, 'method'> = {},
): { readonly filtered: number[][]; readonly predicted: number[][]; readonly logLikelihood: number } {
  const { T, K, transitionAt } = check(chain, 'forwardFilter')
  const run = trace(options, 'forwardFilter', { T, K })
  const filtered: number[][] = []
  const predicted: number[][] = []
  let logL = 0
  for (let t = 0; t < T; t++) {
    let logPrior: number[]
    if (t === 0) {
      const z = logSumExpOf(chain.logInitial)
      logPrior = chain.logInitial.map((l) => l - z)
      logL += z
    } else {
      const A = transitionAt(t - 1)
      const prev = filtered[t - 1] ?? []
      const raw = Array.from({ length: K }, (_, j) => logSumExpOf(prev.map((p, i) => Math.log(p) + (A[i]?.[j] ?? 0))))
      const z = logSumExpOf(raw)
      logPrior = raw.map((l) => l - z)
      logL += z
    }
    predicted.push(logPrior.map(Math.exp))
    const post = logPrior.map((l, k) => l + (chain.logEmission[t]?.[k] ?? 0))
    const z = logSumExpOf(post)
    if (z === Number.NEGATIVE_INFINITY) throw new MdmlError('value', 'chain: the observations have zero probability')
    logL += z
    filtered.push(post.map((l) => Math.exp(l - z)))
    run.message(t, 'forward', { t, values: filtered[t] ?? [] }, { to: `z${t}` })
  }
  run.end('completed', T, { logLikelihood: logL })
  return { filtered, predicted, logLikelihood: logL }
}

/**
 * The Viterbi algorithm: the single most probable state path (max-product on the chain, with back-pointers; ties go
 * to the lower state).
 *
 * @param chain - The chain potentials.
 * @param options - Tracer (one `viterbi` message per step, the best log score of each state).
 * @returns The path and its log potential (log p(x, z) for an HMM).
 * @concept viterbi
 * @example
 * ```ts
 * import { viterbi } from 'mdml-engine/pgm'
 * const chain = {
 *   logInitial: [Math.log(0.6), Math.log(0.4)],
 *   logTransition: [[Math.log(0.7), Math.log(0.3)], [Math.log(0.4), Math.log(0.6)]],
 *   logEmission: [[Math.log(0.1), Math.log(0.6)], [Math.log(0.4), Math.log(0.3)], [Math.log(0.5), Math.log(0.1)]],
 * }
 * viterbi(chain).path // => [1, 0, 0]
 * ```
 */
export function viterbi(
  chain: ChainModel,
  options: Omit<ChainOptions, 'method'> = {},
): { readonly path: number[]; readonly logProbability: number } {
  const { T, K, transitionAt } = check(chain, 'viterbi')
  const run = trace(options, 'viterbi', { T, K })
  if (T === 0) {
    run.end('completed', 0, {})
    return { path: [], logProbability: 0 }
  }
  let score = chain.logInitial.map((l, k) => l + (chain.logEmission[0]?.[k] ?? 0))
  run.message(0, 'viterbi', { t: 0, values: score }, { to: 'z0' })
  const back: number[][] = []
  for (let t = 1; t < T; t++) {
    const A = transitionAt(t - 1)
    const ptr: number[] = []
    const next = Array.from({ length: K }, (_, j) => {
      const cand = score.map((s, i) => s + (A[i]?.[j] ?? 0))
      const best = argmaxOf(cand)
      ptr.push(best)
      return (cand[best] ?? Number.NEGATIVE_INFINITY) + (chain.logEmission[t]?.[j] ?? 0)
    })
    back.push(ptr)
    score = next
    run.message(t, 'viterbi', { t, values: score }, { to: `z${t}` })
  }
  const last = argmaxOf(score)
  const logProbability = score[last] ?? Number.NEGATIVE_INFINITY
  if (logProbability === Number.NEGATIVE_INFINITY)
    throw new MdmlError('value', 'viterbi: every path has zero probability')
  const path = new Array<number>(T)
  path[T - 1] = last
  for (let t = T - 2; t >= 0; t--) path[t] = back[t]?.[path[t + 1] ?? 0] ?? 0
  run.end('completed', T, { logProbability })
  return { path, logProbability }
}

/**
 * Posterior (marginal) decoding: the most probable state at each step on its own, which minimises the expected number
 * of wrong states (and need not be a feasible path).
 *
 * @param chain - The chain potentials.
 * @returns The state of largest posterior at each step.
 * @concept forward-backward
 * @example
 * ```ts
 * import { posteriorDecode } from 'mdml-engine/pgm'
 * const chain = { logInitial: [0, 0], logTransition: [[0, 0], [0, 0]], logEmission: [[0, 1], [2, 0]] }
 * posteriorDecode(chain) // => [1, 0]
 * ```
 */
export function posteriorDecode(chain: ChainModel): number[] {
  return forwardBackward(chain).posteriors.map((p) => argmaxOf(p))
}

/**
 * Forward filtering, backward sampling (Carter and Kohn 1994; Fruhwirth-Schnatter 1994): exact joint samples of the
 * state path from p(z | x): draw z_T from the last filtered marginal, then each z_t from the filtered marginal times
 * the transition into the state drawn after it.
 *
 * @param chain - The chain potentials.
 * @param key - The random key.
 * @param n - Number of paths.
 * @returns The sampled paths, `[n][T]`.
 * @concept forward-backward
 * @example
 * ```ts
 * import { backwardSample } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const chain = { logInitial: [0, -Infinity], logTransition: [[-Infinity, 0], [0, -Infinity]], logEmission: [[0, 0], [0, 0], [0, 0]] }
 * backwardSample(chain, key(0), 2) // => [[0, 1, 0], [0, 1, 0]]
 * ```
 */
export function backwardSample(chain: ChainModel, key: Key, n: number): number[][] {
  const { T, K, transitionAt } = check(chain, 'backwardSample')
  const { filtered } = forwardFilter(chain)
  return Array.from({ length: n }, (_, s) => {
    const u = uniform(foldIn(key, s), [Math.max(T, 1)]).data
    const path = new Array<number>(T)
    // Draw from log weights (normalised by log-sum-exp, so large CRF potentials cannot overflow).
    const draw = (logw: readonly number[], v: number): number => {
      const z = logSumExpOf(logw)
      let acc = 0
      for (let k = 0; k < K; k++) {
        acc += Math.exp((logw[k] ?? Number.NEGATIVE_INFINITY) - z)
        if (v < acc) return k
      }
      return K - 1
    }
    if (T === 0) return path
    path[T - 1] = draw((filtered[T - 1] ?? []).map(Math.log), u[T - 1] ?? 0)
    for (let t = T - 2; t >= 0; t--) {
      const A = transitionAt(t)
      const nextState = path[t + 1] ?? 0
      const logw = (filtered[t] ?? []).map((p, i) => Math.log(p) + (A[i]?.[nextState] ?? Number.NEGATIVE_INFINITY))
      path[t] = draw(logw, u[t] ?? 0)
    }
    return path
  })
}
