/**
 * The factorial hidden Markov model (Ghahramani and Jordan 1997): M independent Markov chains evolve side by side and
 * the observation at each step is Gaussian around a bias plus the sum of one loading per chain, chosen by that chain's
 * current state, with a diagonal noise covariance. Exact inference runs the forward-backward pass on the product chain
 * of K_1 ... K_M states, which is feasible only at small sizes and is the truth for the tests. The structured
 * variational approximation keeps every chain exact and decouples them: each chain is updated by forward-backward
 * with the local evidence the other chains' current expected contributions leave, and the sweep over chains is a
 * coordinate ascent on the evidence lower bound (the ELBO). Both E-steps feed the same closed-form M-step (weighted
 * least squares for the loadings and bias, normalised expected counts for the chains), so EM raises the log likelihood
 * (exact) or the ELBO (variational, warm-started from the previous posterior) at every step.
 */

import { maxOf, OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../../trace/index.ts'
import { solveVector } from '../_dense.ts'
import { forwardBackward } from '../inference/chain.ts'

const vec = (o: number | readonly number[]): readonly number[] => (typeof o === 'number' ? [o] : o)

/**
 * The parameters of a factorial hidden Markov model with M chains of `K_m` states and D-dimensional observations:
 * `start[m][k]`, `transition[m][k][k']`, the loading of chain m in state k `loadings[m][k][d]`, the `bias[d]` and the
 * diagonal noise `variances[d]`. The mean of an observation given the states is the bias plus the loadings. The
 * loadings and the bias are identified only through those sums (shifting a chain's loadings and the bias oppositely
 * changes nothing).
 *
 * @concept hidden-markov-models
 */
export interface FactorialHmmParameters {
  readonly start: number[][]
  readonly transition: number[][][]
  readonly loadings: number[][][]
  readonly bias: number[]
  readonly variances: number[]
}

/** A sequence of observations (a number is a 1-vector). */
export type FactorialSequence = readonly (number | readonly number[])[]

/** The posterior quantities of one sequence that the M-step needs, and the bound or likelihood they came from. */
export interface FactorialMoments {
  /** log p(sequence) for the exact posterior, the ELBO for the variational one. */
  readonly objective: number
  /** Chain marginals `[m][t][k]`. */
  readonly gamma: number[][][]
  /** Chain transition posteriors `[m][t - 1][k][k']`. */
  readonly xi: number[][][][]
  /** Summed over steps, the joint occupancy of two chains: `cross[m][n][k][k']` (for m != n). */
  readonly cross: number[][][][]
}

const sizes = (p: FactorialHmmParameters): number[] => p.start.map((s) => s.length)

/** The mean observation for each state of each chain: bias plus the loadings. */
function logGaussian(x: readonly number[], mean: readonly number[], variances: readonly number[]): number {
  return x.reduce(
    (a, v, d) =>
      a - 0.5 * Math.log(2 * Math.PI * (variances[d] ?? 1)) - (v - (mean[d] ?? 0)) ** 2 / (2 * (variances[d] ?? 1)),
    0,
  )
}

/**
 * Exact inference in a factorial hidden Markov model by the forward-backward pass on the product chain: the log
 * likelihood and, for each chain, the marginals, the transition posteriors, and the joint occupancy of every pair of
 * chains. The cost grows with the product of the chains' sizes.
 *
 * @param params - The parameters.
 * @param sequence - The observations.
 * @returns The log likelihood and the posterior moments.
 * @throws OmaiError with code `value` when the product chain would exceed 4096 states.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const p = { start: [[0.5, 0.5]], transition: [[[0.9, 0.1], [0.1, 0.9]]], loadings: [[[0], [3]]], bias: [0], variances: [1] }
 * Math.round(learning.factorialHmmExact(p, [0, 3]).objective * 1e4) / 1e4 // => -4.6512
 * ```
 */
export function factorialHmmExact(params: FactorialHmmParameters, sequence: FactorialSequence): FactorialMoments {
  const K = sizes(params)
  const M = K.length
  const J = K.reduce((a, v) => a * v, 1)
  if (J > 4096) throw new OmaiError('value', `factorialHmmExact: ${J} joint states is too many`)
  const x = sequence.map(vec)
  const T = x.length
  const tuple = (j: number): number[] => {
    const out: number[] = []
    let r = j
    for (const k of K) {
      out.push(r % k)
      r = Math.floor(r / k)
    }
    return out
  }
  const tuples = Array.from({ length: J }, (_, j) => tuple(j))
  const logInitial = tuples.map((z) => z.reduce((a, k, m) => a + Math.log(params.start[m]?.[k] ?? 0), 0))
  const logTransition = tuples.map((z) =>
    tuples.map((w) => z.reduce((a, k, m) => a + Math.log(params.transition[m]?.[k]?.[w[m] ?? 0] ?? 0), 0)),
  )
  const means = tuples.map((z) =>
    params.bias.map((b, d) => b + z.reduce((a, k, m) => a + (params.loadings[m]?.[k]?.[d] ?? 0), 0)),
  )
  const logEmission = x.map((o) => means.map((mu) => logGaussian(o, mu, params.variances)))
  const fb = forwardBackward({ logInitial, logTransition, logEmission })
  const gamma = Array.from({ length: M }, (_, m) =>
    Array.from({ length: T }, (_, t) => {
      const g = new Array<number>(K[m] ?? 0).fill(0)
      for (let j = 0; j < J; j++) g[tuples[j]?.[m] ?? 0] = (g[tuples[j]?.[m] ?? 0] ?? 0) + (fb.posteriors[t]?.[j] ?? 0)
      return g
    }),
  )
  const xi = Array.from({ length: M }, (_, m) =>
    Array.from({ length: Math.max(0, T - 1) }, (_, t) => {
      const e = Array.from({ length: K[m] ?? 0 }, () => new Array<number>(K[m] ?? 0).fill(0))
      for (let j = 0; j < J; j++)
        for (let i = 0; i < J; i++) {
          const row = e[tuples[j]?.[m] ?? 0] as number[]
          row[tuples[i]?.[m] ?? 0] = (row[tuples[i]?.[m] ?? 0] ?? 0) + (fb.pairwise[t]?.[j]?.[i] ?? 0)
        }
      return e
    }),
  )
  const cross = Array.from({ length: M }, (_, m) =>
    Array.from({ length: M }, (_, n) => {
      const c = Array.from({ length: K[m] ?? 0 }, () => new Array<number>(K[n] ?? 0).fill(0))
      if (m !== n)
        for (let t = 0; t < T; t++)
          for (let j = 0; j < J; j++) {
            const row = c[tuples[j]?.[m] ?? 0] as number[]
            row[tuples[j]?.[n] ?? 0] = (row[tuples[j]?.[n] ?? 0] ?? 0) + (fb.posteriors[t]?.[j] ?? 0)
          }
      return c
    }),
  )
  return { objective: fb.logLikelihood, gamma, xi, cross }
}

/** Options of {@link factorialHmmVariational}. */
export interface VariationalOptions {
  /** The most sweeps over the chains (default 50). */
  readonly maxSweeps?: number
  /** Stop when a sweep raises the ELBO by less than this (default 1e-8). */
  readonly tol?: number
  /** Start from these marginals `[m][t][k]` instead of a keyed jitter of the uniform. */
  readonly start?: number[][][]
  /** The key for the starting jitter. */
  readonly key?: Key
}

/** The result of {@link factorialHmmVariational}: the moments (objective = ELBO) and the ELBO after every sweep. */
export interface VariationalResult extends FactorialMoments {
  readonly history: number[]
}

/**
 * Structured variational inference in a factorial hidden Markov model (mean field across chains, exact within each):
 * the posterior is approximated by a product of one Markov chain per factor. A sweep updates chain m by
 * forward-backward with the local evidence `h_t(k) = w_k' C^-1 (x_t - b - sum of the other chains' expected loadings)
 * - w_k' C^-1 w_k / 2`, which is the exact conditional of that chain given the others' means up to their variance, and
 * the ELBO, computed from the marginals and transition posteriors of the chains, never decreases from sweep to sweep.
 * With a single chain the bound is the log likelihood and the marginals are exact.
 *
 * @param params - The parameters.
 * @param sequence - The observations.
 * @param options - Sweep limit, tolerance, and the start.
 * @returns The variational moments, with the ELBO as `objective` and its history over the sweeps.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * const p = { start: [[0.5, 0.5]], transition: [[[0.9, 0.1], [0.1, 0.9]]], loadings: [[[0], [3]]], bias: [0], variances: [1] }
 * Math.round(learning.factorialHmmVariational(p, [0, 3]).objective * 1e4) / 1e4 // => -4.6512
 * ```
 */
export function factorialHmmVariational(
  params: FactorialHmmParameters,
  sequence: FactorialSequence,
  options: VariationalOptions = {},
): VariationalResult {
  const K = sizes(params)
  const M = K.length
  const x = sequence.map(vec)
  const T = x.length
  const D = params.bias.length
  const tol = options.tol ?? 1e-8
  const maxSweeps = options.maxSweeps ?? 50
  const width = maxOf(K)
  const jitter =
    options.start === undefined ? Array.from(toFlat(uniform(options.key ?? makeKey(0), [M * T * width]))) : []
  // the start: the given marginals, or the uniform tilted by a keyed jitter so that the chains can break symmetry
  const theta: number[][][] =
    options.start !== undefined
      ? options.start.map((c) => c.map((r) => [...r]))
      : K.map((km, m) =>
          Array.from({ length: T }, (_, t) => {
            const r = Array.from({ length: km }, (_, k) => 1 + 0.5 * (jitter[(m * T + t) * width + k] ?? 0))
            const z = r.reduce((a, v) => a + v, 0)
            return r.map((v) => v / z)
          }),
        )
  const invVar = params.variances.map((v) => 1 / v)
  // the expected contribution of chain m at step t: sum_k theta_t(k) w_k, and the expectation of its square
  const contribution = (m: number, t: number, d: number): number =>
    (theta[m]?.[t] ?? []).reduce((a, g, k) => a + g * (params.loadings[m]?.[k]?.[d] ?? 0), 0)
  let xi: number[][][][] = []
  const sweep = (): void => {
    const next: number[][][][] = []
    for (let m = 0; m < M; m++) {
      const evidence = x.map((o, t) =>
        Array.from({ length: K[m] ?? 0 }, (_, k) => {
          let h = 0
          for (let d = 0; d < D; d++) {
            let rest = (o[d] ?? 0) - (params.bias[d] ?? 0)
            for (let n = 0; n < M; n++) if (n !== m) rest -= contribution(n, t, d)
            const w = params.loadings[m]?.[k]?.[d] ?? 0
            h += (invVar[d] ?? 1) * (w * rest - 0.5 * w * w)
          }
          return h
        }),
      )
      const fb = forwardBackward({
        logInitial: (params.start[m] ?? []).map(Math.log),
        logTransition: (params.transition[m] ?? []).map((r) => r.map(Math.log)),
        logEmission: evidence,
      })
      theta[m] = fb.posteriors
      next[m] = fb.pairwise
    }
    xi = next
  }
  const elbo = (): number => {
    let total = 0
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++) {
        const v = params.variances[d] ?? 1
        let mean = params.bias[d] ?? 0
        let spread = 0
        for (let m = 0; m < M; m++) {
          const mu = contribution(m, t, d)
          mean += mu
          spread +=
            (theta[m]?.[t] ?? []).reduce((a, g, k) => a + g * (params.loadings[m]?.[k]?.[d] ?? 0) ** 2, 0) - mu * mu
        }
        total += -0.5 * Math.log(2 * Math.PI * v) - (((x[t]?.[d] ?? 0) - mean) ** 2 + spread) / (2 * v)
      }
    for (let m = 0; m < M; m++) {
      const g = theta[m] ?? []
      for (const [k, p] of (g[0] ?? []).entries()) {
        total += p * Math.log(params.start[m]?.[k] ?? 0)
        if (p > 0) total -= p * Math.log(p)
      }
      for (let t = 0; t + 1 < T; t++)
        for (const [k, row] of (xi[m]?.[t] ?? []).entries())
          for (const [j, p] of row.entries())
            if (p > 0) total += p * (Math.log(params.transition[m]?.[k]?.[j] ?? 0) - Math.log(p / (g[t]?.[k] ?? 1)))
    }
    return total
  }
  const history: number[] = []
  for (let s = 0; s < maxSweeps; s++) {
    sweep()
    history.push(elbo())
    if (s > 0 && (history[s] ?? 0) - (history[s - 1] ?? 0) < tol) break
  }
  const cross = Array.from({ length: M }, (_, m) =>
    Array.from({ length: M }, (_, n) => {
      const c = Array.from({ length: K[m] ?? 0 }, () => new Array<number>(K[n] ?? 0).fill(0))
      if (m !== n)
        for (let t = 0; t < T; t++)
          for (let k = 0; k < (K[m] ?? 0); k++)
            for (let j = 0; j < (K[n] ?? 0); j++)
              (c[k] as number[])[j] = (c[k]?.[j] ?? 0) + (theta[m]?.[t]?.[k] ?? 0) * (theta[n]?.[t]?.[j] ?? 0)
      return c
    }),
  )
  return { objective: history.at(-1) ?? Number.NaN, gamma: theta, xi, cross, history }
}

/** Options of {@link factorialHmmAlgorithm}. */
export interface FactorialHmmOptions {
  /** The E-step: the exact product chain (default) or the structured variational approximation. */
  readonly inference?: 'exact' | 'variational'
  /** Variational sweeps per E-step (default 20). */
  readonly sweeps?: number
  /** The smallest noise variance (default 1e-6). */
  readonly minVariance?: number
  /** Converged when the objective gains less than this in a step (default 1e-4). */
  readonly tol?: number
}

/** The state of {@link factorialHmmAlgorithm}. */
export type FactorialHmmState = {
  params: FactorialHmmParameters
  /** The variational posterior of each sequence, kept to warm-start the next E-step (empty for exact inference). */
  posterior: number[][][][]
  /** The log likelihood (exact) or ELBO (variational) under the parameters the last step started from. */
  objective: number | null
  gain: number | null
}

/**
 * Expectation-maximisation for a factorial hidden Markov model as a traced {@link Algorithm}. Each step computes the
 * posterior moments of every sequence ({@link factorialHmmExact} or {@link factorialHmmVariational}, the latter warm
 * started from the previous step's posterior), then solves for the loadings and bias by regularised weighted least
 * squares on the expected state indicators (their second moments are exact within a chain and, under the variational
 * posterior, products of marginals across chains), the noise variances from the expected squared residuals, and the
 * start and transition distributions of each chain from the expected counts. The objective of the parameters each step
 * started from (the log likelihood, or the ELBO) never decreases. It observes the objective and the `bias` tensor.
 *
 * @param sequences - The training sequences.
 * @param start - The starting parameters.
 * @param options - The E-step, sweeps, variance floor and tolerance.
 * @param key - The key for the variational start's jitter (default `key(0)`).
 * @returns The algorithm.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const p = { start: [[0.5, 0.5]], transition: [[[0.8, 0.2], [0.2, 0.8]]], loadings: [[[0], [2]]], bias: [0], variances: [1] }
 * runAlgorithm(learning.factorialHmmAlgorithm([[0, 0.1, 2, 2.2]], p), { maxSteps: 3 }).steps // => 3
 * ```
 */
export function factorialHmmAlgorithm(
  sequences: readonly FactorialSequence[],
  start: FactorialHmmParameters,
  options: FactorialHmmOptions = {},
  key?: Key,
): Algorithm<FactorialHmmState> {
  const K = sizes(start)
  const M = K.length
  const D = start.bias.length
  const variational = options.inference === 'variational'
  const sweeps = options.sweeps ?? 20
  const floor = options.minVariance ?? 1e-6
  const tol = options.tol ?? 1e-4
  const data = sequences.map((s) => s.map(vec))
  const offsets = K.map((_, m) => K.slice(0, m).reduce((a, v) => a + v, 0))
  const q = K.reduce((a, v) => a + v, 0) + 1
  const norm = (row: readonly number[]): number[] => {
    const z = row.reduce((a, v) => a + v, 0)
    return z > 0 ? row.map((v) => v / z) : row.map(() => 1 / row.length)
  }
  return {
    name: 'factorial-hmm-em',
    config: { chains: K, observation: D, inference: variational ? 'variational' : 'exact', sweeps, tol },
    schema: channels({
      metrics: {
        objective: {
          description: variational
            ? 'evidence lower bound under the parameters the step started from'
            : 'log likelihood under the parameters the step started from',
          direction: 'higher',
        },
      },
      tensors: { bias: { role: 'param', description: 'the bias of the observation mean', shape: [D] } },
    }),
    init: () => ({ params: start, posterior: [], objective: null, gain: null }),
    step: (s, ctx) => {
      const moments = data.map((x, i) =>
        variational
          ? factorialHmmVariational(s.params, x, {
              maxSweeps: sweeps,
              tol: 0,
              ...(s.posterior[i] === undefined ? { key: foldIn(key ?? ctx.key, i) } : { start: s.posterior[i] }),
            })
          : factorialHmmExact(s.params, x),
      )
      const objective = moments.reduce((a, m) => a + m.objective, 0)
      const first = K.map((km) => new Array<number>(km).fill(0))
      const moves = K.map((km) => Array.from({ length: km }, () => new Array<number>(km).fill(0)))
      const occupancy = K.map((km) => new Array<number>(km).fill(0))
      const A = Array.from({ length: q }, () => new Array<number>(q).fill(0))
      const B = Array.from({ length: D }, () => new Array<number>(q).fill(0))
      const xx = new Array<number>(D).fill(0)
      let count = 0
      for (const [i, mo] of moments.entries()) {
        const x = data[i] ?? []
        count += x.length
        for (let m = 0; m < M; m++) {
          for (let k = 0; k < (K[m] ?? 0); k++) {
            ;(first[m] as number[])[k] = (first[m]?.[k] ?? 0) + (mo.gamma[m]?.[0]?.[k] ?? 0)
            for (const [t, o] of x.entries()) {
              const g = mo.gamma[m]?.[t]?.[k] ?? 0
              ;(occupancy[m] as number[])[k] = (occupancy[m]?.[k] ?? 0) + g
              for (let d = 0; d < D; d++)
                (B[d] as number[])[(offsets[m] ?? 0) + k] = (B[d]?.[(offsets[m] ?? 0) + k] ?? 0) + g * (o[d] ?? 0)
            }
          }
          for (const step of mo.xi[m] ?? [])
            for (const [k, row] of step.entries())
              for (const [j, p] of row.entries()) (moves[m]?.[k] as number[])[j] = (moves[m]?.[k]?.[j] ?? 0) + p
          for (let n = 0; n < M; n++)
            for (let k = 0; k < (K[m] ?? 0); k++)
              for (let j = 0; j < (K[n] ?? 0); j++)
                if (m !== n)
                  (A[(offsets[m] ?? 0) + k] as number[])[(offsets[n] ?? 0) + j] =
                    (A[(offsets[m] ?? 0) + k]?.[(offsets[n] ?? 0) + j] ?? 0) + (mo.cross[m]?.[n]?.[k]?.[j] ?? 0)
        }
        for (const o of x)
          for (let d = 0; d < D; d++) {
            xx[d] = (xx[d] ?? 0) + (o[d] ?? 0) ** 2
            ;(B[d] as number[])[q - 1] = (B[d]?.[q - 1] ?? 0) + (o[d] ?? 0)
          }
      }
      for (let m = 0; m < M; m++)
        for (let k = 0; k < (K[m] ?? 0); k++) {
          const a = (offsets[m] ?? 0) + k
          ;(A[a] as number[])[a] = occupancy[m]?.[k] ?? 0
          ;(A[a] as number[])[q - 1] = occupancy[m]?.[k] ?? 0
          ;(A[q - 1] as number[])[a] = occupancy[m]?.[k] ?? 0
        }
      ;(A[q - 1] as number[])[q - 1] = count
      const shift = (1e-8 * A.reduce((a, r, i) => a + (r[i] ?? 0), 0)) / q
      const ridged = A.map((r, i) => r.map((v, j) => v + (i === j ? shift : 0)))
      const solved = B.map((b) => solveVector(ridged, b))
      const loadings = K.map((km, m) =>
        Array.from({ length: km }, (_, k) => solved.map((w) => w[(offsets[m] ?? 0) + k] ?? 0)),
      )
      const bias = solved.map((w) => w[q - 1] ?? 0)
      const variances = solved.map((w, d) => {
        let quad = 0
        for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) quad += (w[a] ?? 0) * (A[a]?.[b] ?? 0) * (w[b] ?? 0)
        const cross = w.reduce((a, v, i) => a + v * (B[d]?.[i] ?? 0), 0)
        return Math.max((xx[d] ?? 0) - 2 * cross + quad, 0) / count + floor
      })
      const params: FactorialHmmParameters = {
        start: first.map(norm),
        transition: moves.map((m) => m.map(norm)),
        loadings,
        bias,
        variances,
      }
      return {
        params,
        posterior: variational ? moments.map((m) => m.gamma) : [],
        objective,
        gain: s.objective === null ? null : objective - s.objective,
      }
    },
    done: (s) => (s.gain !== null && s.gain < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.objective === null ? {} : { objective: s.objective },
      tensors: { bias: { role: 'param', value: s.params.bias, shape: [D] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FactorialHmmState,
  }
}

/**
 * A draw from a factorial hidden Markov model: the state path of each chain (independent) and the observations.
 *
 * @param params - The parameters.
 * @param length - The number of steps.
 * @param key - The random key.
 * @returns The states `[m][t]` and the observations `[t][d]`.
 * @example
 * ```ts
 * import { learning } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const p = { start: [[1, 0]], transition: [[[1, 0], [0, 1]]], loadings: [[[0], [3]]], bias: [1], variances: [1e-12] }
 * learning.factorialHmmSample(p, 2, key(0)).states // => [[0, 0]]
 * ```
 */
export function factorialHmmSample(
  params: FactorialHmmParameters,
  length: number,
  key: Key,
): { readonly states: number[][]; readonly observations: number[][] } {
  const K = sizes(params)
  const M = K.length
  const D = params.bias.length
  const u = Array.from(toFlat(uniform(foldIn(key, 0), [M * length])))
  const z = Array.from(toFlat(normal(foldIn(key, 1), [length, D])))
  const draw = (p: readonly number[], r: number): number => {
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i] ?? 0
      if (r < acc) return i
    }
    return p.length - 1
  }
  const states = K.map((_, m) => {
    const path: number[] = []
    for (let t = 0; t < length; t++)
      path.push(
        draw(
          t === 0 ? (params.start[m] ?? []) : (params.transition[m]?.[path[t - 1] ?? 0] ?? []),
          u[m * length + t] ?? 0,
        ),
      )
    return path
  })
  const observations = Array.from({ length }, (_, t) =>
    params.bias.map(
      (b, d) =>
        b +
        states.reduce((a, path, m) => a + (params.loadings[m]?.[path[t] ?? 0]?.[d] ?? 0), 0) +
        Math.sqrt(params.variances[d] ?? 1) * (z[t * D + d] ?? 0),
    ),
  )
  return { states, observations }
}
