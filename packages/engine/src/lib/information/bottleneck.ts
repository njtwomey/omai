/**
 * The information bottleneck (Tishby, Pereira and Bialek): compress a variable X into a representation T that keeps what
 * X says about a relevance variable Y, minimising I(X; T) - beta I(T; Y). Tishby's self-consistent iteration (a
 * Blahut-Arimoto-like alternation between the encoder q(t | x), the cluster law q(t) and the decoder q(y | t)) as a
 * traced algorithm, the sweep over beta that traces the information curve, the deterministic agglomerative form that
 * merges the pair of clusters costing least relevance, the critical beta at which the trivial one-cluster solution
 * first breaks, and a planted joint distribution whose optimal clustering is known.
 */

import { OmaiError } from '../base/index.ts'
import { svdvals } from '../numerics/linalg/index.ts'
import { foldIn, uniform } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  message,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { float } from './_shared.ts'

const LN2 = Math.LN2

/** A joint distribution p(x, y) in the form the bottleneck works with. */
interface Joint {
  readonly nx: number
  readonly ny: number
  /** p(x, y), row-major, normalised. */
  readonly pxy: Float64Array
  readonly px: Float64Array
  readonly py: Float64Array
}

function readJoint(joint: TensorLike, name: string): Joint {
  const v = float(joint)
  if (v.shape.length !== 2) throw new OmaiError('shape', `${name}: the joint distribution must be a matrix [nx, ny]`)
  const [nx, ny] = v.shape as [number, number]
  const raw = toFlat(v)
  const total = raw.reduce((a, b) => a + b, 0)
  if (raw.some((x) => !(x >= 0)) || !(total > 0))
    throw new OmaiError('value', `${name}: the joint must be non-negative and not all zero`)
  const pxy = Float64Array.from(raw, (x) => x / total)
  const px = new Float64Array(nx)
  const py = new Float64Array(ny)
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      px[i] = (px[i] ?? 0) + (pxy[i * ny + j] ?? 0)
      py[j] = (py[j] ?? 0) + (pxy[i * ny + j] ?? 0)
    }
  }
  return { nx, ny, pxy, px, py }
}

/** The decoder q(y | t), the cluster law q(t) and the two informations (in nats) of an encoder q(t | x). */
interface Marginals {
  readonly qt: Float64Array
  readonly qyt: Float64Array
  readonly ixt: number
  readonly ity: number
}

function marginals(j: Joint, nt: number, enc: ArrayLike<number>): Marginals {
  const qt = new Float64Array(nt)
  const qyt = new Float64Array(nt * j.ny)
  for (let x = 0; x < j.nx; x++) {
    for (let t = 0; t < nt; t++) {
      const w = enc[x * nt + t] ?? 0
      qt[t] = (qt[t] ?? 0) + (j.px[x] ?? 0) * w
      for (let y = 0; y < j.ny; y++) qyt[t * j.ny + y] = (qyt[t * j.ny + y] ?? 0) + w * (j.pxy[x * j.ny + y] ?? 0)
    }
  }
  let ixt = 0
  let ity = 0
  for (let t = 0; t < nt; t++) {
    const mass = qt[t] ?? 0
    if (mass <= 0) continue
    for (let y = 0; y < j.ny; y++) {
      const joint = qyt[t * j.ny + y] ?? 0
      qyt[t * j.ny + y] = joint / mass
      if (joint > 0) ity += joint * Math.log(joint / (mass * (j.py[y] ?? 1)))
    }
  }
  for (let x = 0; x < j.nx; x++) {
    for (let t = 0; t < nt; t++) {
      const w = enc[x * nt + t] ?? 0
      if (w > 0 && (qt[t] ?? 0) > 0) ixt += (j.px[x] ?? 0) * w * Math.log(w / (qt[t] ?? 1))
    }
  }
  return { qt, qyt, ixt, ity }
}

/** One self-consistent update of the encoder: q(t | x) proportional to q(t) exp(-beta KL(p(y | x) || q(y | t))). */
function updateEncoder(j: Joint, nt: number, m: Marginals, beta: number): Float64Array {
  const enc = new Float64Array(j.nx * nt)
  for (let x = 0; x < j.nx; x++) {
    const px = j.px[x] ?? 0
    const logits = new Float64Array(nt)
    let top = Number.NEGATIVE_INFINITY
    for (let t = 0; t < nt; t++) {
      const mass = m.qt[t] ?? 0
      if (mass <= 0 || px <= 0) {
        logits[t] = Number.NEGATIVE_INFINITY
        continue
      }
      let kl = 0
      for (let y = 0; y < j.ny; y++) {
        const p = (j.pxy[x * j.ny + y] ?? 0) / px
        if (p > 0) kl += p * (Math.log(p) - Math.log(Math.max(m.qyt[t * j.ny + y] ?? 0, 1e-300)))
      }
      logits[t] = Math.log(mass) - beta * kl
      top = Math.max(top, logits[t] ?? Number.NEGATIVE_INFINITY)
    }
    let norm = 0
    for (let t = 0; t < nt; t++) {
      const w = Number.isFinite(top) ? Math.exp((logits[t] ?? Number.NEGATIVE_INFINITY) - top) : 1 / nt
      enc[x * nt + t] = w
      norm += w
    }
    for (let t = 0; t < nt; t++) enc[x * nt + t] = (enc[x * nt + t] ?? 0) / norm
  }
  return enc
}

function randomEncoder(
  j: Joint,
  nt: number,
  key: Parameters<typeof uniform>[0],
  perturbation: number,
  base?: Float64Array,
): Float64Array {
  const u = uniform(key, [j.nx * nt]).data
  const enc = new Float64Array(j.nx * nt)
  for (let x = 0; x < j.nx; x++) {
    let norm = 0
    for (let t = 0; t < nt; t++) {
      const w = (base === undefined ? 1 : (base[x * nt + t] ?? 0)) + perturbation * (u[x * nt + t] ?? 0)
      enc[x * nt + t] = w
      norm += w
    }
    for (let t = 0; t < nt; t++) enc[x * nt + t] = (enc[x * nt + t] ?? 0) / norm
  }
  return enc
}

/** Options of {@link informationBottleneckAlgorithm}. */
export interface BottleneckOptions {
  /** The trade-off beta >= 0: how much relevance is worth per unit of compression (default 5). */
  readonly beta?: number
  /** The number of clusters (values of T) allowed (default 4). */
  readonly clusters?: number
  /** Stop when no encoder entry changes by more than this (default 1e-9). */
  readonly tolerance?: number
  /** The size of the random jitter added to the uniform start, which breaks the symmetry between clusters (default 0.1). */
  readonly perturbation?: number
}

/** State of {@link informationBottleneckAlgorithm}. */
export interface BottleneckState {
  /** The encoder q(t | x), row-major `[nx, nt]`. */
  readonly encoder: Float64Array
  /** The compression I(X; T) in bits. */
  readonly compression: number
  /** The relevance I(T; Y) in bits. */
  readonly relevance: number
  /** The largest change of an encoder entry in the last iteration. */
  readonly change: number
  /** Iterations so far. */
  readonly iteration: number
}

/** The paths of {@link informationBottleneckAlgorithm}. */
export type BottleneckPath =
  | 'solver/beta'
  | 'solver/compression'
  | 'solver/relevance'
  | 'solver/objective'
  | 'solver/change'
  | 'solver/encoder'
  | 'solver/converged'

/** The log contract of {@link informationBottleneckAlgorithm}. */
export const bottleneckContract: Contract<BottleneckState, BottleneckPath> = defineContract({
  'solver/beta': scalar({ description: 'the trade-off beta' }),
  'solver/compression': scalar({
    description: 'I(X; T), the bits the representation keeps of X',
    unit: 'bits',
    direction: 'lower',
  }),
  'solver/relevance': scalar({
    description: 'I(T; Y), the bits the representation keeps about Y',
    unit: 'bits',
    direction: 'higher',
  }),
  'solver/objective': scalar({
    description: 'the bottleneck functional I(X; T) - beta I(T; Y) in bits',
    unit: 'bits',
    direction: 'lower',
  }),
  'solver/change': scalar({
    description: 'the largest change of an encoder entry in the iteration',
    direction: 'lower',
  }),
  'solver/encoder': tensor({
    role: 'state',
    shape: ['x', 't'],
    description: 'the encoder q(t | x), one row per value of X',
  }),
  'solver/converged': event({ context: { iteration: 'the iteration at which the encoder stopped changing' } }),
})

/**
 * Tishby's iterative information bottleneck at one beta, one self-consistent update per step:
 * q(t | x) is proportional to q(t) exp(-beta KL(p(y | x) || q(y | t))), then q(t) and q(y | t) are recomputed from it.
 * Each iteration lowers the functional I(X; T) - beta I(T; Y). The solution is a soft clustering of X by the
 * conditional p(y | x): below the critical beta ({@link bottleneckCriticalBeta}) it is trivial (T independent of X,
 * both informations zero); above it clusters appear, and as beta grows toward infinity T becomes as informative about Y
 * as X is (the data-processing bound I(T; Y) <= I(X; Y)). The state serialises.
 *
 * @param joint - The joint distribution p(x, y), a matrix `[nx, ny]` of probabilities or counts.
 * @param options - Beta, the number of clusters, the tolerance and the start's jitter.
 * @returns The algorithm; it converges when the encoder stops changing.
 * @throws OmaiError with code `shape` unless the joint is a matrix, `value` for a negative beta or fewer than two clusters.
 * @concept information-bottleneck
 * @example
 * ```ts
 * import { informationBottleneckAlgorithm } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // X is a noiseless copy of Y: a large beta keeps all of its one bit with two clusters.
 * const run = runAlgorithm(informationBottleneckAlgorithm([[0.5, 0], [0, 0.5]], { beta: 50, clusters: 2 }), { seed: 0, maxSteps: 500 })
 * Math.round(run.state.relevance * 1e6) / 1e6 // => 1
 * ```
 */
export function informationBottleneckAlgorithm(
  joint: TensorLike,
  options: BottleneckOptions = {},
): Algorithm<BottleneckState, BottleneckPath> {
  const j = readJoint(joint, 'informationBottleneckAlgorithm')
  const beta = options.beta ?? 5
  const nt = options.clusters ?? 4
  const tolerance = options.tolerance ?? 1e-9
  const perturbation = options.perturbation ?? 0.1
  if (!(beta >= 0)) throw new OmaiError('value', 'informationBottleneckAlgorithm: beta must be non-negative')
  if (!(Number.isInteger(nt) && nt >= 2))
    throw new OmaiError('value', 'informationBottleneckAlgorithm: need at least two clusters')
  const state = (encoder: Float64Array, change: number, iteration: number): BottleneckState => {
    const m = marginals(j, nt, encoder)
    return { encoder, compression: m.ixt / LN2, relevance: m.ity / LN2, change, iteration }
  }
  return {
    name: 'information/information-bottleneck',
    config: { beta, clusters: nt, tolerance, perturbation, nx: j.nx, ny: j.ny },
    schema: contractSchema(bottleneckContract),
    init: (key) => state(randomEncoder(j, nt, key, perturbation), Number.POSITIVE_INFINITY, 0),
    step: (s, ctx) => {
      const next = updateEncoder(j, nt, marginals(j, nt, s.encoder), beta)
      let change = 0
      for (let i = 0; i < next.length; i++) change = Math.max(change, Math.abs((next[i] ?? 0) - (s.encoder[i] ?? 0)))
      if (change <= tolerance) ctx.emit('solver/converged', { iteration: s.iteration + 1 })
      return state(next, change, s.iteration + 1)
    },
    done: (s) => (s.change <= tolerance ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        'solver/beta': beta,
        'solver/compression': s.compression,
        'solver/relevance': s.relevance,
        'solver/objective': s.compression - beta * s.relevance,
        'solver/change': s.change,
      },
      tensors: { 'solver/encoder': { role: 'state', value: s.encoder, shape: [j.nx, nt] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (json) => stateFromJson(json) as BottleneckState,
  }
}

/** Options of {@link bottleneckCurveAlgorithm}. */
export interface BottleneckCurveOptions {
  /** The number of clusters allowed (default 4). */
  readonly clusters?: number
  /** Iterations allowed at each beta (default 2000). */
  readonly iterations?: number
  /** Convergence tolerance at each beta (default 1e-9). */
  readonly tolerance?: number
  /** Jitter added to the previous solution at each new beta, which lets a collapsed solution split (default 0.05). */
  readonly perturbation?: number
}

/** State of {@link bottleneckCurveAlgorithm}. */
export interface BottleneckCurveState {
  /** Betas done. */
  readonly index: number
  /** The encoder at the last beta. */
  readonly encoder: Float64Array
  /** The points (I(X; T), I(T; Y)) in bits so far, flattened in pairs. */
  readonly points: readonly number[]
  /** Iterations the last beta took. */
  readonly iterations: number
  /** The effective number of clusters at the last beta: 2 to the power of I(X; T), 1 for the trivial solution. */
  readonly effectiveClusters: number
}

/** The paths of {@link bottleneckCurveAlgorithm}. */
export type BottleneckCurvePath =
  | 'solver/beta'
  | 'solver/compression'
  | 'solver/relevance'
  | 'solver/encoder'
  | 'solver/iterations'
  | 'curve/points'
  | 'curve/clusters'

/** The log contract of {@link bottleneckCurveAlgorithm}. */
export const bottleneckCurveContract: Contract<BottleneckCurveState, BottleneckCurvePath> = defineContract({
  'solver/beta': scalar({ description: 'the beta of the step' }),
  'solver/compression': scalar({ description: 'I(X; T) at this beta', unit: 'bits' }),
  'solver/relevance': scalar({ description: 'I(T; Y) at this beta', unit: 'bits' }),
  'solver/encoder': tensor({ role: 'state', shape: ['x', 't'], description: 'the encoder q(t | x) at this beta' }),
  'solver/iterations': scalar({ description: 'iterations the beta took to converge' }),
  'curve/points': tensor({
    role: 'output',
    shape: ['betas', 2],
    description: 'the information curve so far: I(X; T) and I(T; Y) at each beta',
  }),
  'curve/clusters': scalar({ description: 'the effective number of clusters, 2^I(X; T): 1 when T ignores X' }),
})

/**
 * The information curve, one beta per step: for each beta in turn the bottleneck iteration is run to convergence from
 * the previous solution (jittered, so a collapsed solution can split once beta passes the critical value), and the point
 * (I(X; T), I(T; Y)) is added. Ascending betas trace the curve out of the origin, where the first cluster splits off at
 * {@link bottleneckCriticalBeta}; descending betas show the clusters merging as beta falls. The curve is concave and lies
 * under I(T; Y) <= min(I(X; T), I(X; Y)). The state serialises.
 *
 * @param joint - The joint distribution p(x, y).
 * @param betas - The values of beta, in the order they are visited.
 * @param options - Clusters, iterations per beta, tolerance and jitter.
 * @returns The algorithm; it completes after one step per beta.
 * @throws OmaiError with code `shape` unless the joint is a matrix, `value` for no betas, a negative beta or fewer than
 *   two clusters.
 * @concept information-bottleneck
 * @example
 * ```ts
 * import { bottleneckCurveAlgorithm } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(bottleneckCurveAlgorithm([[0.5, 0], [0, 0.5]], [0, 50], { clusters: 2 }), { seed: 0, maxSteps: 5 })
 * run.state.points.map((x) => Math.round(x * 1e6) / 1e6) // => [0, 0, 1, 1]
 * ```
 */
export function bottleneckCurveAlgorithm(
  joint: TensorLike,
  betas: readonly number[],
  options: BottleneckCurveOptions = {},
): Algorithm<BottleneckCurveState, BottleneckCurvePath> {
  const j = readJoint(joint, 'bottleneckCurveAlgorithm')
  const nt = options.clusters ?? 4
  const iterations = options.iterations ?? 2000
  const tolerance = options.tolerance ?? 1e-9
  const perturbation = options.perturbation ?? 0.05
  if (betas.length === 0 || betas.some((b) => !(b >= 0)))
    throw new OmaiError('value', 'bottleneckCurveAlgorithm: need one or more non-negative betas')
  if (!(Number.isInteger(nt) && nt >= 2))
    throw new OmaiError('value', 'bottleneckCurveAlgorithm: need at least two clusters')
  return {
    name: 'information/bottleneck-curve',
    config: { betas: [...betas], clusters: nt, iterations, tolerance, perturbation },
    schema: contractSchema(bottleneckCurveContract),
    init: (key) => ({
      index: 0,
      encoder: randomEncoder(j, nt, key, 0.1),
      points: [],
      iterations: 0,
      effectiveClusters: 1,
    }),
    step: (s, ctx) => {
      const beta = betas[s.index] as number
      let enc = s.index === 0 ? s.encoder : randomEncoder(j, nt, foldIn(ctx.key, 0), perturbation, s.encoder)
      let used = 0
      for (; used < iterations; used++) {
        const next = updateEncoder(j, nt, marginals(j, nt, enc), beta)
        let change = 0
        for (let i = 0; i < next.length; i++) change = Math.max(change, Math.abs((next[i] ?? 0) - (enc[i] ?? 0)))
        enc = next
        if (change <= tolerance) break
      }
      const m = marginals(j, nt, enc)
      return {
        index: s.index + 1,
        encoder: enc,
        points: [...s.points, m.ixt / LN2, m.ity / LN2],
        iterations: used + 1,
        effectiveClusters: 2 ** (m.ixt / LN2),
      }
    },
    done: (s) => (s.index >= betas.length ? 'completed' : undefined),
    observe: (s) => {
      const k = s.points.length / 2
      return {
        metrics: {
          'solver/beta': betas[Math.max(s.index - 1, 0)] ?? 0,
          'solver/compression': s.points[2 * k - 2] ?? 0,
          'solver/relevance': s.points[2 * k - 1] ?? 0,
          'solver/iterations': s.iterations,
          'curve/clusters': s.effectiveClusters,
        },
        tensors: {
          'solver/encoder': { role: 'state', value: s.encoder, shape: [j.nx, nt] },
          'curve/points': { role: 'output', value: s.points, shape: [k, 2] },
        },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (json) => stateFromJson(json) as BottleneckCurveState,
  }
}

/**
 * The critical beta of the bottleneck, beta_c = 1 / lambda_2, where lambda_2 is the second largest eigenvalue of the
 * matrix of the chain x -> y -> x' (the square of the second singular value of p(x, y) / sqrt(p(x) p(y))). The
 * second variation of I(X; T) - beta I(T; Y) at the trivial solution (T independent of X) is positive exactly when
 * beta lambda_2 < 1: below beta_c the trivial solution is the minimum, at it the first cluster splits off. It is at
 * least 1, because I(T; Y) <= I(T; X) makes every beta below 1 trivial; it is 1 when X and Y determine each other
 * (lambda_2 = 1), and infinite when they are independent (lambda_2 = 0: nothing to keep).
 *
 * @param joint - The joint distribution p(x, y).
 * @returns beta_c (infinity when lambda_2 = 0).
 * @throws OmaiError with code `shape` unless the joint is a matrix.
 * @concept information-bottleneck
 * @example
 * ```ts
 * import { bottleneckCriticalBeta } from 'omai-engine/information'
 * bottleneckCriticalBeta([[0.25, 0.25], [0.25, 0.25]]) // => Infinity
 * bottleneckCriticalBeta([[0.5, 0], [0, 0.5]]) // => 1
 * ```
 */
export function bottleneckCriticalBeta(joint: TensorLike): number {
  const j = readJoint(joint, 'bottleneckCriticalBeta')
  const q = new Float64Array(j.nx * j.ny)
  for (let x = 0; x < j.nx; x++) {
    for (let y = 0; y < j.ny; y++) {
      const d = Math.sqrt((j.px[x] ?? 0) * (j.py[y] ?? 0))
      q[x * j.ny + y] = d > 0 ? (j.pxy[x * j.ny + y] ?? 0) / d : 0
    }
  }
  const sv = Array.from(toFlat(svdvals(fromStorage(q, [j.nx, j.ny], 'float64'))))
  const lambda2 = (sv[1] ?? 0) ** 2
  return lambda2 <= 1e-12 ? Number.POSITIVE_INFINITY : 1 / lambda2
}

/** State of {@link agglomerativeBottleneckAlgorithm}. */
export interface AgglomerativeState {
  /** The members of each cluster (values of X). */
  readonly clusters: readonly (readonly number[])[]
  /** The relevance I(T; Y) of the clustering, in bits. */
  readonly relevance: number
  /** The compression I(X; T) = H(T) of the clustering, in bits (the clustering is hard). */
  readonly compression: number
  /** The pair merged last (cluster indices before the merge) and the relevance it cost, in bits. */
  readonly last?: { readonly pair: readonly [number, number]; readonly cost: number }
}

/** The paths of {@link agglomerativeBottleneckAlgorithm}. */
export type AgglomerativePath =
  | 'merge/pair'
  | 'merge/cost'
  | 'clusters/count'
  | 'solver/compression'
  | 'solver/relevance'

/** The log contract of {@link agglomerativeBottleneckAlgorithm}. */
export const agglomerativeContract: Contract<AgglomerativeState, AgglomerativePath> = defineContract({
  'merge/pair': message({
    description: 'the two clusters merged',
    payload: { a: 'the first', b: 'the second', cost: 'the bits of relevance lost' },
  }),
  'merge/cost': scalar({
    description: 'the bits of relevance I(T; Y) the merge lost',
    unit: 'bits',
    direction: 'lower',
  }),
  'clusters/count': scalar({ description: 'clusters left' }),
  'solver/compression': scalar({ description: 'I(X; T) = H(T) of the hard clustering', unit: 'bits' }),
  'solver/relevance': scalar({ description: 'I(T; Y) of the clustering', unit: 'bits' }),
})

function clusterStats(j: Joint, clusters: readonly (readonly number[])[]): { mass: number[]; cond: number[][] } {
  const mass = clusters.map((c) => c.reduce((a, x) => a + (j.px[x] ?? 0), 0))
  const cond = clusters.map((c, k) =>
    Array.from({ length: j.ny }, (_, y) => c.reduce((a, x) => a + (j.pxy[x * j.ny + y] ?? 0), 0) / (mass[k] || 1)),
  )
  return { mass, cond }
}

function hardRelevance(j: Joint, mass: readonly number[], cond: readonly (readonly number[])[]): number {
  let r = 0
  mass.forEach((m, k) => {
    cond[k]?.forEach((p, y) => {
      if (p > 0 && m > 0) r += m * p * Math.log2(p / (j.py[y] ?? 1))
    })
  })
  return r
}

/**
 * The cost in bits of relevance of merging two clusters: (p1 + p2) JS_pi(p(y | t1), p(y | t2)), the Jensen-Shannon
 * divergence weighted by the clusters' shares. The merge lowers I(T; Y) by exactly this amount.
 */
function mergeCost(m1: number, c1: readonly number[], m2: number, c2: readonly number[]): number {
  const total = m1 + m2
  let cost = 0
  for (let y = 0; y < c1.length; y++) {
    const a = c1[y] ?? 0
    const b = c2[y] ?? 0
    const mix = (m1 * a + m2 * b) / total
    if (a > 0) cost += m1 * a * Math.log2(a / mix)
    if (b > 0) cost += m2 * b * Math.log2(b / mix)
  }
  return cost
}

/**
 * The agglomerative information bottleneck (Slonim and Tishby), one merge per step: start with every value of X its own
 * cluster and repeatedly merge the pair whose merge costs least relevance, (p1 + p2) times the weighted Jensen-Shannon
 * divergence of their conditionals p(y | t). The relevance I(T; Y) falls by exactly that cost at each merge, and the
 * compression I(X; T) = H(T) falls as the clusters coarsen: the deterministic limit of the bottleneck as beta goes to
 * infinity, read from the fine end. Values of X with identical p(y | x) merge for free. The state serialises.
 *
 * @param joint - The joint distribution p(x, y).
 * @returns The algorithm; it completes when one cluster is left.
 * @throws OmaiError with code `shape` unless the joint is a matrix.
 * @concept information-bottleneck
 * @example
 * ```ts
 * import { agglomerativeBottleneckAlgorithm } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // Two values of X with the same conditional merge at no cost.
 * const run = runAlgorithm(agglomerativeBottleneckAlgorithm([[0.25, 0.25], [0.125, 0.125], [0.25, 0]]), { seed: 0, maxSteps: 1 })
 * [run.state.clusters, run.state.last?.cost] // => [[[0, 1], [2]], 0]
 * ```
 */
export function agglomerativeBottleneckAlgorithm(joint: TensorLike): Algorithm<AgglomerativeState, AgglomerativePath> {
  const j = readJoint(joint, 'agglomerativeBottleneckAlgorithm')
  const describe = (
    clusters: readonly (readonly number[])[],
    last?: AgglomerativeState['last'],
  ): AgglomerativeState => {
    const { mass, cond } = clusterStats(j, clusters)
    let h = 0
    for (const m of mass) if (m > 0) h -= m * Math.log2(m)
    return {
      clusters,
      relevance: hardRelevance(j, mass, cond),
      compression: h,
      ...(last === undefined ? {} : { last }),
    }
  }
  return {
    name: 'information/agglomerative-bottleneck',
    config: { nx: j.nx, ny: j.ny },
    schema: contractSchema(agglomerativeContract),
    init: () => describe(Array.from({ length: j.nx }, (_, x) => [x])),
    step: (s) => {
      const { mass, cond } = clusterStats(j, s.clusters)
      let best = [0, 1] as [number, number]
      let cost = Number.POSITIVE_INFINITY
      for (let a = 0; a < s.clusters.length; a++) {
        for (let b = a + 1; b < s.clusters.length; b++) {
          const c = mergeCost(mass[a] ?? 0, cond[a] ?? [], mass[b] ?? 0, cond[b] ?? [])
          if (c < cost - 1e-15) {
            cost = c
            best = [a, b]
          }
        }
      }
      const merged = [...(s.clusters[best[0]] ?? []), ...(s.clusters[best[1]] ?? [])].sort((p, q) => p - q)
      const clusters = s.clusters.filter((_, k) => k !== best[0] && k !== best[1])
      clusters.splice(best[0], 0, merged)
      return describe(clusters, { pair: best, cost })
    },
    done: (s) => (s.clusters.length <= 1 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        'merge/cost': s.last?.cost ?? 0,
        'clusters/count': s.clusters.length,
        'solver/compression': s.compression,
        'solver/relevance': s.relevance,
      },
      ...(s.last === undefined
        ? {}
        : { messages: [{ kind: 'merge/pair', payload: { a: s.last.pair[0], b: s.last.pair[1], cost: s.last.cost } }] }),
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (json) => stateFromJson(json) as AgglomerativeState,
  }
}

/** A planted joint distribution and what is true of it. */
export interface PlantedBottleneck {
  /** The joint p(x, y), `[groups x perGroup, outputs]`. */
  readonly joint: Tensor<'float64'>
  /** The group of each value of X. */
  readonly groups: readonly number[]
  /** I(X; Y) in bits: all the relevance there is to keep. */
  readonly mutualInformation: number
  /** H(G) = log2 of the number of groups, the bits the optimal clustering by group needs: I(X; G). */
  readonly groupEntropy: number
}

/** Options of {@link plantedBottleneck}. */
export interface PlantedBottleneckOptions {
  /** The number of groups (default 3). */
  readonly groups?: number
  /** Values of X per group (default 4). */
  readonly perGroup?: number
  /** The number of values of Y (default: the number of groups). */
  readonly outputs?: number
  /** How strongly a group's own output dominates, in [0, 1] (default 0.8): p(y | group g) = signal on y = g (mod outputs), the rest spread evenly. */
  readonly signal?: number
}

/**
 * A joint distribution with planted structure for the bottleneck: X takes `groups x perGroup` equally likely values, and
 * the conditional p(y | x) depends only on the group of x, so all of the information X holds about Y is held by the group:
 * the optimal representation is the group (H(G) bits, keeping I(X; Y) in full) and every finer split of a group is
 * wasted compression. The truth returned is I(X; Y) and H(G).
 *
 * @param options - Number of groups, group size, outputs and signal strength.
 * @returns The joint, the groups and the truth.
 * @throws OmaiError with code `value` for fewer than two groups or a signal outside [0, 1].
 * @concept information-bottleneck
 * @example
 * ```ts
 * import { plantedBottleneck } from 'omai-engine/information'
 * const p = plantedBottleneck({ groups: 2, perGroup: 3, signal: 1 })
 * [p.groups, p.mutualInformation, p.groupEntropy] // => [[0, 0, 0, 1, 1, 1], 1, 1]
 * ```
 */
export function plantedBottleneck(options: PlantedBottleneckOptions = {}): PlantedBottleneck {
  const groups = options.groups ?? 3
  const perGroup = options.perGroup ?? 4
  const outputs = options.outputs ?? groups
  const signal = options.signal ?? 0.8
  if (
    !(
      Number.isInteger(groups) &&
      groups >= 2 &&
      Number.isInteger(perGroup) &&
      perGroup >= 1 &&
      Number.isInteger(outputs) &&
      outputs >= 2
    )
  )
    throw new OmaiError('value', 'plantedBottleneck: need at least two groups, one value per group and two outputs')
  if (!(signal >= 0 && signal <= 1)) throw new OmaiError('value', 'plantedBottleneck: signal must be in [0, 1]')
  const nx = groups * perGroup
  const data = new Float64Array(nx * outputs)
  const labels: number[] = []
  for (let x = 0; x < nx; x++) {
    const g = Math.floor(x / perGroup)
    labels.push(g)
    for (let y = 0; y < outputs; y++)
      data[x * outputs + y] = (1 / nx) * ((y === g % outputs ? signal : 0) + (1 - signal) / outputs)
  }
  // I(X; Y) = I(G; Y): the conditional is the same inside a group.
  const py = new Float64Array(outputs)
  for (let x = 0; x < nx; x++) for (let y = 0; y < outputs; y++) py[y] = (py[y] ?? 0) + (data[x * outputs + y] ?? 0)
  let mi = 0
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < outputs; y++) {
      const p = data[x * outputs + y] ?? 0
      if (p > 0) mi += p * Math.log2(p / ((1 / nx) * (py[y] ?? 1)))
    }
  }
  return {
    joint: fromStorage(data, [nx, outputs], 'float64'),
    groups: labels,
    mutualInformation: mi,
    groupEntropy: Math.log2(groups),
  }
}
