import { jacobian } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { type Distribution, multivariateNormal, normal } from '../../dists/index.ts'
import { cholesky, slogdet, solve } from '../../numerics/linalg/index.ts'
import { gaussianBeliefPropagation, type LinearGaussianNode, linearGaussianBayesNet } from '../../pgm/index.ts'
import { foldIn, type Key, normal as normalDraws } from '../../random/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  concat,
  diag,
  eye,
  fromStorage,
  item,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  square,
  sub,
  swapaxes,
  type Tensor,
  take,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { unexpanded } from '../_expand.ts'
import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { handle, type ProgramTrace, type SampleSite, traceOf } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Message, Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

const LOG_2PI = Math.log(2 * Math.PI)
const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** One site's coordinates in the stacked vector of a linear-Gaussian program. */
export interface GaussianBlock {
  readonly name: string
  readonly observed: boolean
  readonly shape: readonly number[]
  readonly offset: number
  readonly size: number
  /** The markov step of the site, if it is in a loop. */
  readonly step?: number
}

/**
 * A linear-Gaussian program as one linear system: the stacked site values z (in execution order) satisfy
 * z = A z + b + e with e ~ N(0, S), A strictly lower block-triangular (each site depends on earlier ones).
 */
export interface LinearGaussianForm {
  readonly blocks: readonly GaussianBlock[]
  readonly A: Tensor
  readonly b: Tensor
  readonly S: Tensor
  /** The observed values, in the order of the observed blocks. */
  readonly y: Tensor
}

function locOf(site: SampleSite): Value {
  const base = unexpanded(site.fn)
  const loc = base.params['loc']
  if (loc === undefined) throw new WhiteboxError('value', `gaussian: site ${site.name} has no location`)
  return broadcastTo(loc, site.value.shape)
}

function noiseOf(site: SampleSite): Tensor {
  const base = unexpanded(site.fn)
  const n = product(site.value.shape)
  if (base.name === 'normal') {
    const scale = base.params['scale']
    if (scale === undefined) throw new WhiteboxError('value', `gaussian: site ${site.name} has no scale`)
    return primalOf(diag(reshape(square(broadcastTo(primalOf(scale), site.value.shape)), [n])))
  }
  const l = base.params['scaleTril']
  if (l === undefined) throw new WhiteboxError('value', `gaussian: site ${site.name} has no scale`)
  const d = site.value.shape[site.value.shape.length - 1] ?? 1
  const batch = site.value.shape.slice(0, -1)
  const cov = primalOf(matmul(l, swapaxes(l, -1, -2)))
  const count = product(batch)
  const full = primalOf(broadcastTo(cov, [...batch, d, d]))
  const flat = toFlat(full)
  const out = new Float64Array(n * n)
  for (let c = 0; c < count; c++)
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) out[(c * d + i) * n + (c * d + j)] = flat[c * d * d + i * d + j] ?? 0
  return fromStorage(out, [n, n], 'float64')
}

function blockDiagonal(blocks: readonly Tensor[]): Tensor {
  const n = blocks.reduce((s, b) => s + (b.shape[0] ?? 0), 0)
  const out = new Float64Array(n * n)
  let at = 0
  for (const b of blocks) {
    const m = b.shape[0] ?? 0
    const flat = toFlat(b)
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[(at + i) * n + at + j] = flat[i * m + j] ?? 0
    at += m
  }
  return fromStorage(out, [n, n], 'float64')
}

/**
 * Extract the linear system of a linear-Gaussian program: run it with every site's value read from one stacked vector,
 * differentiate the stacked means (the Jacobian is A, exact because the means are affine) and read the noise
 * covariances from the scales.
 *
 * @param model - A linear-Gaussian program (see `analyse(...).linearGaussian`).
 * @param data - Its data.
 * @param prototype - A trace of it (its values are the expansion point).
 * @returns The linear system.
 * @concept linear-gaussian-models
 * @example
 * ```ts
 * import { linearGaussianForm, observe, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { toArray } from 'whitebox-engine/tensor'
 * const model = () => { const x = sample('x', normal(1, 2)); observe('y', normal(x, 1), 0.5) }
 * const f = linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined))
 * toArray(f.A) // => [[0, 0], [1, 0]]
 * ```
 */
export function linearGaussianForm<D>(model: Model<D>, data: D, prototype: ProgramTrace): LinearGaussianForm {
  const blocks: GaussianBlock[] = []
  const sites: SampleSite[] = []
  let offset = 0
  for (const s of prototype.sites.values()) {
    if (s.type !== 'sample') continue
    const size = product(s.value.shape)
    const step = s.markov[0]?.step
    blocks.push({
      name: s.name,
      observed: s.isObserved,
      shape: s.value.shape,
      offset,
      size,
      ...(step === undefined ? {} : { step }),
    })
    sites.push(s)
    offset += size
  }
  const n = offset
  const byName = new Map(blocks.map((b) => [b.name, b]))
  const z0 = concat(sites.map((s) => reshape(primalOf(s.value), [product(s.value.shape)])))
  const means = (z: Value): Value => {
    const placed = handle(model, () => ({
      process: (msg: Message) => {
        if (msg.type !== 'sample') return
        const b = byName.get(msg.name)
        if (b === undefined) return
        msg.value = reshape(sliceAxis(z, 0, b.offset, b.offset + b.size), b.shape)
      },
    }))
    const t = traceOf(placed, data)
    return concat(
      blocks.map((b) => {
        const s = t.sites.get(b.name)
        if (s?.type !== 'sample') throw new WhiteboxError('value', `gaussian: site ${b.name} disappeared`)
        return reshape(locOf(s), [b.size])
      }),
    )
  }
  const A = primalOf(n === 0 ? zeros([0, 0]) : jacobian(means)(z0))
  const b = primalOf(sub(means(z0), matmul(A, z0)))
  const S = blockDiagonal(sites.map(noiseOf))
  const y = primalOf(
    concat(
      blocks.filter((x) => x.observed).map((x) => sliceAxis(z0, 0, x.offset, x.offset + x.size)),
      0,
    ),
  )
  return { blocks, A, b, S, y }
}

function indicesOf(blocks: readonly GaussianBlock[], observed: boolean): number[] {
  return blocks
    .filter((b) => b.observed === observed)
    .flatMap((b) => Array.from({ length: b.size }, (_, i) => b.offset + i))
}

function sub2(m: Tensor, rows: readonly number[], cols: readonly number[]): Tensor {
  return primalOf(take(take(m, rows, 0), cols, 1))
}

/** The exact Gaussian posterior of the latent coordinates: mean, covariance and the log evidence. */
export interface GaussianConditioning {
  readonly mean: Tensor
  readonly covariance: Tensor
  readonly logEvidence: number
  readonly latent: readonly GaussianBlock[]
}

/**
 * Exact Gaussian conditioning of a linear system on its observed coordinates: the joint is z ~ N((I - A)^{-1} b,
 * (I - A)^{-1} S (I - A)^{-T}), and the latent block given y follows from the Schur complement.
 *
 * @param form - The linear system.
 * @returns The posterior mean and covariance of the latent coordinates and log p(y).
 * @concept gaussian-conditioning
 * @example
 * ```ts
 * import { gaussianCondition, linearGaussianForm, observe, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { toArray } from 'whitebox-engine/tensor'
 * const model = () => { const x = sample('x', normal(0, 1)); observe('y', normal(x, 1), 2) }
 * const post = gaussianCondition(linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined)))
 * toArray(post.mean) // => [1]
 * ```
 */
export function gaussianCondition(form: LinearGaussianForm): GaussianConditioning {
  const n = form.b.shape[0] ?? 0
  const M = primalOf(sub(eye(n), form.A))
  const mu = primalOf(solve(M, form.b))
  const Minv = primalOf(solve(M, eye(n)))
  const cov = primalOf(matmul(matmul(Minv, form.S), transpose(Minv)))
  const L = indicesOf(form.blocks, false)
  const O = indicesOf(form.blocks, true)
  const latent = form.blocks.filter((b) => !b.observed)
  const muL = primalOf(take(mu, L, 0))
  const sLL = sub2(cov, L, L)
  if (O.length === 0) return { mean: muL, covariance: sLL, logEvidence: 0, latent }
  const muO = primalOf(take(mu, O, 0))
  const sOO = sub2(cov, O, O)
  const sLO = sub2(cov, L, O)
  const r = primalOf(sub(form.y, muO))
  const gain = primalOf(transpose(solve(sOO, transpose(sLO))))
  const mean = primalOf(add(muL, matmul(gain, r)))
  const covariance = primalOf(symmetric(sub(sLL, matmul(gain, transpose(sLO)))))
  const { logabsdet } = slogdet(sOO)
  const quad = item(matmul(r, solve(sOO, r)))
  const logEvidence = -0.5 * (O.length * LOG_2PI + item(logabsdet) + quad)
  return { mean, covariance, logEvidence, latent }
}

function symmetric(m: Value): Value {
  return mul(add(m, transpose(m)), 0.5)
}

/** Filtered and smoothed moments of a linear-Gaussian chain. */
export interface KalmanResult {
  /** Smoothed means `[T][d]` and covariances `[T][d][d]` of the latent state of each step. */
  readonly means: Tensor[]
  readonly covariances: Tensor[]
  /** Filtered means and covariances. */
  readonly filteredMeans: Tensor[]
  readonly filteredCovariances: Tensor[]
  readonly logEvidence: number
  /** Per step: transition, offset, noise, predicted mean and covariance (for backward sampling). */
  readonly transitions: readonly Tensor[]
  readonly predictedMeans: readonly Tensor[]
  readonly predictedCovariances: readonly Tensor[]
  /** The latent blocks of each step. */
  readonly steps: readonly (readonly GaussianBlock[])[]
}

/**
 * Kalman filtering and Rauch-Tung-Striebel smoothing of a linear-Gaussian chain program: the stacked system is split
 * by markov step into transitions x_t = F_t x_{t-1} + c_t + w_t and observations y_t = H_t x_t + d_t + v_t (within-step
 * dependencies solved out), then filtered forwards (with the log evidence) and smoothed backwards.
 *
 * @param form - The linear system of a chain program (every latent site in one markov loop).
 * @returns The filtered and smoothed moments and log p(y).
 * @throws WhiteboxError with code `value` when the system is not a first-order chain.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { kalmanSmoother, linearGaussianForm, markov, observe, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item, type Value } from 'whitebox-engine/tensor'
 * const ys = [1, 2]
 * const walk = () => markov('t', 2, (t, x: Value | number) => { const z = sample(`x${t}`, normal(x, 1)); observe(`y${t}`, normal(z, 1), ys[t] ?? 0); return z }, 0)
 * const k = kalmanSmoother(linearGaussianForm(walk, undefined, traceOf(seed(walk, key(0)), undefined)))
 * k.means.length // => 2
 * ```
 */
export function kalmanSmoother(form: LinearGaussianForm): KalmanResult {
  const latentSteps = new Map<number, GaussianBlock[]>()
  const obsSteps = new Map<number, GaussianBlock[]>()
  for (const b of form.blocks) {
    if (b.step === undefined) throw new WhiteboxError('value', `kalman: site ${b.name} is outside the markov loop`)
    const m = b.observed ? obsSteps : latentSteps
    m.set(b.step, [...(m.get(b.step) ?? []), b])
  }
  const T = [...latentSteps.keys()].reduce((a, b) => Math.max(a, b), -1) + 1
  const coords = (bs: readonly GaussianBlock[] | undefined): number[] =>
    (bs ?? []).flatMap((b) => Array.from({ length: b.size }, (_, i) => b.offset + i))
  const steps: GaussianBlock[][] = []
  const xIdx: number[][] = []
  const yIdx: number[][] = []
  for (let t = 0; t < T; t++) {
    steps.push(latentSteps.get(t) ?? [])
    xIdx.push(coords(latentSteps.get(t)))
    yIdx.push(coords(obsSteps.get(t)))
  }
  const allowed = (rows: readonly number[], cols: readonly number[]): void => {
    if (rows.length === 0 || cols.length === 0) return
    if (toFlat(sub2(form.A, rows, cols)).some((v) => v !== 0))
      throw new WhiteboxError('value', 'kalman: the program is not a first-order chain of latent states')
  }
  const n = form.b.shape[0] ?? 0
  const everything = Array.from({ length: n }, (_, i) => i)
  const yPos = new Map(indicesOf(form.blocks, true).map((c, i) => [c, i]))
  const transitions: Tensor[] = []
  const predictedMeans: Tensor[] = []
  const predictedCovariances: Tensor[] = []
  const filteredMeans: Tensor[] = []
  const filteredCovariances: Tensor[] = []
  let logEvidence = 0
  let m: Tensor | undefined
  let P: Tensor | undefined
  for (let t = 0; t < T; t++) {
    const xi = xIdx[t] ?? []
    const yi = yIdx[t] ?? []
    const prev = t > 0 ? (xIdx[t - 1] ?? []) : []
    allowed(
      xi,
      everything.filter((c) => !xi.includes(c) && !prev.includes(c)),
    )
    allowed(
      yi,
      everything.filter((c) => !xi.includes(c) && !yi.includes(c)),
    )
    const d = xi.length
    const W = primalOf(sub(eye(d), sub2(form.A, xi, xi)))
    const F = t > 0 ? primalOf(solve(W, sub2(form.A, xi, prev))) : zeros([d, 0])
    const c = primalOf(solve(W, take(form.b, xi, 0)))
    const Winv = primalOf(solve(W, eye(d)))
    const Q = primalOf(matmul(matmul(Winv, sub2(form.S, xi, xi)), transpose(Winv)))
    const mPred = m === undefined || t === 0 ? c : primalOf(add(matmul(F, m), c))
    const PPred = P === undefined || t === 0 ? Q : primalOf(symmetric(add(matmul(matmul(F, P), transpose(F)), Q)))
    transitions.push(F)
    predictedMeans.push(mPred)
    predictedCovariances.push(PPred)
    m = mPred
    P = PPred
    if (yi.length > 0) {
      const V = primalOf(sub(eye(yi.length), sub2(form.A, yi, yi)))
      const H = primalOf(solve(V, sub2(form.A, yi, xi)))
      const dOff = primalOf(solve(V, take(form.b, yi, 0)))
      const Vinv = primalOf(solve(V, eye(yi.length)))
      const R = primalOf(matmul(matmul(Vinv, sub2(form.S, yi, yi)), transpose(Vinv)))
      const y = primalOf(
        take(
          form.y,
          yi.map((c2) => yPos.get(c2) ?? 0),
          0,
        ),
      )
      const innov = primalOf(sub(y, add(matmul(H, m), dOff)))
      const Sinn = primalOf(symmetric(add(matmul(matmul(H, P), transpose(H)), R)))
      const K = primalOf(transpose(solve(Sinn, matmul(H, P))))
      m = primalOf(add(m, matmul(K, innov)))
      P = primalOf(symmetric(sub(P, matmul(matmul(K, H), P))))
      const { logabsdet } = slogdet(Sinn)
      logEvidence += -0.5 * (yi.length * LOG_2PI + item(logabsdet) + item(matmul(innov, solve(Sinn, innov))))
    }
    filteredMeans.push(m)
    filteredCovariances.push(P)
  }
  const means = [...filteredMeans]
  const covariances = [...filteredCovariances]
  for (let t = T - 2; t >= 0; t--) {
    const Pf = filteredCovariances[t] as Tensor
    const F = transitions[t + 1] as Tensor
    const Pp = predictedCovariances[t + 1] as Tensor
    const G = primalOf(transpose(solve(Pp, matmul(F, Pf))))
    means[t] = primalOf(
      add(filteredMeans[t] as Tensor, matmul(G, sub(means[t + 1] as Tensor, predictedMeans[t + 1] as Tensor))),
    )
    covariances[t] = primalOf(
      symmetric(add(Pf, matmul(matmul(G, sub(covariances[t + 1] as Tensor, Pp)), transpose(G)))),
    )
  }
  return {
    means,
    covariances,
    filteredMeans,
    filteredCovariances,
    logEvidence,
    transitions,
    predictedMeans,
    predictedCovariances,
    steps,
  }
}

function mvnDraws(key: Key, mean: Tensor, cov: Tensor, n: number): Tensor {
  const d = mean.shape[0] ?? 0
  if (d === 0) return zeros([n, 0])
  const L = primalOf(cholesky(add(cov, mul(eye(d), 1e-12))))
  const z = normalDraws(key, [n, d])
  return primalOf(add(matmul(z, transpose(L)), mean))
}

function siteDraws(blocks: readonly GaussianBlock[], flat: Tensor, n: number): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  let at = 0
  for (const b of blocks) {
    out[b.name] = primalOf(reshape(sliceAxis(flat, 1, at, at + b.size), [n, ...b.shape]))
    at += b.size
  }
  return out
}

function marginalsOf(blocks: readonly GaussianBlock[], mean: Tensor, cov: Tensor): Record<string, Distribution> {
  const out: Record<string, Distribution> = {}
  let at = 0
  for (const b of blocks) {
    const idx = Array.from({ length: b.size }, (_, i) => at + i)
    const m = primalOf(take(mean, idx, 0))
    const c = sub2(cov, idx, idx)
    out[b.name] =
      b.shape.length === 0
        ? normal(reshape(m, []), Math.sqrt(item(reshape(c, []))))
        : multivariateNormal(m, { covariance: c })
    at += b.size
  }
  return out
}

/** Options of {@link gaussianInference}. */
export interface GaussianOptions {
  /** `joint` (dense conditioning), `kalman` (filtering and smoothing of chains), `bp` (Gaussian belief propagation),
   * or `auto` (default: Kalman for chain programs, else joint). */
  readonly method?: 'auto' | 'joint' | 'kalman' | 'bp'
  readonly tracer?: EngineRunOptions['tracer']
  readonly policy?: EngineRunOptions['policy']
}

function isChain(a: ProgramAnalysis): boolean {
  const loop = a.markov[0]
  return a.markov.length === 1 && loop !== undefined && loop.outsideLatents.length === 0 && loop.after.length === 0
}

/**
 * Exact inference for a linear-Gaussian program: the stacked linear system conditioned on the observations, by dense
 * Gaussian conditioning, by Kalman filtering and smoothing for chain programs (with exact joint draws by backward
 * sampling), or by Gaussian belief propagation in `pgm` (exact means, and exact variances on trees).
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The method, tracer and policy.
 * @returns The posterior (exact; marginals are normal or multivariate normal per site).
 * @throws WhiteboxError with code `value` when the program is not linear-Gaussian.
 * @concept gaussian-conditioning
 * @example
 * ```ts
 * import { gaussianInference, observe, plate, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const model = (y: number[]) => { const mu = sample('mu', normal(0, 1)); plate('n', y.length, () => observe('y', normal(mu, 1), y)) }
 * item(gaussianInference(model, [1, 2, 3]).mean('mu')) // => 1.5
 * ```
 */
export function gaussianInference<D>(
  model: Model<D>,
  data: D,
  options: GaussianOptions & { readonly analysis?: ProgramAnalysis } = {},
): Posterior {
  const a = options.analysis ?? analyse(model, data)
  if (!a.linearGaussian.linear)
    throw new WhiteboxError(
      'value',
      `gaussianInference: the program is not linear-Gaussian (${a.linearGaussian.reason})`,
    )
  const form = linearGaussianForm(model, data, a.prototype)
  const latent = form.blocks.filter((b) => !b.observed)
  const method =
    options.method === undefined || options.method === 'auto' ? (isChain(a) ? 'kalman' : 'joint') : options.method
  if (method === 'kalman') {
    const k = kalmanSmoother(form)
    const marginals: Record<string, Distribution> = {}
    k.steps.forEach((bs, t) => {
      const mean = k.means[t] as Tensor
      const cov = k.covariances[t] as Tensor
      Object.assign(marginals, marginalsOf(bs, mean, cov))
    })
    const sampler = (key: Key, n: number): Record<string, Tensor> => {
      const T = k.steps.length
      const xs: Tensor[] = new Array<Tensor>(T)
      xs[T - 1] = mvnDraws(
        foldIn(key, T - 1),
        k.filteredMeans[T - 1] as Tensor,
        k.filteredCovariances[T - 1] as Tensor,
        n,
      )
      for (let t = T - 2; t >= 0; t--) {
        const Pf = k.filteredCovariances[t] as Tensor
        const F = k.transitions[t + 1] as Tensor
        const Pp = k.predictedCovariances[t + 1] as Tensor
        const G = primalOf(transpose(solve(Pp, matmul(F, Pf))))
        const cond = primalOf(symmetric(sub(Pf, matmul(matmul(G, Pp), transpose(G)))))
        const noise = mvnDraws(foldIn(key, t), zeros([Pf.shape[0] ?? 0]), cond, n)
        const centred = primalOf(sub(xs[t + 1] as Tensor, k.predictedMeans[t + 1] as Tensor))
        xs[t] = primalOf(add(add(matmul(centred, transpose(G)), k.filteredMeans[t] as Tensor), noise))
      }
      const out: Record<string, Tensor> = {}
      for (const [t, bs] of k.steps.entries()) Object.assign(out, siteDraws(bs, xs[t] as Tensor, n))
      return out
    }
    return definePosterior({
      form: 'exact',
      engine: 'gaussian',
      sites: latent.map((b) => b.name),
      marginals,
      sampler,
      logEvidence: { value: k.logEvidence, standardError: 0, method: 'Kalman filter' },
      diagnostics: { method: 'kalman', steps: k.steps.length },
    })
  }
  if (method === 'bp') {
    const n = form.b.shape[0] ?? 0
    const Af = toFlat(form.A)
    const Sf = toFlat(form.S)
    const bf = toFlat(form.b)
    const name = (i: number): string => `z${i}`
    const nodes: LinearGaussianNode[] = []
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++)
        if (i !== j && (Sf[i * n + j] ?? 0) !== 0)
          throw new WhiteboxError('value', 'gaussian bp: correlated noise within a site; use the joint method')
      const parents: string[] = []
      const weights: number[] = []
      for (let j = 0; j < n; j++) {
        const w = Af[i * n + j] ?? 0
        if (w !== 0) {
          parents.push(name(j))
          weights.push(w)
        }
      }
      nodes.push({ name: name(i), parents, weights, intercept: bf[i] ?? 0, variance: Sf[i * n + i] ?? 1 })
    }
    const evidence: Record<string, number> = {}
    const yf = toFlat(form.y)
    indicesOf(form.blocks, true).forEach((c, i) => {
      evidence[name(c)] = yf[i] ?? 0
    })
    const r = gaussianBeliefPropagation(linearGaussianBayesNet(nodes), {
      evidence,
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
    })
    const marginals: Record<string, Distribution> = {}
    for (const b of latent) {
      const idx = Array.from({ length: b.size }, (_, i) => b.offset + i)
      const mean = fromStorage(
        Float64Array.from(idx, (i) => r.means[name(i)] ?? Number.NaN),
        b.shape,
        'float64',
      )
      const sd = fromStorage(
        Float64Array.from(idx, (i) => Math.sqrt(r.variances[name(i)] ?? Number.NaN)),
        b.shape,
        'float64',
      )
      marginals[b.name] = normal(mean, sd)
    }
    return definePosterior({
      form: a.graphShape === 'loopy' ? 'variational' : 'exact',
      engine: 'gaussian',
      sites: latent.map((b) => b.name),
      marginals,
      status: r.converged ? 'converged' : r.status,
      diagnostics: { method: 'bp', iterations: r.iterations, converged: r.converged },
    })
  }
  const post = gaussianCondition(form)
  return definePosterior({
    form: 'exact',
    engine: 'gaussian',
    sites: latent.map((b) => b.name),
    marginals: marginalsOf(latent, post.mean, post.covariance),
    sampler: (key, n) => siteDraws(latent, mvnDraws(key, post.mean, post.covariance, n), n),
    logProb: (values) => {
      const x = concat(
        latent.map((b) => {
          const v = values[b.name]
          if (v === undefined) throw new WhiteboxError('value', `gaussian posterior: no value for ${b.name}`)
          return reshape(asTensor(v), [b.size])
        }),
      )
      return item(multivariateNormal(post.mean, { covariance: post.covariance }).logProb(x))
    },
    logEvidence: { value: post.logEvidence, standardError: 0, method: 'Gaussian conditioning' },
    diagnostics: { method: 'joint', coordinates: form.b.shape[0] ?? 0 },
  })
}

/**
 * The Gaussian engine: exact inference for linear-Gaussian programs (cost: the cube of the number of coordinates, or
 * steps times the cube of the state size for chains).
 *
 * @param options - The method.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { analyse, gaussianEngine, observe, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * const a = analyse(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }, undefined)
 * gaussianEngine().applies(a).applies // => true
 * ```
 */
export function gaussianEngine(options: Pick<GaussianOptions, 'method'> = {}): InferenceEngine {
  return {
    name: 'gaussian',
    form: 'exact',
    description: 'exact Gaussian conditioning, Kalman smoothing and Gaussian belief propagation',
    applies: (a: ProgramAnalysis) => {
      if (!a.linearGaussian.linear) return feasibility(false, 'exact', 0, a.linearGaussian.reason)
      const n = a.sites
        .filter((s) => s.role === 'latent' || s.role === 'observed')
        .reduce((s, x) => s + x.elements * Math.max(1, x.eventShape[0] ?? 1), 0)
      const loop = a.markov[0]
      const cost = isChain(a) && loop !== undefined ? loop.length * (n / loop.length) ** 3 : n ** 3
      return feasibility(true, 'exact', Math.max(1, cost), a.linearGaussian.reason)
    },
    run: <D>(model: Model<D>, data: D, a: ProgramAnalysis, run: EngineRunOptions) =>
      gaussianInference(model, data, {
        analysis: a,
        ...(options.method === undefined ? {} : { method: options.method }),
        ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
        ...(run.policy === undefined ? {} : { policy: run.policy }),
      }),
  }
}
