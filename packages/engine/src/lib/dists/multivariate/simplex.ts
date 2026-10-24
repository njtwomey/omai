import { WhiteboxError } from '../../base/index.ts'
import { gaussHermite } from '../../numerics/quadrature/index.ts'
import { digamma, lgamma, xlogy } from '../../numerics/special/index.ts'
import { dirichlet as dirichletDraw, foldIn, type Key, multinomial as multinomialDraw } from '../../random/index.ts'
import {
  add,
  concat,
  div,
  exp,
  expandDims,
  eye,
  floor,
  fromStorage,
  isfinite,
  log,
  matmul,
  mul,
  neg,
  promoteFloat,
  reshape,
  sliceAxis,
  softmax,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { discreteSample, gammaSample } from '../_sampling.ts'
import { broadcastBatch, type Distribution, defineDistribution, inSupport, validateParams } from '../distribution.ts'
import { type CovarianceParams, multivariateNormal } from './normal.ts'

/**
 * The Dirichlet distribution over the probability simplex, with concentrations along the last axis
 * (scipy.stats.dirichlet, torch.distributions.Dirichlet).
 *
 * @param concentration - Concentrations `[..., K]` (positive).
 * @returns The distribution; `rsample` normalises implicitly reparameterised gamma samples, so samples differentiate in
 *   the concentrations.
 * @concept distributions
 * @example
 * ```ts
 * import { dirichlet } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(dirichlet([1, 3]).mean()) // => [0.25, 0.75]
 * ```
 */
export function dirichlet(concentration: TensorLike): Distribution {
  const [alpha] = promoteFloat(concentration) as [Value]
  if (alpha.shape.length === 0) throw new WhiteboxError('shape', 'dirichlet: concentrations need a category axis')
  validateParams('dirichlet', { concentration: [alpha, 'positive'] })
  const k = alpha.shape[alpha.shape.length - 1] ?? 0
  const total = sum(alpha, -1)
  const totalKeep = sum(alpha, -1, true)
  const logNorm = sub(sum(lgamma(alpha), -1), lgamma(total))
  return defineDistribution({
    name: 'dirichlet',
    params: { concentration: alpha },
    support: { kind: 'simplex' },
    batchShape: alpha.shape.slice(0, -1),
    eventShape: [k],
    logProb: (x) =>
      where(
        inSupport({ kind: 'simplex' }, x),
        sub(sum(xlogy(sub(alpha, 1), x), -1), logNorm),
        Number.NEGATIVE_INFINITY,
      ),
    rsample: (key, shape) => {
      const g = gammaSample(key, alpha, shape)
      return div(g, sum(g, -1, true))
    },
    mean: () => div(alpha, totalKeep),
    variance: () => div(mul(alpha, sub(totalKeep, alpha)), mul(mul(totalKeep, totalKeep), add(totalKeep, 1))),
    mode: () => div(sub(alpha, 1), sub(totalKeep, k)),
    entropy: () => add(sub(logNorm, sum(mul(sub(alpha, 1), digamma(alpha)), -1)), mul(sub(total, k), digamma(total))),
  })
}

/** Every vector of k non-negative integers summing to n (for exact entropies of count vectors). */
function compositions(n: number, k: number): number[][] {
  if (k === 1) return [[n]]
  const out: number[][] = []
  for (let first = 0; first <= n; first++) for (const rest of compositions(n - first, k - 1)) out.push([first, ...rest])
  return out
}

/** -sum p log p over every count vector of total n, for each batch member (outcomes on a new leading axis). */
function entropyByEnumeration(logPmf: (x: Value) => Value, n: number, k: number, batchRank: number): Value {
  const outcomes = promoteFloat(compositions(n, k))[0] as Value
  const m = outcomes.shape[0] ?? 0
  const x = reshape(outcomes, [m, ...new Array<number>(batchRank).fill(1), k])
  const lp = logPmf(x)
  const finite = isfinite(lp)
  const safe = where(finite, lp, 0)
  return neg(sum(where(finite, mul(exp(safe), safe), 0), 0))
}

/**
 * The multinomial distribution of counts in K categories from n draws, with probabilities along the last axis
 * (scipy.stats.multinomial, torch.distributions.Multinomial).
 *
 * @param n - Number of draws (a non-negative integer).
 * @param probs - Category probabilities `[..., K]` (normalised for you).
 * @returns The distribution (no reparameterisation; `entropy` enumerates outcomes and suits small n and K).
 * @example
 * ```ts
 * import { multinomial } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(multinomial(10, [0.2, 0.8]).mean()) // => [2, 8]
 * ```
 */
export function multinomial(n: number, probs: TensorLike): Distribution {
  if (!(Number.isInteger(n) && n >= 0))
    throw new WhiteboxError('value', 'multinomial: n must be a non-negative integer')
  const [raw] = promoteFloat(probs) as [Value]
  validateParams('multinomial', { probs: [raw, 'nonNegative'] })
  const p = div(raw, sum(raw, -1, true))
  const k = p.shape[p.shape.length - 1] ?? 0
  const batchShape = p.shape.slice(0, -1)
  const support = { kind: 'counts', total: n } as const
  const logPmf = (x: Value): Value => {
    const lp = add(sub(lgamma(n + 1), sum(lgamma(add(x, 1)), -1)), sum(xlogy(x, p), -1))
    return where(inSupport(support, x), lp, Number.NEGATIVE_INFINITY)
  }
  return defineDistribution({
    name: 'multinomial',
    params: { probs: p },
    support,
    batchShape,
    eventShape: [k],
    logProb: logPmf,
    sample: (key: Key, shape) =>
      countsSample(key, shape, batchShape, k, [p], ([pt]) => (row) => {
        const counts = multinomialDraw(row.key, n, row.slice(pt as Tensor), 1).data
        return Array.from(counts)
      }),
    mean: () => mul(p, n),
    variance: () => mul(mul(p, sub(1, p)), n),
    mode: () => floor(mul(p, n + 1)),
    entropy: () => entropyByEnumeration(logPmf, n, k, batchShape.length),
  })
}

/**
 * The Dirichlet-multinomial (Polya) distribution: multinomial counts whose probabilities are Dirichlet
 * (scipy.stats.dirichlet_multinomial), an overdispersed multinomial.
 *
 * @param n - Number of draws (a non-negative integer).
 * @param concentration - Dirichlet concentrations `[..., K]` (positive).
 * @returns The distribution (no reparameterisation; `entropy` enumerates outcomes and suits small n and K).
 * @example
 * ```ts
 * import { dirichletMultinomial } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(dirichletMultinomial(10, [1, 3]).mean()) // => [2.5, 7.5]
 * ```
 */
export function dirichletMultinomial(n: number, concentration: TensorLike): Distribution {
  if (!(Number.isInteger(n) && n >= 0))
    throw new WhiteboxError('value', 'dirichletMultinomial: n must be a non-negative integer')
  const [alpha] = promoteFloat(concentration) as [Value]
  if (alpha.shape.length === 0)
    throw new WhiteboxError('shape', 'dirichletMultinomial: concentrations need a category axis')
  validateParams('dirichletMultinomial', { concentration: [alpha, 'positive'] })
  const k = alpha.shape[alpha.shape.length - 1] ?? 0
  const batchShape = alpha.shape.slice(0, -1)
  const total = sum(alpha, -1)
  const totalKeep = sum(alpha, -1, true)
  const support = { kind: 'counts', total: n } as const
  const logPmf = (x: Value): Value => {
    const lp = add(
      sub(add(lgamma(n + 1), lgamma(total)), lgamma(add(total, n))),
      sum(sub(lgamma(add(x, alpha)), add(lgamma(add(x, 1)), lgamma(alpha))), -1),
    )
    return where(inSupport(support, x), lp, Number.NEGATIVE_INFINITY)
  }
  const share = div(alpha, totalKeep)
  return defineDistribution({
    name: 'dirichletMultinomial',
    params: { concentration: alpha },
    support,
    batchShape,
    eventShape: [k],
    logProb: logPmf,
    sample: (key: Key, shape) =>
      countsSample(key, shape, batchShape, k, [alpha], ([at]) => (row) => {
        const probs = Array.from(dirichletDraw(foldIn(row.key, 0), row.slice(at as Tensor), 1).data)
        return Array.from(multinomialDraw(foldIn(row.key, 1), n, probs, 1).data)
      }),
    mean: () => mul(share, n),
    variance: () => mul(mul(mul(share, sub(1, share)), n), div(add(totalKeep, n), add(totalKeep, 1))),
    mode: () => floor(mul(share, n + 1)),
    entropy: () => entropyByEnumeration(logPmf, n, k, batchShape.length),
  })
}

interface Row {
  readonly key: Key
  slice(t: Tensor): number[]
}

/** Draw `[...shape, ...batch, k]` count vectors, one row at a time, from concrete parameters. */
function countsSample(
  key: Key,
  shape: readonly number[],
  batchShape: readonly number[],
  k: number,
  params: readonly Value[],
  draw: (values: readonly Tensor[]) => (row: Row) => number[],
): Value {
  const full = [...shape, ...batchShape]
  return discreteSample(params, [...full, k], 'float64', (values) => {
    const rows = full.reduce((a, b) => a * b, 1)
    const batchRows = batchShape.reduce((a, b) => a * b, 1)
    const one = draw(values)
    const out = new Float64Array(rows * k)
    for (let r = 0; r < rows; r++) {
      const b = r % batchRows
      const counts = one({
        key: foldIn(key, r),
        slice: (t) => {
          const flat = toFlat(broadcastBatch(t, [...batchShape, k]))
          return flat.slice(b * k, (b + 1) * k)
        },
      })
      for (let j = 0; j < k; j++) out[r * k + j] = counts[j] ?? 0
    }
    return fromStorage(out, [...full, k], 'float64')
  })
}

/**
 * The logistic-normal distribution on the simplex (Aitchison's additive log-ratio): X = softmax([Y, 0]) for
 * Y ~ N(loc, Sigma) in K - 1 dimensions, so log(X_i / X_K) = Y_i.
 *
 * @param loc - Mean of the log-ratios `[..., K - 1]`.
 * @param params - The covariance of the log-ratios: `{ scale }` (independent, `[..., K - 1]`) or any
 *   {@link CovarianceParams}.
 * @returns The distribution with event shape `[K]`; `rsample` maps the multivariate normal's reparameterised draw.
 *   The mean, variance and entropy (which have no closed form) use a product Gauss-Hermite rule over Y, of at most
 *   about 20000 points (exact to rounding for K up to 4, approximate above).
 * @example
 * ```ts
 * import { logisticNormal } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(logisticNormal([0], { scale: [1] }).mean()).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0.5]
 * ```
 */
export function logisticNormal(
  loc: TensorLike,
  params: { readonly scale: TensorLike } | CovarianceParams,
): Distribution {
  const [mu] = promoteFloat(loc) as [Value]
  const d = mu.shape[mu.shape.length - 1] ?? 0
  const base =
    'scale' in params
      ? multivariateNormal(mu, { scaleTril: diagEmbed(promoteFloat(params.scale)[0] as Value, d) })
      : multivariateNormal(mu, params)
  const k = d + 1
  const batchShape = base.batchShape
  const toSimplex = (y: Value): Value => softmax(concat([y, zeros([...y.shape.slice(0, -1), 1])], -1), -1)
  const logRatio = (x: Value): Value => {
    const parts = log(x)
    return sub(sliceAxis(parts, -1, 0, d), sliceAxis(parts, -1, d, k))
  }
  // E[g(X)] over Y ~ base by a product Gauss-Hermite rule.
  const expect = (g: (x: Value) => Value): Value => {
    const n = Math.max(3, Math.min(40, Math.floor(20000 ** (1 / Math.max(1, d)))))
    const rule = gaussHermite(n)
    const nodes = toFlat(rule.nodes)
    const weights = toFlat(rule.weights)
    const total = n ** d
    const z = new Float64Array(total * d)
    const w = new Float64Array(total)
    for (let i = 0; i < total; i++) {
      let rest = i
      let weight = 1
      for (let j = 0; j < d; j++) {
        const idx = rest % n
        rest = Math.floor(rest / n)
        z[i * d + j] = Math.SQRT2 * (nodes[idx] ?? 0)
        weight *= (weights[idx] ?? 0) / Math.sqrt(Math.PI)
      }
      w[i] = weight
    }
    const zt = reshape(fromStorage(z, [total, d], 'float64'), [
      total,
      ...new Array<number>(batchShape.length).fill(1),
      d,
    ])
    const y = add(base.loc, squeeze(matmul(base.scaleTril, expandDims(zt, -1)), -1))
    const gx = g(toSimplex(y))
    const wt = reshape(fromStorage(w, [total], 'float64'), [total, ...new Array<number>(gx.shape.length - 1).fill(1)])
    return sum(mul(wt, gx), 0)
  }
  const mean = (): Value => expect((x) => x)
  return defineDistribution({
    name: 'logisticNormal',
    params: { loc: base.loc, scaleTril: base.scaleTril },
    support: { kind: 'simplex' },
    batchShape,
    eventShape: [k],
    logProb: (x) =>
      where(
        inSupport({ kind: 'simplex' }, x),
        sub(base.logProb(logRatio(x)), sum(log(x), -1)),
        Number.NEGATIVE_INFINITY,
      ),
    rsample: (key, shape) => toSimplex(base.rsample(key, shape)),
    mean,
    variance: () => {
      const m = mean()
      return sub(
        expect((x) => square(x)),
        square(m),
      )
    },
    mode: () => {
      throw new WhiteboxError('not-implemented', 'logisticNormal: the mode has no closed form')
    },
    // H(X) = H(Y) + E[sum_i log X_i], the log-Jacobian of the additive log-ratio map.
    entropy: () =>
      add(
        base.entropy(),
        expect((x) => sum(log(x), -1)),
      ),
  })
}

/** A diagonal matrix `[..., d, d]` from its diagonal `[..., d]`. */
const diagEmbed = (scale: Value, d: number): Value => mul(expandDims(scale, -1), eye(d))
