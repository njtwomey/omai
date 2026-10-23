import { WhiteboxError } from '../../base/index.ts'
import { digamma, lgamma, xlogy } from '../../numerics/special/index.ts'
import { foldIn, type Key, multinomial as multinomialDraw } from '../../random/index.ts'
import {
  add,
  all,
  div,
  equal,
  floor,
  fromStorage,
  greaterEqual,
  isTensor,
  logicalAnd,
  mul,
  promoteFloat,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discreteSample, gammaSample } from '../_sampling.ts'
import { type Distribution, defineDistribution } from '../distribution.ts'

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
    logProb: (x) => sub(sum(xlogy(sub(alpha, 1), x), -1), logNorm),
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

/** Every vector of k non-negative integers summing to n (for exact multinomial entropies). */
function compositions(n: number, k: number): number[][] {
  if (k === 1) return [[n]]
  const out: number[][] = []
  for (let first = 0; first <= n; first++) for (const rest of compositions(n - first, k - 1)) out.push([first, ...rest])
  return out
}

/**
 * The multinomial distribution of counts in K categories from n draws, with probabilities along the last axis
 * (scipy.stats.multinomial).
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
  const p = div(raw, sum(raw, -1, true))
  const k = p.shape[p.shape.length - 1] ?? 0
  const batchShape = p.shape.slice(0, -1)
  const logPmf = (x: Value): Value => {
    const valid = logicalAnd(all(logicalAnd(greaterEqual(x, 0), equal(floor(x), x)), -1), equal(sum(x, -1), n))
    const lp = add(sub(lgamma(n + 1), sum(lgamma(add(x, 1)), -1)), sum(xlogy(x, p), -1))
    return where(valid, lp, Number.NEGATIVE_INFINITY)
  }
  return defineDistribution({
    name: 'multinomial',
    params: { probs: p },
    support: { kind: 'simplex' },
    batchShape,
    eventShape: [k],
    logProb: logPmf,
    sample: (key: Key, shape) => {
      const full = [...shape, ...batchShape]
      return discreteSample([p], [...full, k], 'float64', ([probsT]) => {
        const pt = probsT as Tensor
        const rows = full.reduce((a, b) => a * b, 1)
        const batchRows = batchShape.reduce((a, b) => a * b, 1)
        const out = new Float64Array(rows * k)
        for (let r = 0; r < rows; r++) {
          const b = r % batchRows
          const row = Array.from(pt.data.subarray(b * k, (b + 1) * k))
          const counts = multinomialDraw(foldIn(key, r), n, row, 1).data
          for (let j = 0; j < k; j++) out[r * k + j] = counts[j] ?? 0
        }
        return fromStorage(out, [...full, k], 'float64')
      })
    },
    mean: () => mul(p, n),
    variance: () => mul(mul(p, sub(1, p)), n),
    mode: () => floor(mul(p, n + 1)),
    entropy: () => {
      if (!isTensor(p))
        throw new WhiteboxError('not-differentiable', 'multinomial: entropy needs concrete probabilities')
      const outcomes = promoteFloat(compositions(n, k))[0] as Value
      const rows = batchShape.reduce((a, b) => a * b, 1)
      const values = toFlat(p)
      const result = new Float64Array(rows)
      for (let r = 0; r < rows; r++) {
        const lp = toFlat(logPmfRow(outcomes, values.slice(r * k, (r + 1) * k), n))
        result[r] = -lp.reduce((s, v) => s + (v === Number.NEGATIVE_INFINITY ? 0 : Math.exp(v) * v), 0)
      }
      return fromStorage(result, batchShape, 'float64')
    },
  })
}

function logPmfRow(outcomes: Value, row: readonly number[], n: number): Value {
  const p = promoteFloat(row)[0] as Value
  return add(sub(lgamma(n + 1), sum(lgamma(add(outcomes, 1)), -1)), sum(xlogy(outcomes, p), -1))
}
