// Broadcasting a distribution to a larger batch shape (torch's and NumPyro's `Distribution.expand`), which plates use to
// give every element of a plate its own independent draw. Singleton batch axes that grow get independent draws, never
// copies.
import { MdmlError } from '../base/index.ts'
import { type Distribution, defineDistribution } from '../dists/index.ts'
import type { Key } from '../random/index.ts'
import { broadcastShapes, broadcastTo, reshape, shapesEqual, transpose, type Value } from '../tensor/index.ts'

/** A distribution broadcast to a larger batch shape, remembering the distribution it came from. */
export interface ExpandedDistribution extends Distribution {
  readonly expanded: true
  readonly base: Distribution
}

function draw(
  base: Distribution,
  batchShape: readonly number[],
  sampler: (key: Key, shape: readonly number[]) => Value,
  key: Key,
  shape: readonly number[],
): Value {
  const b = base.batchShape
  const n = batchShape.length
  const k = b.length
  const prefix = batchShape.slice(0, n - k)
  const inflated: number[] = []
  for (let i = 0; i < k; i++) if (b[i] === 1 && batchShape[n - k + i] !== 1) inflated.push(i)
  const extra = inflated.map((i) => batchShape[n - k + i] ?? 1)
  const x = sampler(key, [...shape, ...prefix, ...extra])
  if (inflated.length === 0) return x
  // x is [...shape, ...prefix, ...extra, ...b, ...event]: put each extra axis where its singleton batch axis was.
  const s = shape.length + prefix.length
  const m = inflated.length
  const rank = x.shape.length
  const perm: number[] = Array.from({ length: s }, (_, i) => i)
  for (const i of inflated) perm.push(s + m + i)
  for (let i = 0; i < k; i++) {
    const j = inflated.indexOf(i)
    if (j < 0) perm.push(s + m + i)
    else perm.push(s + j)
  }
  for (let a = s + m + k; a < rank; a++) perm.push(a)
  return reshape(transpose(x, perm), [...shape, ...batchShape, ...base.eventShape])
}

/**
 * Broadcast a distribution to `batchShape` (which its own batch shape must broadcast to).
 *
 * @param base - The distribution.
 * @param batchShape - The new batch shape.
 * @returns The expanded distribution (`base` itself when the shapes already agree).
 * @throws MdmlError with code `shape` when the batch shapes do not broadcast.
 */
export function expandDistribution(base: Distribution, batchShape: readonly number[]): Distribution {
  if (shapesEqual(base.batchShape, batchShape)) return base
  const joined = broadcastShapes(base.batchShape, batchShape)
  if (!shapesEqual(joined, batchShape))
    throw new MdmlError(
      'shape',
      `expand: batch shape [${base.batchShape.join(', ')}] does not broadcast to [${batchShape.join(', ')}]`,
    )
  const event = base.eventShape
  const batched = (v: Value): Value => broadcastTo(v, broadcastShapes(v.shape, batchShape))
  const moment = (v: Value): Value => broadcastTo(v, [...batchShape, ...event])
  const d = defineDistribution({
    name: base.name,
    params: base.params,
    support: base.support,
    batchShape: [...batchShape],
    eventShape: event,
    logProb: (x) => batched(base.logProb(x)),
    cdf: (x) => batched(base.cdf(x)),
    logCdf: (x) => batched(base.logCdf(x)),
    sf: (x) => batched(base.sf(x)),
    logSf: (x) => batched(base.logSf(x)),
    ppf: (q) => batched(base.ppf(q)),
    isf: (q) => batched(base.isf(q)),
    ...(base.hasRsample
      ? { rsample: (key: Key, shape: readonly number[]) => draw(base, batchShape, base.rsample, key, shape) }
      : {}),
    sample: (key, shape) => draw(base, batchShape, base.sample, key, shape),
    mean: () => moment(base.mean()),
    variance: () => moment(base.variance()),
    stddev: () => moment(base.stddev()),
    mode: () => moment(base.mode()),
    entropy: () => broadcastTo(base.entropy(), [...batchShape]),
  })
  const out: ExpandedDistribution = { ...d, expanded: true, base }
  return out
}

/**
 * The distribution under any expansions (plates broadcast the distributions of their sites).
 *
 * @param d - A distribution, possibly expanded.
 * @returns The innermost base distribution.
 */
export function unexpanded(d: Distribution): Distribution {
  let cur = d
  while (isExpanded(cur)) cur = cur.base
  return cur
}

/**
 * Whether a distribution was made by {@link expandDistribution}.
 *
 * @param d - A distribution.
 * @returns True for an expanded distribution.
 */
export function isExpanded(d: Distribution): d is ExpandedDistribution {
  return (d as Partial<ExpandedDistribution>).expanded === true
}
