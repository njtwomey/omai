/**
 * Where to sample along rays, the first half of volume rendering: stratified samples (one uniform draw per equal bin
 * between the near and far planes, NeRF's coarse pass) and hierarchical samples (inverse-transform sampling of the
 * piecewise-constant density that a coarse pass's weights define, NeRF's fine pass). `samplePdf` is the inverse CDF of
 * a piecewise-constant density on arbitrary bins and stands on its own. Samples are concrete numbers: no gradient flows
 * through where the samples fell, as in the NeRF paper.
 */

import { MdmlError } from '../base/index.ts'
import { type Key, uniform } from '../random/index.ts'
import { fromStorage, type Tensor, toFlat, type Value } from '../tensor/index.ts'

const flat = (v: ArrayLike<number> | Value): ArrayLike<number> =>
  Array.isArray(v) || ArrayBuffer.isView(v) ? (v as ArrayLike<number>) : toFlat(v as Value)

/** The near and far distances of rays: one number for all, or one per ray. */
export type Planes = number | ArrayLike<number>

const planeAt = (p: Planes, r: number): number => (typeof p === 'number' ? p : (p[r] as number))

/**
 * Stratified samples along rays: the interval from `near` to `far` is cut into `samples` equal bins and one distance is
 * drawn uniformly in each (with `perturb` false, at its midpoint), so samples cover the ray evenly while the network is
 * trained on continuously varying positions. Each row is increasing and within the planes.
 *
 * @param key - The key (unused without `perturb`).
 * @param rays - The number of rays.
 * @param samples - The samples per ray.
 * @param near - The near distance (a number, or one per ray).
 * @param far - The far distance.
 * @param perturb - Draw within the bins (default true) or take their midpoints.
 * @returns The distances `[rays, samples]`.
 * @throws MdmlError with code `value` for fewer than one sample or planes that are not increasing.
 * @concept volume-rendering
 * @example
 * ```ts
 * import { stratifiedSamples } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(stratifiedSamples(key(0), 1, 4, 2, 6, false)) // => [[2.5, 3.5, 4.5, 5.5]]
 * ```
 */
export function stratifiedSamples(
  key: Key,
  rays: number,
  samples: number,
  near: Planes,
  far: Planes,
  perturb = true,
): Tensor {
  if (!(Number.isInteger(samples) && samples >= 1))
    throw new MdmlError('value', 'stratifiedSamples: need at least one sample')
  const u = perturb ? uniform(key, [rays, samples]).data : undefined
  const out = new Float64Array(rays * samples)
  for (let r = 0; r < rays; r++) {
    const lo = planeAt(near, r)
    const hi = planeAt(far, r)
    if (!(hi > lo)) throw new MdmlError('value', 'stratifiedSamples: far must exceed near')
    for (let i = 0; i < samples; i++)
      out[r * samples + i] = lo + ((hi - lo) * (i + (u === undefined ? 0.5 : (u[r * samples + i] as number)))) / samples
  }
  return fromStorage(out, [rays, samples], 'float64')
}

/**
 * Samples from a piecewise-constant density by inverse transform: on each row the bins `bins[i] .. bins[i + 1]` carry
 * the probability proportional to `weights[i]`, the CDF is piecewise linear, and the sample for a uniform `u` is where
 * the CDF reaches `u`, found by linear interpolation inside its bin. With `key` the `u` are random; without, they are the
 * evenly spaced quantiles `(i + 0.5) / n`, so the result is deterministic and is the quantile function itself. A row
 * whose weights are all zero is sampled uniformly over its bins.
 *
 * @param bins - Bin edges `[rows, m + 1]` (increasing along each row).
 * @param weights - Bin weights `[rows, m]` (non-negative; need not sum to one).
 * @param samples - The samples per row.
 * @param key - A key for random quantiles; omit for deterministic ones.
 * @returns The samples `[rows, samples]`, increasing along each row.
 * @throws MdmlError with code `shape` when the weights are not one fewer than the edges, `value` for a negative weight.
 * @concept volume-rendering
 * @example
 * ```ts
 * import { samplePdf } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * // Half the mass in [0, 1) and half in [1, 4): the median is at the shared edge, the quartiles at 0.5 and 2.5.
 * toArray(samplePdf([[0, 1, 4]], [[1, 1]], 3)) // => [[0.3333333333333333, 1, 3]]
 * ```
 */
export function samplePdf(
  bins: Value | ArrayLike<ArrayLike<number>>,
  weights: Value | ArrayLike<ArrayLike<number>>,
  samples: number,
  key?: Key,
): Tensor {
  const b = asRows(bins)
  const w = asRows(weights)
  if (b.rows !== w.rows || b.cols !== w.cols + 1)
    throw new MdmlError('shape', `samplePdf: ${b.cols} edges need ${b.cols - 1} weights, got [${w.rows}, ${w.cols}]`)
  const m = w.cols
  const u = key === undefined ? undefined : uniform(key, [b.rows, samples]).data
  const out = new Float64Array(b.rows * samples)
  const cdf = new Float64Array(m + 1)
  for (let r = 0; r < b.rows; r++) {
    let total = 0
    for (let i = 0; i < m; i++) {
      const wi = w.data[r * m + i] as number
      if (wi < 0) throw new MdmlError('value', 'samplePdf: weights must be non-negative')
      total += wi
    }
    cdf[0] = 0
    for (let i = 0; i < m; i++)
      cdf[i + 1] = (cdf[i] as number) + (total > 0 ? (w.data[r * m + i] as number) / total : 1 / m)
    for (let s = 0; s < samples; s++) {
      const q = u === undefined ? (s + 0.5) / samples : (u[r * samples + s] as number)
      // The bin whose CDF interval contains q (the last bin at q = 1).
      let lo = 0
      let hi = m
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1
        if ((cdf[mid] as number) <= q) lo = mid
        else hi = mid
      }
      const c0 = cdf[lo] as number
      const width = (cdf[lo + 1] as number) - c0
      const frac = width > 0 ? (q - c0) / width : 0
      const left = b.data[r * b.cols + lo] as number
      const right = b.data[r * b.cols + lo + 1] as number
      out[r * samples + s] = left + frac * (right - left)
    }
    if (u !== undefined) out.subarray(r * samples, (r + 1) * samples).sort()
  }
  return fromStorage(out, [b.rows, samples], 'float64')
}

function asRows(x: Value | ArrayLike<ArrayLike<number>>): { data: ArrayLike<number>; rows: number; cols: number } {
  if (Array.isArray(x)) {
    const rows = x.length
    const cols = rows === 0 ? 0 : (x[0] as ArrayLike<number>).length
    return { data: Float64Array.from(x.flatMap((row) => Array.from(row as ArrayLike<number>))), rows, cols }
  }
  const v = x as Value
  if (v.shape.length !== 2)
    throw new MdmlError('shape', `samplePdf: expected a matrix, got shape [${v.shape.join(', ')}]`)
  return { data: flat(v), rows: v.shape[0] as number, cols: v.shape[1] as number }
}

/**
 * NeRF's hierarchical samples: the weights a coarse pass gave its samples define a piecewise-constant density along the
 * ray (bins between the coarse samples' midpoints, the weights of the interior samples), and `fine` more distances are
 * drawn from it by inverse transform; the result merges them with the coarse distances in increasing order, so the
 * fine network sees the coarse samples too. No gradient flows through the new positions.
 *
 * @param coarse - The coarse distances `[rays, n]` (increasing).
 * @param weights - The coarse weights `[rays, n]` from `volumeRender`.
 * @param fine - How many fine samples per ray.
 * @param key - A key for random quantiles; omit for evenly spaced ones.
 * @returns The merged distances `[rays, n + fine]`, increasing along each ray.
 * @throws MdmlError with code `shape` when the weights do not match the distances, or there are fewer than three coarse samples.
 * @concept volume-rendering
 * @reference Mildenhall et al. (2020), NeRF, section 5.2 (hierarchical volume sampling).
 * @example
 * ```ts
 * import { hierarchicalSamples } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * // All the weight on the third of five coarse samples: every fine sample lands near it.
 * const t = [[1, 2, 3, 4, 5]]
 * (toArray(hierarchicalSamples(t, [[0, 0, 1, 0, 0]], 2))[0] as number[]).map((v) => Math.round(v * 100) / 100) // => [1, 2, 2.75, 3, 3.25, 4, 5]
 * ```
 */
export function hierarchicalSamples(
  coarse: Value | ArrayLike<ArrayLike<number>>,
  weights: Value | ArrayLike<ArrayLike<number>>,
  fine: number,
  key?: Key,
): Tensor {
  const t = asRows(coarse)
  const w = asRows(weights)
  if (t.rows !== w.rows || t.cols !== w.cols || t.cols < 3)
    throw new MdmlError(
      'shape',
      'hierarchicalSamples: need distances and weights of the same shape, at least three samples',
    )
  const n = t.cols
  const mids = new Float64Array(t.rows * (n - 1))
  const inner = new Float64Array(t.rows * (n - 2))
  for (let r = 0; r < t.rows; r++) {
    for (let i = 0; i < n - 1; i++)
      mids[r * (n - 1) + i] = 0.5 * ((t.data[r * n + i] as number) + (t.data[r * n + i + 1] as number))
    for (let i = 0; i < n - 2; i++) inner[r * (n - 2) + i] = (w.data[r * n + i + 1] as number) + 1e-5
  }
  const extra = samplePdf(
    fromStorage(mids, [t.rows, n - 1], 'float64'),
    fromStorage(inner, [t.rows, n - 2], 'float64'),
    fine,
    key,
  ).data
  const out = new Float64Array(t.rows * (n + fine))
  for (let r = 0; r < t.rows; r++) {
    const row = new Float64Array(n + fine)
    for (let i = 0; i < n; i++) row[i] = t.data[r * n + i] as number
    for (let i = 0; i < fine; i++) row[n + i] = extra[r * fine + i] as number
    row.sort()
    out.set(row, r * (n + fine))
  }
  return fromStorage(out, [t.rows, n + fine], 'float64')
}
