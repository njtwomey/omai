/**
 * Perturbation attributions (captum's perturbation family and global screening): feature ablation over groups of
 * features with any baseline, occlusion by a sliding window over an image (or a series), RISE's randomised masks, and
 * Morris elementary effects. Each needs only a prediction function; masks from `modalities.ts` carry them to text,
 * images and series.
 */
import { OmaiError } from '../base/index.ts'
import { keyFor } from '../learn/_shared.ts'
import { matrixView } from '../learn/validation.ts'
import { foldIn, type Key, permutation, randint, uniform } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { baselineRows, targetOutputs } from './_outputs.ts'
import type { Target } from './gradients.ts'
import type { PredictFunction } from './shapley.ts'

/** Group ids per feature: validated, and the groups in first-seen order. */
function groupsOf(mask: readonly number[] | undefined, d: number, who: string): number[][] {
  const ids = mask ?? Array.from({ length: d }, (_, j) => j)
  if (ids.length !== d) throw new OmaiError('shape', `${who}: the feature mask needs one group per feature`)
  const order: number[] = []
  const members = new Map<number, number[]>()
  ids.forEach((g, j) => {
    if (!Number.isInteger(g)) throw new OmaiError('value', `${who}: group ids must be integers`)
    const list = members.get(g)
    if (list === undefined) {
      members.set(g, [j])
      order.push(g)
    } else list.push(j)
  })
  return order.map((g) => members.get(g) ?? [])
}

/** Options of {@link featureAblation}. */
export interface FeatureAblationOptions {
  readonly target?: Target
  /** What an ablated feature becomes (default 0): a number, a row `[d]`, or a row per input `[n, d]`. */
  readonly baseline?: TensorLike | number
  /** A group id per feature (default each feature alone); a group is ablated together. */
  readonly featureMask?: readonly number[]
}

/**
 * Feature ablation (captum FeatureAblation): replace each feature (or group of features sharing a mask id) by its
 * baseline and record how far the target output falls, f(x) - f(x with the group ablated); every feature of a group
 * gets the group's value. Groups make it apply to superpixels, words and time windows.
 *
 * @param predict - The model.
 * @param x - Inputs `[n, d]`.
 * @param options - target, baseline and featureMask.
 * @returns Attributions `[n][d]`.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[2 * i] - x.data[2 * i + 1]))
 * explain.featureAblation(linear, [[1, 3]]) // => [[2, -3]]
 * explain.featureAblation(linear, [[1, 3]], { featureMask: [0, 0] }) // => [[-1, -1]]
 * ```
 */
export function featureAblation(
  predict: PredictFunction,
  x: TensorLike,
  options: FeatureAblationOptions = {},
): number[][] {
  const m = matrixView(x, 'featureAblation')
  const base = baselineRows(options.baseline, m.n, m.d, 'featureAblation')
  const groups = groupsOf(options.featureMask, m.d, 'featureAblation')
  const fx = targetOutputs(predict, Float64Array.from(m.data), m.n, m.d, options.target)
  const out = Array.from({ length: m.n }, () => new Array<number>(m.d).fill(0))
  for (const g of groups) {
    const data = Float64Array.from(m.data)
    for (let i = 0; i < m.n; i++) for (const j of g) data[i * m.d + j] = base[i * m.d + j] ?? 0
    const f = targetOutputs(predict, data, m.n, m.d, options.target)
    for (let i = 0; i < m.n; i++) for (const j of g) (out[i] as number[])[j] = (fx[i] ?? 0) - (f[i] ?? 0)
  }
  return out
}

/** Options of {@link occlusion}. */
export interface OcclusionOptions {
  readonly target?: Target
  /** The input's shape as an image `[h, w]` (or a series `[t]`); the rows hold it flattened, row-major. */
  readonly shape: readonly number[]
  /** The occluding window, one size per axis. */
  readonly window: readonly number[]
  /** The window's step per axis (default 1). */
  readonly strides?: readonly number[]
  /** What occluded values become (default 0). */
  readonly baseline?: TensorLike | number
}

/**
 * Occlusion (Zeiler and Fergus 2014; captum Occlusion): slide a window over the input (an image `[h, w]` or a series
 * `[t]`), replace what it covers by the baseline, and credit every covered element with the fall in the target
 * output; each element's attribution is the mean over the windows that covered it. As captum, a last window that
 * would overrun an edge is clipped to it.
 *
 * @param predict - The model (on flattened inputs).
 * @param x - Inputs `[n, prod(shape)]`.
 * @param options - shape, window, strides, baseline and target.
 * @returns Attributions `[n][prod(shape)]`.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const total = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[3 * i] + x.data[3 * i + 1] + x.data[3 * i + 2]))
 * // Windows of two over [1, 2, 3]: [1, 2] removes 3, [2, 3] removes 5.
 * explain.occlusion(total, [[1, 2, 3]], { shape: [3], window: [2] }) // => [[3, 4, 5]]
 * ```
 */
export function occlusion(predict: PredictFunction, x: TensorLike, options: OcclusionOptions): number[][] {
  const m = matrixView(x, 'occlusion')
  const shape = options.shape
  const size = shape.reduce((a, b) => a * b, 1)
  if (size !== m.d) throw new OmaiError('shape', `occlusion: shape ${shape.join('x')} does not hold ${m.d} values`)
  if (options.window.length !== shape.length) throw new OmaiError('shape', 'occlusion: one window size per axis')
  const strides = options.strides ?? shape.map(() => 1)
  const starts = shape.map((s, a) => {
    const w = options.window[a] ?? 1
    const st = strides[a] ?? 1
    if (!(Number.isInteger(w) && w >= 1 && w <= s)) throw new OmaiError('value', 'occlusion: bad window size')
    // captum's count, ceil((s - w) / stride) + 1: a last window that would overrun the edge is clipped to it.
    const count = Math.ceil((s - w) / st) + 1
    return Array.from({ length: count }, (_, i) => i * st)
  })
  const base = baselineRows(options.baseline, m.n, m.d, 'occlusion')
  const fx = targetOutputs(predict, Float64Array.from(m.data), m.n, m.d, options.target)
  const sum = new Float64Array(m.n * m.d)
  const count = new Float64Array(m.d)
  const positions = starts.reduce<number[][]>((acc, s) => acc.flatMap((p) => s.map((v) => [...p, v])), [[]])
  for (const pos of positions) {
    // The flat indices the window covers.
    let cells: number[][] = [[]]
    pos.forEach((p, a) => {
      const span = Math.min(options.window[a] ?? 1, (shape[a] ?? 1) - p)
      cells = cells.flatMap((c) => Array.from({ length: span }, (_, o) => [...c, p + o]))
    })
    const idx = cells.map((c) => c.reduce((acc, v, a) => acc * (shape[a] ?? 1) + v, 0))
    const data = Float64Array.from(m.data)
    for (let i = 0; i < m.n; i++) for (const j of idx) data[i * m.d + j] = base[i * m.d + j] ?? 0
    const f = targetOutputs(predict, data, m.n, m.d, options.target)
    for (const j of idx) count[j] = (count[j] ?? 0) + 1
    for (let i = 0; i < m.n; i++)
      for (const j of idx) sum[i * m.d + j] = (sum[i * m.d + j] ?? 0) + (fx[i] ?? 0) - (f[i] ?? 0)
  }
  return Array.from({ length: m.n }, (_, i) =>
    Array.from({ length: m.d }, (_, j) => ((count[j] ?? 0) === 0 ? 0 : (sum[i * m.d + j] ?? 0) / (count[j] ?? 1))),
  )
}

/** Options of {@link rise}. */
export interface RiseOptions {
  readonly target?: Target
  /** The image's shape `[h, w]`; rows hold it flattened. */
  readonly shape: readonly [number, number]
  /** Random masks drawn (default 1000). */
  readonly nMasks?: number
  /** Cells of the coarse random grid per side (default 7). */
  readonly cells?: number
  /** The probability a coarse cell is kept (default 0.5). */
  readonly keep?: number
  /** What masked-out pixels become (default 0). */
  readonly baseline?: TensorLike | number
  readonly key?: Key
}

/** One RISE mask `[h * w]`: a coarse random grid, bilinearly upsampled and randomly shifted. */
function riseMask(key: Key, h: number, w: number, s: number, p: number): Float64Array {
  const u = toFlat(uniform(foldIn(key, 0), [s * s]))
  const grid = u.map((v) => (v < p ? 1 : 0))
  const ch = Math.ceil(h / s)
  const cw = Math.ceil(w / s)
  const shift = toFlat(uniform(foldIn(key, 1), [2]))
  const dy = (shift[0] ?? 0) * ch
  const dx = (shift[1] ?? 0) * cw
  const at = (r: number, c: number): number =>
    grid[Math.min(s - 1, Math.max(0, r)) * s + Math.min(s - 1, Math.max(0, c))] ?? 0
  const out = new Float64Array(h * w)
  for (let i = 0; i < h; i++)
    for (let j = 0; j < w; j++) {
      // Position in grid units (cell centres at integers), shifted.
      const gy = (i + dy) / ch - 0.5
      const gx = (j + dx) / cw - 0.5
      const y0 = Math.floor(gy)
      const x0 = Math.floor(gx)
      const fy = gy - y0
      const fx = gx - x0
      out[i * w + j] =
        (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) +
        fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
    }
  return out
}

/**
 * RISE (Petsiuk, Das and Saenko 2018): randomised input sampling for explanation. Draw smooth random masks (a coarse
 * random grid of kept cells, upsampled and shifted), blend each image with its baseline through the mask, and average
 * the masks weighted by the target output: the saliency of a pixel is E[f(x . M) | M(pixel) = 1], estimated as
 * sum f(x . M_i) M_i / (N p). Needs no gradients, only many forward passes.
 *
 * @param predict - The model (on flattened images).
 * @param x - Images `[n, h * w]`.
 * @param options - shape, nMasks, cells, keep, baseline, target and key.
 * @returns Saliency maps `[n][h * w]`.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * // The model reads only the top-left pixel of a 4 x 4 image: it gets the largest saliency.
 * const corner = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[16 * i]))
 * const s = explain.rise(corner, [new Array(16).fill(1)], { shape: [4, 4], nMasks: 200, cells: 2 })[0]
 * s.indexOf(Math.max(...s)) // => 0
 * ```
 */
export function rise(predict: PredictFunction, x: TensorLike, options: RiseOptions): number[][] {
  const m = matrixView(x, 'rise')
  const [h, w] = options.shape
  if (h * w !== m.d) throw new OmaiError('shape', `rise: shape ${h}x${w} does not hold ${m.d} values`)
  const nMasks = options.nMasks ?? 1000
  const s = options.cells ?? 7
  const p = options.keep ?? 0.5
  if (!(p > 0 && p <= 1)) throw new OmaiError('value', 'rise: keep must be in (0, 1]')
  const key = options.key ?? keyFor({}, 'rise')
  const base = baselineRows(options.baseline, m.n, m.d, 'rise')
  const sal = new Float64Array(m.n * m.d)
  const batch = 50
  for (let start = 0; start < nMasks; start += batch) {
    const masks = Array.from({ length: Math.min(batch, nMasks - start) }, (_, b) =>
      riseMask(foldIn(key, start + b), h, w, s, p),
    )
    const rows = masks.length * m.n
    const data = new Float64Array(rows * m.d)
    masks.forEach((mask, b) => {
      for (let i = 0; i < m.n; i++)
        for (let j = 0; j < m.d; j++) {
          const v = mask[j] ?? 0
          data[(b * m.n + i) * m.d + j] = v * (m.data[i * m.d + j] ?? 0) + (1 - v) * (base[i * m.d + j] ?? 0)
        }
    })
    const f = targetOutputs(predict, data, rows, m.d, options.target, (r) => r % m.n)
    masks.forEach((mask, b) => {
      for (let i = 0; i < m.n; i++) {
        const fi = f[b * m.n + i] ?? 0
        for (let j = 0; j < m.d; j++) sal[i * m.d + j] = (sal[i * m.d + j] ?? 0) + fi * (mask[j] ?? 0)
      }
    })
  }
  return Array.from({ length: m.n }, (_, i) =>
    Array.from({ length: m.d }, (_, j) => (sal[i * m.d + j] ?? 0) / (nMasks * p)),
  )
}

/** Options of {@link morrisEffects}. */
export interface MorrisOptions {
  /** Trajectories (default 20), each moving every feature once. */
  readonly nTrajectories?: number
  /** Grid levels per feature (default 4; even, so the step is half the range plus one level). */
  readonly levels?: number
  readonly target?: number
  readonly key?: Key
}

/** Morris screening: per feature the mean, mean absolute and spread of its elementary effects. */
export interface MorrisResult {
  /** The mean elementary effect. */
  readonly mu: readonly number[]
  /** The mean absolute elementary effect (Campolongo's mu*), the importance. */
  readonly muStar: readonly number[]
  /** The standard deviation of the effects: non-linearity or interaction. */
  readonly sigma: readonly number[]
  /** effects[feature][trajectory]. */
  readonly effects: readonly (readonly number[])[]
}

/**
 * Morris elementary effects (Morris 1991; Campolongo, Cariboni and Saltelli 2007; SALib's morris): a global screening
 * of a model over a box. Each trajectory starts at a random point of a p-level grid over the box and moves the
 * features one at a time, in random order, by the step Delta = p / (2 (p - 1)) of the unit range; each move gives an
 * elementary effect (f(x + Delta e_i) - f(x)) / Delta (in unit-range units). Their mean absolute value ranks the
 * features, their spread flags non-linearity and interactions, at (d + 1) evaluations per trajectory.
 *
 * @param predict - The model.
 * @param bounds - The box, `[low, high]` per feature.
 * @param options - nTrajectories, levels, target and key.
 * @returns mu, mu*, sigma and every effect.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * // Linear: every effect is the coefficient times the range.
 * const f = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 3 * x.data[2 * i] - x.data[2 * i + 1]))
 * explain.morrisEffects(f, [[0, 1], [0, 2]], { nTrajectories: 4 }).muStar.map((v) => Math.round(v * 1e9) / 1e9) // => [3, 2]
 * ```
 */
export function morrisEffects(
  predict: PredictFunction,
  bounds: readonly (readonly [number, number])[],
  options: MorrisOptions = {},
): MorrisResult {
  const d = bounds.length
  if (d === 0) throw new OmaiError('value', 'morrisEffects: no features')
  const r = options.nTrajectories ?? 20
  const p = options.levels ?? 4
  if (!(Number.isInteger(p) && p >= 2)) throw new OmaiError('value', 'morrisEffects: levels must be >= 2')
  const delta = p / (2 * (p - 1))
  const key = options.key ?? keyFor({}, 'morrisEffects')
  // Grid levels from which a step of delta stays inside [0, 1].
  const starts = Array.from({ length: p }, (_, l) => l / (p - 1)).filter((v) => v + delta <= 1 + 1e-12)
  const effects = Array.from({ length: d }, () => new Array<number>(r).fill(0))
  const scale = (u: readonly number[]): number[] =>
    u.map((v, j) => (bounds[j]?.[0] ?? 0) + v * ((bounds[j]?.[1] ?? 1) - (bounds[j]?.[0] ?? 0)))
  for (let t = 0; t < r; t++) {
    const k = foldIn(key, t)
    const pick = toFlat(randint(foldIn(k, 0), [d], 0, starts.length))
    const order = toFlat(permutation(foldIn(k, 1), d))
    const u = Array.from({ length: d }, (_, j) => starts[pick[j] ?? 0] ?? 0)
    const pts: number[] = [...scale(u)]
    for (const j of order) {
      u[j] = (u[j] ?? 0) + delta
      pts.push(...scale(u))
    }
    const f = targetOutputs(predict, Float64Array.from(pts), d + 1, d, options.target ?? 0)
    order.forEach((j, s) => {
      ;(effects[j] as number[])[t] = ((f[s + 1] ?? 0) - (f[s] ?? 0)) / delta
    })
  }
  const mu = effects.map((e) => e.reduce((a, b) => a + b, 0) / r)
  const muStar = effects.map((e) => e.reduce((a, b) => a + Math.abs(b), 0) / r)
  const sigma = effects.map((e, j) =>
    r < 2 ? 0 : Math.sqrt(e.reduce((a, b) => a + (b - (mu[j] ?? 0)) ** 2, 0) / (r - 1)),
  )
  return { mu, muStar, sigma, effects }
}
