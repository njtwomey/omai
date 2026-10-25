/**
 * Modality helpers: feature masks that group the raw features of text, images and series into the units an
 * explanation speaks of (tokens, superpixels, time windows), and predictors over those units, so every attribution in
 * this domain (ablation, Shapley values, LIME) applies to them unchanged.
 */
import { MdmlError } from '../base/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { PredictFunction } from './shapley.ts'

/** A text split into tokens, with where each starts and ends in the text. */
export interface Tokens {
  readonly tokens: readonly string[]
  readonly starts: readonly number[]
  readonly ends: readonly number[]
}

/**
 * Split a text into word tokens (runs of letters, digits and apostrophes), keeping their positions, as lime's text
 * explainer splits on non-word characters.
 *
 * @param text - The text.
 * @returns The tokens and their spans.
 * @example
 * ```ts
 * import { tokenise } from 'mdml-engine/explain'
 * tokenise('Not bad, really.').tokens // => ['Not', 'bad', 'really']
 * ```
 */
export function tokenise(text: string): Tokens {
  const tokens: string[] = []
  const starts: number[] = []
  const ends: number[] = []
  for (const m of text.matchAll(/[\p{L}\p{N}']+/gu)) {
    tokens.push(m[0])
    starts.push(m.index ?? 0)
    ends.push((m.index ?? 0) + m[0].length)
  }
  return { tokens, starts, ends }
}

/**
 * A predictor over token presence: rows `[m, T]` of 0/1 (or fractions, read as present above one half) choose which
 * tokens of a text survive; the text is rebuilt without the absent ones (or with a mask token in their place) and
 * classified. Ablation, Shapley values and LIME then explain the text token by token, with the all-ones row as the
 * text itself and the all-zeros row as the empty text.
 *
 * @param tokens - The tokens of the explained text.
 * @param classify - The model on texts: one score per text.
 * @param options - mask: the token absent tokens become (default removed).
 * @returns The predictor.
 * @example
 * ```ts
 * import { tokenPredictor } from 'mdml-engine/explain'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * const f = tokenPredictor(['good', 'not', 'bad'], (texts) => texts.map((t) => t.split(' ').filter((w) => w === 'good').length))
 * toFlat(f(tensor([[1, 1, 1], [0, 1, 1]]))) // => [1, 0]
 * ```
 */
export function tokenPredictor(
  tokens: readonly string[],
  classify: (texts: readonly string[]) => readonly number[],
  options: { readonly mask?: string } = {},
): PredictFunction {
  return (input: Tensor) => {
    const z = asTensor(input)
    const m = z.shape[0] ?? 0
    const T = z.shape[1] ?? tokens.length
    if (T !== tokens.length) throw new MdmlError('shape', `tokenPredictor: ${T} columns for ${tokens.length} tokens`)
    const v = toFlat(z)
    const texts = Array.from({ length: m }, (_, i) =>
      tokens
        .flatMap((t, j) => ((v[i * T + j] ?? 0) > 0.5 ? [t] : options.mask === undefined ? [] : [options.mask]))
        .join(' '),
    )
    const out = classify(texts)
    if (out.length !== m) throw new MdmlError('shape', 'tokenPredictor: the classifier must score every text')
    return fromStorage(Float64Array.from(out), [m], 'float64')
  }
}

/**
 * A grid of square superpixels over an image: each pixel's segment id, row-major (the regular segmentation captum's
 * examples use; cells at the edges are smaller when the size does not divide).
 *
 * @param shape - The image `[h, w]`.
 * @param cell - The side of a cell in pixels.
 * @returns A segment id per pixel `[h * w]`.
 * @example
 * ```ts
 * import { gridSuperpixels } from 'mdml-engine/explain'
 * gridSuperpixels([2, 4], 2) // => [0, 0, 1, 1, 0, 0, 1, 1]
 * ```
 */
export function gridSuperpixels(shape: readonly [number, number], cell: number): number[] {
  const [h, w] = shape
  if (!(Number.isInteger(cell) && cell >= 1)) throw new MdmlError('value', 'gridSuperpixels: cell must be >= 1')
  const across = Math.ceil(w / cell)
  return Array.from(
    { length: h * w },
    (_, p) => Math.floor(Math.floor(p / w) / cell) * across + Math.floor((p % w) / cell),
  )
}

/** Options of {@link slicSuperpixels}. */
export interface SlicOptions {
  /** Segments wanted (default 16; the result may have fewer). */
  readonly nSegments?: number
  /** The weight of position against intensity (default 10, as skimage's compactness for [0, 1] images ~ 0.1 x 100). */
  readonly compactness?: number
  /** k-means iterations (default 10). */
  readonly iterations?: number
}

/**
 * SLIC superpixels (Achanta et al. 2012, on a grey image): k-means in (row, column, intensity) started from a regular
 * grid, each pixel compared only with centres within two grid steps, distance sqrt(dI^2 + (ds / S)^2 m^2) for grid
 * step S and compactness m; segment ids are renumbered from 0 in raster order of first appearance.
 *
 * @param image - The image `[h, w]` as nested rows, or flat with `shape`.
 * @param shape - The image's shape `[h, w]`.
 * @param options - nSegments, compactness and iterations.
 * @returns A segment id per pixel `[h * w]`.
 * @example
 * ```ts
 * import { slicSuperpixels } from 'mdml-engine/explain'
 * // A bright left half and a dark right half split at the edge.
 * const image = [0.9, 0.9, 0, 0, 0.9, 0.9, 0, 0]
 * slicSuperpixels(image, [2, 4], { nSegments: 2 }) // => [0, 0, 1, 1, 0, 0, 1, 1]
 * ```
 */
export function slicSuperpixels(
  image: TensorLike,
  shape: readonly [number, number],
  options: SlicOptions = {},
): number[] {
  const [h, w] = shape
  const px = toFlat(asTensor(image))
  if (px.length !== h * w) throw new MdmlError('shape', `slicSuperpixels: ${px.length} pixels for ${h}x${w}`)
  const k = Math.max(1, options.nSegments ?? 16)
  const S = Math.max(1, Math.sqrt((h * w) / k))
  const mWeight = options.compactness ?? 10
  const rows = Math.max(1, Math.round(h / S))
  const cols = Math.max(1, Math.round(w / S))
  type Centre = { y: number; x: number; v: number }
  let centres: Centre[] = []
  for (let a = 0; a < rows; a++)
    for (let b = 0; b < cols; b++) {
      const y = Math.min(h - 1, Math.floor(((a + 0.5) * h) / rows))
      const x = Math.min(w - 1, Math.floor(((b + 0.5) * w) / cols))
      centres.push({ y, x, v: px[y * w + x] ?? 0 })
    }
  const label = new Int32Array(h * w).fill(-1)
  const step = Math.max(h / rows, w / cols)
  for (let it = 0; it < (options.iterations ?? 10); it++) {
    const best = new Float64Array(h * w).fill(Number.POSITIVE_INFINITY)
    centres.forEach((c, ci) => {
      const y0 = Math.max(0, Math.floor(c.y - 2 * step))
      const y1 = Math.min(h - 1, Math.ceil(c.y + 2 * step))
      const x0 = Math.max(0, Math.floor(c.x - 2 * step))
      const x1 = Math.min(w - 1, Math.ceil(c.x + 2 * step))
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) {
          const p = y * w + x
          const dv = ((px[p] ?? 0) - c.v) * 100
          const ds = ((y - c.y) ** 2 + (x - c.x) ** 2) / (step * step)
          const dist = dv * dv + ds * mWeight * mWeight
          if (dist < (best[p] ?? 0)) {
            best[p] = dist
            label[p] = ci
          }
        }
    })
    const acc = centres.map(() => ({ y: 0, x: 0, v: 0, n: 0 }))
    for (let p = 0; p < h * w; p++) {
      const a = acc[label[p] ?? 0]
      if (a === undefined) continue
      a.y += Math.floor(p / w)
      a.x += p % w
      a.v += px[p] ?? 0
      a.n += 1
    }
    centres = centres.map((c, ci) => {
      const a = acc[ci]
      return a === undefined || a.n === 0 ? c : { y: a.y / a.n, x: a.x / a.n, v: a.v / a.n }
    })
  }
  const renumber = new Map<number, number>()
  return Array.from(label, (l) => {
    const known = renumber.get(l)
    if (known !== undefined) return known
    renumber.set(l, renumber.size)
    return renumber.size - 1
  })
}

/**
 * Windows over a series: each time step's window id, consecutive windows of `window` steps (the last may be shorter);
 * with `channels`, the same window over every channel of a row-major `[t, channels]` series.
 *
 * @param length - The series length.
 * @param window - Steps per window.
 * @param channels - Channels per step (default 1).
 * @returns A window id per value `[length * channels]`.
 * @example
 * ```ts
 * import { timeWindows } from 'mdml-engine/explain'
 * timeWindows(5, 2) // => [0, 0, 1, 1, 2]
 * timeWindows(2, 1, 2) // => [0, 0, 1, 1]
 * ```
 */
export function timeWindows(length: number, window: number, channels = 1): number[] {
  if (!(Number.isInteger(window) && window >= 1)) throw new MdmlError('value', 'timeWindows: window must be >= 1')
  return Array.from({ length: length * channels }, (_, i) => Math.floor(Math.floor(i / channels) / window))
}

/**
 * The groups of a mask: the feature indices of each id, ids in the order they first appear.
 *
 * @param mask - A group id per feature.
 * @returns The features of each group.
 * @example
 * ```ts
 * import { maskGroups } from 'mdml-engine/explain'
 * maskGroups([3, 3, 1, 3]) // => [[0, 1, 3], [2]]
 * ```
 */
export function maskGroups(mask: readonly number[]): number[][] {
  const order: number[] = []
  const members = new Map<number, number[]>()
  mask.forEach((g, j) => {
    const list = members.get(g)
    if (list === undefined) {
      members.set(g, [j])
      order.push(g)
    } else list.push(j)
  })
  return order.map((g) => members.get(g) ?? [])
}

/**
 * A predictor over groups of one input: rows `[m, G]` of group presence (1 keeps a group's features at the input's
 * values, 0 sets them to the baseline; fractions blend) mapped to full rows and predicted. With it, Kernel SHAP, exact
 * Shapley values or LIME over the G groups explain an image by superpixels or a series by windows; the all-ones row is
 * the input and the all-zeros row its baseline.
 *
 * @param predict - The model on full rows `[m, d]`.
 * @param x - The input `[d]`.
 * @param mask - A group id per feature (`gridSuperpixels`, `slicSuperpixels`, `timeWindows`).
 * @param baseline - What absent features become (default 0): a number or a row `[d]`.
 * @returns The predictor over groups and the groups (feature indices of each, in id order of first appearance).
 * @example
 * ```ts
 * import { maskedPredictor } from 'mdml-engine/explain'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * const total = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[3 * i] + x.data[3 * i + 1] + x.data[3 * i + 2]))
 * const { predict } = maskedPredictor(total, [1, 2, 3], [0, 0, 1])
 * toFlat(predict(tensor([[1, 0], [0, 1]]))) // => [3, 3]
 * ```
 */
export function maskedPredictor(
  predict: PredictFunction,
  x: TensorLike,
  mask: readonly number[],
  baseline: TensorLike | number = 0,
): { readonly predict: PredictFunction; readonly groups: readonly (readonly number[])[] } {
  const xv = toFlat(asTensor(x))
  const d = xv.length
  if (mask.length !== d) throw new MdmlError('shape', 'maskedPredictor: one group id per feature')
  const b = typeof baseline === 'number' ? new Array<number>(d).fill(baseline) : toFlat(asTensor(baseline))
  if (b.length !== d) throw new MdmlError('shape', 'maskedPredictor: the baseline must be a number or [d]')
  const groups = maskGroups(mask)
  const G = groups.length
  return {
    groups,
    predict: (input: Tensor) => {
      const z = asTensor(input)
      const m = z.shape[0] ?? 0
      if ((z.shape[1] ?? G) !== G) throw new MdmlError('shape', `maskedPredictor: rows must have ${G} groups`)
      const zv = toFlat(z)
      const data = new Float64Array(m * d)
      for (let i = 0; i < m; i++)
        groups.forEach((g, a) => {
          const keep = zv[i * G + a] ?? 0
          for (const j of g) data[i * d + j] = keep * (xv[j] ?? 0) + (1 - keep) * (b[j] ?? 0)
        })
      return predict(fromStorage(data, [m, d], 'float64'))
    },
  }
}

/**
 * Spread group attributions back over the features: each feature of a group takes the group's value (captum's
 * convention), or its share of it (`share`), so a map over superpixels draws as an image.
 *
 * @param values - One value per group.
 * @param groups - The features of each group.
 * @param d - The number of features.
 * @param options - share: divide a group's value among its features (default false).
 * @returns One value per feature.
 * @example
 * ```ts
 * import { expandGroups } from 'mdml-engine/explain'
 * expandGroups([2, 5], [[0, 1], [2]], 3) // => [2, 2, 5]
 * expandGroups([2, 5], [[0, 1], [2]], 3, { share: true }) // => [1, 1, 5]
 * ```
 */
export function expandGroups(
  values: readonly number[],
  groups: readonly (readonly number[])[],
  d: number,
  options: { readonly share?: boolean } = {},
): number[] {
  const out = new Array<number>(d).fill(0)
  groups.forEach((g, a) => {
    for (const j of g) out[j] = (values[a] ?? 0) / (options.share === true ? g.length : 1)
  })
  return out
}
