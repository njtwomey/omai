/**
 * GrabCut-lite (Rother, Kolmogorov and Blake 2004) as a traced algorithm: Gaussian mixture colour models of the
 * foreground and background (fitted by `mixture`'s EM), alternated with an exact binary graph cut (by `graph`'s minimum
 * cut) of the colour likelihoods plus a contrast-sensitive smoothness term; pixels outside the user's rectangle stay
 * background. The mixture fits are seeded by k-means from an explicit key.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { digraph, minCut } from '../graph/index.ts'
import { gaussianEmAlgorithm, initialMixture, type MixtureParameters, mixturePosterior } from '../mixture/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { imageTensor, rasterOf } from './_image.ts'

/** Options of {@link grabCutAlgorithm}. */
export interface GrabCutOptions {
  /** Gaussians per colour model (default 3). */
  readonly components?: number
  /** The weight of the smoothness term (default 50, the paper's). */
  readonly gamma?: number
  /** The key of the mixture initialisations (default key 0). */
  readonly key?: Key
}

/** The state of {@link grabCutAlgorithm}: the foreground mask and the energy of the last cut. */
export interface GrabCutState {
  readonly mask: Uint8Array
  readonly energy: number
  readonly iteration: number
  /** Pixels whose label the last cut changed. */
  readonly changed: number
}

/**
 * GrabCut as an algorithm: starting with the rectangle as possible foreground, each step fits a Gaussian mixture to the
 * colours of each side, then labels the pixels by the minimum cut of E = sum -log p(colour | side) + gamma sum over
 * 8-neighbours of exp(-beta |dc|^2) / distance [labels differ], beta = 1 / (2 mean |dc|^2); outside the rectangle is
 * fixed background.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param rect - The rectangle [x, y, width, height] containing the object.
 * @param options - Components, smoothness weight and key.
 * @returns The algorithm (stopping when the mask stops changing).
 * @concept graph-cuts
 * @example
 * ```ts
 * import { grabCutAlgorithm } from 'mdml-engine/vision'
 * grabCutAlgorithm([[0, 1], [1, 0]], [0, 0, 2, 2]).name // => 'grabcut'
 * ```
 */
export function grabCutAlgorithm(
  x: TensorLike,
  rect: readonly [number, number, number, number],
  options: GrabCutOptions = {},
): Algorithm<GrabCutState> {
  const r = rasterOf(x, 'grabCut')
  const { h, w, c } = r
  const n = h * w
  const [rx, ry, rw, rh] = rect
  const inside = (p: number): boolean => {
    const y = Math.floor(p / w)
    const xx = p % w
    return xx >= rx && xx < rx + rw && y >= ry && y < ry + rh
  }
  const K = options.components ?? 3
  const gamma = options.gamma ?? 50
  const key = options.key ?? makeKey(0)
  const px = (p: number): number[] => Array.from(r.data.subarray(p * c, (p + 1) * c))
  // Pairwise weights over 8-neighbours, computed once.
  const pairs: [number, number, number][] = []
  let sum = 0
  let count = 0
  const diff2 = (p: number, q: number): number => {
    let s = 0
    for (let k = 0; k < c; k++) s += ((r.data[p * c + k] ?? 0) - (r.data[q * c + k] ?? 0)) ** 2
    return s
  }
  const neighbours: [number, number, number][] = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, Math.SQRT2],
    [1, -1, Math.SQRT2],
  ]
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++)
      for (const [dy, dx] of neighbours) {
        const yy = y + dy
        const x2 = xx + dx
        if (yy >= h || x2 < 0 || x2 >= w) continue
        sum += diff2(y * w + xx, yy * w + x2)
        count++
      }
  const beta = sum > 0 ? 1 / (2 * (sum / count)) : 0
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++)
      for (const [dy, dx, dist] of neighbours) {
        const yy = y + dy
        const x2 = xx + dx
        if (yy >= h || x2 < 0 || x2 >= w) continue
        const p = y * w + xx
        const q = yy * w + x2
        pairs.push([p, q, (gamma * Math.exp(-beta * diff2(p, q))) / dist])
      }
  const fit = (idx: readonly number[], t: number, side: number): MixtureParameters | undefined => {
    if (idx.length === 0) return undefined
    const pts = idx.map(px)
    const k = Math.min(K, pts.length)
    const start = initialMixture(pts, { nComponents: k, regCovar: 1e-4 }, foldIn(key, 2 * t + side))
    return runAlgorithm(gaussianEmAlgorithm(pts, start, { regCovar: 1e-4 }), { maxSteps: 10 }).state
  }
  return {
    name: 'grabcut',
    config: { components: K, gamma, rect: [...rect] } as Json,
    schema: channels({
      metrics: {
        energy: { description: 'the energy of the minimum cut', direction: 'lower' },
        foreground: { description: 'foreground pixels' },
      },
      tensors: { mask: { role: 'state', description: 'the foreground mask', shape: ['h', 'w'] } },
    }),
    init: () => {
      const mask = new Uint8Array(n)
      for (let p = 0; p < n; p++) if (inside(p)) mask[p] = 1
      if (mask.every((v) => v === 1) || mask.every((v) => v === 0))
        throw new MdmlError('value', 'grabCut: the rectangle must leave some background')
      return { mask, energy: Number.POSITIVE_INFINITY, iteration: 0, changed: n }
    },
    step: (s) => {
      const fgIdx: number[] = []
      const bgIdx: number[] = []
      for (let p = 0; p < n; p++) ((s.mask[p] ?? 0) === 1 ? fgIdx : bgIdx).push(p)
      const fg = fit(fgIdx, s.iteration, 0)
      const bg = fit(bgIdx, s.iteration, 1)
      const all = Array.from({ length: n }, (_, p) => px(p))
      const lf = fg === undefined ? new Array<number>(n).fill(-1e9) : mixturePosterior(all, fg).logNorm
      const lb = bg === undefined ? new Array<number>(n).fill(-1e9) : mixturePosterior(all, bg).logNorm
      const S = n
      const T = n + 1
      const big = 1e9
      const edges: [number, number, number][] = []
      for (let p = 0; p < n; p++) {
        // Cutting S -> p labels p background and pays its background cost; cutting p -> T pays the foreground cost.
        const costBg = inside(p) ? -(lb[p] ?? 0) : 0
        const costFg = inside(p) ? -(lf[p] ?? 0) : big
        const m = Math.min(costBg, costFg)
        edges.push([S, p, costBg - m], [p, T, costFg - m])
      }
      for (const [p, q, wgt] of pairs) edges.push([p, q, wgt], [q, p, wgt])
      const cut = minCut(digraph(edges), S, T)
      const mask = new Uint8Array(n)
      for (const p of cut.partition[0]) if (p < n) mask[p] = 1
      let changed = 0
      for (let p = 0; p < n; p++) if (mask[p] !== s.mask[p]) changed++
      return { mask, energy: cut.value, iteration: s.iteration + 1, changed }
    },
    done: (s) => (s.iteration > 0 && s.changed === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { energy: Number.isFinite(s.energy) ? s.energy : -1, foreground: s.mask.reduce((a, b) => a + b, 0) },
      tensors: { mask: { role: 'state', value: Array.from(s.mask), shape: [h, w] } },
    }),
  }
}

/**
 * Segment the object in a rectangle by GrabCut ({@link grabCutAlgorithm} for a number of iterations).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param rect - The rectangle [x, y, width, height].
 * @param options - GrabCut's options, iterations (default 5) and tracing.
 * @returns The foreground mask `[H, W]`.
 * @concept graph-cuts
 * @example
 * ```ts
 * import { grabCut } from 'mdml-engine/vision'
 * import { toFlat } from 'mdml-engine/tensor'
 * const img = Array.from({ length: 12 }, (_, i) => Array.from({ length: 12 }, (_, j) => (i > 3 && i < 8 && j > 3 && j < 8 ? [0.9, 0.2, 0.2] : [0.1, 0.3, 0.8])))
 * toFlat(grabCut(img, [2, 2, 8, 8])).reduce((a, b) => a + b, 0) // => 16
 * ```
 */
export function grabCut(
  x: TensorLike,
  rect: readonly [number, number, number, number],
  options: GrabCutOptions & { readonly iterations?: number } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
    > = {},
): Tensor {
  const r = rasterOf(x, 'grabCut')
  const st = runAlgorithm(grabCutAlgorithm(x, rect, options), {
    maxSteps: options.iterations ?? 5,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  return imageTensor(Float64Array.from(st.mask), r.h, r.w)
}
