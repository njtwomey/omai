/**
 * Differentiable spatial sampling: `affineGrid` and `gridSample` (the pieces of a spatial transformer network, sampling
 * an image at arbitrary locations with bilinear, nearest or bicubic interpolation), and the region-of-interest pooling
 * of object detectors, `roiAlign` (Mask R-CNN) and `roiPool` (Fast R-CNN), as in torchvision.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  abs,
  add,
  asTensor,
  clip,
  floor,
  fromStorage,
  matmul,
  max,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  takeAlongAxis,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'

/** Normalised coordinates of n pixels along an axis, in [-1, 1]. */
const baseCoords = (n: number, alignCorners: boolean): number[] =>
  Array.from({ length: n }, (_, i) => (alignCorners ? (n > 1 ? -1 + (2 * i) / (n - 1) : 0) : (2 * i + 1) / n - 1))

/**
 * The sampling grid of an affine transform (torch.nn.functional.affine_grid; spatial transformer networks): the
 * normalised output pixel coordinates, mapped by theta.
 *
 * @param theta - `[N, 2, 3]` (2-D) or `[N, 3, 4]` (3-D).
 * @param size - The output size `[N, C, H, W]` or `[N, C, D, H, W]`.
 * @param alignCorners - Corner pixels at -1 and 1 (default false: pixel edges there).
 * @returns The grid `[N, H, W, 2]` (x, y) or `[N, D, H, W, 3]` (x, y, z), differentiable in theta.
 * @concept spatial-transformer
 * @reference Jaderberg, Simonyan, Zisserman and Kavukcuoglu (2015), Spatial transformer networks.
 * @example
 * ```ts
 * import { affineGrid } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(affineGrid([[[1, 0, 0], [0, 1, 0]]], [1, 1, 1, 2], true)) // => [[[[-1, 0], [1, 0]]]]
 * ```
 */
export function affineGrid(theta: TensorLike, size: readonly number[], alignCorners = false): Value {
  const [t] = promoteFloat(theta) as [Value]
  const d = size.length - 2
  if (d !== 2 && d !== 3) throw new WhiteboxError('shape', 'affineGrid: size must be 4-D or 5-D')
  const n = size[0] ?? 1
  const spatial = size.slice(2)
  // Homogeneous base coordinates [P, d + 1] in (x, y[, z], 1) order, x varying fastest.
  const axes = spatial.map((s) => baseCoords(s, alignCorners))
  const points = spatial.reduce((a, b) => a * b, 1)
  const base = new Float64Array(points * (d + 1))
  for (let p = 0; p < points; p++) {
    let rem = p
    const coords = new Array<number>(d)
    for (let i = d - 1; i >= 0; i--) {
      const s = spatial[i] ?? 1
      coords[i] = axes[i]?.[rem % s] ?? 0
      rem = Math.floor(rem / s)
    }
    // spatial is (D, H, W) or (H, W); the grid lists (x, y, z) = (W, H, D).
    for (let j = 0; j < d; j++) base[p * (d + 1) + j] = coords[d - 1 - j] ?? 0
    base[p * (d + 1) + d] = 1
  }
  const b = fromStorage(base, [1, points, d + 1], 'float64')
  return reshape(matmul(b, swapaxes(t, 1, 2)), [n, ...spatial, d])
}

/** Options of {@link gridSample}. */
export interface GridSampleOptions {
  readonly mode?: 'bilinear' | 'nearest' | 'bicubic'
  /** Out-of-range samples: zeros, the border value, or reflection. */
  readonly paddingMode?: 'zeros' | 'border' | 'reflection'
  readonly alignCorners?: boolean
}

const unnormalise = (g: Value, size: number, alignCorners: boolean): Value =>
  alignCorners ? mul(add(g, 1), (size - 1) / 2) : div2(sub(mul(add(g, 1), size), 1))
const div2 = (v: Value): Value => mul(v, 0.5)

/** Reflect coordinates into [low, high] (PyTorch's reflect_coordinates with twice the bounds). */
function reflect(x: Value, twiceLow: number, twiceHigh: number): Value {
  if (twiceLow === twiceHigh) return mul(x, 0)
  const min = twiceLow / 2
  const span = (twiceHigh - twiceLow) / 2
  const shifted = abs(sub(x, min))
  const flips = floor(mul(primalOf(shifted), 1 / span))
  const extra = sub(shifted, mul(flips, span))
  const even = toFlat(flips).map((f) => f % 2 === 0)
  const evenMask = fromStorage(Uint8Array.from(even, Number), x.shape, 'bool')
  return where(evenMask, add(extra, min), add(mul(extra, -1), span + min))
}

function pad(x: Value, size: number, mode: 'zeros' | 'border' | 'reflection', alignCorners: boolean): Value {
  if (mode === 'border') return clip(x, 0, size - 1)
  if (mode === 'reflection') {
    const r = alignCorners ? reflect(x, 0, 2 * (size - 1)) : reflect(x, -1, 2 * size - 1)
    return clip(r, 0, size - 1)
  }
  return x
}

const cubicWeights = (t: Value): Value[] => {
  const a = -0.75
  // Keys's kernel at distances t + 1, t, 1 - t, 2 - t.
  const k1 = (s: Value): Value => add(mul(mul(sub(mul(s, a + 2), a + 3), s), s), 1)
  const k2 = (s: Value): Value => sub(mul(add(mul(sub(mul(s, a), 5 * a), s), 8 * a), s), 4 * a)
  return [k2(add(t, 1)), k1(t), k1(sub(1, t)), k2(sub(2, t))]
}

/**
 * Sample an input at arbitrary normalised locations (torch.nn.functional.grid_sample, 2-D): bilinear, nearest or
 * bicubic interpolation, with zeros, border or reflection padding. Differentiable in the input and (for bilinear and
 * bicubic) in the grid, so a spatial transformer can learn its transform.
 *
 * @param input - `[N, C, H, W]`.
 * @param grid - `[N, Ho, Wo, 2]`, (x, y) in [-1, 1].
 * @param options - Mode (default bilinear), padding mode (default zeros) and alignCorners (default false).
 * @returns `[N, C, Ho, Wo]`.
 * @concept spatial-transformer
 * @example
 * ```ts
 * import { gridSample } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(gridSample([[[[1, 3]]]], [[[[0, 0]]]], { alignCorners: true })) // => [[[[2]]]]
 * ```
 */
export function gridSample(input: TensorLike, grid: TensorLike, options: GridSampleOptions = {}): Value {
  const [x, g] = promoteFloat(input, grid) as [Value, Value]
  if (x.shape.length !== 4 || g.shape.length !== 4 || g.shape[3] !== 2)
    throw new WhiteboxError('shape', 'gridSample: input [N, C, H, W] and grid [N, Ho, Wo, 2] (volumes: TODO)')
  const [n = 1, c = 1, h = 1, w = 1] = x.shape
  const [, ho = 1, wo = 1] = g.shape
  const m = ho * wo
  const align = options.alignCorners ?? false
  const padding = options.paddingMode ?? 'zeros'
  const mode = options.mode ?? 'bilinear'
  const flat = reshape(x, [n, c, h * w])
  const gx = reshape(sliceAxis(g, 3, 0, 1), [n, 1, m])
  const gy = reshape(sliceAxis(g, 3, 1, 2), [n, 1, m])
  let ix = unnormalise(gx, w, align)
  let iy = unnormalise(gy, h, align)
  /** Gather values at integer pixel coordinates (concrete), zero where out of range. */
  const tap = (xs: readonly number[], ys: readonly number[]): Value => {
    const idx = new Int32Array(n * c * m)
    const valid = new Uint8Array(n * m)
    for (let b = 0; b < n; b++) {
      for (let k = 0; k < m; k++) {
        const xi = xs[b * m + k] ?? 0
        const yi = ys[b * m + k] ?? 0
        const inside = xi >= 0 && xi < w && yi >= 0 && yi < h
        valid[b * m + k] = inside ? 1 : 0
        const at = inside ? yi * w + xi : 0
        for (let ch = 0; ch < c; ch++) idx[(b * c + ch) * m + k] = at
      }
    }
    const v = takeAlongAxis(flat, fromStorage(idx, [n, c, m], 'int32'), 2)
    return where(fromStorage(valid, [n, 1, m], 'bool'), v, 0)
  }
  let out: Value
  if (mode === 'bicubic') {
    // PyTorch pads each integer tap, not the continuous coordinate.
    const fx = floor(primalOf(ix))
    const fy = floor(primalOf(iy))
    const wx = cubicWeights(sub(ix, fx))
    const wy = cubicWeights(sub(iy, fy))
    const bx = toFlat(fx)
    const by = toFlat(fy)
    const padIndex = (v: number, size: number): number => {
      if (padding === 'border') return Math.min(Math.max(v, 0), size - 1)
      if (padding === 'reflection') {
        const r = toFlat(pad(asTensor([v]), size, 'reflection', align))[0] ?? 0
        return Math.round(r)
      }
      return v
    }
    let acc: Value | undefined
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const xs = bx.map((v) => padIndex(v - 1 + j, w))
        const ys = by.map((v) => padIndex(v - 1 + i, h))
        const term = mul(tap(xs, ys), mul(wx[j] as Value, wy[i] as Value))
        acc = acc === undefined ? term : add(acc, term)
      }
    }
    out = acc as Value
  } else {
    ix = pad(ix, w, padding, align)
    iy = pad(iy, h, padding, align)
    if (mode === 'nearest') {
      // Round half to even, as PyTorch's nearbyint.
      const even = (v: number): number => {
        const r = Math.round(v)
        return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r
      }
      out = tap(toFlat(primalOf(ix)).map(even), toFlat(primalOf(iy)).map(even))
    } else {
      const x0 = floor(primalOf(ix))
      const y0 = floor(primalOf(iy))
      const wx1 = sub(ix, x0)
      const wy1 = sub(iy, y0)
      const wx0 = sub(1, wx1)
      const wy0 = sub(1, wy1)
      const bx = toFlat(x0)
      const by = toFlat(y0)
      const plus = (a: number[]): number[] => a.map((v) => v + 1)
      out = add(
        add(mul(tap(bx, by), mul(wx0, wy0)), mul(tap(plus(bx), by), mul(wx1, wy0))),
        add(mul(tap(bx, plus(by)), mul(wx0, wy1)), mul(tap(plus(bx), plus(by)), mul(wx1, wy1))),
      )
    }
  }
  return reshape(out, [n, c, ho, wo])
}

/** Options of {@link roiAlign}. */
export interface RoiAlignOptions {
  /** Box coordinates times this give pixel coordinates (1 / the feature stride; default 1). */
  readonly spatialScale?: number
  /** Samples per bin along each axis (default: adaptive, ceil(roi size / output size)). */
  readonly samplingRatio?: number
  /** Shift boxes by half a pixel (torchvision's aligned=True, the correct alignment). */
  readonly aligned?: boolean
}

const boxList = (boxes: TensorLike): number[][] => {
  const b = asTensor(boxes)
  if (b.shape.length === 1 && b.shape[0] === 0) return []
  if (b.shape.length !== 2 || b.shape[1] !== 5) throw new WhiteboxError('shape', 'roi: boxes must be [K, 5]')
  const v = toFlat(b)
  return Array.from({ length: b.shape[0] ?? 0 }, (_, k) => v.slice(k * 5, k * 5 + 5))
}

/**
 * ROI Align (torchvision.ops.roi_align, Mask R-CNN): pool each box of a feature map into a fixed grid of bins by
 * averaging bilinear samples, without quantising the box. Differentiable in the features.
 *
 * @param input - Features `[N, C, H, W]`.
 * @param boxes - `[K, 5]` rows (batch index, x1, y1, x2, y2).
 * @param outputSize - `[ph, pw]` (or one number).
 * @param options - Spatial scale, sampling ratio and aligned.
 * @returns `[K, C, ph, pw]`.
 * @reference He, Gkioxari, Dollar and Girshick (2017), Mask R-CNN.
 * @example
 * ```ts
 * import { roiAlign } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(roiAlign([[[[1, 1], [1, 1]]]], [[0, 0, 0, 1, 1]], 1)) // => [[[[1]]]]
 * ```
 */
export function roiAlign(
  input: TensorLike,
  boxes: TensorLike,
  outputSize: number | readonly [number, number],
  options: RoiAlignOptions = {},
): Value {
  const [x] = promoteFloat(input) as [Value]
  const [, c = 1, h = 1, w = 1] = x.shape
  const [ph, pw] = typeof outputSize === 'number' ? [outputSize, outputSize] : outputSize
  const scale = options.spatialScale ?? 1
  const offset = options.aligned === true ? 0.5 : 0
  const flat = reshape(x, [x.shape[0] ?? 1, c, h * w])
  const rois = boxList(boxes).map((r) => {
    const [bi = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0] = r
    const sx = x1 * scale - offset
    const sy = y1 * scale - offset
    let rw = x2 * scale - offset - sx
    let rh = y2 * scale - offset - sy
    if (options.aligned !== true) {
      rw = Math.max(rw, 1)
      rh = Math.max(rh, 1)
    }
    const bh = rh / ph
    const bw = rw / pw
    const gh =
      options.samplingRatio !== undefined && options.samplingRatio > 0 ? options.samplingRatio : Math.ceil(rh / ph)
    const gw =
      options.samplingRatio !== undefined && options.samplingRatio > 0 ? options.samplingRatio : Math.ceil(rw / pw)
    const count = Math.max(gh * gw, 1)
    // A sparse matrix from pixels to bins, as (bin, pixel, weight) entries.
    const weights = new Float64Array(ph * pw * h * w)
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        for (let iy = 0; iy < gh; iy++) {
          const yy = sy + py * bh + ((iy + 0.5) * bh) / gh
          for (let ix = 0; ix < gw; ix++) {
            const xx = sx + px * bw + ((ix + 0.5) * bw) / gw
            for (const [pix, wt] of bilinearTaps(yy, xx, h, w)) {
              const at = (py * pw + px) * h * w + pix
              weights[at] = (weights[at] ?? 0) + wt / count
            }
          }
        }
      }
    }
    const row = reshape(take(flat, [Math.trunc(bi)], 0), [c, h * w])
    return reshape(matmul(row, swapaxes(fromStorage(weights, [ph * pw, h * w], 'float64'), 0, 1)), [c, ph, pw])
  })
  return rois.length === 0 ? zeros([0, c, ph, pw]) : stack(rois, 0)
}

/** torchvision's bilinear taps for ROI Align: zero outside [-1, size], clamped at the border. */
function bilinearTaps(y0: number, x0: number, h: number, w: number): [number, number][] {
  if (y0 < -1 || y0 > h || x0 < -1 || x0 > w) return []
  let y = Math.max(y0, 0)
  let x = Math.max(x0, 0)
  let yl = Math.floor(y)
  let xl = Math.floor(x)
  let yh: number
  let xh: number
  if (yl >= h - 1) {
    yl = h - 1
    yh = h - 1
    y = yl
  } else yh = yl + 1
  if (xl >= w - 1) {
    xl = w - 1
    xh = w - 1
    x = xl
  } else xh = xl + 1
  const ly = y - yl
  const lx = x - xl
  return [
    [yl * w + xl, (1 - ly) * (1 - lx)],
    [yl * w + xh, (1 - ly) * lx],
    [yh * w + xl, ly * (1 - lx)],
    [yh * w + xh, ly * lx],
  ]
}

/**
 * ROI pooling (torchvision.ops.roi_pool, Fast R-CNN): max over each quantised bin of each box.
 *
 * @param input - Features `[N, C, H, W]`.
 * @param boxes - `[K, 5]` rows (batch index, x1, y1, x2, y2).
 * @param outputSize - `[ph, pw]` (or one number).
 * @param spatialScale - Box-to-feature scale (default 1).
 * @returns `[K, C, ph, pw]`; empty bins give 0.
 * @reference Girshick (2015), Fast R-CNN.
 * @example
 * ```ts
 * import { roiPool } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(roiPool([[[[1, 2], [3, 4]]]], [[0, 0, 0, 1, 1]], 1)) // => [[[[4]]]]
 * ```
 */
export function roiPool(
  input: TensorLike,
  boxes: TensorLike,
  outputSize: number | readonly [number, number],
  spatialScale = 1,
): Value {
  const [x] = promoteFloat(input) as [Value]
  const [, c = 1, h = 1, w = 1] = x.shape
  const [ph, pw] = typeof outputSize === 'number' ? [outputSize, outputSize] : outputSize
  const rows = boxList(boxes).map((r) => {
    const [bi = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0] = r
    const sx = Math.round(x1 * spatialScale)
    const sy = Math.round(y1 * spatialScale)
    const rw = Math.max(Math.round(x2 * spatialScale) - sx + 1, 1)
    const rh = Math.max(Math.round(y2 * spatialScale) - sy + 1, 1)
    const fm = reshape(take(x, [Math.trunc(bi)], 0), [c, h, w])
    const bins: Value[] = []
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        const hs = Math.min(Math.max(Math.floor((py * rh) / ph) + sy, 0), h)
        const he = Math.min(Math.max(Math.ceil(((py + 1) * rh) / ph) + sy, 0), h)
        const ws = Math.min(Math.max(Math.floor((px * rw) / pw) + sx, 0), w)
        const we = Math.min(Math.max(Math.ceil(((px + 1) * rw) / pw) + sx, 0), w)
        bins.push(
          he <= hs || we <= ws
            ? reshape(mul(sum(sliceAxis(fm, 1, 0, 1), [1, 2]), 0), [c])
            : max(sliceAxis(sliceAxis(fm, 1, hs, he), 2, ws, we), [1, 2]),
        )
      }
    }
    return reshape(stack(bins, 1), [c, ph, pw])
  })
  return stack(rows, 0)
}

/**
 * Identity affine transforms `[n, 2, 3]` (the usual initialisation of a spatial transformer's localisation head).
 *
 * @param n - The batch size.
 * @returns n copies of [[1, 0, 0], [0, 1, 0]].
 * @example
 * ```ts
 * import { identityTheta } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(identityTheta(1)) // => [[[1, 0, 0], [0, 1, 0]]]
 * ```
 */
export function identityTheta(n: number): Tensor {
  return fromStorage(
    Float64Array.from({ length: n * 6 }, (_, i) => (i % 6 === 0 || i % 6 === 4 ? 1 : 0)),
    [n, 2, 3],
    'float64',
  )
}
