/**
 * The grid of a trained Kolmogorov-Arnold network, changed without changing what it computes. `refineGrid` is the
 * paper's grid extension: a network trained on a coarse grid is rebuilt on a finer one (more intervals, or a higher
 * degree) and every edge's coefficients are refitted by least squares so the new function matches the old one, after
 * which training continues with more capacity. `updateGrid` is pykan's `update_grid_from_samples`: the range of every
 * layer's inputs (and quantile knots) is moved to where the data now falls, which matters for the hidden layers, whose
 * inputs drift as the layers below train. Both fit the new coefficients on the sample's inputs to each layer plus an
 * even mesh over the new range, so the edge is matched everywhere on the grid and not only where the data is; for a
 * finer grid with the same range and a nested family (splines on a refined uniform grid, polynomials of a higher
 * degree) the old function lies in the new space and is reproduced to the precision of the solve.
 */

import { OmaiError } from '../../base/index.ts'
import { lstsq } from '../../numerics/linalg/index.ts'
import { key as makeKey } from '../../random/index.ts'
import { asTensor, fromStorage, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { KAN, type KanState } from './kan.ts'

/** Options of {@link updateGrid} and {@link refineGrid}. */
export interface RegridOptions {
  /** `updateGrid`: widen each layer's data range by this share of its width on each side (default 0.05). */
  readonly margin?: number
  /** Mesh points per coefficient added to the sample for the fit (default 4, at least 32 per input). */
  readonly mesh?: number
  /** The ridge on the coefficients, as the squared weight of one extra row per coefficient (default 1e-12). */
  readonly ridge?: number
}

/** A network rebuilt on a new grid, with how far its function moved on the sample. */
export interface Regridded {
  readonly kan: KAN
  readonly state: KanState
  /** The root-mean-square and largest difference between the old and the new network's outputs on the sample. */
  readonly drift: { readonly rms: number; readonly max: number }
}

const layerOf = (tree: Params, l: number): Params => (tree[String(l)] ?? {}) as Params

type Range = readonly (readonly [number, number])[]

/** The range of each column of `[n, d]` values, widened by `margin` of its width on each side. */
function columnRanges(values: ArrayLike<number>, d: number, margin: number): [number, number][] {
  const n = Math.floor(values.length / d)
  return Array.from({ length: d }, (_, i) => {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let r = 0; r < n; r++) {
      const v = values[r * d + i] as number
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    if (!(hi > lo)) return [lo - 0.5, lo + 0.5] as [number, number]
    return [lo - margin * (hi - lo), hi + margin * (hi - lo)] as [number, number]
  })
}

function regrid(
  kan: KAN,
  state: KanState,
  x: TensorLike,
  change: { readonly grid?: number; readonly degree?: number },
  keepRange: boolean,
  options: RegridOptions,
): Regridded {
  const xs = asTensor(x) as Value
  const old = kan.trace(state.params, xs, state.buffers)
  const margin = options.margin ?? 0.05
  const { widths: _w, ...structure } = kan.spec
  const layerSpecs = kan.layers.map((layer, l) => {
    const h = old.inputs[l] as Value
    return {
      sample: h,
      range: (keepRange
        ? layer.inputRange(layerOf(state.buffers, l))
        : columnRanges(toFlat(h), layer.inFeatures, margin)) as Range,
    }
  })
  const next = new KAN(kan.widths, {
    ...structure,
    ...(change.grid === undefined ? {} : { grid: change.grid }),
    ...(change.degree === undefined ? {} : { degree: change.degree }),
    layers: layerSpecs,
    key: makeKey(0),
  })
  const params: Record<string, unknown> = {}
  const buffers: Record<string, unknown> = {}
  kan.layers.forEach((oldLayer, l) => {
    const newLayer = next.layers[l]
    if (newLayer === undefined) throw new OmaiError('value', 'regrid: layer mismatch')
    const [din, dout, kNew] = [oldLayer.inFeatures, oldLayer.outFeatures, newLayer.edge.basisSize]
    const oldP = layerOf(state.params, l)
    const oldB = layerOf(state.buffers, l)
    const newP = layerOf(next.params, l)
    const newB = layerOf(next.buffers, l)
    // The points: the sample's inputs to this layer, then an even mesh over the new range of each input.
    const h = toFlat(old.inputs[l] as Value)
    const rows = Math.floor(h.length / din)
    const mesh = Math.max(32, (options.mesh ?? 4) * kNew)
    const ranges = newLayer.inputRange(newB)
    const points = rows + mesh
    const at = new Float64Array(points * din)
    for (let p = 0; p < points; p++)
      for (let i = 0; i < din; i++) {
        const [lo, hi] = ranges[i] as [number, number]
        at[p * din + i] = p < rows ? (h[p * din + i] as number) : lo + ((hi - lo) * (p - rows)) / (mesh - 1)
      }
    const pts = fromStorage(at, [points, din], 'float64')
    const target = toFlat(oldLayer.edge.terms(oldP['edge'] as Params, pts, oldB['edge'] as Params | undefined))
    const basis = toFlat(newLayer.edge.basis(newP['edge'] as Params, pts, newB['edge'] as Params | undefined))
    const root = Math.sqrt(options.ridge ?? 1e-12)
    const coef = new Float64Array(dout * din * kNew)
    for (let i = 0; i < din; i++) {
      const a = new Float64Array((points + kNew) * kNew)
      for (let p = 0; p < points; p++)
        for (let k = 0; k < kNew; k++) a[p * kNew + k] = basis[(p * din + i) * kNew + k] as number
      for (let k = 0; k < kNew; k++) a[(points + k) * kNew + k] = root
      const rhs = new Float64Array((points + kNew) * dout)
      for (let p = 0; p < points; p++)
        for (let o = 0; o < dout; o++) rhs[p * dout + o] = target[(p * din + i) * dout + o] as number
      const c = toFlat(
        lstsq(fromStorage(a, [points + kNew, kNew], 'float64'), fromStorage(rhs, [points + kNew, dout], 'float64')),
      )
      for (let o = 0; o < dout; o++)
        for (let k = 0; k < kNew; k++) coef[(o * din + i) * kNew + k] = c[k * dout + o] as number
    }
    const carried: Record<string, unknown> = {}
    for (const name of ['wSpline', 'wBase', 'bias']) if (oldP[name] !== undefined) carried[name] = oldP[name]
    params[String(l)] = {
      ...newP,
      ...carried,
      edge: { ...(newP['edge'] as Params), coef: fromStorage(coef, [dout, din, kNew], 'float64') },
    }
    buffers[String(l)] = { ...newB, mask: oldB['mask'] }
  })
  const result: KanState = { params: params as Params, buffers: buffers as Params }
  const before = toFlat(old.output)
  const after = toFlat(next.trace(result.params, xs, result.buffers).output)
  let sq = 0
  let max = 0
  for (let k = 0; k < before.length; k++) {
    const d = (after[k] as number) - (before[k] as number)
    sq += d * d
    max = Math.max(max, Math.abs(d))
  }
  return { kan: next, state: result, drift: { rms: Math.sqrt(sq / Math.max(1, before.length)), max } }
}

/**
 * Pykan's `update_grid_from_samples`: rebuild the network with every layer's input range (and quantile knots) taken from
 * the data as it now flows through the network, refitting each edge so the function is kept where the old grid covered it.
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - Inputs `[n, n_in]`.
 * @param options - The margin around the data range, the mesh and the ridge of the fit.
 * @returns The rebuilt network, its state and how far the function moved on `x`.
 * @concept splines
 * @example
 * ```ts
 * import { KAN, updateGrid } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const net = new KAN([1, 1], { grid: 4, key: key(0) })
 * const x = [[-3], [0], [3]]
 * const moved = updateGrid(net, net.state, x)
 * moved.kan.layers[0]?.inputRange()[0]?.map((v) => Math.round(v * 100) / 100) // => [-3.3, 3.3]
 * ```
 */
export function updateGrid(kan: KAN, state: KanState, x: TensorLike, options: RegridOptions = {}): Regridded {
  return regrid(kan, state, x, {}, false, options)
}

/**
 * The paper's grid extension: the same network on a finer grid (a larger `grid`, or a higher `degree`), with each edge's
 * coefficients refitted by least squares to the old function. The ranges stay as they were. With a nested family (cubic
 * splines on a uniform grid whose size is a multiple of the old, Chebyshev or Fourier terms) the function is unchanged up
 * to the solve; otherwise it is the best fit in the new space. Training then continues from the new state with a fresh
 * optimiser.
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param grid - The new grid size.
 * @param x - Inputs `[n, n_in]` (their hidden images locate the fit).
 * @param options - A new `degree`, the mesh and the ridge of the fit.
 * @returns The rebuilt network, its state and how far the function moved on `x`.
 * @concept splines
 * @example
 * ```ts
 * import { KAN, refineGrid } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const net = new KAN([2, 3, 1], { grid: 3, noise: 2, key: key(0) })
 * const x = [[0.1, 0.2], [-0.4, 0.7], [0.5, -0.5]]
 * const fine = refineGrid(net, net.state, 6, x)
 * fine.kan.spec.grid // => 6
 * fine.drift.max < 1e-6 // => true
 * ```
 */
export function refineGrid(
  kan: KAN,
  state: KanState,
  grid: number,
  x: TensorLike,
  options: RegridOptions & { readonly degree?: number } = {},
): Regridded {
  return regrid(
    kan,
    state,
    x,
    { grid, ...(options.degree === undefined ? {} : { degree: options.degree }) },
    true,
    options,
  )
}
