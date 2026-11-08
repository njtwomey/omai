/**
 * Coordinate encodings for neural fields: the maps that lift a low-dimensional coordinate to features an MLP can use,
 * which is what lets a small network fit fine detail (a plain ReLU MLP on raw coordinates is biased towards low
 * frequencies). The NeRF positional encoding (octaves of sines and cosines), Gaussian random Fourier features
 * (Tancik et al. 2020) and the multiresolution hash grid of Instant-NGP (Mueller et al. 2022), whose trainable table is
 * read by trilinear interpolation of integer-hashed lattice corners. All are differentiable in the input, and the hash
 * grid in its table, through the library's autodiff.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  asTensor,
  clip,
  concat,
  cos,
  expandDims,
  fromStorage,
  matmul,
  mul,
  primalOf,
  reshape,
  sin,
  sliceAxis,
  squeeze,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'

/** The rows of a value of shape `[..., d]` as a matrix `[n, d]`, with the way back to the leading shape. */
function asRows(x: TensorLike, who: string): { rows: Value; lead: number[]; d: number } {
  const v = asTensor(x) as Value
  if (v.shape.length === 0) throw new OmaiError('shape', `${who}: expected coordinates [..., d], got a scalar`)
  const d = v.shape[v.shape.length - 1] as number
  const lead = v.shape.slice(0, -1)
  return { rows: v.shape.length === 2 ? v : reshape(v, [-1, d]), lead, d }
}

/** Options of {@link positionalEncoding}. */
export interface PositionalEncodingOptions {
  /** The number of octaves `L` (frequencies `2^0 pi, ..., 2^(L-1) pi`). */
  readonly octaves: number
  /** Prepend the input itself (NeRF's `include_input`; default false). */
  readonly includeInput?: boolean
}

/**
 * The width of {@link positionalEncoding}: `d * 2 L`, plus `d` when the input is included.
 *
 * @param inputDim - The coordinate dimension `d`.
 * @param options - The octaves and `includeInput`.
 * @returns The output width.
 * @example
 * ```ts
 * import { positionalEncodingWidth } from 'omai-engine/nn'
 * positionalEncodingWidth(3, { octaves: 10, includeInput: true }) // => 63
 * ```
 */
export function positionalEncodingWidth(inputDim: number, options: PositionalEncodingOptions): number {
  return inputDim * 2 * options.octaves + (options.includeInput === true ? inputDim : 0)
}

/**
 * The NeRF positional encoding (Mildenhall et al. 2020), applied to every coordinate:
 * `gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^(L-1) pi p), cos(2^(L-1) pi p))`, laid out octave by octave
 * (all coordinates' sines, then all cosines, then the next octave), with the input itself first when `includeInput`.
 * Differentiable in `x`.
 *
 * @param x - Coordinates `[..., d]`, conventionally scaled to `[-1, 1]`.
 * @param options - Octaves `L` and whether to include the input.
 * @returns The encoding `[..., d * 2 L (+ d)]`.
 * @throws OmaiError with code `value` for fewer than one octave.
 * @concept coordinate-encoding
 * @reference Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi and Ng (2020), NeRF: representing scenes as neural
 *   radiance fields for view synthesis.
 * @example
 * ```ts
 * import { positionalEncoding } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * // One coordinate, two octaves, at x = 0.5: sin(pi/2), cos(pi/2), sin(pi), cos(pi).
 * toArray(positionalEncoding([[0.5]], { octaves: 2 })).map((row) => (row as number[]).map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, 0, 0, -1]]
 * ```
 */
export function positionalEncoding(x: TensorLike, options: PositionalEncodingOptions): Value {
  const L = options.octaves
  if (!(Number.isInteger(L) && L >= 1))
    throw new OmaiError('value', 'positionalEncoding: octaves must be an integer >= 1')
  const { rows, lead, d } = asRows(x, 'positionalEncoding')
  const freqs = asTensor(Array.from({ length: L }, (_, l) => 2 ** l * Math.PI)) as Value
  // [n, L, d]: every coordinate at every frequency.
  const angles = mul(expandDims(rows, 1), reshape(freqs, [L, 1]))
  const body = reshape(stack([sin(angles), cos(angles)], 2), [-1, L * 2 * d])
  const out = options.includeInput === true ? concat([rows, body], -1) : body
  return reshape(out, [...lead, out.shape[1] as number])
}

/**
 * A Gaussian frequency matrix `B` `[d, m]` with entries `N(0, sigma^2)` for {@link fourierFeatures}. The scale `sigma`
 * sets the bandwidth of what the network can fit: small values give a smooth fit, large ones noise (Tancik et al. 2020).
 *
 * @param key - The key.
 * @param inputDim - The coordinate dimension `d`.
 * @param features - The number of frequencies `m`.
 * @param sigma - The standard deviation of the frequencies (default 10).
 * @returns `B`.
 * @concept coordinate-encoding
 * @example
 * ```ts
 * import { gaussianFrequencies } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * gaussianFrequencies(key(0), 2, 16, 5).shape // => [2, 16]
 * ```
 */
export function gaussianFrequencies(key: Key, inputDim: number, features: number, sigma = 10): Tensor {
  return normal(foldIn(key, 0), [inputDim, features], { scale: sigma })
}

/**
 * Random Fourier features of coordinates (Tancik et al. 2020): `gamma(v) = (sin(2 pi v B), cos(2 pi v B))` for a fixed
 * frequency matrix `B` `[d, m]` (draw one with {@link gaussianFrequencies}). With Gaussian `B` the dot product of two
 * encodings is a stationary kernel of their difference. Differentiable in `v` and in `B`.
 *
 * @param v - Coordinates `[..., d]`.
 * @param frequencies - `B` `[d, m]`.
 * @returns The encoding `[..., 2 m]`: the `m` sines, then the `m` cosines.
 * @throws OmaiError with code `shape` when the coordinate dimension is not `B`'s first.
 * @concept coordinate-encoding
 * @reference Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron and Ng (2020),
 *   Fourier features let networks learn high frequency functions in low dimensional domains.
 * @example
 * ```ts
 * import { fourierFeatures } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * // B = [[0.25]]: the angle is 2 pi * 1 * 0.25 = pi / 2.
 * toArray(fourierFeatures([[1]], [[0.25]])).map((row) => (row as number[]).map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, 0]]
 * ```
 */
export function fourierFeatures(v: TensorLike, frequencies: TensorLike): Value {
  const { rows, lead, d } = asRows(v, 'fourierFeatures')
  const b = asTensor(frequencies) as Value
  if (b.shape.length !== 2 || b.shape[0] !== d)
    throw new OmaiError('shape', `fourierFeatures: B must be [${d}, m], got [${b.shape.join(', ')}]`)
  const angles = mul(matmul(rows, b), 2 * Math.PI)
  const out = concat([sin(angles), cos(angles)], -1)
  return reshape(out, [...lead, 2 * (b.shape[1] as number)])
}

/** The configuration of a multiresolution hash grid. */
export interface HashGridConfig {
  /** The coordinate dimension (1, 2 or 3). */
  readonly dim: number
  /** The number of resolution levels `L` (default 8). */
  readonly levels?: number
  /** The table has `2^log2TableSize` entries per level (default 14). */
  readonly log2TableSize?: number
  /** Feature channels per entry `F` (default 2). */
  readonly featuresPerLevel?: number
  /** The coarsest grid's cells per side (default 4). */
  readonly baseResolution?: number
  /** The finest grid's cells per side (default 128). */
  readonly finestResolution?: number
  /** The coordinate range `[low, high]` mapped to the unit cube (default `[-1, 1]`); inputs outside are clamped. */
  readonly bounds?: readonly [number, number]
}

/** A {@link HashGridConfig} with its defaults filled in. */
export type ResolvedHashGrid = Required<HashGridConfig>

/**
 * A hash grid configuration with its defaults filled in and its constraints checked.
 *
 * @param config - The configuration.
 * @returns The complete configuration.
 * @throws OmaiError with code `value` for a dimension outside 1 to 3, no levels or features, or a finest resolution
 *   below the base resolution.
 * @example
 * ```ts
 * import { resolveHashGrid } from 'omai-engine/nn'
 * resolveHashGrid({ dim: 2 }).log2TableSize // => 14
 * ```
 */
export function resolveHashGrid(config: HashGridConfig): ResolvedHashGrid {
  const c: ResolvedHashGrid = {
    dim: config.dim,
    levels: config.levels ?? 8,
    log2TableSize: config.log2TableSize ?? 14,
    featuresPerLevel: config.featuresPerLevel ?? 2,
    baseResolution: config.baseResolution ?? 4,
    finestResolution: config.finestResolution ?? 128,
    bounds: config.bounds ?? [-1, 1],
  }
  if (!(Number.isInteger(c.dim) && c.dim >= 1 && c.dim <= 3))
    throw new OmaiError('value', 'hashGrid: the dimension must be 1, 2 or 3')
  if (!(Number.isInteger(c.levels) && c.levels >= 1 && Number.isInteger(c.featuresPerLevel) && c.featuresPerLevel >= 1))
    throw new OmaiError('value', 'hashGrid: levels and featuresPerLevel must be positive integers')
  if (!(Number.isInteger(c.log2TableSize) && c.log2TableSize >= 1 && c.log2TableSize <= 24))
    throw new OmaiError('value', 'hashGrid: log2TableSize must be an integer in 1..24')
  if (!(c.baseResolution >= 1 && c.finestResolution >= c.baseResolution))
    throw new OmaiError('value', 'hashGrid: need 1 <= baseResolution <= finestResolution')
  if (!(c.bounds[1] > c.bounds[0])) throw new OmaiError('value', 'hashGrid: bounds must be increasing')
  return c
}

/**
 * The cells per side of each level, growing geometrically from the base to the finest resolution:
 * `N_l = floor(N_min b^l)` with `b = exp((ln N_max - ln N_min) / (L - 1))`.
 *
 * @param config - The configuration.
 * @returns The resolutions, coarse to fine.
 * @example
 * ```ts
 * import { hashGridResolutions } from 'omai-engine/nn'
 * hashGridResolutions({ dim: 2, levels: 3, baseResolution: 4, finestResolution: 16 }) // => [4, 8, 16]
 * ```
 */
export function hashGridResolutions(config: HashGridConfig): number[] {
  const c = resolveHashGrid(config)
  const growth =
    c.levels === 1 ? 1 : Math.exp((Math.log(c.finestResolution) - Math.log(c.baseResolution)) / (c.levels - 1))
  // The small tolerance keeps exact powers (b^l = 2 exactly) from rounding down.
  return Array.from({ length: c.levels }, (_, l) => Math.floor(c.baseResolution * growth ** l + 1e-9))
}

/** The spatial-hash primes of Teschner et al.: 1 for the first coordinate keeps the cache-friendly order. */
const PRIMES = [1, 2654435761, 805459861] as const

/**
 * The table slot of an integer lattice vertex at a level: the dense row-major index when the level's `(N + 1)^d`
 * vertices fit in the table (no collisions), otherwise the spatial hash `(x_1 pi_1 xor x_2 pi_2 xor ...) mod T` with
 * 32-bit integer arithmetic (deterministic, and `T` a power of two so the modulus is a mask).
 *
 * @param coords - The vertex's integer coordinates (`dim` of them, in `0..resolution`).
 * @param resolution - The level's cells per side `N`.
 * @param log2TableSize - `log2` of the table size `T`.
 * @returns The slot in `0..T-1`.
 * @example
 * ```ts
 * import { hashGridIndex } from 'omai-engine/nn'
 * hashGridIndex([2, 1], 4, 14) // => 7
 * hashGridIndex([2, 1], 4000, 4) < 16 // => true
 * ```
 */
export function hashGridIndex(coords: readonly number[], resolution: number, log2TableSize: number): number {
  const size = 2 ** log2TableSize
  const side = resolution + 1
  if (side ** coords.length <= size) {
    let index = 0
    for (let k = coords.length - 1; k >= 0; k--) index = index * side + (coords[k] as number)
    return index
  }
  let h = 0
  for (let k = 0; k < coords.length; k++) h ^= Math.imul(coords[k] as number, PRIMES[k] as number)
  return (h >>> 0) & (size - 1)
}

/**
 * The shape of a hash grid's table: `[L * 2^log2TableSize, F]`, level `l` in rows `l T .. (l + 1) T - 1`.
 *
 * @param config - The configuration.
 * @returns The table shape.
 * @example
 * ```ts
 * import { hashGridTableShape } from 'omai-engine/nn'
 * hashGridTableShape({ dim: 2, levels: 4, log2TableSize: 6, featuresPerLevel: 2 }) // => [256, 2]
 * ```
 */
export function hashGridTableShape(config: HashGridConfig): [number, number] {
  const c = resolveHashGrid(config)
  return [c.levels * 2 ** c.log2TableSize, c.featuresPerLevel]
}

/**
 * The multiresolution hash encoding of Instant-NGP (Mueller et al. 2022): at each of `L` levels the point falls in a
 * cell of a grid with `N_l` cells per side; the `2^d` corners of the cell are looked up in a trainable table (hashed
 * to a fixed size, so a fine level shares slots between corners instead of storing the whole grid) and combined by
 * multilinear interpolation. The features of all levels are concatenated. The result is differentiable in the table
 * (the gradient of a point reaches only its corners' rows) and, through the interpolation weights, piecewise linear in
 * the coordinates.
 *
 * @param x - Coordinates `[..., d]` within the configuration's bounds (outside values are clamped to the border).
 * @param table - The table `[L * 2^log2TableSize, F]` (see {@link hashGridTableShape}).
 * @param config - The grid.
 * @returns The features `[..., L F]`.
 * @throws OmaiError with code `shape` when `x` or the table disagrees with the configuration.
 * @concept coordinate-encoding
 * @reference Mueller, Evans, Schied and Keller (2022), Instant neural graphics primitives with a multiresolution hash
 *   encoding.
 * @example
 * ```ts
 * import { hashGridEncode } from 'omai-engine/nn'
 * import { fromStorage, toArray } from 'omai-engine/tensor'
 * // A 1-D grid with 2 cells and a 3-vertex table holding 0, 1, 4: the value at x = 0.75 (half way from vertex 1 to 2) is 2.5.
 * const config = { dim: 1, levels: 1, log2TableSize: 2, featuresPerLevel: 1, baseResolution: 2, finestResolution: 2, bounds: [0, 1] as const }
 * const table = fromStorage(Float64Array.of(0, 1, 4, 0), [4, 1], 'float64')
 * toArray(hashGridEncode([[0.75]], table, config)) // => [[2.5]]
 * ```
 */
export function hashGridEncode(x: TensorLike, table: Value, config: HashGridConfig): Value {
  const c = resolveHashGrid(config)
  const T = 2 ** c.log2TableSize
  const F = c.featuresPerLevel
  const { rows, lead, d } = asRows(x, 'hashGridEncode')
  if (d !== c.dim) throw new OmaiError('shape', `hashGridEncode: expected ${c.dim} coordinates, got ${d}`)
  if (table.shape.length !== 2 || table.shape[0] !== c.levels * T || table.shape[1] !== F)
    throw new OmaiError(
      'shape',
      `hashGridEncode: the table must be [${c.levels * T}, ${F}], got [${table.shape.join(', ')}]`,
    )
  const n = rows.shape[0] as number
  const [lo, hi] = c.bounds
  // The unit-cube position, clamped (the clamp passes the gradient inside the bounds and stops it outside).
  const clamped = clip(mul(sub(rows, lo), 1 / (hi - lo)), 0, 1)
  const corners = 2 ** d
  const levels: Value[] = []
  for (const [l, N] of hashGridResolutions(c).entries()) {
    const pos = mul(clamped, N)
    const concrete = primalOf(pos).data
    const base = new Int32Array(n * d)
    for (let i = 0; i < n * d; i++) base[i] = Math.min(Math.floor(concrete[i] as number), N - 1)
    // The fractional position: differentiable in x; the lattice base is a constant.
    const frac = sub(pos, fromStorage(Float64Array.from(base), [n, d], 'float64'))
    const index = new Int32Array(n * corners)
    const vertex = new Array<number>(d)
    for (let i = 0; i < n; i++) {
      for (let corner = 0; corner < corners; corner++) {
        for (let k = 0; k < d; k++) vertex[k] = (base[i * d + k] as number) + ((corner >> k) & 1)
        index[i * corners + corner] = l * T + hashGridIndex(vertex, N, c.log2TableSize)
      }
    }
    const fk = Array.from({ length: d }, (_, k) => squeeze(sliceAxis(frac, -1, k, k + 1), -1))
    const weights = stack(
      Array.from({ length: corners }, (_, corner) =>
        fk.reduce<Value>((w, f, k) => mul(w, (corner >> k) & 1 ? f : sub(1, f)), asTensor(1) as Value),
      ),
      1,
    )
    const gathered = reshape(take(table, fromStorage(index, [n * corners], 'int32')), [n, corners, F])
    levels.push(sum(mul(gathered, expandDims(weights, -1)), 1))
  }
  const out = levels.length === 1 ? (levels[0] as Value) : concat(levels, -1)
  return reshape(out, [...lead, c.levels * F])
}
