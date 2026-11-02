/**
 * The NeRF network and its renderer. `NerfField` maps a 3-D position and a viewing direction to a density and a
 * view-dependent colour: the position (encoded by a slot, the positional encoding with 10 octaves by default) goes
 * through a trunk of ReLU layers to the density and a feature vector, which joins the encoded direction (4 octaves by
 * default) in a small head for the colour, so the density depends on position alone and the colour on both, as in the
 * paper. `renderRays` casts rays through a model by stratified sampling, optionally followed by hierarchical resampling
 * with a second (fine) network, and composites by `volumeRender`.
 */

import { type ApplyContext, Composite, initKey, Linear, mlp, type Params, ReLU, Sequential } from '../nn/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Vec3 } from '../scene/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { concat, fromStorage, primalOf, relu, reshape, sigmoid, softplus, type Value } from '../tensor/index.ts'
import { hierarchicalSamples, type Planes, stratifiedSamples } from './sampling.ts'
import { fieldEncodings } from './tables.ts'
import { type VolumeRender, volumeRender } from './volume.ts'

const ZERO_KEY: Key = makeKey(0)

/** The options of {@link NerfField}. */
export interface NerfOptions {
  /** A `fieldEncodings` configuration for the position (default the positional encoding, 10 octaves, with the input). */
  readonly positionEncoding?: SpaceValue
  /** A `fieldEncodings` configuration for the direction (default 4 octaves, with the input). */
  readonly directionEncoding?: SpaceValue
  /** The width of the trunk and the feature vector (default 64; the paper's is 256). */
  readonly width?: number
  /** The trunk's layers (default 4; the paper's is 8, with a skip connection). */
  readonly depth?: number
  /** How the density is made non-negative: `softplus` (default, never exactly zero) or `relu` (the paper's). */
  readonly densityActivation?: 'softplus' | 'relu'
  readonly key?: Key
}

/** What a {@link NerfField} computes at points: densities `[n]` and colours `[n, 3]` in (0, 1). */
export interface NerfOutput {
  readonly density: Value
  readonly colour: Value
}

const POSITION = { kind: 'frequency', octaves: 10, includeInput: true } as const
const DIRECTION = { kind: 'frequency', octaves: 4, includeInput: true } as const

/**
 * A neural radiance field: density from position, colour from position and direction. Its input is the pair
 * `[positions, directions]`, each `[n, 3]`; its output the densities `[n]` (non-negative) and the colours `[n, 3]`
 * (through a sigmoid).
 *
 * @concept neural-radiance-fields
 * @reference Mildenhall et al. (2020), NeRF: representing scenes as neural radiance fields for view synthesis.
 * @example
 * ```ts
 * import { NerfField } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const f = new NerfField({ width: 16, depth: 2, positionEncoding: { kind: 'frequency', octaves: 3 }, key: key(0) })
 * const out = f.forward([[[0.1, 0.2, 0.3]], [[0, 0, 1]]] as never)
 * [out.density.shape, out.colour.shape] // => [[1], [1, 3]]
 * ```
 */
export class NerfField extends Composite<readonly [Value, Value], NerfOutput> {
  readonly kind = 'NerfField'
  readonly width: number
  readonly densityActivation: 'softplus' | 'relu'

  constructor(options: NerfOptions = {}) {
    const key = initKey('NerfField', options.key)
    const width = options.width ?? 64
    const depth = options.depth ?? 4
    const position = fieldEncodings.make(options.positionEncoding ?? POSITION)(3, foldIn(key, 0))
    const direction = fieldEncodings.make(options.directionEncoding ?? DIRECTION)(3, foldIn(key, 1))
    const half = Math.max(2, Math.floor(width / 2))
    super({
      positionEncoding: position.module,
      directionEncoding: direction.module,
      trunk: new Sequential(
        mlp([position.outputDim, ...Array.from({ length: depth }, () => width)], { key: foldIn(key, 2) }),
        new ReLU(),
      ),
      densityHead: new Linear(width, 1, { key: foldIn(key, 3) }),
      featureHead: new Linear(width, width, { key: foldIn(key, 4) }),
      colourHidden: new Sequential(new Linear(width + direction.outputDim, half, { key: foldIn(key, 5) }), new ReLU()),
      colourHead: new Linear(half, 3, { key: foldIn(key, 6) }),
    })
    this.width = width
    this.densityActivation = options.densityActivation ?? 'softplus'
  }

  protected override prepare(x: unknown): unknown {
    const [p, d] = x as readonly [Value, Value]
    return [this.asValue(p), this.asValue(d)]
  }

  private asValue(v: unknown): Value {
    return Array.isArray(v)
      ? (fromStorage(
          Float64Array.from((v as number[][]).flat()),
          [v.length, (v[0] as number[]).length],
          'float64',
        ) as Value)
      : (v as Value)
  }

  apply(params: Params, x: readonly [Value, Value], ctx: ApplyContext = {}): NerfOutput {
    const run = this.parts(params, ctx)
    const h = run.run<Value, Value>('trunk', run.run<Value, Value>('positionEncoding', x[0]))
    const raw = run.run<Value, Value>('densityHead', h)
    const density = reshape(this.densityActivation === 'relu' ? relu(raw) : softplus(raw), [-1])
    const feature = run.run<Value, Value>('featureHead', h)
    const view = run.run<Value, Value>('directionEncoding', x[1])
    const colour = sigmoid(
      run.run<Value, Value>('colourHead', run.run<Value, Value>('colourHidden', concat([feature, view], -1))),
    )
    run.done()
    return { density, colour }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), 4]
  }

  override extraRepr(): string {
    return `width=${this.width}, density=${this.densityActivation}`
  }
}

/** A NeRF as a coarse network and, optionally, a fine one for hierarchical sampling. */
export interface NerfModel {
  readonly coarse: NerfField
  readonly fine?: NerfField
}

/** The parameters of a {@link NerfModel}. */
export interface NerfParams {
  readonly coarse: Params
  readonly fine?: Params
}

/**
 * A NeRF model: one network, or a coarse and a fine one with independent weights.
 *
 * @param options - The network options, and whether to add the fine network.
 * @returns The model.
 * @concept neural-radiance-fields
 * @index structure A NeRF model: one network, or coarse and fine networks.
 * @example
 * ```ts
 * import { nerfModel } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * nerfModel({ width: 8, depth: 1, fine: true, key: key(0) }).fine !== undefined // => true
 * ```
 */
export function nerfModel(options: NerfOptions & { readonly fine?: boolean } = {}): NerfModel {
  const { fine, key, ...rest } = options
  const k = initKey('NerfModel', key)
  return {
    coarse: new NerfField({ ...rest, key: foldIn(k, 0) }),
    ...(fine === true ? { fine: new NerfField({ ...rest, key: foldIn(k, 1) }) } : {}),
  }
}

/**
 * The parameters of a model's networks.
 *
 * @param model - The model.
 * @returns Their parameter trees.
 * @example
 * ```ts
 * import { nerfModel, nerfParams } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * Object.keys(nerfParams(nerfModel({ width: 8, depth: 1, fine: true, key: key(0) }))) // => ['coarse', 'fine']
 * ```
 */
export function nerfParams(model: NerfModel): NerfParams {
  return { coarse: model.coarse.params, ...(model.fine === undefined ? {} : { fine: model.fine.params }) }
}

/** A batch of rays: origins and unit directions, three numbers per ray. */
export interface RayBatch {
  readonly origins: ArrayLike<number>
  readonly directions: ArrayLike<number>
}

/** Options of {@link renderRays}. */
export interface RenderRaysOptions {
  /** The near and far distances along the rays (numbers, or one per ray). */
  readonly near: Planes
  readonly far: Planes
  /** Coarse samples per ray (default 32). */
  readonly samples?: number
  /** Fine samples per ray added by hierarchical sampling when the model has a fine network (default as the coarse). */
  readonly fineSamples?: number
  /** Draw the samples at random within their bins and the fine ones randomly (training); omit for midpoints. */
  readonly key?: Key
  /** Composite onto white (default false: black). */
  readonly whiteBackground?: boolean
}

/** What a render of rays gives: the coarse pass and, with a fine network, the fine pass, which is the model's answer. */
export interface NerfRender {
  readonly coarse: NerfPass
  readonly fine?: NerfPass
}

/** One pass of a render: the volume rendering of the samples, their distances and the densities that were composited. */
export type NerfPass = VolumeRender & { readonly t: Value; readonly density: Value }

/** The densities and colours of a network at the points along rays, shaped `[rays, n]` and `[rays, n, 3]`. */
function query(net: NerfField, params: Params, rays: RayBatch, t: Value): { density: Value; colour: Value } {
  const [r, n] = t.shape as [number, number]
  const times = primalOf(t).data
  const points = new Float64Array(r * n * 3)
  const views = new Float64Array(r * n * 3)
  for (let i = 0; i < r; i++)
    for (let s = 0; s < n; s++)
      for (let c = 0; c < 3; c++) {
        points[(i * n + s) * 3 + c] =
          (rays.origins[3 * i + c] as number) + (rays.directions[3 * i + c] as number) * (times[i * n + s] as number)
        views[(i * n + s) * 3 + c] = rays.directions[3 * i + c] as number
      }
  const out = net.apply(
    params,
    [fromStorage(points, [r * n, 3], 'float64'), fromStorage(views, [r * n, 3], 'float64')] as const,
    { train: true },
  )
  return { density: reshape(out.density, [r, n]), colour: reshape(out.colour, [r, n, 3]) }
}

/**
 * Render rays through a NeRF: stratified distances, the coarse network's densities and colours composited by volume
 * rendering and, when the model has a fine network, the coarse weights resampled (no gradient through where the new
 * samples fall) for a second pass over the merged distances. Differentiable in the parameters.
 *
 * @param model - The coarse (and fine) network.
 * @param params - Their parameters.
 * @param rays - The rays.
 * @param options - Planes, sample counts, the key for random samples and the background.
 * @returns The coarse render and, with a fine network, the fine one.
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { nerfModel, nerfParams, renderRays } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const model = nerfModel({ width: 8, depth: 1, positionEncoding: { kind: 'frequency', octaves: 2 }, key: key(0) })
 * const rays = { origins: [0, 0, 4, 0, 0, 4], directions: [0, 0, -1, 0, 0, -1] }
 * renderRays(model, nerfParams(model), rays, { near: 2, far: 6, samples: 5 }).coarse.colour.shape // => [2, 3]
 * ```
 */
export function renderRays(
  model: NerfModel,
  params: NerfParams,
  rays: RayBatch,
  options: RenderRaysOptions,
): NerfRender {
  const count = rays.origins.length / 3
  const samples = options.samples ?? 32
  const volume = options.whiteBackground === true ? { whiteBackground: true } : {}
  const t = stratifiedSamples(
    foldIn(options.key ?? ZERO_KEY, 0),
    count,
    samples,
    options.near,
    options.far,
    options.key !== undefined,
  )
  const first = query(model.coarse, params.coarse, rays, t)
  const coarse = { ...volumeRender(first.density, first.colour, t, volume), t, density: first.density }
  if (model.fine === undefined || params.fine === undefined) return { coarse }
  const merged = hierarchicalSamples(
    t,
    primalOf(coarse.weights),
    options.fineSamples ?? samples,
    options.key === undefined ? undefined : foldIn(options.key, 1),
  )
  const second = query(model.fine, params.fine, rays, merged)
  return {
    coarse,
    fine: { ...volumeRender(second.density, second.colour, merged, volume), t: merged, density: second.density },
  }
}

/** The density and colour of a radiance field on a regular grid of points, from {@link nerfDensityGrid}. */
export interface NerfDensityGrid {
  /** Points per axis. */
  readonly resolution: number
  /** The box the grid fills: the grid points are the cell centres. */
  readonly bounds: readonly [Vec3, Vec3]
  /** Density at each point, `x` varying fastest, then `y`, then `z`: index `i + n (j + n k)`. */
  readonly density: readonly number[]
  /** Colour at each point, interleaved red, green, blue, in the same order. */
  readonly colour: readonly number[]
  readonly maxDensity: number
}

/**
 * Ask a radiance field for its density and colour on a regular grid of points in a box: the learned scene as a volume
 * rather than as rendered pictures. A point's colour depends on the viewing direction, so one is given (the colour the
 * point shows to a camera looking along it). Uses the fine network when the model has one.
 *
 * @param model - The model.
 * @param params - Its parameters.
 * @param options - The box (default `[-1, -1, -1]` to `[1, 1, 1]`), points per axis (default 16) and the viewing
 *   direction (default along `-z`).
 * @returns The densities and colours on the grid, and the largest density.
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { nerfDensityGrid, nerfModel, nerfParams } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const model = nerfModel({ width: 8, depth: 1, positionEncoding: { kind: 'frequency', octaves: 2 }, key: key(0) })
 * const g = nerfDensityGrid(model, nerfParams(model), { resolution: 3 })
 * [g.density.length, g.colour.length] // => [27, 81]
 * ```
 */
export function nerfDensityGrid(
  model: NerfModel,
  params: NerfParams,
  options: {
    readonly bounds?: readonly [Vec3, Vec3]
    readonly resolution?: number
    readonly direction?: Vec3
  } = {},
): NerfDensityGrid {
  const n = options.resolution ?? 16
  const bounds = options.bounds ?? [
    [-1, -1, -1],
    [1, 1, 1],
  ]
  const direction = options.direction ?? [0, 0, -1]
  const total = n * n * n
  const points = new Float64Array(total * 3)
  const views = new Float64Array(total * 3)
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const at = i + n * (j + n * k)
        const cell = [i, j, k]
        for (let c = 0; c < 3; c++) {
          const [lo, hi] = [bounds[0][c] as number, bounds[1][c] as number]
          points[3 * at + c] = lo + ((hi - lo) * ((cell[c] as number) + 0.5)) / n
          views[3 * at + c] = direction[c] as number
        }
      }
  const net = model.fine ?? model.coarse
  const netParams = (model.fine === undefined ? params.coarse : params.fine) as Params
  const out = net.apply(
    netParams,
    [fromStorage(points, [total, 3], 'float64'), fromStorage(views, [total, 3], 'float64')] as const,
    { train: false },
  )
  const density = Array.from(primalOf(out.density).data)
  return {
    resolution: n,
    bounds,
    density,
    colour: Array.from(primalOf(out.colour).data),
    maxDensity: density.reduce((a, b) => Math.max(a, b), 0),
  }
}
