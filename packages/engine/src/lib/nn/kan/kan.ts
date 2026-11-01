/**
 * `KAN`: a Kolmogorov-Arnold network, a stack of `KanLayer`s given by its widths `[n_in, h_1, ..., n_out]`. Every edge
 * between consecutive layers carries its own learnable univariate function and the nodes sum, so the whole network is a
 * composition of sums of one-dimensional functions and can be read edge by edge: `edgeCurve` gives any edge's function
 * on a grid, `trace` the inputs and the activations of every layer, `edgeScales` the average magnitude of each edge (the
 * quantity the sparsity regulariser, pruning and the plots of the paper are made of). The structure (widths, grid,
 * degree, the families of the edges and the base functions) is a `KanSpec`, plain data from which the network is
 * rebuilt, so it travels with a saved state and changes when the grid is refined or nodes are pruned.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { defaults, type ModeConfig, type OptionTable, type SpaceValue } from '../../space/index.ts'
import { abs, asTensor, mean, type Tensor, type TensorLike, transpose, type Value } from '../../tensor/index.ts'
import { type ApplyContext, assemble, childScope, initKey, Module, type Params } from '../module/module.ts'
import { kanBases } from './edge-bases.ts'
import { KanLayer, kanBaseFunctions } from './kan-layer.ts'

/** The parameters and buffers of a network: what training changes and what grid updates and pruning change. */
export interface KanState {
  readonly params: Params
  readonly buffers: Params
}

/** The structure of a network as plain data. */
export interface KanSpec {
  /** The widths `[n_in, h_1, ..., n_out]`. */
  readonly widths: readonly number[]
  readonly grid: number
  readonly degree: number
  /** The `kanBases` option (complete: every knob present). */
  readonly basis: SpaceValue
  /** The `kanBaseFunctions` option. */
  readonly base: SpaceValue
  /** The `splineKnots` option. */
  readonly knots: SpaceValue
  readonly outside: 'clamp' | 'linear'
  readonly bias: boolean
  readonly noise: number
  readonly scaleBaseMu: number
  readonly scaleBaseSigma: number
  readonly scaleSpline: number
}

/** Options of {@link KAN}: the structure (all optional) and what places the grids. */
export interface KanOptions extends Partial<Omit<KanSpec, 'widths'>> {
  /** The range of the inputs of every layer (default: the range of `sample` for the first layer, `[-1, 1]` for the others). */
  readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
  /** A sample of the network's inputs `[n, n_in]`: sets the range of the first layer (and quantile knots). */
  readonly sample?: TensorLike
  /**
   * Per layer, the range of its inputs and/or a sample of them (the quantile knots, and the range when none is given):
   * what a rebuilt network takes from the one it replaces. Takes precedence over `range` and `sample`.
   */
  readonly layers?: readonly (
    | {
        readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
        readonly sample?: TensorLike
      }
    | undefined
  )[]
  readonly key?: Key
}

/** A configuration of an option table with the knobs it leaves out at their defaults. */
function complete<T, P>(table: OptionTable<T, P>, config: SpaceValue | undefined): SpaceValue {
  const given = (typeof config === 'object' && config !== null && !Array.isArray(config) ? config : {}) as ModeConfig
  const chosen = table.optionOf(given)
  return { ...(defaults(chosen.space) as ModeConfig), ...given, [table.tag]: chosen.name }
}

/**
 * A structure with every default filled in.
 *
 * @param widths - The widths `[n_in, ..., n_out]`.
 * @param options - Any of the structure's fields.
 * @returns The complete structure.
 * @throws MdmlError `value` for fewer than two widths or a width below 1.
 * @example
 * ```ts
 * import { kanSpec } from 'mdml-engine/nn'
 * kanSpec([2, 5, 1]).grid // => 5
 * kanSpec([2, 1], { basis: { kind: 'fourier' } }).basis // => { kind: 'fourier' }
 * ```
 */
export function kanSpec(widths: readonly number[], options: Partial<Omit<KanSpec, 'widths'>> = {}): KanSpec {
  if (widths.length < 2 || widths.some((w) => !(Number.isInteger(w) && w >= 1)))
    throw new MdmlError('value', 'kanSpec: widths need at least [n_in, n_out], each an integer >= 1')
  return {
    widths: [...widths],
    grid: options.grid ?? 5,
    degree: options.degree ?? 3,
    basis: complete(kanBases, options.basis),
    base: complete(kanBaseFunctions, options.base),
    knots: options.knots ?? { kind: 'uniform' },
    outside: options.outside ?? 'linear',
    bias: options.bias ?? false,
    noise: options.noise ?? 0.3,
    scaleBaseMu: options.scaleBaseMu ?? 0,
    scaleBaseSigma: options.scaleBaseSigma ?? 1,
    scaleSpline: options.scaleSpline ?? 1,
  }
}

/** What a forward pass through a network leaves behind, layer by layer. */
export interface KanTrace {
  /** The input of every layer `[n, in_l]` (the first is the network's input). */
  readonly inputs: readonly Value[]
  /** The activation of every edge of every layer `[n, in_l, out_l]` (before the sum at the nodes). */
  readonly activations: readonly Value[]
  /** The network's output `[n, n_out]`. */
  readonly output: Value
}

/**
 * A Kolmogorov-Arnold network: layers of learnable univariate functions on the edges, summed at the nodes
 * (`KAN([2, 5, 1])` is the paper's example for `exp(sin(pi x) + y^2)`). Parameters are the layers' under `0`, `1`, ...;
 * the buffers are the layers' ranges, fixed knots and edge masks.
 *
 * @concept splines
 * @example
 * ```ts
 * import { KAN } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new KAN([2, 5, 1], { grid: 5, degree: 3, key: key(0) })
 * net.forward([[0.1, 0.2], [0.3, -0.5]]).shape // => [2, 1]
 * net.edgeCurve(0, 1, 3, [-1, 0, 1]).shape // => [3]
 * ```
 */
export class KAN extends Module {
  readonly kind = 'KAN'
  readonly spec: KanSpec
  readonly layers: readonly KanLayer[]

  constructor(widths: readonly number[], options: KanOptions = {}) {
    super()
    const { range, sample, layers, key, ...structure } = options
    this.spec = kanSpec(widths, structure)
    const root = initKey('KAN', key)
    const placement = (
      l: number,
    ): { range?: readonly [number, number] | readonly (readonly [number, number])[]; sample?: TensorLike } => {
      const given = layers?.[l]
      if (given !== undefined && (given.range !== undefined || given.sample !== undefined))
        return {
          ...(given.range === undefined ? {} : { range: given.range }),
          ...(given.sample === undefined ? {} : { sample: given.sample }),
        }
      if (range !== undefined) return { range }
      return l === 0 && sample !== undefined ? { sample } : {}
    }
    this.layers = this.spec.widths.slice(0, -1).map(
      (w, l) =>
        new KanLayer({
          inFeatures: w,
          outFeatures: this.spec.widths[l + 1] as number,
          grid: this.spec.grid,
          degree: this.spec.degree,
          basis: this.spec.basis,
          base: this.spec.base,
          knots: this.spec.knots,
          outside: this.spec.outside,
          bias: this.spec.bias,
          noise: this.spec.noise,
          scaleBaseMu: this.spec.scaleBaseMu,
          scaleBaseSigma: this.spec.scaleBaseSigma,
          scaleSpline: this.spec.scaleSpline,
          ...placement(l),
          key: foldIn(root, l),
        }),
    )
    const named = this.layers.map((layer, l) => [String(l), layer] as const)
    this.params = assemble(named.map(([n, c]) => [n, c.params] as const))
    this.buffers = assemble(named.map(([n, c]) => [n, c.buffers] as const))
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return this.layers.map((layer, l) => [String(l), layer] as const)
  }

  /** The widths `[n_in, ..., n_out]`. */
  get widths(): readonly number[] {
    return this.spec.widths
  }

  /** The state the network holds. */
  get state(): KanState {
    return { params: this.params, buffers: this.buffers }
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const y = this.layers.reduce(
      (h, layer, l) => layer.apply((params[String(l)] ?? {}) as Params, h, scope.child(String(l), l)),
      x,
    )
    scope.done()
    return y
  }

  /**
   * A forward pass that keeps what each layer saw and did: the layer inputs and the edge activations.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, n_in]`.
   * @param buffers - The buffers (default the network's).
   * @returns The inputs and activations of every layer and the output.
   */
  trace(params: Params, x: TensorLike, buffers?: Params): KanTrace {
    const b = buffers ?? this.buffers
    let h = asTensor(x) as Value
    const inputs: Value[] = []
    const activations: Value[] = []
    this.layers.forEach((layer, l) => {
      const p = (params[String(l)] ?? {}) as Params
      const lb = (b[String(l)] ?? {}) as Params
      inputs.push(h)
      const acts = layer.activations(p, h, lb)
      activations.push(acts)
      h = layer.apply(p, h, { buffers: lb, train: false })
    })
    return { inputs, activations, output: h }
  }

  /**
   * The average magnitude of every edge on a sample, `E_x |phi_{o,i}(x_i)|`, per layer `[out, in]`: the paper's L1 norm
   * of an activation function, and the quantity that sparsification, pruning and edge plots read.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, n_in]`.
   * @param buffers - The buffers (default the network's).
   * @returns One `[out, in]` array per layer.
   */
  edgeScales(params: Params, x: TensorLike, buffers?: Params): Value[] {
    return this.trace(params, x, buffers).activations.map((a) => transpose(mean(abs(a), 0)))
  }

  /**
   * The function on the edge from node `i` of layer `layer` to node `j` of the next layer, on a grid.
   *
   * @param layer - The layer (0 is the first).
   * @param i - The source node.
   * @param j - The target node.
   * @param grid - The points.
   * @param state - The parameters and buffers (default the network's).
   * @returns The values `phi_{j,i}(grid)`.
   */
  edgeCurve(layer: number, i: number, j: number, grid: ArrayLike<number>, state: Partial<KanState> = {}): Tensor {
    const l = this.layers[layer]
    if (l === undefined) throw new MdmlError('value', `KAN.edgeCurve: no layer ${layer}`)
    if (!(i >= 0 && i < l.inFeatures && j >= 0 && j < l.outFeatures))
      throw new MdmlError('value', `KAN.edgeCurve: no edge ${i} -> ${j} in layer ${layer}`)
    const params = (state.params ?? this.params)[String(layer)] as Params
    const buffers = (state.buffers ?? this.buffers)[String(layer)] as Params
    return l.edgeCurve(params, grid, i, j, buffers)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.spec.widths[this.spec.widths.length - 1] as number]
  }

  override extraRepr(): string {
    return `widths=[${this.spec.widths.join(', ')}], grid=${this.spec.grid}, degree=${this.spec.degree}`
  }
}
