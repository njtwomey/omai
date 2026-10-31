/**
 * Receptive-field analysis for stacks of convolution and pooling layers: the theoretical receptive field (its size, the
 * spacing of its centres, where the first one sits) computed layer by layer from kernels, strides, dilations and
 * paddings, from specifications or straight from modules, and the effective receptive field, the magnitude of the
 * gradient of one output unit with respect to the input, which is concentrated in the middle and smaller than the
 * theoretical one.
 */

import { grad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import {
  abs,
  asTensor,
  div,
  reshape,
  sizeOf,
  sum,
  type Tensor,
  type TensorLike,
  take,
  type Value,
} from '../tensor/index.ts'
import type { Module } from './module/module.ts'

/** One layer's geometry along each spatial axis (a number applies to every axis). */
export interface ConvLayerSpec {
  readonly kernel: number | readonly number[]
  readonly stride?: number | readonly number[]
  readonly dilation?: number | readonly number[]
  /** Symmetric padding in each axis, or `same`. */
  readonly padding?: number | readonly number[] | 'same'
}

/** The receptive field at one level of a stack, per spatial axis. */
export interface ReceptiveFieldLevel {
  /** Input units that one unit at this level depends on. */
  readonly size: readonly number[]
  /** Input distance between neighbouring units at this level (the product of the strides so far). */
  readonly jump: readonly number[]
  /** Input coordinate of the centre of the first unit's receptive field (negative when padding reaches outside). */
  readonly start: readonly number[]
}

const perAxis = (v: number | readonly number[] | undefined, d: number, fallback: number): number[] =>
  v === undefined ? new Array<number>(d).fill(fallback) : typeof v === 'number' ? new Array<number>(d).fill(v) : [...v]

/**
 * The theoretical receptive field after each layer of a stack (Araujo et al. 2019): with effective kernel
 * `k' = d (k - 1) + 1`, the size grows by `(k' - 1) j` where `j` is the jump (the product of the strides so far), the
 * jump multiplies by the stride, and the first centre moves by `((k' - 1) / 2 - p) j`.
 *
 * @param layers - The layers, input first.
 * @param dims - The number of spatial axes (default: the longest array among the specifications, or 1).
 * @returns One level for the input (size 1, jump 1, start 0) then one per layer.
 * @reference Araujo, Norris and Sim (2019), Computing receptive fields of convolutional neural networks. Distill.
 * @example
 * ```ts
 * import { receptiveField } from 'mdml-engine/nn'
 * receptiveField([{ kernel: 3 }, { kernel: 3, stride: 2 }, { kernel: 3 }]).at(-1)?.size // => [9]
 * ```
 */
export function receptiveField(layers: readonly ConvLayerSpec[], dims?: number): ReceptiveFieldLevel[] {
  const d =
    dims ??
    Math.max(
      1,
      ...layers.flatMap((l) =>
        [l.kernel, l.stride, l.dilation, l.padding].map((v) => (Array.isArray(v) ? v.length : 1)),
      ),
    )
  let size: number[] = new Array<number>(d).fill(1)
  let jump: number[] = new Array<number>(d).fill(1)
  let start: number[] = new Array<number>(d).fill(0)
  const levels: ReceptiveFieldLevel[] = [{ size, jump, start }]
  for (const layer of layers) {
    const k = perAxis(layer.kernel, d, 1)
    const s = perAxis(layer.stride, d, 1)
    const dil = perAxis(layer.dilation, d, 1)
    const keff = k.map((v, i) => (dil[i] as number) * (v - 1) + 1)
    const p =
      layer.padding === 'same'
        ? keff.map((v) => (v - 1) / 2)
        : perAxis(layer.padding as number | readonly number[] | undefined, d, 0)
    const j = jump
    size = size.map((r, i) => r + ((keff[i] as number) - 1) * (j[i] as number))
    start = start.map((a, i) => a + (((keff[i] as number) - 1) / 2 - (p[i] as number)) * (j[i] as number))
    jump = jump.map((v, i) => v * (s[i] as number))
    levels.push({ size, jump, start })
  }
  return levels
}

/**
 * The size of the theoretical receptive field of a stack along one axis.
 *
 * @param layers - The layers, input first.
 * @param axis - The spatial axis (default 0).
 * @returns The number of input units one output unit depends on.
 * @example
 * ```ts
 * import { receptiveFieldSize } from 'mdml-engine/nn'
 * receptiveFieldSize([1, 2, 4, 8].map((dilation) => ({ kernel: 2, dilation }))) // => 16
 * ```
 */
export function receptiveFieldSize(layers: readonly ConvLayerSpec[], axis = 0): number {
  return (receptiveField(layers).at(-1)?.size[axis] ?? 1) as number
}

interface Geometry {
  readonly kernelSize: readonly number[]
  readonly stride: readonly number[]
  readonly dilation?: readonly number[]
  readonly padding?: readonly number[] | 'same' | 'valid'
}

const geometry = (m: unknown): Geometry | undefined => {
  const g = m as Partial<Geometry>
  return g.kernelSize !== undefined && g.stride !== undefined ? (g as Geometry) : undefined
}

/**
 * The layer specifications of a module: a convolution or pooling layer is one, a container is the concatenation of its
 * children's, and layers that do not move information between positions (activations, norms, dropout) are skipped.
 * Transposed convolutions and layers that mix all positions (attention, recurrent, linear on a flattened input) are
 * not supported and throw.
 *
 * @param module - The module.
 * @returns The specifications, input first.
 * @example
 * ```ts
 * import { layerSpecs, Conv1d, Sequential, ReLU } from 'mdml-engine/nn'
 * layerSpecs(new Sequential(new Conv1d(1, 2, 3, { dilation: 2 }), new ReLU())) // => [{ kernel: [3], stride: [1], dilation: [2], padding: [0] }]
 * ```
 */
export function layerSpecs(module: Module<never, unknown>): ConvLayerSpec[] {
  const kind = module.kind
  if (/Transpose/.test(kind)) throw new MdmlError('value', `layerSpecs: ${kind} is not supported`)
  const g = geometry(module)
  if (g !== undefined) {
    return [
      {
        kernel: g.kernelSize,
        stride: g.stride,
        ...(g.dilation === undefined ? {} : { dilation: g.dilation }),
        ...(g.padding === undefined ? {} : { padding: g.padding === 'valid' ? 0 : g.padding }),
      },
    ]
  }
  const children = module.children()
  if (children.length > 0) return children.flatMap(([, child]) => layerSpecs(child))
  if (/Attention|LSTM|GRU|RNN|Linear|Embedding/.test(kind))
    throw new MdmlError('value', `layerSpecs: ${kind} has no finite receptive field`)
  return []
}

/**
 * The receptive field after each layer of a convolutional module (see {@link layerSpecs}).
 *
 * @param module - A convolution, pooling or a container of them.
 * @returns The levels, as {@link receptiveField}.
 * @example
 * ```ts
 * import { receptiveFieldOf, Conv2d, Sequential } from 'mdml-engine/nn'
 * receptiveFieldOf(new Sequential(new Conv2d(1, 4, 3), new Conv2d(4, 4, 3))).at(-1)?.size // => [5, 5]
 * ```
 */
export function receptiveFieldOf(module: Module<never, unknown>): ReceptiveFieldLevel[] {
  return receptiveField(layerSpecs(module))
}

/**
 * The effective receptive field of one output unit: the absolute gradient of that unit with respect to the input,
 * normalised to sum to one (Luo et al. 2016). Its mass concentrates in the middle and decays like a Gaussian, so it
 * is smaller than the theoretical field.
 *
 * @param f - The network, input to output.
 * @param input - A reference input (the gradient of a linear network does not depend on it).
 * @param unit - The flat index of the output unit into `f(input)`.
 * @returns The normalised absolute gradient, the shape of `input`.
 * @reference Luo, Li, Urtasun and Zemel (2016), Understanding the effective receptive field in deep convolutional neural networks.
 * @example
 * ```ts
 * import { effectiveReceptiveField } from 'mdml-engine/nn'
 * import { mul, sum, toArray } from 'mdml-engine/tensor'
 * toArray(effectiveReceptiveField((x) => mul(x, 2), [1, 2], 0)) // => [1, 0]
 * ```
 */
export function effectiveReceptiveField(f: (x: Value) => Value, input: TensorLike, unit: number): Value {
  const x = asTensor(input) as Tensor
  if (sizeOf(f(x).shape) <= unit) throw new MdmlError('value', `effectiveReceptiveField: no output unit ${unit}`)
  const g = grad((v: Value) => take(reshape(f(v), [-1]), unit, 0))(x) as Value
  const a = abs(g)
  return div(a, sum(a))
}
