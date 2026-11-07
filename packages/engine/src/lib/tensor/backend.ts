/**
 * The backend contract: the kernel vocabulary (elementwise, comparison, predicate and reduction kernels) that a backend
 * implements for tensors to run on it.
 */

import type { DType, Storage } from './dtype.ts'
import type { Shape } from './shape.ts'
import type { Tensor } from './tensor.ts'

/** Elementwise functions of one float tensor. */
export type UnaryKernel =
  | 'neg'
  | 'abs'
  | 'sign'
  | 'exp'
  | 'expm1'
  | 'log'
  | 'log1p'
  | 'log2'
  | 'log10'
  | 'sqrt'
  | 'sin'
  | 'cos'
  | 'tan'
  | 'asin'
  | 'acos'
  | 'atan'
  | 'sinh'
  | 'cosh'
  | 'tanh'
  | 'asinh'
  | 'acosh'
  | 'atanh'
  | 'sigmoid'
  | 'floor'
  | 'ceil'
  | 'round'
  | 'trunc'

/** Elementwise functions of two broadcast-compatible tensors of one dtype. */
export type BinaryKernel = 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'max' | 'min' | 'atan2' | 'floorDiv' | 'mod'

/** Elementwise comparisons; results are bool. */
export type CompareKernel = 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge'

/** Elementwise tests; results are bool. */
export type PredicateKernel = 'isnan' | 'isfinite' | 'isinf'

/** Reductions over axes. */
export type ReduceKernel = 'sum' | 'prod' | 'max' | 'min'

/**
 * The kernel vocabulary a backend implements. Primitives call these on concrete tensors; transformations never see them.
 * A new backend (WASM, WebGPU) implements this interface, and may fall back to the CPU backend for kernels it lacks.
 *
 * @concept tensor-backend
 */
export interface Backend {
  readonly name: string
  wrap<D extends DType>(dtype: D, shape: Shape, data: Storage<D>): Tensor<D>
  unary<D extends DType>(op: UnaryKernel, x: Tensor<D>): Tensor<D>
  predicate(op: PredicateKernel, x: Tensor): Tensor<'bool'>
  binary<D extends DType>(op: BinaryKernel, a: Tensor<D>, b: Tensor<D>): Tensor<D>
  compare(op: CompareKernel, a: Tensor, b: Tensor): Tensor<'bool'>
  map<D extends DType>(x: Tensor, fn: (v: number) => number, dtype: D): Tensor<D>
  map2<D extends DType>(a: Tensor, b: Tensor, fn: (x: number, y: number) => number, dtype: D): Tensor<D>
  reduce<D extends DType>(op: ReduceKernel, x: Tensor<D>, axes: readonly number[], keepDims: boolean): Tensor<D>
  argReduce(op: 'max' | 'min', x: Tensor, axis: number, keepDims: boolean): Tensor<'int32'>
  scan<D extends DType>(op: 'sum' | 'prod', x: Tensor<D>, axis: number, reverse: boolean): Tensor<D>
  matmul<D extends DType>(a: Tensor<D>, b: Tensor<D>): Tensor<D>
  transpose<D extends DType>(x: Tensor<D>, perm: readonly number[]): Tensor<D>
  broadcastTo<D extends DType>(x: Tensor<D>, shape: Shape): Tensor<D>
  slice<D extends DType>(
    x: Tensor<D>,
    starts: readonly number[],
    stops: readonly number[],
    steps: readonly number[],
  ): Tensor<D>
  sliceScatter<D extends DType>(
    values: Tensor<D>,
    shape: Shape,
    starts: readonly number[],
    steps: readonly number[],
  ): Tensor<D>
  concat<D extends DType>(xs: readonly Tensor<D>[], axis: number): Tensor<D>
  pad<D extends DType>(x: Tensor<D>, widths: readonly (readonly [number, number])[], value: number): Tensor<D>
  take<D extends DType>(x: Tensor<D>, indices: Int32Array, axis: number): Tensor<D>
  takeScatter<D extends DType>(updates: Tensor<D>, indices: Int32Array, axis: number, size: number): Tensor<D>
  takeAlongAxis<D extends DType>(x: Tensor<D>, indices: Tensor<'int32'>, axis: number): Tensor<D>
  putAlongAxisAdd<D extends DType>(updates: Tensor<D>, indices: Tensor<'int32'>, axis: number, shape: Shape): Tensor<D>
  where<D extends DType>(condition: Tensor<'bool'>, a: Tensor<D>, b: Tensor<D>): Tensor<D>
  cast<D extends DType>(x: Tensor, dtype: D): Tensor<D>
  argsort(x: Tensor, axis: number, descending: boolean): Tensor<'int32'>
}
