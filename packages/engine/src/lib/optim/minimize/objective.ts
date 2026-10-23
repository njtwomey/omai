import { hvp as hessianVector, jacfwd, valueAndGrad } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { idamax } from '../../numerics/linalg/index.ts'
import {
  asTensor,
  fromStorage,
  isValue,
  item,
  primalOf,
  reshape,
  sizeOf,
  sliceAxis,
  type TensorLike,
  toFlat,
  treeFlatten,
  treeUnflatten,
  type Value,
} from '../../tensor/index.ts'
import type { RunOptions, RunStatus } from '../../trace/index.ts'
import type { Tree } from '../transform.ts'

/** An objective: a scalar function of a parameter tree (a vector, a matrix or nested records of tensors). */
export type Objective = (params: Tree) => Value

/** Options every vector method accepts: an iteration cap, the gradient tolerance and the run options. */
export interface VectorMethodOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The iteration cap (scipy's maxiter); the default depends on the method. */
  readonly maxIterations?: number
  /** Stop once the infinity norm of the gradient (or projected gradient) falls to this (scipy's gtol). */
  readonly gradTol?: number
}

/** A minimisation method as a function value: `minimize(f, x0, { method: bfgs })` calls `bfgs(f, x0, options)`. */
export type MinimizeMethod<O> = (f: Objective, x0: Parameters, options?: O) => MinimizeResult

/**
 * An objective of any parameter tree seen as a function of one flat float64 vector, the form the vector methods
 * (line searches, quasi-Newton, Newton, trust regions, derivative-free and global methods) work in. Gradients,
 * Hessian-vector products and Hessians come from automatic differentiation of the original objective.
 */
export interface FlatObjective {
  /** The number of parameters. */
  readonly n: number
  /** The starting point, flattened. */
  readonly x0: Float64Array
  /** Rebuild a parameter tree (concrete tensors) from a flat vector. */
  readonly unravel: (x: ArrayLike<number>) => Tree
  /** The objective of a flat value (differentiable: usable inside transformations). */
  readonly ofValue: (x: Value) => Value
  /** f(x). */
  value(x: ArrayLike<number>): number
  /** f(x) and its gradient. */
  valueAndGrad(x: ArrayLike<number>): { readonly value: number; readonly grad: Float64Array }
  /** H(x) v, by forward-over-reverse differentiation. */
  hvp(x: ArrayLike<number>, v: ArrayLike<number>): Float64Array
  /** The Hessian, n x n row-major. */
  hessian(x: ArrayLike<number>): Float64Array
}

/** Parameters as given by a caller: a parameter tree, or anything tensor-like (numbers, nested arrays, typed arrays). */
export type Parameters = Tree | TensorLike

const isNumeric = (x: unknown): boolean =>
  typeof x === 'number' || ArrayBuffer.isView(x) || (Array.isArray(x) && x.length > 0 && x.every(isNumeric))

/** Convert numeric leaves (numbers, typed arrays, nested number arrays) of a parameter tree to tensors. */
function asTree(x0: unknown): Tree {
  if (isValue(x0)) return x0
  if (isNumeric(x0)) return asTensor(x0 as TensorLike)
  if (Array.isArray(x0)) return x0.map(asTree)
  if (typeof x0 === 'object' && x0 !== null) {
    const out: Record<string, Tree> = {}
    for (const [k, v] of Object.entries(x0)) out[k] = asTree(v)
    return out
  }
  throw new WhiteboxError('value', `parameters must be numbers, arrays or tensors, got ${typeof x0}`)
}

const vectorOf = (x: ArrayLike<number>): Value => fromStorage(Float64Array.from(x), [x.length], 'float64')

const flatOf = (v: Value): Float64Array => Float64Array.from(toFlat(primalOf(v)))

/**
 * View an objective of a parameter tree as a function of one flat vector, with derivatives by autodiff.
 *
 * @param f - The objective, a scalar function of a tree.
 * @param x0 - The starting parameters (a tree, or anything tensor-like such as `[1, 2]`).
 * @returns The flat objective.
 * @throws WhiteboxError with code `shape` when `f` does not return a single value.
 * @concept optimisation
 * @example
 * ```ts
 * import { flatObjective } from 'whitebox-engine/optim'
 * import { add, mul, square, sum, type Value } from 'whitebox-engine/tensor'
 * const obj = flatObjective((p) => {
 *   const { a, b } = p as { a: Value; b: Value }
 *   return add(sum(square(a)), mul(b, 3))
 * }, { a: [1, 2], b: 0 })
 * obj.n // => 3
 * Array.from(obj.valueAndGrad([1, 2, 1]).grad) // => [2, 4, 3]
 * ```
 */
export function flatObjective(f: Objective, x0: Parameters): FlatObjective {
  const raw = treeFlatten(asTree(x0))
  const leaves = raw.leaves.map((l) => asTensor(l))
  const shapes = leaves.map((l) => l.shape)
  const sizes = shapes.map((s) => sizeOf(s))
  const offsets = sizes.map((_, i) => sizes.slice(0, i).reduce((a, b) => a + b, 0))
  const n = sizes.reduce((a, b) => a + b, 0)
  const single = leaves.length === 1 && raw.def.kind === 'leaf'
  const start = new Float64Array(n)
  for (const [i, leaf] of leaves.entries()) start.set(toFlat(primalOf(leaf)), offsets[i] ?? 0)
  const toTree = (v: Value): Tree => {
    if (single) return reshape(v, shapes[0] ?? [])
    const parts = shapes.map((s, i) =>
      reshape(sliceAxis(v, 0, offsets[i] ?? 0, (offsets[i] ?? 0) + (sizes[i] ?? 0)), s),
    )
    return treeUnflatten(raw.def, parts)
  }
  const ofValue = (v: Value): Value => {
    const out = f(toTree(v))
    if (sizeOf(out.shape) !== 1)
      throw new WhiteboxError('shape', `objective must return a single value, got shape [${out.shape.join(', ')}]`)
    return reshape(out, [])
  }
  const vg = valueAndGrad(ofValue)
  return {
    n,
    x0: start,
    unravel: (x) => toTree(vectorOf(x)),
    ofValue,
    value: (x) => item(primalOf(ofValue(vectorOf(x)))),
    valueAndGrad: (x) => {
      const r = vg(vectorOf(x))
      return { value: item(r.value), grad: flatOf(r.grad) }
    },
    hvp: (x, v) => flatOf(hessianVector(ofValue, vectorOf(x), vectorOf(v))),
    hessian: (x) => flatOf(jacfwd(vg as unknown as (x: Value) => Value)(vectorOf(x))),
  }
}

/** Result of a minimisation (scipy's OptimizeResult). */
export interface MinimizeResult {
  /** The solution, with the structure of the starting parameters. */
  readonly x: Tree
  /** The objective at the solution. */
  readonly fun: number
  /** The gradient at the solution (absent for derivative-free methods). */
  readonly grad?: Tree
  /** The gradient norm at the solution (the infinity norm for vector methods, NaN when no gradient is known). */
  readonly gradNorm: number
  /** Iterations taken. */
  readonly steps: number
  /** Objective evaluations. */
  readonly nfev: number
  /** Gradient evaluations. */
  readonly ngev: number
  /** Hessian (or Hessian-vector product) evaluations. */
  readonly nhev: number
  /** The run's status. */
  readonly status: RunStatus
  /** Whether a convergence test was met. */
  readonly converged: boolean
  /** Why the method stopped. */
  readonly message: string
}

/**
 * The infinity norm of a vector (its largest absolute entry): scipy's gradient test for the vector methods.
 *
 * @param x - The vector.
 * @returns max |x_i| (0 for an empty vector).
 * @example
 * ```ts
 * import { infNorm } from 'whitebox-engine/optim'
 * infNorm([1, -3, 2]) // => 3
 * ```
 */
export function infNorm(x: ArrayLike<number>): number {
  const i = idamax(x)
  return i < 0 ? 0 : Math.abs(x[i] ?? 0)
}
