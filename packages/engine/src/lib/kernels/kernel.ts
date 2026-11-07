/**
 * The `Kernel` contract and kernel algebra: a positive semi-definite kernel with its hyperparameters as data,
 * `defineKernel`, sums, products, scaling and powers of kernels, kernels restricted to some input columns and automatic
 * relevance determination (one length scale per dimension), hyperparameter listing and the log-parameterised `theta`
 * vector, as scikit-learn and GPyTorch, and Gram matrices with their gradients.
 *
 * @index structure The Kernel contract and algebra: hyperparameters as data, sums, products, scaling.
 */

import { jacfwd } from '../autodiff/index.ts'
import { type Json, MdmlError } from '../base/index.ts'
import {
  add,
  concat,
  exp,
  eye,
  fromStorage,
  isTensor,
  isValue,
  mul,
  ones,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  sizeOf,
  sliceAxis,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  treeFlatten,
  treePaths,
  treeUnflatten,
  type Value,
  zeros,
} from '../tensor/index.ts'

/**
 * How a hyperparameter is optimised: `positive` ones are log-parameterised in {@link theta}, `real` ones enter as they
 * are, and `fixed` ones are held constant (left out of `theta`).
 */
export type Constraint = 'positive' | 'real' | 'fixed'

/**
 * Whether a kernel's Gram matrices are positive semi-definite: `always` for any points and any valid hyperparameters,
 * `conditional` when it holds only for some inputs or settings (the kernel says which in `psdDimension`).
 */
export type KernelPsd = 'always' | 'conditional'

/** Kernel hyperparameters as a pytree: records of values, nested records and arrays of records (one per part). */
export interface KernelParams {
  readonly [name: string]: Value | KernelParams | readonly KernelParams[]
}

/**
 * A kernel (positive semi-definite where `psd` says so) with its hyperparameters as data: `k(x, y)` evaluates it, `diag(x)` gives k(x_i, x_i)
 * without forming the matrix, and `withParams` rebuilds it from a pytree of the same structure, so gradients with
 * respect to hyperparameters are ordinary `grad` calls. Points are rows: `[d]` is one point, `[n, d]` are n points.
 */
export interface Kernel {
  /** The kernel's name ('rbf', 'matern', 'sum', ...). */
  readonly name: string
  /** Hyperparameters in natural units (length scales, variances), as a pytree of values. */
  readonly params: KernelParams
  /** The constraint of each hyperparameter, keyed by its pytree path (as `treePaths` names it). */
  readonly constraints: Readonly<Record<string, Constraint>>
  /** Whether k(x, y) depends on x - y only. */
  readonly stationary: boolean
  /**
   * Whether every Gram matrix of the kernel is positive semi-definite (`always`), or only in some cases
   * (`conditional`: the periodic and cosine kernels of several dimensions, the sigmoid kernel). Consumers that factorise
   * a Gram matrix can add jitter or warn for a `conditional` kernel.
   */
  readonly psd: KernelPsd
  /**
   * For a `conditional` kernel, the largest input dimension in which it is guaranteed positive semi-definite (1 for the
   * periodic and cosine kernels); absent when there is no such guarantee (the sigmoid kernel) and for `always`.
   */
  readonly psdDimension?: number
  /** Settings that are not hyperparameters (orders, degrees, exponents). */
  readonly config: Readonly<Record<string, Json>>
  /** The kernels this one combines (empty for a base kernel). */
  readonly parts: readonly Kernel[]
  /**
   * The cross-covariance between the rows of x `[m, d]` and y `[n, d]`, `[m, n]`. Without y it is the Gram matrix of
   * x, which differs only for white noise (whose cross-covariance between separate sets is zero, as scikit-learn).
   * A single point `[d]` on either side drops that axis.
   */
  k(x: TensorLike, y?: TensorLike): Value
  /** k(x_i, x_i) for each row `[m]` (a scalar for one point). */
  diag(x: TensorLike): Value
  /** The same kernel with new hyperparameters of the same pytree structure (possibly traced). */
  withParams(params: KernelParams): Kernel
}

/** The definition of a kernel from its matrix function, for {@link defineKernel}. */
export interface KernelSpec<P extends KernelParams> {
  readonly name: string
  readonly params: P
  /** Constraints by pytree path (default `positive` for every hyperparameter). */
  readonly constraints?: Readonly<Record<string, Constraint>>
  readonly stationary?: boolean
  /** Positive semi-definiteness (default `always`), see {@link Kernel.psd}. */
  readonly psd?: KernelPsd
  /** The largest dimension a `conditional` kernel is guaranteed positive semi-definite in, see {@link Kernel.psdDimension}. */
  readonly psdDimension?: number
  readonly config?: Readonly<Record<string, Json>>
  /** The parts of a combination, rebuilt from the hyperparameters. */
  readonly parts?: (params: P) => readonly Kernel[]
  /** The matrix between rows of x `[m, d]` and y `[n, d]` (y undefined: x with itself), `[m, n]`. */
  readonly matrix: (params: P, x: Value, y: Value | undefined, parts: readonly Kernel[]) => Value
  /** The diagonal `[m]` (default: the matrix of each row with itself). */
  readonly diag?: (params: P, x: Value, parts: readonly Kernel[]) => Value
}

/**
 * Points as rows `[n, d]`, remembering whether a single point `[d]` (or a scalar, a point in one dimension) was given:
 * the input convention of every kernel and feature map.
 *
 * @param x - Points `[n, d]`, one point `[d]` or a scalar.
 * @param name - The calling function, for the error message.
 * @returns The rows and whether the input was a single point.
 * @throws MdmlError `shape` for inputs of rank 3 or more.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { asRows } from 'mdml-engine/kernels'
 * asRows([1, 2, 3], 'example').rows.shape // => [1, 3]
 * asRows([[1], [2]], 'example').single // => false
 * ```
 */
export function asRows(x: TensorLike, name: string): { readonly rows: Value; readonly single: boolean } {
  const [v] = promoteFloat(x) as [Value]
  if (v.shape.length === 0) return { rows: reshape(v, [1, 1]), single: true }
  if (v.shape.length === 1) return { rows: reshape(v, [1, -1]), single: true }
  if (v.shape.length === 2) return { rows: v, single: false }
  throw new MdmlError('shape', `${name}: points must be [d] or [n, d], got [${v.shape.join(', ')}]`)
}

function structureOf(params: KernelParams): string {
  return JSON.stringify(treeFlatten(params).def)
}

/** Hyperparameters that are scales of a distance or a period: zero is not "switched off" but undefined (0 / 0 on the diagonal). */
const STRICTLY_POSITIVE: ReadonlySet<string> = new Set(['lengthScale', 'period', 'alpha', 'decay'])

/**
 * Resolve constraints: the given ones, `positive` by default, and `fixed` for a variance-like positive hyperparameter
 * set to 0. Concrete values must be finite and non-negative, and a length scale, period, alpha or decay strictly
 * positive.
 */
function resolveConstraints(name: string, params: KernelParams, given: Readonly<Record<string, Constraint>>) {
  const out: Record<string, Constraint> = {}
  for (const [path, leaf] of Object.entries(treePaths(params))) {
    let c = given[path] ?? 'positive'
    if (isTensor(leaf)) {
      const values = Array.from(leaf.data)
      if (values.some((v) => !Number.isFinite(v)))
        throw new MdmlError('domain', `${name}: hyperparameter ${path} must be finite`, { values })
      if (c === 'positive') {
        if (values.some((v) => v < 0))
          throw new MdmlError('domain', `${name}: hyperparameter ${path} must be non-negative`, { values })
        const zero = values.length > 0 && values.every((v) => v === 0)
        if (values.includes(0) && STRICTLY_POSITIVE.has(path.slice(path.lastIndexOf('.') + 1)))
          throw new MdmlError('domain', `${name}: hyperparameter ${path} must be positive, got 0`, { values })
        // log(0) is not finite, so a variance-like positive hyperparameter set to exactly zero is held fixed.
        if (zero) c = 'fixed'
      }
    }
    out[path] = c
  }
  return out
}

/**
 * Define a kernel from its matrix function: the result checks point shapes, handles single points, rebuilds itself
 * from new hyperparameters and supplies a diagonal. Every kernel in mdml-js is made this way, and so can new ones.
 *
 * @param spec - Name, hyperparameters, constraints, the matrix function and optionally a diagonal.
 * @returns The kernel.
 * @throws MdmlError `domain` for a negative or non-finite hyperparameter, or a zero length scale, period, alpha or decay;
 *   `shape` for bad points.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { defineKernel } from 'mdml-engine/kernels'
 * import { matmul, mul, swapaxes, tensor, toArray } from 'mdml-engine/tensor'
 * const scaledLinear = defineKernel({
 *   name: 'scaled-linear',
 *   params: { variance: tensor(2) },
 *   matrix: (p, x, y) => mul(p.variance, matmul(x, swapaxes(y ?? x, 0, 1))),
 * })
 * toArray(scaledLinear.k([[1, 2], [3, 4]], [[1, 0]])) // => [[2], [6]]
 * ```
 */
export function defineKernel<P extends KernelParams>(spec: KernelSpec<P>): Kernel {
  const params = spec.params
  const constraints = resolveConstraints(spec.name, params, spec.constraints ?? {})
  const parts = spec.parts === undefined ? [] : spec.parts(params)
  const structure = structureOf(params)
  const diag = (rows: Value): Value => {
    if (spec.diag !== undefined) return spec.diag(params, rows, parts)
    const n = rows.shape[0] ?? 0
    if (n === 0) return zeros([0])
    const each = Array.from({ length: n }, (_, i) => {
      const row = sliceAxis(rows, 0, i, i + 1)
      return reshape(spec.matrix(params, row, row, parts), [1])
    })
    return concat(each, 0)
  }
  return {
    name: spec.name,
    params,
    constraints,
    stationary: spec.stationary ?? false,
    psd: spec.psd ?? 'always',
    ...(spec.psd === 'conditional' && spec.psdDimension !== undefined ? { psdDimension: spec.psdDimension } : {}),
    config: spec.config ?? {},
    parts,
    k(x: TensorLike, y?: TensorLike): Value {
      const a = asRows(x, spec.name)
      if (y === undefined) {
        const out = spec.matrix(params, a.rows, undefined, parts)
        return a.single ? reshape(out, []) : out
      }
      const b = asRows(y, spec.name)
      if (a.rows.shape[1] !== b.rows.shape[1])
        throw new MdmlError('shape', `${spec.name}: points of dimension ${a.rows.shape[1]} and ${b.rows.shape[1]}`)
      const out = spec.matrix(params, a.rows, b.rows, parts)
      if (a.single && b.single) return reshape(out, [])
      return a.single || b.single ? reshape(out, [-1]) : out
    },
    diag(x: TensorLike): Value {
      const a = asRows(x, spec.name)
      const out = diag(a.rows)
      return a.single ? reshape(out, []) : out
    },
    withParams(next: KernelParams): Kernel {
      if (structureOf(next) !== structure)
        throw new MdmlError('value', `${spec.name}: withParams needs hyperparameters of the same structure`)
      return defineKernel({ ...spec, params: next as P, constraints })
    },
  }
}

/** Constraints of parts under a prefix: `terms[0].lengthScale`, or `base.lengthScale` for a single part. */
function prefixed(prefix: string, kernels: readonly Kernel[], indexed: boolean): Record<string, Constraint> {
  const out: Record<string, Constraint> = {}
  kernels.forEach((kernel, i) => {
    const head = indexed ? `${prefix}[${i}]` : prefix
    for (const [path, c] of Object.entries(kernel.constraints)) out[`${head}.${path}`] = c
  })
  return out
}

function termsOf(p: KernelParams): readonly KernelParams[] {
  return p['terms'] as readonly KernelParams[]
}

function baseOf(p: KernelParams): KernelParams {
  return p['base'] as KernelParams
}

/** The positive semi-definiteness of a sum or product of kernels: PSD in the dimensions where every part is. */
function combinedPsd(kernels: readonly Kernel[]): Pick<KernelSpec<KernelParams>, 'psd' | 'psdDimension'> {
  const weak = kernels.filter((k) => k.psd === 'conditional')
  if (weak.length === 0) return { psd: 'always' }
  const dims = weak.map((k) => k.psdDimension)
  return dims.some((d) => d === undefined)
    ? { psd: 'conditional' }
    : { psd: 'conditional', psdDimension: Math.min(...(dims as number[])) }
}

function combine(name: 'sum' | 'product', kernels: readonly Kernel[]): Kernel {
  if (kernels.length === 0) throw new MdmlError('value', `${name}Kernel needs at least one kernel`)
  const join = name === 'sum' ? add : mul
  const fold = (values: readonly Value[]): Value =>
    values.slice(1).reduce((acc: Value, v) => join(acc, v), values[0] as Value)
  return defineKernel({
    name,
    params: { terms: kernels.map((k) => k.params) },
    constraints: prefixed('terms', kernels, true),
    stationary: kernels.every((k) => k.stationary),
    ...combinedPsd(kernels),
    parts: (p) => kernels.map((k, i) => k.withParams(termsOf(p)[i] as KernelParams)),
    matrix: (_p, x, y, parts) => fold(parts.map((k) => k.k(x, y))),
    diag: (_p, x, parts) => fold(parts.map((k) => k.diag(x))),
  })
}

/**
 * The sum of kernels, k(x, y) = sum_i k_i(x, y) (scikit-learn Sum, GPyTorch AdditiveKernel): positive semi-definite
 * when every term is. Its hyperparameters are `{ terms: [...] }`.
 *
 * @param kernels - The terms (at least one).
 * @returns The sum kernel.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { constantKernel, linearKernel, sumKernel } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sumKernel(linearKernel(), constantKernel({ value: 1 })).k([[1, 2]], [[3, 4]])) // => [[12]]
 * ```
 */
export function sumKernel(...kernels: readonly Kernel[]): Kernel {
  return combine('sum', kernels)
}

/**
 * The product of kernels, k(x, y) = prod_i k_i(x, y) (scikit-learn Product, GPyTorch ProductKernel): positive
 * semi-definite when every factor is (the Schur product theorem). Its hyperparameters are `{ terms: [...] }`.
 *
 * @param kernels - The factors (at least one).
 * @returns The product kernel.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { constantKernel, linearKernel, productKernel } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(productKernel(linearKernel(), constantKernel({ value: 2 })).k([[1, 2]], [[3, 4]])) // => [[22]]
 * ```
 */
export function productKernel(...kernels: readonly Kernel[]): Kernel {
  return combine('product', kernels)
}

/**
 * A kernel scaled by an output variance, s k(x, y) (GPyTorch ScaleKernel; scikit-learn ConstantKernel * k).
 * Hyperparameters `{ outputScale, base }`.
 *
 * @param kernel - The base kernel.
 * @param outputScale - The variance s (default 1).
 * @returns The scaled kernel.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { rbfKernel, scaleKernel } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(scaleKernel(rbfKernel(), 3).k([0], [0])) // => 3
 * ```
 */
export function scaleKernel(kernel: Kernel, outputScale: TensorLike = 1): Kernel {
  const [s] = promoteFloat(outputScale) as [Value]
  return defineKernel({
    name: 'scale',
    params: { outputScale: s, base: kernel.params },
    constraints: { outputScale: 'positive', ...prefixed('base', [kernel], false) },
    stationary: kernel.stationary,
    ...combinedPsd([kernel]),
    parts: (p) => [kernel.withParams(baseOf(p))],
    matrix: (p, x, y, parts) => mul(p.outputScale as Value, (parts[0] as Kernel).k(x, y)),
    diag: (p, x, parts) => mul(p.outputScale as Value, (parts[0] as Kernel).diag(x)),
  })
}

/**
 * A kernel raised to a fixed power, k(x, y)^p (scikit-learn Exponentiation). For a positive integer p it is positive
 * semi-definite whenever k is; other powers keep that only for special kernels (such as infinitely divisible ones).
 * Hyperparameters `{ base }`.
 *
 * @param kernel - The base kernel.
 * @param exponent - The power p (a setting, not a hyperparameter).
 * @returns The exponentiated kernel.
 * @throws MdmlError `domain` for an exponent that is not finite.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { linearKernel, powerKernel } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(powerKernel(linearKernel(), 2).k([1, 2], [3, 1])) // => 25
 * ```
 */
export function powerKernel(kernel: Kernel, exponent: number): Kernel {
  if (!Number.isFinite(exponent))
    throw new MdmlError('domain', `powerKernel: the exponent must be finite, got ${exponent}`)
  return defineKernel({
    name: 'power',
    params: { base: kernel.params },
    constraints: prefixed('base', [kernel], false),
    stationary: kernel.stationary,
    // A non-negative integer power of a PSD kernel is PSD (Schur); other powers are only for special kernels.
    ...(Number.isInteger(exponent) && exponent >= 0 ? combinedPsd([kernel]) : { psd: 'conditional' as const }),
    config: { exponent },
    parts: (p) => [kernel.withParams(baseOf(p))],
    matrix: (_p, x, y, parts) => pow((parts[0] as Kernel).k(x, y), exponent),
    diag: (_p, x, parts) => pow((parts[0] as Kernel).diag(x), exponent),
  })
}

/**
 * A kernel that looks only at some input columns (GPyTorch active_dims), for additive models over feature groups:
 * `sumKernel(onDimensions(rbfKernel(), [0]), onDimensions(periodicKernel(), [1]))`. The hyperparameters are those of
 * the base.
 *
 * @param kernel - The base kernel (of the selected dimension).
 * @param dims - Column indices.
 * @returns The restricted kernel.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { linearKernel, onDimensions } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(onDimensions(linearKernel(), [1]).k([5, 2], [7, 3])) // => 6
 * ```
 */
export function onDimensions(kernel: Kernel, dims: readonly number[]): Kernel {
  const index = fromStorage(Int32Array.from(dims), [dims.length], 'int32')
  const pick = (x: Value): Value => take(x, index, 1)
  return defineKernel({
    name: 'dimensions',
    params: kernel.params,
    constraints: kernel.constraints,
    stationary: kernel.stationary,
    // The base sees only the selected columns, so it matters whether it is PSD in that many dimensions.
    ...(kernel.psd === 'always' || (kernel.psdDimension ?? 0) >= dims.length
      ? { psd: 'always' as const }
      : { psd: 'conditional' as const }),
    config: { dims: [...dims] },
    parts: (p) => [kernel.withParams(p)],
    matrix: (_p, x, y, parts) => (parts[0] as Kernel).k(pick(x), y === undefined ? undefined : pick(y)),
    diag: (_p, x, parts) => (parts[0] as Kernel).diag(pick(x)),
  })
}

/**
 * Automatic relevance determination: every scalar `lengthScale` of a base kernel in the expression becomes one length
 * scale per input dimension, initialised to the scalar, so that optimising them weighs each feature (GPyTorch
 * ard_num_dims; scikit-learn anisotropic length scales). Kernels restricted with {@link onDimensions} get one per
 * selected column; isotropic kernels (periodic ones) and kernels without length scales are left as they are.
 *
 * @param kernel - A kernel or kernel expression.
 * @param dim - The input dimension.
 * @returns The same kernel with `[dim]` length scales.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { ard, hyperparameters, rbfKernel } from 'mdml-engine/kernels'
 * hyperparameters(ard(rbfKernel({ lengthScale: 2 }), 3))[0]?.shape // => [3]
 * ```
 */
export function ard(kernel: Kernel, dim: number): Kernel {
  const widen = (k: Kernel, p: KernelParams, d: number): KernelParams => {
    const part = (i: number): Kernel => k.parts[i] as Kernel
    switch (k.name) {
      case 'sum':
      case 'product':
        return { terms: termsOf(p).map((t, i) => widen(part(i), t, d)) }
      case 'scale':
      case 'power':
        return { ...p, base: widen(part(0), baseOf(p), d) }
      case 'dimensions':
        return widen(part(0), p, (k.config['dims'] as readonly number[]).length)
      default: {
        const l = p['lengthScale']
        if (k.parts.length > 0 || k.config['isotropic'] === true || !isValue(l) || l.shape.length !== 0) return p
        return { ...p, lengthScale: mul(ones([d]), l) }
      }
    }
  }
  return kernel.withParams(widen(kernel, kernel.params, dim))
}

/**
 * Hold hyperparameters constant (scikit-learn's `bounds="fixed"`): they keep their values and leave {@link theta}.
 *
 * @param kernel - The kernel.
 * @param paths - Pytree paths of the hyperparameters to fix (as in `kernel.constraints`).
 * @returns The kernel with those hyperparameters fixed, also after `withParams`.
 * @throws MdmlError `value` for an unknown path.
 * @concept kernel-hyperparameters
 * @example
 * ```ts
 * import { fixHyperparameters, rationalQuadraticKernel, theta } from 'mdml-engine/kernels'
 * theta(fixHyperparameters(rationalQuadraticKernel(), ['alpha'])).shape // => [1]
 * ```
 */
export function fixHyperparameters(kernel: Kernel, paths: readonly string[]): Kernel {
  for (const path of paths) {
    if (!(path in kernel.constraints)) throw new MdmlError('value', `fixHyperparameters: no hyperparameter ${path}`)
  }
  const constraints = { ...kernel.constraints }
  for (const path of paths) constraints[path] = 'fixed'
  return {
    ...kernel,
    constraints,
    withParams: (next: KernelParams): Kernel => fixHyperparameters(kernel.withParams(next), paths),
  }
}

/** A hyperparameter of a kernel, as {@link hyperparameters} lists them. */
export interface Hyperparameter {
  /** Pytree path ('lengthScale', 'terms[1].noiseLevel'). */
  readonly name: string
  readonly constraint: Constraint
  readonly shape: readonly number[]
  /** Current values (natural units), flattened. */
  readonly value: readonly number[]
  /** Position in {@link theta} (meaningless for a fixed one). */
  readonly offset: number
  /** Number of entries in theta (0 when fixed). */
  readonly size: number
}

/**
 * The hyperparameters of a kernel in a fixed order (sorted pytree paths, parts in order), with their constraints and
 * their place in {@link theta}.
 *
 * @param kernel - The kernel.
 * @returns One entry per hyperparameter leaf.
 * @concept kernel-hyperparameters
 * @example
 * ```ts
 * import { hyperparameters, rationalQuadraticKernel } from 'mdml-engine/kernels'
 * hyperparameters(rationalQuadraticKernel()).map((h) => h.name) // => ['alpha', 'lengthScale']
 * ```
 */
export function hyperparameters(kernel: Kernel): Hyperparameter[] {
  let offset = 0
  return Object.entries(treePaths(kernel.params)).map(([name, leaf]) => {
    const constraint = kernel.constraints[name] ?? 'positive'
    const size = constraint === 'fixed' ? 0 : sizeOf(leaf.shape)
    const h = { name, constraint, shape: [...leaf.shape], value: toFlat(primalOf(leaf)), offset, size }
    offset += size
    return h
  })
}

/**
 * The free hyperparameters as one flat vector, positive ones as logarithms (scikit-learn's `theta`): the coordinates
 * in which marginal likelihoods are optimised without constraints.
 *
 * @param kernel - The kernel.
 * @returns theta `[p]`.
 * @concept kernel-hyperparameters
 * @example
 * ```ts
 * import { rbfKernel, theta } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(theta(rbfKernel({ lengthScale: Math.E }))) // => [1]
 * ```
 */
export function theta(kernel: Kernel): Tensor<'float64'> {
  const out: number[] = []
  for (const h of hyperparameters(kernel)) {
    if (h.constraint === 'fixed') continue
    for (const v of h.value) out.push(h.constraint === 'positive' ? Math.log(v) : v)
  }
  return fromStorage(Float64Array.from(out), [out.length], 'float64')
}

/**
 * The kernel at a point of {@link theta} coordinates (positive hyperparameters are exp(theta)). Differentiable in
 * theta, so `grad((t) => objective(withTheta(kernel, t)))(theta(kernel))` gives log-scale gradients.
 *
 * @param kernel - The kernel.
 * @param t - theta `[p]` (possibly traced).
 * @returns The kernel with those hyperparameters.
 * @throws MdmlError `shape` when t has the wrong length.
 * @concept kernel-hyperparameters
 * @example
 * ```ts
 * import { hyperparameters, rbfKernel, withTheta } from 'mdml-engine/kernels'
 * hyperparameters(withTheta(rbfKernel(), [1]))[0]?.value[0] === Math.E // => true
 * ```
 */
export function withTheta(kernel: Kernel, t: TensorLike): Kernel {
  const [v] = promoteFloat(t) as [Value]
  const { leaves, def } = treeFlatten(kernel.params)
  const names = Object.keys(treePaths(kernel.params))
  let offset = 0
  const next = leaves.map((leaf, i) => {
    const constraint = kernel.constraints[names[i] ?? ''] ?? 'positive'
    if (constraint === 'fixed') return leaf
    const size = sizeOf(leaf.shape)
    if (offset + size > (v.shape[0] ?? 0)) throw new MdmlError('shape', 'withTheta: theta is too short')
    const slice = reshape(sliceAxis(v, 0, offset, offset + size), leaf.shape)
    offset += size
    return constraint === 'positive' ? exp(slice) : slice
  })
  if (v.shape.length !== 1 || offset !== v.shape[0])
    throw new MdmlError('shape', `withTheta: theta must have ${offset} entries, got [${v.shape.join(', ')}]`)
  return kernel.withParams(treeUnflatten(def, next) as KernelParams)
}

/** Options of {@link gram}. */
export interface GramOptions {
  /** Added to the diagonal, for a stable Cholesky factorisation (default 0). */
  readonly jitter?: number
}

/**
 * The Gram matrix K_ij = k(x_i, x_j) of a set of points.
 *
 * @param kernel - The kernel.
 * @param x - Points `[n, d]` (or one point `[d]`).
 * @param options - Diagonal jitter.
 * @returns `[n, n]` (`[1, 1]` for a single point `[d]`, `[0, 0]` for no points), differentiable in the points and the
 *   hyperparameters.
 * @throws MdmlError `value` for a jitter that is negative or not finite.
 * @concept kernel-functions
 * @index function Gram matrix of a kernel over a set of points.
 * @example
 * ```ts
 * import { gram, linearKernel } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gram(linearKernel(), [[1, 0], [1, 1]], { jitter: 0.5 })) // => [[1.5, 1], [1, 2.5]]
 * ```
 */
export function gram(kernel: Kernel, x: TensorLike, options: GramOptions = {}): Value {
  const jitter = options.jitter ?? 0
  if (!Number.isFinite(jitter) || jitter < 0)
    throw new MdmlError('value', `gram: the jitter must be finite and non-negative, got ${jitter}`)
  // A single point [d] is one row: the matrix is [1, 1], not the scalar `kernel.k(x)` drops the axes to.
  const k = kernel.k(asRows(x, 'gram').rows)
  return jitter === 0 ? k : add(k, mul(eye(k.shape[0] ?? 0), jitter))
}

/**
 * The cross-covariance K_ij = k(x_i, y_j) between two sets of points.
 *
 * @param kernel - The kernel.
 * @param x - Points `[m, d]`.
 * @param y - Points `[n, d]`.
 * @returns `[m, n]`.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { crossGram, linearKernel } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(crossGram(linearKernel(), [[1, 2]], [[1, 0], [0, 1], [1, 1]])) // => [[1, 2, 3]]
 * ```
 */
export function crossGram(kernel: Kernel, x: TensorLike, y: TensorLike): Value {
  return kernel.k(x, y)
}

/** A Gram matrix with its derivatives in theta, as {@link gramGradient} returns it. */
export interface GramGradient {
  /** K `[n, n]`. */
  readonly gram: Value
  /** dK / dtheta `[n, n, p]`, in the order of {@link theta}. */
  readonly gradient: Value
}

/**
 * The Gram matrix and its gradient with respect to the log-parameterised hyperparameters (scikit-learn's
 * `kernel(X, eval_gradient=True)`), by forward-mode differentiation through the kernel.
 *
 * @param kernel - The kernel.
 * @param x - Points `[n, d]`.
 * @returns K and dK/dtheta.
 * @concept kernel-hyperparameters
 * @example
 * ```ts
 * import { gramGradient, rbfKernel } from 'mdml-engine/kernels'
 * gramGradient(rbfKernel(), [[0], [1], [3]]).gradient.shape // => [3, 3, 1]
 * ```
 */
export function gramGradient(kernel: Kernel, x: TensorLike): GramGradient {
  const { rows } = asRows(x, 'gramGradient')
  const n = rows.shape[0] ?? 0
  const gramMatrix = kernel.k(rows)
  const t = theta(kernel)
  const p = t.shape[0] ?? 0
  if (p === 0) return { gram: gramMatrix, gradient: zeros([n, n, 0]) }
  const jac = jacfwd((v: Value) => reshape(withTheta(kernel, v).k(rows), [-1]))(t)
  return { gram: gramMatrix, gradient: reshape(jac, [n, n, p]) }
}

function formatNumber(v: number): string {
  return String(Number(v.toPrecision(4)))
}

/**
 * A readable formula for a kernel, in the manner of scikit-learn's kernel repr: `2 * rbf(lengthScale=1.5) +
 * white(noiseLevel=0.1)`.
 *
 * @param kernel - The kernel.
 * @returns The description.
 * @concept kernel-algebra
 * @example
 * ```ts
 * import { formatKernel, rbfKernel, scaleKernel, sumKernel, whiteNoiseKernel } from 'mdml-engine/kernels'
 * formatKernel(sumKernel(scaleKernel(rbfKernel({ lengthScale: 1.5 }), 2), whiteNoiseKernel({ noiseLevel: 0.1 })))
 * // => '2 * rbf(lengthScale=1.5) + white(noiseLevel=0.1)'
 * ```
 */
export function formatKernel(kernel: Kernel): string {
  const parts = kernel.parts.map(formatKernel)
  switch (kernel.name) {
    case 'sum':
      return parts.join(' + ')
    case 'product':
      return kernel.parts.map((p, i) => (p.name === 'sum' ? `(${parts[i]})` : parts[i])).join(' * ')
    case 'scale': {
      const s = toFlat(primalOf(kernel.params['outputScale'] as Value))[0] ?? Number.NaN
      const inner = kernel.parts[0]?.name === 'sum' ? `(${parts[0]})` : (parts[0] ?? '')
      return `${formatNumber(s)} * ${inner}`
    }
    case 'power': {
      const inner = (kernel.parts[0]?.parts.length ?? 0) > 0 ? `(${parts[0]})` : (parts[0] ?? '')
      return `${inner} ** ${String(kernel.config['exponent'])}`
    }
    case 'dimensions': {
      const inner = (kernel.parts[0]?.parts.length ?? 0) > 0 ? `(${parts[0]})` : (parts[0] ?? '')
      return `${inner}@[${(kernel.config['dims'] as readonly number[]).join(', ')}]`
    }
    default: {
      const args = Object.entries(treePaths(kernel.params)).map(([name, leaf]) => {
        const values = toFlat(primalOf(leaf)).map(formatNumber)
        return `${name}=${leaf.shape.length === 0 ? values[0] : `[${values.join(', ')}]`}`
      })
      const settings = Object.entries(kernel.config)
        .filter(([key]) => key !== 'isotropic')
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      return `${kernel.name}(${[...args, ...settings].join(', ')})`
    }
  }
}
