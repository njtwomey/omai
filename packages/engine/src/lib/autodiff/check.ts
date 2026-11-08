/**
 * The conformance checks for derivative rules: central finite-difference gradients, the dot-product (adjoint) test that
 * forward and reverse mode agree, `gradCheck` for a scalar function, and `checkPrimitive`, the test every primitive in
 * omai passes (finite-difference vjp and jvp, their consistency, and no mutation of inputs).
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../random/index.ts'
import {
  asTensor,
  fromStorage,
  isTensor,
  mul,
  sizeOf,
  stack,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { grad, jvp, vjp } from './transforms.ts'
import { vmap } from './vmap.ts'

function concrete(x: Value, where: string): Tensor {
  if (!isTensor(x)) throw new OmaiError('not-differentiable', `${where} needs concrete values`)
  return x
}

function dotFlat(a: Value, b: Value): number {
  const x = toFlat(a)
  const y = toFlat(b)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (y[i] ?? 0)
  return s
}

function perturb(x: Tensor, i: number, h: number): Tensor {
  const data = Float64Array.from(x.data)
  data[i] = (data[i] ?? 0) + h
  return fromStorage(data, x.shape, 'float64')
}

/**
 * Central finite-difference gradient of a scalar function, the reference for checking derivative rules.
 *
 * @param f - A function of one value returning a single-element value.
 * @param x - The point.
 * @param eps - Relative step (default 1e-6, scaled by `max(1, |x_i|)`).
 * @returns The estimated gradient, with the shape of `x`.
 * @concept gradient-check
 * @example
 * ```ts
 * import { numericalGradient } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(numericalGradient((v: Value) => mul(mul(v, v), v), tensor([2]))) // => [12.000000]
 * ```
 */
export function numericalGradient(f: (x: Value) => Value, x: TensorLike, eps = 1e-6): Tensor<'float64'> {
  const t = concrete(asTensor(x, 'float64'), 'numericalGradient')
  const out = new Float64Array(t.data.length)
  for (let i = 0; i < out.length; i++) {
    const h = eps * Math.max(1, Math.abs(t.data[i] ?? 0))
    const plus = toFlat(f(perturb(t, i, h)))[0] ?? Number.NaN
    const minus = toFlat(f(perturb(t, i, -h)))[0] ?? Number.NaN
    out[i] = (plus - minus) / (2 * h)
  }
  return fromStorage(out, t.shape, 'float64')
}

/** The outcome of a derivative check. */
export interface CheckReport {
  readonly passed: boolean
  readonly maxAbsError: number
  readonly maxRelError: number
  readonly failures: readonly string[]
}

/** Tolerances for {@link checkPrimitive}. */
export interface CheckOptions {
  readonly eps?: number
  readonly rtol?: number
  readonly atol?: number
  readonly key?: Key
  /** Indices of inputs to differentiate with respect to (default: every float input). */
  readonly wrt?: readonly number[]
  /** Also check `vmap` against a loop over a small batch (default true). */
  readonly vmap?: boolean
  /**
   * `2` also checks the second derivative of the rules (default: first order only): the Hessian-vector product by
   * forward over reverse and by reverse over forward, each against a finite difference of the first-order gradient.
   * Use it on smooth primitives; a rule that calls a kernel or reads `primalOf` instead of operations fails it.
   */
  readonly order?: 1 | 2
}

// Collects comparisons against a reference with one tolerance, tracking the largest errors.
function comparer(rtol: number, atol: number) {
  const failures: string[] = []
  let maxAbs = 0
  let maxRel = 0
  const compare = (what: string, got: number, want: number): void => {
    const err = Math.abs(got - want)
    const rel = err / Math.max(Math.abs(want), 1e-12)
    if (Number.isFinite(err)) {
      maxAbs = Math.max(maxAbs, err)
      maxRel = Math.max(maxRel, rel)
    }
    if (!(err <= atol + rtol * Math.abs(want)) && !(Number.isNaN(got) && Number.isNaN(want))) {
      failures.push(`${what}: got ${got}, expected ${want}`)
    }
  }
  const report = (): CheckReport => ({
    passed: failures.length === 0,
    maxAbsError: maxAbs,
    maxRelError: maxRel,
    failures,
  })
  return { compare, failures, report }
}

// Scales of the examples in the vmap check: scaling keeps every structural property of the point (symmetry, definiteness,
// signs, zero patterns), so a valid input stays valid, while the examples still differ.
const BATCH_SCALES = [1, 1 + 1e-3, 1 - 2e-3]

function vmapAgainstLoop(
  f: (...xs: Value[]) => Value,
  xs: readonly Tensor[],
  wrt: readonly number[],
  axis: number,
  failures: string[],
): void {
  const examples = BATCH_SCALES.map((c) => xs.map((x, i) => (wrt.includes(i) ? mul(x, c) : x)))
  let want: Tensor
  try {
    want = concrete(stack(examples.map((args) => f(...args))), 'checkPrimitive')
  } catch {
    // The function cannot be evaluated at the scaled points, so there is nothing to compare against.
    return
  }
  const args = xs.map((x, i) =>
    wrt.includes(i)
      ? stack(
          examples.map((e) => e[i] as Value),
          axis,
        )
      : x,
  )
  const inAxes = xs.map((_, i) => (wrt.includes(i) ? axis : null))
  let got: Tensor
  try {
    got = concrete(vmap(f, inAxes)(...args), 'checkPrimitive')
  } catch (e) {
    failures.push(`vmap (batch axis ${axis}) threw: ${e instanceof Error ? e.message : String(e)}`)
    return
  }
  if (got.shape.join(',') !== want.shape.join(',')) {
    failures.push(`vmap (batch axis ${axis}): shape [${got.shape.join(', ')}], expected [${want.shape.join(', ')}]`)
    return
  }
  const g = toFlat(got)
  const w = toFlat(want)
  for (let k = 0; k < w.length; k++) {
    const a = g[k] ?? Number.NaN
    const b = w[k] ?? Number.NaN
    const close = Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)) || (Number.isNaN(a) && Number.isNaN(b)) || a === b
    if (!close) {
      failures.push(`vmap (batch axis ${axis})[${k}]: got ${a}, loop gives ${b}`)
      return
    }
  }
}

// The second-order check of one input: the Hessian-vector product `H t` of `<g, f>` by forward over reverse and by
// reverse over forward, against a five-point difference of the first-order gradient along `t`.
function secondOrder(
  fi: (v: Value) => Value,
  xi: Tensor,
  g: Tensor,
  direction: Tensor,
  i: number,
  eps: number,
  compare: (what: string, got: number, want: number) => void,
  failures: string[],
): void {
  const gradAt = (v: Value): Value => vjp(fi, v).pullback(g)
  // A five-point stencil with a wide step: the gradient of a special function carries about 1e-10 of error, which a
  // two-point difference at a small step would amplify into the second derivative.
  const h = 100 * eps
  // Each coordinate moves in proportion to its own size, so an input that mixes 0.5 and 200 is differenced well in both.
  const t = fromStorage(
    Float64Array.from(xi.data, (v, k) => Math.max(1, Math.abs(v)) * (direction.data[k] ?? 0)),
    xi.shape,
    'float64',
  )
  const at = (s: number): Value =>
    fromStorage(
      Float64Array.from(xi.data, (v, k) => v + s * h * (t.data[k] ?? 0)),
      xi.shape,
      'float64',
    )
  try {
    const grads = [-2, -1, 1, 2].map((s) => toFlat(gradAt(at(s))))
    const forwardOverReverse = toFlat(concrete(jvp(gradAt, xi as Value, t as Value).tangent, 'checkPrimitive'))
    const reverseOverForward = toFlat(
      concrete(grad((v: Value) => sum(mul(g, jvp(fi, v, t as Value).tangent)))(xi), 'checkPrimitive'),
    )
    for (let k = 0; k < xi.data.length; k++) {
      const [m2, m1, p1, p2] = grads.map((gr) => gr[k] ?? Number.NaN) as [number, number, number, number]
      const fd = (m2 - 8 * m1 + 8 * p1 - p2) / (12 * h)
      compare(`hvp (forward over reverse) input ${i}[${k}]`, forwardOverReverse[k] ?? Number.NaN, fd)
      compare(`hvp (reverse over forward) input ${i}[${k}]`, reverseOverForward[k] ?? Number.NaN, fd)
    }
  } catch (e) {
    failures.push(`second order input ${i} threw: ${e instanceof Error ? e.message : String(e)}`)
  }
}

/**
 * The dot-product (adjoint) test: for random directions `v` and `u`, the forward-mode product and the reverse-mode
 * product must satisfy `<u, J v> = <J^T u, v>`. It links the two modes without any finite differences, so it holds to
 * rounding error and catches a vjp that is not the transpose of its jvp.
 *
 * @param f - A function of one value.
 * @param x - The point (a concrete float value).
 * @param options - Tolerances and the key for the random directions.
 * @returns A report with the discrepancy.
 * @concept gradient-check
 * @example
 * ```ts
 * import { dotProductTest } from 'omai-engine/autodiff'
 * import { matmul, sin, tensor, type Value } from 'omai-engine/tensor'
 * dotProductTest((x: Value) => sin(matmul(tensor([[1, 2], [3, 4]]), x)), tensor([0.1, 0.2])).passed // => true
 * ```
 */
export function dotProductTest(f: (x: Value) => Value, x: TensorLike, options: CheckOptions = {}): CheckReport {
  const point = concrete(asTensor(x), 'dotProductTest')
  const baseKey = options.key ?? makeKey(0)
  const { compare, report } = comparer(options.rtol ?? 1e-10, options.atol ?? 1e-12)
  const v = normal(foldIn(baseKey, 2), point.shape)
  const { out, tangent } = jvp(f, point as Value, v as Value)
  const u = normal(foldIn(baseKey, 3), out.shape)
  const back = concrete(vjp(f, point as Value).pullback(u), 'dotProductTest')
  compare('<u, J v> against <J^T u, v>', dotFlat(u, tangent), dotFlat(back, v))
  return report()
}

/**
 * Check the derivative rules of a function (usually a single primitive) at a point, the conformance test every
 * primitive in omai passes:
 *
 * 1. the reverse-mode derivative against central finite differences of `<g, f(x)>` for a random cotangent `g`;
 * 2. the forward-mode derivative against finite differences along a random tangent;
 * 3. the dot-product test `<g, J t> = <J^T g, t>` linking the two;
 * 4. `vmap` against a loop over a small batch of scaled copies of the point, batched on the first and on the last
 *    axis (this exercises the batch rules, or the per-example fallback);
 * 5. the inputs are not mutated;
 * 6. with `order: 2`, the second derivative of the rules: the Hessian-vector product of `<g, f>` along a random
 *    direction by forward over reverse and by reverse over forward, against finite differences of the first-order
 *    gradient (so a rule that is right at first order but calls a kernel or `primalOf` fails).
 *
 * @param f - A function of several values returning one value.
 * @param inputs - The point (concrete float values for the differentiated inputs).
 * @param options - Step, tolerances, key for the random directions, which inputs to check, whether to check vmap and
 *   whether to check second derivatives (`order: 2`).
 * @returns A report with the largest errors and any failures.
 * @concept gradient-check
 * @index function Conformance test every primitive's jvp and vjp must pass.
 * @example
 * ```ts
 * import { checkPrimitive } from 'omai-engine/autodiff'
 * import { mul, tensor } from 'omai-engine/tensor'
 * checkPrimitive(mul, [tensor([1, 2]), tensor([3, 4])]).passed // => true
 * ```
 */
export function checkPrimitive(
  f: (...xs: Value[]) => Value,
  inputs: readonly TensorLike[],
  options: CheckOptions = {},
): CheckReport {
  const eps = options.eps ?? 1e-6
  const rtol = options.rtol ?? 1e-5
  const atol = options.atol ?? 1e-7
  const baseKey = options.key ?? makeKey(0)
  const xs = inputs.map((x) => concrete(asTensor(x), 'checkPrimitive'))
  const before = xs.map((x) => Array.from(x.data))
  const wrt =
    options.wrt ?? xs.map((x, i) => (x.dtype === 'float64' || x.dtype === 'float32' ? i : -1)).filter((i) => i >= 0)
  const { compare, failures, report } = comparer(rtol, atol)

  const out = concrete(f(...xs), 'checkPrimitive')
  const g = normal(foldIn(baseKey, 1), out.shape)
  const fOf =
    (i: number) =>
    (v: Value): Value =>
      f(...xs.map((x, j) => (j === i ? v : x)))

  for (const i of wrt) {
    const xi = xs[i] as Tensor
    // 1. reverse mode against finite differences of <g, f>
    const { pullback } = vjp((v: Value) => fOf(i)(v), xi as Value)
    const gradRev = concrete(pullback(g), 'checkPrimitive')
    const scalar = (v: Value): Value => fromStorage(new Float64Array([dotFlat(fOf(i)(v), g)]), [], 'float64')
    const gradFd = numericalGradient(scalar, xi, eps)
    const rev = toFlat(gradRev)
    const fd = toFlat(gradFd)
    for (let k = 0; k < fd.length; k++) compare(`vjp input ${i}[${k}]`, rev[k] ?? Number.NaN, fd[k] ?? Number.NaN)

    // 2. forward mode against finite differences along a random tangent
    const t = normal(foldIn(baseKey, 100 + i), xi.shape)
    const tangent = concrete(jvp((v: Value) => fOf(i)(v), xi as Value, t as Value).tangent, 'checkPrimitive')
    const h = eps * Math.max(1, maxOf(Array.from(xi.data, Math.abs)))
    const step = (s: number): Value =>
      fromStorage(
        Float64Array.from(xi.data, (v, k) => v + s * h * (t.data[k] ?? 0)),
        xi.shape,
        'float64',
      )
    const plus = toFlat(fOf(i)(step(1)))
    const minus = toFlat(fOf(i)(step(-1)))
    const tan = toFlat(tangent)
    for (let k = 0; k < tan.length; k++) {
      compare(`jvp input ${i}[${k}]`, tan[k] ?? Number.NaN, ((plus[k] ?? 0) - (minus[k] ?? 0)) / (2 * h))
    }

    // 3. dot-product test: <g, J t> = <J^T g, t>
    compare(`dot product input ${i}`, dotFlat(g, tangent), dotFlat(gradRev, t))

    // 6. second order: Hessian-vector products against finite differences of the gradient
    if (options.order === 2) {
      secondOrder(fOf(i), xi, g, normal(foldIn(baseKey, 200 + i), xi.shape) as Tensor, i, eps, compare, failures)
    }
  }

  // 4. vmap against a loop, with the batch first and last
  if (options.vmap !== false && wrt.length > 0) {
    vmapAgainstLoop(f, xs, wrt, 0, failures)
    vmapAgainstLoop(f, xs, wrt, -1, failures)
  }

  xs.forEach((x, i) => {
    if (Array.from(x.data).some((v, k) => !Object.is(v, before[i]?.[k]))) failures.push(`input ${i} was mutated`)
  })
  if (sizeOf(out.shape) === 0) failures.push('empty output: nothing was checked')
  return report()
}

/**
 * Compare the gradient of a scalar function with finite differences.
 *
 * @param f - A scalar function of one value.
 * @param x - The point.
 * @param options - Step and tolerances.
 * @returns A report.
 * @concept gradient-check
 * @example
 * ```ts
 * import { gradCheck } from 'omai-engine/autodiff'
 * import { sum, tanh, tensor, type Value } from 'omai-engine/tensor'
 * gradCheck((x: Value) => sum(tanh(x)), tensor([0.1, -0.4])).passed // => true
 * ```
 */
export function gradCheck(f: (x: Value) => Value, x: TensorLike, options: CheckOptions = {}): CheckReport {
  return checkPrimitive(f, [x], options)
}
