/**
 * The conformance checks for derivative rules: central finite-difference gradients, the dot-product (adjoint) test that
 * forward and reverse mode agree, `gradCheck` for a scalar function, and `checkPrimitive`, the test every primitive in
 * mdml-js passes (finite-difference vjp and jvp, their consistency, and no mutation of inputs).
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../random/index.ts'
import {
  asTensor,
  fromStorage,
  isTensor,
  mul,
  sizeOf,
  stack,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { jvp, vjp } from './transforms.ts'
import { vmap } from './vmap.ts'

function concrete(x: Value, where: string): Tensor {
  if (!isTensor(x)) throw new MdmlError('not-differentiable', `${where} needs concrete values`)
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
 * import { numericalGradient } from 'mdml-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'mdml-engine/tensor'
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
 * import { dotProductTest } from 'mdml-engine/autodiff'
 * import { matmul, sin, tensor, type Value } from 'mdml-engine/tensor'
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
 * primitive in mdml-js passes:
 *
 * 1. the reverse-mode derivative against central finite differences of `<g, f(x)>` for a random cotangent `g`;
 * 2. the forward-mode derivative against finite differences along a random tangent;
 * 3. the dot-product test `<g, J t> = <J^T g, t>` linking the two;
 * 4. `vmap` against a loop over a small batch of scaled copies of the point, batched on the first and on the last
 *    axis (this exercises the batch rules, or the per-example fallback);
 * 5. the inputs are not mutated.
 *
 * @param f - A function of several values returning one value.
 * @param inputs - The point (concrete float values for the differentiated inputs).
 * @param options - Step, tolerances, key for the random directions, which inputs to check, and whether to check vmap.
 * @returns A report with the largest errors and any failures.
 * @concept gradient-check
 * @example
 * ```ts
 * import { checkPrimitive } from 'mdml-engine/autodiff'
 * import { mul, tensor } from 'mdml-engine/tensor'
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
 * import { gradCheck } from 'mdml-engine/autodiff'
 * import { sum, tanh, tensor, type Value } from 'mdml-engine/tensor'
 * gradCheck((x: Value) => sum(tanh(x)), tensor([0.1, -0.4])).passed // => true
 * ```
 */
export function gradCheck(f: (x: Value) => Value, x: TensorLike, options: CheckOptions = {}): CheckReport {
  return checkPrimitive(f, [x], options)
}
