import { maxOf, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../random/index.ts'
import {
  asTensor,
  fromStorage,
  isTensor,
  sizeOf,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { jvp, vjp } from './transforms.ts'

function concrete(x: Value, where: string): Tensor {
  if (!isTensor(x)) throw new WhiteboxError('not-differentiable', `${where} needs concrete values`)
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
 * import { numericalGradient } from 'whitebox-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'whitebox-engine/tensor'
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
}

/**
 * Check the derivative rules of a function (usually a single primitive) at a point, the conformance test every
 * primitive in whitebox-js passes:
 *
 * 1. the reverse-mode derivative against central finite differences of `<g, f(x)>` for a random cotangent `g`;
 * 2. the forward-mode derivative against finite differences along a random tangent;
 * 3. the dot-product test `<g, J t> = <J^T g, t>` linking the two;
 * 4. the inputs are not mutated.
 *
 * @param f - A function of several values returning one value.
 * @param inputs - The point (concrete float values for the differentiated inputs).
 * @param options - Step, tolerances, key for the random directions, and which inputs to check.
 * @returns A report with the largest errors and any failures.
 * @concept gradient-check
 * @example
 * ```ts
 * import { checkPrimitive } from 'whitebox-engine/autodiff'
 * import { mul, tensor } from 'whitebox-engine/tensor'
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
    toFlat(gradFd).forEach((want, k) => {
      compare(`vjp input ${i}[${k}]`, rev[k] ?? Number.NaN, want)
    })

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
    toFlat(tangent).forEach((got, k) => {
      compare(`jvp input ${i}[${k}]`, got, ((plus[k] ?? 0) - (minus[k] ?? 0)) / (2 * h))
    })

    // 3. dot-product test: <g, J t> = <J^T g, t>
    compare(`dot product input ${i}`, dotFlat(g, tangent), dotFlat(gradRev, t))
  }

  xs.forEach((x, i) => {
    if (Array.from(x.data).some((v, k) => !Object.is(v, before[i]?.[k]))) failures.push(`input ${i} was mutated`)
  })
  if (sizeOf(out.shape) === 0) failures.push('empty output: nothing was checked')
  return { passed: failures.length === 0, maxAbsError: maxAbs, maxRelError: maxRel, failures }
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
 * import { gradCheck } from 'whitebox-engine/autodiff'
 * import { sum, tanh, tensor, type Value } from 'whitebox-engine/tensor'
 * gradCheck((x: Value) => sum(tanh(x)), tensor([0.1, -0.4])).passed // => true
 * ```
 */
export function gradCheck(f: (x: Value) => Value, x: TensorLike, options: CheckOptions = {}): CheckReport {
  return checkPrimitive(f, [x], options)
}
