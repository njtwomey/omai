// Sampling as primitives: the kernel draws from concrete parameter values (under any trace), and the derivative rule
// says how a sample moves with the parameters.
import { jvp } from '../autodiff/index.ts'
import { gammainc, lgamma } from '../numerics/special/index.ts'
import { foldIn, gammaVariates, type Key, uniform } from '../random/index.ts'
import {
  add,
  arg,
  bind1,
  broadcastTo,
  definePrimitive,
  div,
  exp,
  type FloatDType,
  fromStorage,
  full,
  greater,
  isfinite,
  log,
  logicalAnd,
  mul,
  neg,
  type Primitive,
  stopGradient,
  sub,
  type Tensor,
  type Value,
  where,
} from '../tensor/index.ts'

// dx/dalpha of a unit-rate gamma sample x, holding its CDF value fixed (implicit reparameterisation, Figurnov et al.
// 2018): -(dP/dalpha) / p(x), with dP/dalpha the exact shape derivative of the regularised incomplete gamma function.
function gammaSampleDerivative(alpha: Value, x: Value): Value {
  const ones = full(alpha.shape, 1, { dtype: alpha.dtype as FloatDType })
  const dPda = jvp((a: Value) => gammainc(a, x), alpha, ones).tangent
  const density = exp(sub(sub(mul(sub(alpha, 1), log(x)), x), lgamma(alpha)))
  return neg(div(dPda, density))
}

const gammaSamplePrimitive: Primitive<{ readonly key: Key }> = definePrimitive<{ readonly key: Key }>({
  name: 'gammaSample',
  abstractEval: (xs) => [arg(xs, 0)],
  impl: ([alpha], { key }) => {
    const a = alpha as Tensor
    const draws = gammaVariates(key, a.data)
    const out = a.dtype === 'float32' ? Float32Array.from(draws) : draws
    return [a.backend.wrap(a.dtype, a.shape, out as never)]
  },
  jvp: ([alpha], [t], _p, [x]) => [
    t === null || t === undefined ? null : mul(t, gammaSampleDerivative(alpha as Value, x as Value)),
  ],
  vjp: ([g], [alpha], _p, [x]) => [
    g === null || g === undefined ? null : mul(g, gammaSampleDerivative(alpha as Value, x as Value)),
  ],
  describe: () => 'implicit reparameterisation',
})

/**
 * Unit-rate gamma samples, one per entry of `alpha` broadcast to `[...shape, ...alpha.shape]`, differentiable in
 * `alpha` by implicit reparameterisation.
 */
export function gammaSample(key: Key, alpha: Value, shape: readonly number[]): Value {
  const full = shape.length === 0 ? alpha : broadcastTo(alpha, [...shape, ...alpha.shape])
  return bind1(gammaSamplePrimitive, [full], { key })
}

/** The parameters of a discrete sampler: a function of the concrete parameter values. */
interface DrawParams {
  readonly draw: (values: readonly Tensor[]) => Tensor
  readonly shape: readonly number[]
  readonly dtype: 'int32' | 'float64' | 'float32' | 'bool'
}

const discreteSamplePrimitive: Primitive<DrawParams> = definePrimitive<DrawParams>({
  name: 'discreteSample',
  abstractEval: (_xs, p) => [{ shape: [...p.shape], dtype: p.dtype }],
  impl: (xs, p) => [p.draw(xs)],
  // Samples are piecewise constant in the parameters.
  zeroDerivative: true,
})

/**
 * Draw from concrete parameter values (under any trace) with a zero derivative, for discrete distributions.
 */
export function discreteSample(
  values: readonly Value[],
  shape: readonly number[],
  dtype: DrawParams['dtype'],
  draw: (values: readonly Tensor[]) => Tensor,
): Value {
  return bind1(discreteSamplePrimitive, values, { draw, shape, dtype })
}

/**
 * Draw `[...shape, ...batchShape]` values, one per entry, from concrete parameters broadcast to that shape, with a zero
 * derivative (for discrete families whose scalar samplers take one parameter set at a time). Entry i uses the key
 * `foldIn(key, i)`.
 */
export function elementwiseSample(
  key: Key,
  params: readonly Value[],
  shape: readonly number[],
  batchShape: readonly number[],
  drawOne: (k: Key, ...p: number[]) => number,
): Value {
  const full = [...shape, ...batchShape]
  return discreteSample(params, full, 'float64', (values) => {
    const broadcast = values.map((v) =>
      v.shape.length === full.length && v.shape.every((d, i) => d === full[i])
        ? v
        : (v.backend.broadcastTo(v, full) as Tensor),
    )
    const size = full.reduce((a, b) => a * b, 1)
    const out = new Float64Array(size)
    for (let i = 0; i < size; i++) out[i] = drawOne(foldIn(key, i), ...broadcast.map((b) => b.data[i] ?? Number.NaN))
    return fromStorage(out, full, 'float64')
  })
}

/** Uniform noise in the open interval (0, 1): a draw of exactly 0 becomes 2^-54, so logarithms and quantiles stay finite. */
export function uniformOpen(key: Key, shape: readonly number[]): Tensor {
  const u = uniform(key, shape)
  const data = u.data
  for (let i = 0; i < data.length; i++) if (data[i] === 0) data[i] = 2 ** -54
  return u
}

/**
 * Implicit reparameterisation (Figurnov et al., 2018) of samples drawn from concrete parameter values: a value equal to
 * `x` whose derivative with respect to the parameters is -(dF/dtheta) / p(x), the motion of a sample that keeps its CDF
 * value fixed. Needs a CDF that differentiates in the parameters.
 */
export function implicitReparam(x: Value, cdf: (x: Value) => Value, prob: (x: Value) => Value): Value {
  const xs = stopGradient(x)
  const f = cdf(xs)
  const p = stopGradient(prob(xs))
  const ok = logicalAnd(greater(p, 0), isfinite(p))
  return add(xs, where(ok, neg(div(sub(f, stopGradient(f)), where(ok, p, 1))), 0))
}
