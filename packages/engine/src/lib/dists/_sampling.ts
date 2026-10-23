// Sampling as primitives: the kernel draws from concrete parameter values (under any trace), and the derivative rule
// says how a sample moves with the parameters.
import { jvp } from '../autodiff/index.ts'
import { gammainc, lgamma } from '../numerics/special/index.ts'
import { gammaVariates, type Key } from '../random/index.ts'
import {
  arg,
  bind1,
  broadcastTo,
  definePrimitive,
  div,
  exp,
  type FloatDType,
  full,
  log,
  mul,
  neg,
  type Primitive,
  sub,
  type Tensor,
  type Value,
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
