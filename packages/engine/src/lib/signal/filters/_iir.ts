// The one filtering primitive of the signal domain: a rational filter A(z) y = B(z) x applied to rows `[batch, n]`
// from rest (zero initial state). Everything else (initial conditions, final states, second-order sections, forward
// and backward filtering) is composed from it with differentiable tensor operations.
//
// Derivatives follow from A y = B x:
//   tangent:   dy = (B dx + dB x - dA y) / A, three runs of the same filter;
//   cotangent: with w = (1 / A)^T g (the filter run backwards in time), x-bar = B^T w (likewise backwards),
//              b-bar_k = sum_n w[n] x[n - k] and a-bar_k = -sum_n w[n] y[n - k].
// Both rules are written with this primitive and tensor operations, so higher derivatives work.
import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  bind1,
  definePrimitive,
  flip,
  fromStorage,
  mul,
  neg,
  type Primitive,
  sliceAxis,
  stack,
  sum,
  type Tensor,
  type Value,
  zeros,
} from '../../tensor/index.ts'

function run(b: ArrayLike<number>, a: ArrayLike<number>, x: Tensor): Tensor {
  const [batch = 1, n = 0] = x.shape
  const a0 = a[0] ?? 1
  if (a0 === 0) throw new WhiteboxError('value', 'lfilter: the leading denominator coefficient must be non-zero')
  const k = Math.max(b.length, a.length) - 1
  const bn = Float64Array.from({ length: k + 1 }, (_, i) => (b[i] ?? 0) / a0)
  const an = Float64Array.from({ length: k + 1 }, (_, i) => (a[i] ?? 0) / a0)
  const out = new Float64Array(batch * n)
  const z = new Float64Array(k + 1)
  for (let r = 0; r < batch; r++) {
    z.fill(0)
    const off = r * n
    for (let t = 0; t < n; t++) {
      const xt = x.data[off + t] ?? 0
      const yt = (bn[0] ?? 0) * xt + (z[0] ?? 0)
      for (let i = 0; i < k; i++) z[i] = (bn[i + 1] ?? 0) * xt - (an[i + 1] ?? 0) * yt + (z[i + 1] ?? 0)
      out[off + t] = yt
    }
  }
  return fromStorage(out, [batch, n], 'float64')
}

/** sum over rows and time of w[r, n] v[r, n - k], for k = 0 .. count - 1, as a `[count]` value. */
function laggedProducts(w: Value, v: Value, count: number): Value {
  const n = w.shape[1] ?? 0
  const terms: Value[] = []
  for (let k = 0; k < count; k++) {
    terms.push(k >= n ? zeros([]) : sum(mul(sliceAxis(w, 1, k, n), sliceAxis(v, 1, 0, n - k))))
  }
  return stack(terms)
}

const ONE: Tensor = fromStorage(Float64Array.of(1), [1], 'float64')

/** The transpose of filtering: run backwards in time. */
function backwards(b: Value, a: Value, g: Value): Value {
  return flip(iir(b, a, flip(g, 1)), 1)
}

const iirPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'lfilter',
  abstractEval: ([, , x]) => [x as NonNullable<typeof x>],
  impl: ([b, a, x]) => [run((b as Tensor).data, (a as Tensor).data, x as Tensor)],
  jvp: ([b, a, x], tangents, _p, [y]) => {
    const [db = null, da = null, dx = null] = tangents
    const parts: Value[] = []
    if (dx !== null) parts.push(iir(b as Value, a as Value, dx))
    if (db !== null) parts.push(iir(db, a as Value, x as Value))
    if (da !== null) parts.push(neg(iir(da, a as Value, y as Value)))
    const first = parts[0]
    if (first === undefined) return [null]
    return [parts.slice(1).reduce((s, p) => add(s, p), first)]
  },
  vjp: ([g], [b, a, x], _p, [y]) => {
    if (g === null || g === undefined) return [null, null, null]
    const w = backwards(ONE, a as Value, g)
    const xbar = backwards(b as Value, ONE, w)
    const bbar = laggedProducts(w, x as Value, (b as Value).shape[0] ?? 0)
    const abar = neg(laggedProducts(w, y as Value, (a as Value).shape[0] ?? 0))
    return [bbar, abar, xbar]
  },
  describe: () => 'lfilter',
})

/**
 * Filter the rows of x `[batch, n]` from rest by B(z) / A(z), coefficients in powers of z^-1 (a[0] non-zero).
 * Differentiable in b, a and x.
 */
export function iir(b: Value, a: Value, x: Value): Value {
  if (b.shape.length !== 1 || a.shape.length !== 1 || x.shape.length !== 2)
    throw new WhiteboxError('shape', 'lfilter: coefficients are 1-D and signals [batch, n]')
  return bind1(iirPrimitive, [b, a, x], null)
}
