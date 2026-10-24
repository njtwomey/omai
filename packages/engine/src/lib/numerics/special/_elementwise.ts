import {
  abs,
  add,
  allocate,
  arg,
  batchElementwise,
  bind1,
  broadcastShapes,
  definePrimitive,
  div,
  type FloatDType,
  type MaybeZero,
  maximum,
  mul,
  type Primitive,
  promoteFloat,
  sizeOf,
  sub,
  sumToShape,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/** d out / d (one argument), written with operations so that it is itself differentiable. */
export type Partial = (xs: readonly Value[], out: Value) => Value

function evaluate(kernel: (...args: number[]) => number, xs: readonly Tensor[]): Tensor {
  const first = xs[0] as Tensor
  const dtype = first.dtype as FloatDType
  if (xs.length === 1) return first.backend.map(first, kernel, dtype)
  if (xs.length === 2) return first.backend.map2(first, xs[1] as Tensor, kernel, dtype)
  const shape = xs.reduce<number[]>((s, x) => broadcastShapes(s, x.shape), [])
  const full = xs.map((x) => first.backend.broadcastTo(x, shape))
  const out = allocate(dtype, sizeOf(shape))
  const args = new Array<number>(xs.length)
  for (let i = 0; i < out.length; i++) {
    for (let j = 0; j < full.length; j++) args[j] = (full[j] as Tensor).data[i] ?? 0
    out[i] = kernel(...args)
  }
  return first.backend.wrap(dtype, shape, out)
}

/**
 * An elementwise special function of any number of broadcast arguments as a primitive. Each argument has a partial
 * derivative written with operations, or `null` for a central finite difference of the function itself (step
 * `1e-6 max(1, |x|)`, relative accuracy about 1e-8 to 1e-10). A finite-difference partial is used only where no
 * convenient closed form exists (derivatives with respect to orders, degrees of freedom of some inverses, and second
 * derivatives of derivative kernels); each such case is stated in the function's documentation.
 */
export function special(
  name: string,
  kernel: (...args: number[]) => number,
  partials: readonly (Partial | null)[],
): (...xs: TensorLike[]) => Value {
  const apply = (xs: readonly Value[]): Value => bind1(primitive, xs, null)
  const fd =
    (i: number): Partial =>
    (xs) => {
      const x = xs[i] as Value
      const h = mul(maximum(abs(x), 1), 1e-6)
      const up = apply(xs.map((v, j) => (j === i ? add(v, h) : v)))
      const down = apply(xs.map((v, j) => (j === i ? sub(v, h) : v)))
      return div(sub(up, down), mul(h, 2))
    }
  const rules = partials.map((p, i) => p ?? fd(i))
  const primitive: Primitive<null> = definePrimitive<null>({
    name,
    abstractEval: (xs) => [
      { shape: xs.reduce<number[]>((s, x) => broadcastShapes(s, x.shape), []), dtype: arg(xs, 0).dtype },
    ],
    impl: (xs) => [evaluate(kernel, xs)],
    jvp: (xs, ts, _p, outs) => {
      const out = arg(outs, 0)
      let total: MaybeZero = null
      ts.forEach((t, i) => {
        if (t === null) return
        const term = mul(t, (rules[i] as Partial)(xs, out))
        total = total === null ? term : add(total, term)
      })
      return [total]
    },
    vjp: (gs, xs, _p, outs) => {
      const g = arg(gs, 0)
      if (g === null) return xs.map(() => null)
      const out = arg(outs, 0)
      return xs.map((x, i) => sumToShape(mul(g, (rules[i] as Partial)(xs, out)), x.shape))
    },
    batch: (xs, ds) => batchElementwise(primitive, xs, ds, null),
  })
  return (...xs) => apply(promoteFloat(...xs))
}
