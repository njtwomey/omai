// Guarded elementwise functions for kernel formulas whose derivatives blow up at points where the kernel itself is
// smooth or where a subgradient is the sensible choice (the "double where" pattern: the unsafe branch never sees the
// bad input, so neither the value nor the derivative is NaN). Private to the kernels domain.
import {
  add,
  arccos,
  div,
  equal,
  greaterEqual,
  lessEqual,
  logicalOr,
  mul,
  sqrt,
  square,
  sub,
  type Value,
  where,
} from '../tensor/index.ts'

/** sqrt(x) for x >= 0, with value and derivative 0 where x <= 0. */
export function safeSqrt(x: Value): Value {
  const zero = lessEqual(x, 0)
  return where(zero, 0, sqrt(where(zero, 1, x)))
}

/** n / d, 0 where d = 0 (with zero derivative there). */
export function safeDivide(n: Value, d: Value): Value {
  const zero = equal(d, 0)
  return where(zero, 0, div(n, where(zero, 1, d)))
}

/** arccos of a cosine clipped to [-1, 1], with zero derivative at the ends (where the angle cannot move). */
export function safeArccos(c: Value): Value {
  const hi = greaterEqual(c, 1)
  const lo = lessEqual(c, -1)
  const inside = where(logicalOr(hi, lo), 0, c)
  return where(hi, 0, where(lo, Math.PI, arccos(inside)))
}

/** sin(theta) from cos(theta) for theta in [0, pi]: sqrt(1 - c^2), zero derivative at the ends. */
export function sinFromCos(c: Value): Value {
  return safeSqrt(sub(1, square(c)))
}

/** The arc-cosine profile J_n(theta) of Cho and Saul, as a function of c = cos(theta), for n = 0, 1, 2. */
export function arcCosineJ(order: 0 | 1 | 2, c: Value): Value {
  const theta = safeArccos(c)
  const rest = sub(Math.PI, theta)
  if (order === 0) return rest
  const s = sinFromCos(c)
  if (order === 1) return add(s, mul(rest, c))
  // 3 sin(theta) cos(theta) + (pi - theta)(1 + 2 cos^2(theta))
  return add(mul(mul(s, c), 3), mul(rest, add(mul(square(c), 2), 1)))
}
