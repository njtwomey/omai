import {
  add,
  div,
  equal,
  mul,
  neg,
  promoteFloat,
  sin,
  sqrt,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as e from './_elliptic.ts'

// Each derivative in m has a removable singularity at m = 0: the limit is used there, and the general formula is
// evaluated at a safe point so that the unselected branch contributes no NaN.
const atZero = (m: Value, limit: (s: Value) => Value, general: (s: Value) => Value): Value => {
  const zero = equal(m, 0)
  const safe = where(zero, 0.5, m)
  return where(zero, limit(m), general(safe))
}
const constant = (like: Value, c: number): Value => add(mul(like, 0), c)
const delta = (phi: Value, m: Value): Value => sqrt(sub(1, mul(m, square(sin(phi)))))

const ellipkP = special('ellipk', e.ellipk, [
  // dK/dm = (E - (1 - m) K) / (2 m (1 - m)); pi / 8 at m = 0.
  ([m]) =>
    atZero(
      m as Value,
      (s) => constant(s, Math.PI / 8),
      (s) => div(sub(ellipe(s), mul(sub(1, s), ellipk(s))), mul(mul(s, sub(1, s)), 2)),
    ),
])
const ellipeP = special('ellipe', e.ellipe, [
  // dE/dm = (E - K) / (2 m); -pi / 8 at m = 0.
  ([m]) =>
    atZero(
      m as Value,
      (s) => constant(s, -Math.PI / 8),
      (s) => div(sub(ellipe(s), ellipk(s)), mul(s, 2)),
    ),
])
const ellipkincP = special('ellipkinc', e.ellipkinc, [
  ([phi, m]) => div(1, delta(phi as Value, m as Value)),
  // dF/dm = E / (2 m (1 - m)) - F / (2 m) - sin(2 phi) / (4 (1 - m) delta); (2 phi - sin 2 phi) / 8 at m = 0.
  ([phi, m]) => {
    const p = phi as Value
    return atZero(
      m as Value,
      () => mul(sub(mul(p, 2), sin(mul(p, 2))), 0.125),
      (s) =>
        sub(
          sub(div(ellipeinc(p, s), mul(mul(s, sub(1, s)), 2)), div(ellipkinc(p, s), mul(s, 2))),
          div(sin(mul(p, 2)), mul(mul(sub(1, s), 4), delta(p, s))),
        ),
    )
  },
])
const ellipeincP = special('ellipeinc', e.ellipeinc, [
  ([phi, m]) => delta(phi as Value, m as Value),
  // dE/dm = (E - F) / (2 m); -(2 phi - sin 2 phi) / 8 at m = 0.
  ([phi, m]) => {
    const p = phi as Value
    return atZero(
      m as Value,
      () => mul(sub(mul(p, 2), sin(mul(p, 2))), -0.125),
      (s) => div(sub(ellipeinc(p, s), ellipkinc(p, s)), mul(s, 2)),
    )
  },
])
// Jacobi elliptic functions: exact derivatives in u, finite differences in m.
const snP = special('ellipjSn', (u, m) => e.ellipj(u, m).sn, [
  ([u, m]) => mul(cnP(u as Value, m as Value), dnP(u as Value, m as Value)),
  null,
])
const cnP = special('ellipjCn', (u, m) => e.ellipj(u, m).cn, [
  ([u, m]) => neg(mul(snP(u as Value, m as Value), dnP(u as Value, m as Value))),
  null,
])
const dnP = special('ellipjDn', (u, m) => e.ellipj(u, m).dn, [
  ([u, m]) => neg(mul(mul(m as Value, snP(u as Value, m as Value)), cnP(u as Value, m as Value))),
  null,
])
const amP = special('ellipjPh', (u, m) => e.ellipj(u, m).ph, [([u, m]) => dnP(u as Value, m as Value), null])

/**
 * The complete elliptic integral of the first kind K(m) = int_0^{pi/2} (1 - m sin^2 t)^{-1/2} dt (parameter m = k^2,
 * as scipy).
 *
 * @param m - Parameter (m < 1).
 * @returns K(m), differentiable.
 * @concept special-functions
 * @example
 * ```ts
 * import { ellipk } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ellipk(0)) === Math.PI / 2 // => true
 * ```
 */
export function ellipk(m: TensorLike): Value {
  return ellipkP(m)
}

/**
 * The complete elliptic integral of the second kind E(m) = int_0^{pi/2} (1 - m sin^2 t)^{1/2} dt.
 *
 * @param m - Parameter (m <= 1).
 * @returns E(m), differentiable.
 * @example
 * ```ts
 * import { ellipe } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ellipe(1)) // => 1
 * ```
 */
export function ellipe(m: TensorLike): Value {
  return ellipeP(m)
}

/**
 * The incomplete elliptic integral of the first kind F(phi | m) (Carlson's R_F).
 *
 * @param phi - Amplitude.
 * @param m - Parameter.
 * @returns F(phi | m), differentiable in both arguments.
 * @example
 * ```ts
 * import { ellipkinc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ellipkinc(0.3, 0)) // => 0.3
 * ```
 */
export function ellipkinc(phi: TensorLike, m: TensorLike): Value {
  return ellipkincP(phi, m)
}

/**
 * The incomplete elliptic integral of the second kind E(phi | m) (Carlson's R_F and R_D).
 *
 * @param phi - Amplitude.
 * @param m - Parameter.
 * @returns E(phi | m), differentiable in both arguments.
 * @example
 * ```ts
 * import { ellipeinc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ellipeinc(0.3, 0)) // => 0.3
 * ```
 */
export function ellipeinc(phi: TensorLike, m: TensorLike): Value {
  return ellipeincP(phi, m)
}

/**
 * The Jacobi elliptic functions sn, cn, dn and the amplitude am at (u, m), 0 <= m <= 1 (the exact motion of a
 * pendulum: theta(t) = 2 arcsin(k sn(omega t, k^2))).
 *
 * @param u - Argument.
 * @param m - Parameter.
 * @returns `{ sn, cn, dn, am }`, differentiable (exactly in u, by finite differences in m).
 * @example
 * ```ts
 * import { ellipj } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ellipj(0, 0.5).cn) // => 1
 * ```
 */
export function ellipj(u: TensorLike, m: TensorLike): { sn: Value; cn: Value; dn: Value; am: Value } {
  const [uu, mm] = promoteFloat(u, m) as [Value, Value]
  return { sn: snP(uu, mm), cn: cnP(uu, mm), dn: dnP(uu, mm), am: amP(uu, mm) }
}
