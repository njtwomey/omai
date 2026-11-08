/**
 * Light bending around a mass: null geodesics of the Schwarzschild metric in the Binet form u'' + u = 3 M u^2 (u = 1/r,
 * the independent variable the angle phi, units G = c = 1 so that M is half the Schwarzschild radius), integrated as a stepped
 * algorithm that logs the ray's path, its closest approach and its capture. The deflection angle by exact quadrature (the
 * truth), the weak-field expansion 4M/b + 15 pi M^2 / (4 b^2), the photon sphere and the critical impact parameter inside
 * which rays are captured, and Flamm's paraboloid, the embedding of the curved space.
 */

import { OmaiError } from '../../base/index.ts'
import { fixedQuad } from '../../numerics/quadrature/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { add, div, item, tensor as makeTensor, mul, sliceAxis, sqrt, stack, sub, sum } from '../../tensor/index.ts'
import { type Algorithm, type Contract, defineContract, event, scalar, tensor } from '../../trace/index.ts'
import { type FlowState, flowAlgorithm } from '../ode/flow.ts'
import type { OdeVectorField } from '../ode/problem.ts'

/**
 * The radius of the photon sphere, 3 M: the circular (unstable) orbit of light.
 *
 * @param M - The mass (G = c = 1).
 * @returns 3 M.
 * @example
 * ```ts
 * import { photonSphereRadius } from 'omai-engine/dynamics/celestial'
 * photonSphereRadius(1) // => 3
 * ```
 */
export function photonSphereRadius(M: number): number {
  return 3 * M
}

/**
 * The critical impact parameter 3 sqrt(3) M: a ray with a smaller one is captured by the black hole, a larger one is
 * deflected and escapes. It is the radius of the black hole's shadow as seen from far away.
 *
 * @param M - The mass.
 * @returns 3 sqrt(3) M.
 * @example
 * ```ts
 * import { criticalImpactParameter } from 'omai-engine/dynamics/celestial'
 * criticalImpactParameter(1) // => 5.196152422706632
 * ```
 */
export function criticalImpactParameter(M: number): number {
  return 3 * Math.sqrt(3) * M
}

/**
 * The weak-field deflection angle of light passing a mass at impact parameter b: 4 M / b (Einstein's 1915 value, twice the
 * Newtonian one) and with the second-order term 15 pi M^2 / (4 b^2).
 *
 * @param b - The impact parameter.
 * @param M - The mass.
 * @param order - 1 or 2 (default 1).
 * @returns The deflection in radians.
 * @concept gravitational-lensing
 * @example
 * ```ts
 * import { weakFieldDeflection } from 'omai-engine/dynamics/celestial'
 * weakFieldDeflection(1000, 1) // => 0.004
 * ```
 */
export function weakFieldDeflection(b: number, M: number, order: 1 | 2 = 1): number {
  const first = (4 * M) / b
  return order === 1 ? first : first + (15 * Math.PI * M * M) / (4 * b * b)
}

/** The turning point 1 / r_min of a ray with impact parameter b, the smaller positive root of 1/b^2 - u^2 + 2 M u^3. */
function turningPoint(b: number, M: number): number {
  const f = (u: number): number => 1 / (b * b) - u * u + 2 * M * u ** 3
  return brentq(f, 0, 1 / (3 * M), { xtol: 1e-16, rtol: 4 * Number.EPSILON }).root
}

/**
 * The exact deflection angle of a light ray passing a Schwarzschild mass at impact parameter b: twice the angle swept from
 * infinity to the closest approach, 2 integral du / sqrt(1/b^2 - u^2 + 2 M u^3) from 0 to the turning point, minus pi. The
 * square-root singularity at the turning point is removed by u = u_max (1 - s^2) and the smooth integrand integrated by
 * Gauss-Legendre quadrature. Infinite for a captured ray (b at or below the critical value).
 *
 * @param b - The impact parameter.
 * @param M - The mass.
 * @returns The deflection in radians (Infinity when the ray is captured).
 * @throws OmaiError with code `value` for non-positive arguments.
 * @concept gravitational-lensing
 * @example
 * ```ts
 * import { deflectionAngle } from 'omai-engine/dynamics/celestial'
 * // Far from the mass the exact deflection is the weak-field 4 M / b plus 15 pi M^2 / (4 b^2) and terms of order (M / b)^3.
 * Math.abs(deflectionAngle(1000, 1) - 0.004 - (15 * Math.PI) / 4e6) < 1e-7 // => true
 * ```
 */
export function deflectionAngle(b: number, M: number): number {
  if (!(b > 0) || !(M > 0)) throw new OmaiError('value', 'deflectionAngle: b and M must be positive')
  if (b <= criticalImpactParameter(M)) return Number.POSITIVE_INFINITY
  const um = turningPoint(b, M)
  const c = 1 - 2 * M * um
  const integral = item(
    fixedQuad(
      (s) => {
        // h(u) = f(u) / (u_max - u) at u = u_max (1 - s^2); the integrand is 2 sqrt(u_max) / sqrt(h).
        const u = mul(sub(1, mul(s, s)), um)
        const h = add(add(mul(mul(u, u), -2 * M), mul(u, c)), c * um)
        return div(2 * Math.sqrt(um), sqrt(h))
      },
      0,
      1,
      96,
    ),
  )
  return 2 * integral - Math.PI
}

/** The angle swept from infinity to the radius 1 / u0 on the way in: the integral of du / sqrt(1/b^2 - u^2 + 2 M u^3). */
function startAngle(b: number, M: number, u0: number): number {
  return item(
    fixedQuad(
      (s) => {
        const u = mul(s, u0)
        return div(u0, sqrt(add(sub(1 / (b * b), mul(u, u)), mul(mul(mul(u, u), u), 2 * M))))
      },
      0,
      1,
      24,
    ),
  )
}

/**
 * The vector field of the Schwarzschild null geodesic in the Binet form on `(u, du/dphi)`: u' = du/dphi and
 * (du/dphi)' = -u + 3 M u^2, with the angle phi as the independent variable (the `t` of the field). The quantity
 * (du/dphi)^2 + u^2 - 2 M u^3 = 1/b^2 is conserved.
 *
 * @param M - The mass.
 * @returns The field.
 * @concept gravitational-lensing
 * @example
 * ```ts
 * import { schwarzschildNullGeodesicField } from 'omai-engine/dynamics/celestial'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * // At u = 1 / (3 M) (the photon sphere) with no radial motion the ray stays: u'' = 0.
 * toFlat(schwarzschildNullGeodesicField(1)(0, tensor([1 / 3, 0])))[1] // => 0
 * ```
 */
export function schwarzschildNullGeodesicField(M: number): OdeVectorField {
  return (_phi, s) => {
    const u = sum(sliceAxis(s, 0, 0, 1))
    const v = sum(sliceAxis(s, 0, 1, 2))
    return stack([v, add(mul(u, -1), mul(mul(u, u), 3 * M))])
  }
}

/**
 * The conserved quantity of the Binet form, (du/dphi)^2 + u^2 - 2 M u^3, which equals 1/b^2 for a ray of impact parameter b.
 *
 * @param state - `[u, du/dphi]`.
 * @param M - The mass.
 * @returns The value of the invariant.
 * @example
 * ```ts
 * import { nullInvariant } from 'omai-engine/dynamics/celestial'
 * nullInvariant([0, 0.5], 1) // => 0.25
 * ```
 */
export function nullInvariant(state: ArrayLike<number>, M: number): number {
  const u = state[0] ?? 0
  const v = state[1] ?? 0
  return v * v + u * u - 2 * M * u ** 3
}

/**
 * Flamm's paraboloid: the embedding of the equatorial plane of the Schwarzschild space in flat three-space, z(r) =
 * 2 sqrt(2 M (r - 2 M)), a surface of revolution that flares out at the horizon r = 2M. Light follows the geodesics of
 * this surface (and the time part of the metric bends them further).
 *
 * @param M - The mass.
 * @param r - The radius (at least 2 M).
 * @returns The height z.
 * @throws OmaiError with code `value` for a radius inside the horizon.
 * @example
 * ```ts
 * import { flammParaboloid } from 'omai-engine/dynamics/celestial'
 * flammParaboloid(1, 6) // => 5.656854249492381
 * ```
 */
export function flammParaboloid(M: number, r: number): number {
  if (r < 2 * M) throw new OmaiError('value', 'flammParaboloid: inside the horizon')
  return 2 * Math.sqrt(2 * M * (r - 2 * M))
}

/** The state of {@link lightRayAlgorithm}: the Binet state at an angle, and what has become of the ray. */
export interface RayState extends FlowState {
  /** The status of the ray: still travelling, captured by the horizon or escaped to the start radius. */
  readonly fate: 'travelling' | 'captured' | 'escaped'
  /** The closest approach so far, 1 / u_max. */
  readonly closest: number
  /** The angle at which the ray reached the start radius again, when it has. */
  readonly exitAngle: number
}

/** The paths of {@link lightRayAlgorithm}. */
export type RayPath =
  | 'ray/impactParameter'
  | 'ray/phi'
  | 'ray/radius'
  | 'ray/position'
  | 'ray/deflection'
  | 'ray/weakField'
  | 'ray/captured'

/** What a light ray logs. */
export const rayContract: Contract<RayState, RayPath> = defineContract({
  'ray/impactParameter': scalar({ description: 'the impact parameter b = L / E of the ray' }),
  'ray/phi': scalar({ description: 'the angle swept' }),
  'ray/radius': scalar({ description: 'the radius r = 1 / u' }),
  'ray/position': tensor({ role: 'state', shape: [2], description: 'the position (x, y) in the plane of the orbit' }),
  'ray/deflection': scalar({
    description: 'the angle swept beyond pi, less the part not yet travelled (the bend so far)',
  }),
  'ray/weakField': scalar({ description: 'the weak-field prediction 4 M / b' }),
  'ray/captured': event({ description: 'the ray crossed the horizon', context: { phi: 'where' } }),
})

/** Options of {@link lightRayAlgorithm}. */
export interface RayOptions {
  /** The angular step (default 0.002). */
  readonly dphi?: number
  /** The radius at which the ray starts and is taken to have left (default 200 b: far enough for the bend to be the whole of it). */
  readonly startRadius?: number
  /** The most steps to take (default 20000). */
  readonly maxSteps?: number
}

/**
 * One light ray past a black hole as a stepped algorithm. It starts incoming at the start radius, with the angle chosen
 * so that the incoming asymptote lies along phi = 0, and is integrated in the angle by fourth-order Runge-Kutta; it logs the
 * radius, the position, the bend so far and the weak-field value, fires `ray/captured` when it crosses the horizon (r = 2M)
 * and stops when it has returned to the start radius or been captured. The exit angle is refined inside the last step.
 * A fan of rays, one per step of a sweep over b, is the picture of lensing: bent more the closer they pass, captured inside
 * 3 sqrt(3) M.
 *
 * @param M - The mass.
 * @param b - The impact parameter.
 * @param options - Step, start radius and step budget.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for non-positive arguments or a start radius not far outside the ray.
 * @concept gravitational-lensing
 * @example
 * ```ts
 * import { lightRayAlgorithm } from 'omai-engine/dynamics/celestial'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const { state } = runAlgorithm(lightRayAlgorithm(1, 8), { maxSteps: 20000 })
 * state.fate // => 'escaped'
 * ```
 */
export function lightRayAlgorithm(M: number, b: number, options: RayOptions = {}): Algorithm<RayState, RayPath> {
  if (!(M > 0) || !(b > 0)) throw new OmaiError('value', 'lightRayAlgorithm: M and b must be positive')
  const dphi = options.dphi ?? 0.002
  const rStart = options.startRadius ?? 200 * b
  const u0 = 1 / rStart
  if (!(u0 * b < 0.1))
    throw new OmaiError('value', 'lightRayAlgorithm: start well outside the ray (radius at least 10 b)')
  const field = schwarzschildNullGeodesicField(M)
  const f0 = (u: number): number => Math.sqrt(1 / (b * b) - u * u + 2 * M * u ** 3)
  // The angle from infinity to the start radius, so that the incoming asymptote is at phi = 0.
  const phiStart = startAngle(b, M, u0)
  const horizon = 1 / (2 * M)
  const inner = flowAlgorithm<RayPath>(field, [u0, f0(u0)], {
    dt: dphi,
    steps: options.maxSteps ?? 20000,
    t0: phiStart,
    name: 'lightRay',
    config: { M, b, dphi, startRadius: rStart },
  })
  return {
    name: 'lightRay',
    config: inner.config,
    init: (k) => {
      const s = inner.init(k)
      return { ...s, fate: 'travelling', closest: rStart, exitAngle: Number.NaN }
    },
    step: (s, ctx) => {
      const next = inner.step(s, ctx)
      const u = next.y[0] as number
      const phi = next.t
      const r = 1 / u
      const closest = Math.min(s.closest, r)
      let fate: RayState['fate'] = 'travelling'
      let exitAngle = Number.NaN
      if (u >= horizon) {
        fate = 'captured'
        ctx.emit('ray/captured', { phi })
      } else if (u <= u0 && closest < rStart * 0.5 && s.y[1] !== undefined && (s.y[1] as number) < 0) {
        // Back out at the start radius (u falling through u0): refine the crossing in the last step by linear interpolation.
        fate = 'escaped'
        const ua = s.y[0] as number
        exitAngle = s.t + (dphi * (ua - u0)) / (ua - u)
      }
      ctx.log('ray/impactParameter', b)
      ctx.log('ray/phi', phi)
      ctx.log('ray/radius', r)
      ctx.log('ray/position', makeTensor([r * Math.cos(phi), r * Math.sin(phi)]))
      ctx.log('ray/deflection', phi + phiStart - Math.PI)
      ctx.log('ray/weakField', weakFieldDeflection(b, M))
      return { ...next, fate, closest, exitAngle }
    },
    done: (s) => (s.fate === 'travelling' ? undefined : s.fate === 'captured' ? 'stopped' : 'completed'),
    serialise: (s) => inner.serialise?.(s) as never,
    deserialise: (j) => inner.deserialise?.(j) as RayState,
  }
}

/**
 * The deflection of a ray as the stepped integration measures it: the angle from the incoming asymptote to the exit
 * direction, less pi.
 *
 * @param state - The final state of a {@link lightRayAlgorithm} run.
 * @param impactParameter - The ray's b.
 * @param M - The mass.
 * @param startRadius - The radius the ray started at.
 * @returns The deflection, or Infinity for a captured ray.
 * @concept gravitational-lensing
 * @example
 * ```ts
 * import { lightRayAlgorithm, measuredDeflection } from 'omai-engine/dynamics/celestial'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const { state } = runAlgorithm(lightRayAlgorithm(1, 30, { startRadius: 3000 }), { maxSteps: 40000 })
 * Math.abs(measuredDeflection(state, 30, 1, 3000) - 4 / 30) < 0.02 // => true
 * ```
 */
export function measuredDeflection(state: RayState, impactParameter: number, M: number, startRadius: number): number {
  if (state.fate === 'captured') return Number.POSITIVE_INFINITY
  return state.exitAngle + startAngle(impactParameter, M, 1 / startRadius) - Math.PI
}
