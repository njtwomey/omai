/**
 * Orbits of the two-body problem as the six classical elements (semi-major axis, eccentricity, inclination, longitude of
 * the ascending node, argument of periapsis, true anomaly) and as a position and velocity: the conversions both ways
 * for ellipses, parabolas and hyperbolas with the degenerate circular and equatorial cases handled, the analytic
 * propagation through Kepler's equation (the truth the integrators are measured against), and the conserved quantities
 * and standard orbit facts (vis-viva, period, escape speed, the area swept).
 */

import { MdmlError } from '../../base/index.ts'
import { conicKind, meanFromTrue, solveKepler, trueFromMean } from './kepler.ts'

/** A vector in three dimensions. */
export type Vec3 = readonly [number, number, number]

/**
 * The six elements of an orbit about a body of gravitational parameter mu. Angles are in radians. For a hyperbola
 * `a` is negative; for a parabola `a` is infinite and the semi-latus rectum `p` is what fixes the size.
 */
export interface OrbitalElements {
  /** Semi-major axis (negative for a hyperbola, `Infinity` for a parabola). */
  readonly a: number
  /** Eccentricity. */
  readonly e: number
  /** Inclination, in [0, pi]. */
  readonly i: number
  /** Longitude of the ascending node. */
  readonly raan: number
  /** Argument of periapsis. */
  readonly argp: number
  /** True anomaly. */
  readonly nu: number
  /** Semi-latus rectum a (1 - e^2); required when e = 1, derived otherwise. */
  readonly p?: number
}

/** A position and velocity. */
export interface StateVector {
  readonly r: Vec3
  readonly v: Vec3
}

const dot = (u: Vec3, v: Vec3): number => u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
const norm = (u: Vec3): number => Math.hypot(u[0], u[1], u[2])
const cross = (u: Vec3, v: Vec3): Vec3 => [
  u[1] * v[2] - u[2] * v[1],
  u[2] * v[0] - u[0] * v[2],
  u[0] * v[1] - u[1] * v[0],
]
const scale = (u: Vec3, s: number): Vec3 => [u[0] * s, u[1] * s, u[2] * s]
const minus = (u: Vec3, v: Vec3): Vec3 => [u[0] - v[0], u[1] - v[1], u[2] - v[2]]

/** The angle from u to v turning about the axis (right-handed), in (-pi, pi]. */
function signedAngle(u: Vec3, v: Vec3, axis: Vec3): number {
  return Math.atan2(dot(cross(u, v), axis) / norm(axis), dot(u, v))
}

/**
 * The semi-latus rectum of an element set: the given `p`, or a (1 - e^2).
 *
 * @param el - The elements.
 * @returns p.
 * @throws MdmlError with code `value` for a parabola without `p`, or a non-positive size.
 * @example
 * ```ts
 * import { semiLatusRectum } from 'mdml-engine/dynamics/celestial'
 * semiLatusRectum({ a: 2, e: 0.5, i: 0, raan: 0, argp: 0, nu: 0 }) // => 1.5
 * ```
 */
export function semiLatusRectum(el: OrbitalElements): number {
  const kind = conicKind(el.e)
  const p = el.p ?? (kind === 'parabolic' ? Number.NaN : el.a * (1 - el.e * el.e))
  if (!(p > 0) || !Number.isFinite(p)) {
    throw new MdmlError(
      'value',
      kind === 'parabolic' ? 'a parabola needs its semi-latus rectum p' : 'the orbit has no size: check a and e',
    )
  }
  return p
}

/**
 * The state vector from the classical elements: the perifocal position r = p / (1 + e cos nu) and velocity
 * sqrt(mu / p) (-sin nu, e + cos nu) rotated into the inertial frame by the node, inclination and periapsis angles.
 *
 * @param mu - Gravitational parameter G M of the central body.
 * @param el - The elements.
 * @returns The position and velocity.
 * @throws MdmlError with code `value` for a non-positive mu, a hyperbolic true anomaly beyond the asymptote, or a
 * parabola without p.
 * @concept orbital-elements
 * @example
 * ```ts
 * import { stateFromElements } from 'mdml-engine/dynamics/celestial'
 * // A circular orbit of radius 1 about a unit mass, at the node: the speed is 1, along y.
 * stateFromElements(1, { a: 1, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }) // => { r: [1, 0, 0], v: [0, 1, 0] }
 * ```
 */
export function stateFromElements(mu: number, el: OrbitalElements): StateVector {
  if (!(mu > 0)) throw new MdmlError('value', 'stateFromElements: mu must be positive')
  const p = semiLatusRectum(el)
  const { e, nu } = el
  const denominator = 1 + e * Math.cos(nu)
  if (!(denominator > 0)) throw new MdmlError('value', 'stateFromElements: the true anomaly is beyond the asymptote')
  const radius = p / denominator
  const speed = Math.sqrt(mu / p)
  const cn = Math.cos(el.raan)
  const sn = Math.sin(el.raan)
  const ci = Math.cos(el.i)
  const si = Math.sin(el.i)
  const cw = Math.cos(el.argp)
  const sw = Math.sin(el.argp)
  // The columns of the rotation R3(-raan) R1(-i) R3(-argp), taking perifocal axes P, Q to inertial.
  const P: Vec3 = [cn * cw - sn * sw * ci, sn * cw + cn * sw * ci, sw * si]
  const Q: Vec3 = [-cn * sw - sn * cw * ci, -sn * sw + cn * cw * ci, cw * si]
  const xr = radius * Math.cos(nu)
  const yr = radius * Math.sin(nu)
  const xv = -speed * Math.sin(nu)
  const yv = speed * (e + Math.cos(nu))
  const mix = (a: number, b: number): Vec3 => [a * P[0] + b * Q[0], a * P[1] + b * Q[1], a * P[2] + b * Q[2]]
  return { r: mix(xr, yr), v: mix(xv, yv) }
}

/** Below this an eccentricity or inclination counts as zero in {@link elementsFromState}. */
const DEGENERATE = 1e-11

/**
 * The classical elements from a state vector. Where an angle is undefined the usual convention is used: for a circular
 * orbit the argument of periapsis is 0 and the true anomaly is the argument of latitude (the angle from the node); for
 * an equatorial orbit the node is 0 and the argument of periapsis is measured from the x axis; for both, the true
 * anomaly is the true longitude.
 *
 * @param mu - Gravitational parameter.
 * @param r - Position.
 * @param v - Velocity.
 * @returns The elements, with `p` always set.
 * @throws MdmlError with code `value` for a rectilinear orbit (zero angular momentum) or non-positive mu.
 * @concept orbital-elements
 * @example
 * ```ts
 * import { elementsFromState } from 'mdml-engine/dynamics/celestial'
 * const el = elementsFromState(1, [1, 0, 0], [0, 1.1, 0])
 * [Math.round(el.a * 1e6) / 1e6, Math.round(el.e * 1e6) / 1e6] // => [1.265823, 0.21]
 * ```
 */
export function elementsFromState(mu: number, r: Vec3, v: Vec3): OrbitalElements {
  if (!(mu > 0)) throw new MdmlError('value', 'elementsFromState: mu must be positive')
  const rn = norm(r)
  const h = cross(r, v)
  const hn = norm(h)
  if (!(hn > 0)) throw new MdmlError('value', 'elementsFromState: zero angular momentum (a rectilinear orbit)')
  const node: Vec3 = [-h[1], h[0], 0]
  const nn = norm(node)
  const rv = dot(r, v)
  const evec = scale(minus(scale(r, dot(v, v) - mu / rn), scale(v, rv)), 1 / mu)
  const e = norm(evec)
  const energy = dot(v, v) / 2 - mu / rn
  const p = (hn * hn) / mu
  const a = Math.abs(e - 1) <= DEGENERATE ? Number.POSITIVE_INFINITY : -mu / (2 * energy)
  const i = Math.atan2(Math.hypot(h[0], h[1]), h[2])
  const equatorial = nn <= DEGENERATE * hn
  const circular = e <= DEGENERATE
  const raan = equatorial ? 0 : Math.atan2(node[1], node[0])
  // Angles are measured in the direction of motion about h, from the node where there is one and else from the x axis.
  const origin: Vec3 = equatorial ? [1, 0, 0] : scale(node, 1 / nn)
  const argp = circular ? 0 : signedAngle(origin, evec, h)
  const nu = circular ? signedAngle(origin, r, h) : signedAngle(evec, r, h)
  const wrap = (x: number): number => (x < 0 ? x + 2 * Math.PI : x)
  return { a, e, i, raan: wrap(raan), argp: wrap(argp), nu: conicKind(e) === 'elliptic' ? wrap(nu) : nu, p }
}

/**
 * The mean motion n = sqrt(mu / |a|^3) (for a parabola the rate 2 sqrt(mu / p^3) of the parabolic mean anomaly).
 *
 * @param mu - Gravitational parameter.
 * @param el - The elements.
 * @returns The rate at which the mean anomaly advances.
 * @example
 * ```ts
 * import { meanMotion } from 'mdml-engine/dynamics/celestial'
 * meanMotion(1, { a: 1, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }) // => 1
 * ```
 */
export function meanMotion(mu: number, el: OrbitalElements): number {
  if (conicKind(el.e) === 'parabolic') return 2 * Math.sqrt(mu / semiLatusRectum(el) ** 3)
  return Math.sqrt(mu / Math.abs(el.a) ** 3)
}

/**
 * The elements a time dt later, with nothing changed but the true anomaly: the mean anomaly advances at the mean motion
 * and Kepler's equation turns it back into a position (on an ellipse the true anomaly is returned in [0, 2 pi)). This is the analytic solution of the two-body problem, the
 * truth against which a numerical orbit is measured.
 *
 * @param mu - Gravitational parameter.
 * @param el - The elements now.
 * @param dt - Time to advance (negative goes back).
 * @returns The elements at t + dt.
 * @concept keplers-equation
 * @example
 * ```ts
 * import { propagateKepler } from 'mdml-engine/dynamics/celestial'
 * // A full period later the planet is where it started.
 * const el = { a: 1, e: 0.3, i: 0.2, raan: 0.1, argp: 0.5, nu: 1 }
 * Math.abs(propagateKepler(1, el, 2 * Math.PI).nu - 1) < 1e-12 // => true
 * ```
 */
export function propagateKepler(mu: number, el: OrbitalElements, dt: number): OrbitalElements {
  const kind = conicKind(el.e)
  const M = meanFromTrue(el.nu, el.e) + meanMotion(mu, el) * dt
  const nu = kind === 'parabolic' ? solveKepler(M, el.e).trueAnomaly : trueFromMean(M, el.e)
  return { ...el, nu: kind === 'elliptic' ? nu - 2 * Math.PI * Math.floor(nu / (2 * Math.PI)) : nu }
}

/**
 * The state vector at time dt after the elements, by the analytic solution.
 *
 * @param mu - Gravitational parameter.
 * @param el - The elements at time 0.
 * @param dt - The time.
 * @returns The position and velocity.
 * @example
 * ```ts
 * import { keplerState } from 'mdml-engine/dynamics/celestial'
 * const s = keplerState(1, { a: 1, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }, Math.PI / 2)
 * [Math.round(s.r[1] * 1e12) / 1e12, Math.round(s.v[0] * 1e12) / 1e12] // => [1, -1]
 * ```
 */
export function keplerState(mu: number, el: OrbitalElements, dt: number): StateVector {
  return stateFromElements(mu, propagateKepler(mu, el, dt))
}

/**
 * The period of an elliptic orbit, 2 pi sqrt(a^3 / mu).
 *
 * @param mu - Gravitational parameter.
 * @param a - Semi-major axis (positive).
 * @returns The period.
 * @throws MdmlError with code `value` for a non-positive a (an open orbit has no period).
 * @example
 * ```ts
 * import { orbitalPeriod } from 'mdml-engine/dynamics/celestial'
 * orbitalPeriod(1, 1) // => 6.283185307179586
 * ```
 */
export function orbitalPeriod(mu: number, a: number): number {
  if (!(a > 0)) throw new MdmlError('value', 'orbitalPeriod: only a bound orbit (a > 0) has a period')
  return 2 * Math.PI * Math.sqrt((a * a * a) / mu)
}

/**
 * The speed at distance r on an orbit of semi-major axis a (vis-viva): v^2 = mu (2 / r - 1 / a). Use `a = Infinity`
 * for the parabola.
 *
 * @param mu - Gravitational parameter.
 * @param r - Distance from the central body.
 * @param a - Semi-major axis.
 * @returns The speed.
 * @example
 * ```ts
 * import { visViva } from 'mdml-engine/dynamics/celestial'
 * visViva(1, 1, 1) // => 1
 * ```
 */
export function visViva(mu: number, r: number, a: number): number {
  return Math.sqrt(mu * (2 / r - 1 / a))
}

/**
 * The escape speed at distance r, sqrt(2 mu / r): the speed of the parabola there.
 *
 * @param mu - Gravitational parameter.
 * @param r - Distance.
 * @returns The speed.
 * @example
 * ```ts
 * import { escapeSpeed } from 'mdml-engine/dynamics/celestial'
 * escapeSpeed(1, 2) // => 1
 * ```
 */
export function escapeSpeed(mu: number, r: number): number {
  return Math.sqrt((2 * mu) / r)
}

/** The conserved quantities of a Kepler orbit, from its state. */
export interface KeplerInvariants {
  /** The specific energy v^2 / 2 - mu / r. */
  readonly energy: number
  /** The specific angular momentum vector r x v. */
  readonly h: Vec3
  /** Its magnitude. */
  readonly hNorm: number
  /** The eccentricity (Laplace-Runge-Lenz) vector, pointing at periapsis. */
  readonly eccentricity: Vec3
}

/**
 * The quantities a Kepler orbit conserves: the specific energy, the angular momentum (Kepler's second law: the area
 * swept per time is half its magnitude) and the Laplace-Runge-Lenz eccentricity vector (the orbit closes because it
 * is fixed).
 *
 * @param mu - Gravitational parameter.
 * @param r - Position.
 * @param v - Velocity.
 * @returns The invariants.
 * @concept two-body-problem
 * @example
 * ```ts
 * import { keplerInvariants } from 'mdml-engine/dynamics/celestial'
 * keplerInvariants(1, [1, 0, 0], [0, 1, 0]).energy // => -0.5
 * ```
 */
export function keplerInvariants(mu: number, r: Vec3, v: Vec3): KeplerInvariants {
  const rn = norm(r)
  const h = cross(r, v)
  const eccentricity = scale(minus(scale(r, dot(v, v) - mu / rn), scale(v, dot(r, v))), 1 / mu)
  return { energy: dot(v, v) / 2 - mu / rn, h, hNorm: norm(h), eccentricity }
}
