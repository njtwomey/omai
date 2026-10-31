/**
 * Named n-body configurations that carry their truth: a two-body orbit chosen by its launch speed (circular, elliptic,
 * parabolic, hyperbolic) with the period, energy and eccentricity it must have; the Lagrange equilateral triangle that
 * rotates rigidly; the figure-eight choreography of Chenciner and Montgomery with its period; the Pythagorean problem of
 * Burrau; and the planets from their mean elements, in astronomical units, days and solar masses.
 */

import { MdmlError } from '../../base/index.ts'
import { type OrbitalElements, orbitalPeriod, stateFromElements } from './elements.ts'
import { trueFromMean, wrapAngle } from './kepler.ts'
import { type NBodyOptions, type NBodySystem, nBodyHamiltonian } from './nbody.ts'

/** A starting configuration of n bodies, with what is known about its motion. */
export interface NBodyConfiguration {
  readonly name: string
  readonly description: string
  readonly masses: readonly number[]
  /** Positions, one row per body. */
  readonly positions: readonly (readonly number[])[]
  /** Velocities, one row per body. */
  readonly velocities: readonly (readonly number[])[]
  readonly G: number
  readonly dim: 2 | 3
  /** The period after which the motion returns to the start (as a whole or in shape), when it has one. */
  readonly period?: number
  /** The kind of motion, in words (`circular`, `choreography`, ...). */
  readonly kind: string
  /** The total energy the configuration must have (exact where the truth gives it). */
  readonly energy?: number
}

/**
 * The system of a configuration: its Hamiltonian with the configuration's G.
 *
 * @param c - The configuration.
 * @param options - Softening.
 * @returns The system.
 * @example
 * ```ts
 * import { configurationSystem, PYTHAGOREAN } from 'mdml-engine/dynamics/celestial'
 * configurationSystem(PYTHAGOREAN).masses // => [3, 4, 5]
 * ```
 */
export function configurationSystem(c: NBodyConfiguration, options: Pick<NBodyOptions, 'softening'> = {}): NBodySystem {
  return nBodyHamiltonian(c.masses, c.dim, { G: c.G, ...options })
}

/**
 * The positions and momenta of a configuration as flat arrays, `[n * dim]` each (momentum is mass times velocity).
 *
 * @param c - The configuration.
 * @returns Positions and momenta.
 * @example
 * ```ts
 * import { configurationState, PYTHAGOREAN } from 'mdml-engine/dynamics/celestial'
 * Array.from(configurationState(PYTHAGOREAN).q) // => [1, 3, -2, -1, 1, -1]
 * ```
 */
export function configurationState(c: NBodyConfiguration): { readonly q: Float64Array; readonly p: Float64Array } {
  const q = Float64Array.from(c.positions.flat())
  const p = Float64Array.from(c.velocities.flatMap((v, i) => v.map((x) => x * (c.masses[i] as number))))
  return { q, p }
}

/**
 * The Pythagorean three-body problem (Burrau, 1913): masses 3, 4 and 5 at rest at the vertices of a 3-4-5 right triangle
 * opposite the sides of those lengths. The motion is a chaotic dance that ends with two bodies bound and the third
 * ejected. G = 1; the energy is -(12/5 + 15/4 + 20/3).
 */
export const PYTHAGOREAN: NBodyConfiguration = {
  name: 'pythagorean',
  description: 'Masses 3, 4, 5 released from rest at the corners of a 3-4-5 triangle: chaos, then an ejection.',
  masses: [3, 4, 5],
  positions: [
    [1, 3],
    [-2, -1],
    [1, -1],
  ],
  velocities: [
    [0, 0],
    [0, 0],
    [0, 0],
  ],
  G: 1,
  dim: 2,
  kind: 'chaotic',
  energy: -(12 / 5 + 15 / 4 + 20 / 3),
}

/**
 * The figure-eight choreography (Chenciner and Montgomery, 2000; Simo's numerical values): three unit masses chase one
 * another along a single figure-eight curve. G = 1, period 6.32591398 (the initial conditions are given to eight
 * digits, so the motion closes to about that accuracy).
 */
export const FIGURE_EIGHT: NBodyConfiguration = {
  name: 'figure-eight',
  description: 'Three equal masses chasing each other round one figure-eight, a stable periodic orbit.',
  masses: [1, 1, 1],
  positions: [
    [-0.97000436, 0.24308753],
    [0.97000436, -0.24308753],
    [0, 0],
  ],
  velocities: [
    [0.466203685, 0.43236573],
    [0.466203685, 0.43236573],
    [-0.93240737, -0.86473146],
  ],
  G: 1,
  dim: 2,
  period: 6.32591398,
  kind: 'choreography',
}

/**
 * Lagrange's equilateral triangle: n = 3 bodies of mass m at the corners of an equilateral triangle of side s, spinning
 * rigidly about their centre of mass with angular speed sqrt(3 G m / s^3) (period 2 pi / that). It is an exact solution
 * of the three-body problem (and unstable for equal masses).
 *
 * @param s - The side (default 1).
 * @param m - The mass of each body (default 1).
 * @param G - The gravitational constant (default 1).
 * @returns The configuration, with its period and energy.
 * @throws MdmlError with code `value` for a non-positive side, mass or G.
 * @example
 * ```ts
 * import { lagrangeTriangle } from 'mdml-engine/dynamics/celestial'
 * lagrangeTriangle().period // => 3.6275987284684357
 * ```
 */
export function lagrangeTriangle(s = 1, m = 1, G = 1): NBodyConfiguration {
  if (!(s > 0 && m > 0 && G > 0)) throw new MdmlError('value', 'lagrangeTriangle: side, mass and G must be positive')
  const omega = Math.sqrt((3 * G * m) / s ** 3)
  const radius = s / Math.sqrt(3)
  const angles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3]
  return {
    name: 'lagrange-triangle',
    description: 'Three equal masses at the corners of an equilateral triangle that turns rigidly about its centre.',
    masses: [m, m, m],
    positions: angles.map((a) => [radius * Math.cos(a), radius * Math.sin(a)]),
    velocities: angles.map((a) => [-radius * omega * Math.sin(a), radius * omega * Math.cos(a)]),
    G,
    dim: 2,
    period: (2 * Math.PI) / omega,
    kind: 'rigid rotation',
    energy: (-3 * G * m * m) / s + 3 * 0.5 * m * (radius * omega) ** 2,
  }
}

/**
 * Two bodies launched from their closest approach at a multiple of the circular speed: 1 is a circle, between 1 and
 * sqrt(2) an ellipse, sqrt(2) a parabola, above it a hyperbola (below 1 the launch point is the far end of the
 * ellipse). The pair is in its centre-of-mass frame (the centre at rest at the origin). The truth: the semi-major axis
 * rp / (2 - f^2), the eccentricity |f^2 - 1|, and for a bound orbit the period and the total energy
 * -G m1 m2 / (2 a).
 *
 * @param speed - The launch speed as a multiple of the circular speed at the separation.
 * @param options - The masses (default 1 and 1), the closest separation (default 1) and G (default 1).
 * @returns The configuration with its kind, and its period and energy where the orbit is bound.
 * @throws MdmlError with code `value` for non-positive arguments.
 * @concept two-body-problem
 * @example
 * ```ts
 * import { twoBody } from 'mdml-engine/dynamics/celestial'
 * const c = twoBody(1.1)
 * [c.kind, Math.round(c.period ?? 0)] // => ['elliptic', 6]
 * ```
 */
export function twoBody(
  speed: number,
  options: { readonly m1?: number; readonly m2?: number; readonly rp?: number; readonly G?: number } = {},
): NBodyConfiguration {
  const { m1 = 1, m2 = 1, rp = 1, G = 1 } = options
  if (!(speed > 0 && m1 > 0 && m2 > 0 && rp > 0 && G > 0))
    throw new MdmlError('value', 'twoBody: arguments must be positive')
  const M = m1 + m2
  const mu = G * M
  const vc = Math.sqrt(mu / rp)
  const v = speed * vc
  const e = Math.abs(speed * speed - 1)
  const f1 = m2 / M
  const f2 = m1 / M
  const kind =
    Math.abs(speed - Math.SQRT2) < 1e-12
      ? 'parabolic'
      : speed < Math.SQRT2
        ? Math.abs(speed - 1) < 1e-12
          ? 'circular'
          : 'elliptic'
        : 'hyperbolic'
  const a = rp / (2 - speed * speed)
  const bound = speed < Math.SQRT2 - 1e-12
  return {
    name: 'two-body',
    description: `Two bodies launched at ${speed} times the circular speed: a ${kind} orbit of eccentricity ${e.toFixed(3)}.`,
    masses: [m1, m2],
    positions: [
      [-f1 * rp, 0],
      [f2 * rp, 0],
    ],
    velocities: [
      [0, -f1 * v],
      [0, f2 * v],
    ],
    G,
    dim: 2,
    kind,
    ...(bound ? { period: orbitalPeriod(mu, a), energy: (-G * m1 * m2) / (2 * a) } : {}),
    ...(kind === 'parabolic' ? { energy: 0 } : {}),
  }
}

/** The Gauss gravitational constant k, in AU^(3/2) per solar mass^(1/2) per day: k^2 is G in AU^3 / (solar mass day^2). */
export const GAUSS_K = 0.01720209895

/** The mean orbital elements of a planet at J2000 (Standish; valid 1800 to 2050) and its mass in solar masses. */
export interface Planet {
  readonly name: string
  /** Mass in solar masses. */
  readonly mass: number
  /** Semi-major axis, AU. */
  readonly a: number
  readonly e: number
  /** Inclination to the ecliptic, degrees. */
  readonly inclination: number
  /** Mean longitude, degrees. */
  readonly meanLongitude: number
  /** Longitude of perihelion, degrees. */
  readonly longitudePerihelion: number
  /** Longitude of the ascending node, degrees. */
  readonly node: number
}

/** The eight planets (the Earth entry is the Earth-Moon barycentre's orbit with the Earth's mass). */
export const PLANETS: readonly Planet[] = [
  {
    name: 'Mercury',
    mass: 1.6601e-7,
    a: 0.38709927,
    e: 0.20563593,
    inclination: 7.00497902,
    meanLongitude: 252.2503235,
    longitudePerihelion: 77.45779628,
    node: 48.33076593,
  },
  {
    name: 'Venus',
    mass: 2.4478e-6,
    a: 0.72333566,
    e: 0.00677672,
    inclination: 3.39467605,
    meanLongitude: 181.9790995,
    longitudePerihelion: 131.60246718,
    node: 76.67984255,
  },
  {
    name: 'Earth',
    mass: 3.0035e-6,
    a: 1.00000261,
    e: 0.01671123,
    inclination: -0.00001531,
    meanLongitude: 100.46457166,
    longitudePerihelion: 102.93768193,
    node: 0,
  },
  {
    name: 'Mars',
    mass: 3.2271e-7,
    a: 1.52371034,
    e: 0.0933941,
    inclination: 1.84969142,
    meanLongitude: -4.55343205,
    longitudePerihelion: -23.94362959,
    node: 49.55953891,
  },
  {
    name: 'Jupiter',
    mass: 9.5479e-4,
    a: 5.202887,
    e: 0.04838624,
    inclination: 1.30439695,
    meanLongitude: 34.39644051,
    longitudePerihelion: 14.72847983,
    node: 100.47390909,
  },
  {
    name: 'Saturn',
    mass: 2.8588e-4,
    a: 9.53667594,
    e: 0.05386179,
    inclination: 2.48599187,
    meanLongitude: 49.95424423,
    longitudePerihelion: 92.59887831,
    node: 113.66242448,
  },
  {
    name: 'Uranus',
    mass: 4.3662e-5,
    a: 19.18916464,
    e: 0.04725744,
    inclination: 0.77263783,
    meanLongitude: 313.23810451,
    longitudePerihelion: 170.9542763,
    node: 74.01692503,
  },
  {
    name: 'Neptune',
    mass: 5.1514e-5,
    a: 30.06992276,
    e: 0.00859048,
    inclination: 1.77004347,
    meanLongitude: -55.12002969,
    longitudePerihelion: 44.96476227,
    node: 131.78422574,
  },
]

/** The classical elements of a planet in radians, with the true anomaly from its mean longitude by Kepler's equation. */
export function planetElements(planet: Planet): OrbitalElements {
  const rad = Math.PI / 180
  const argp = (planet.longitudePerihelion - planet.node) * rad
  const M = wrapAngle((planet.meanLongitude - planet.longitudePerihelion) * rad)
  return {
    a: planet.a,
    e: planet.e,
    i: planet.inclination * rad,
    raan: planet.node * rad,
    argp,
    nu: trueFromMean(M, planet.e),
  }
}

/**
 * The Sun and planets at J2000 from their mean elements (heliocentric elements turned into states with Kepler's
 * equation, the Sun's reflex motion set so that the barycentre is at rest at the origin). Units: AU, days, solar
 * masses, with G = k^2. These are mean elements, not an ephemeris: the positions are right to the accuracy of the
 * elements (a fraction of a degree), which is enough for the shapes and periods and for testing integrators, not for
 * pointing a telescope. The period of each planet is its Kepler period about the Sun.
 *
 * @param names - Which planets (default all eight).
 * @returns The configuration, Sun first.
 * @throws MdmlError with code `value` for an unknown name.
 * @concept solar-system
 * @example
 * ```ts
 * import { solarSystem } from 'mdml-engine/dynamics/celestial'
 * solarSystem(['Earth']).masses.length // => 2
 * ```
 */
export function solarSystem(names?: readonly string[]): NBodyConfiguration {
  const chosen =
    names === undefined
      ? PLANETS
      : names.map((n) => {
          const p = PLANETS.find((q) => q.name.toLowerCase() === n.toLowerCase())
          if (p === undefined) throw new MdmlError('value', `solarSystem: unknown planet '${n}'`)
          return p
        })
  const G = GAUSS_K * GAUSS_K
  const masses = [1, ...chosen.map((p) => p.mass)]
  const total = masses.reduce((a, b) => a + b, 0)
  const rows = chosen.map((p) => stateFromElements(G * (1 + p.mass), planetElements(p)))
  const centre = [0, 0, 0]
  const drift = [0, 0, 0]
  rows.forEach((s, i) => {
    for (let k = 0; k < 3; k++) {
      centre[k] = (centre[k] ?? 0) + ((chosen[i] as Planet).mass * (s.r[k] as number)) / total
      drift[k] = (drift[k] ?? 0) + ((chosen[i] as Planet).mass * (s.v[k] as number)) / total
    }
  })
  const positions = [[...centre].map((c) => -c), ...rows.map((s) => s.r.map((x, k) => x - (centre[k] as number)))]
  const velocities = [[...drift].map((c) => -c), ...rows.map((s) => s.v.map((x, k) => x - (drift[k] as number)))]
  return {
    name: 'solar-system',
    description: 'The Sun and planets at J2000 from their mean elements, in AU, days and solar masses.',
    masses,
    positions,
    velocities,
    G,
    dim: 3,
    kind: 'planetary',
  }
}
