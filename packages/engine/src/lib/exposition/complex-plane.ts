/**
 * The exposition record of the complex plane (`complexPlane`): the poles and zeros of a transfer function, the roots
 * of a polynomial or the eigenvalues of a matrix as points, with the branches of a root locus and the boundary of the
 * stable region (the imaginary axis for a continuous-time system, the unit circle for a discrete-time one). A session
 * probes it at each step to watch the poles move as a gain or a parameter changes.
 */

import { type Complex, MdmlError } from '../base/index.ts'
import { seal } from './_check.ts'

export type { Complex }

/** One branch of a root locus: the path a pole takes as the parameter runs. */
export interface LocusBranch {
  readonly name?: string
  readonly path: readonly Complex[]
}

/** Poles, zeros and branches in the complex plane. */
export interface ComplexPlane {
  readonly kind: 'complexPlane'
  /** `s` (continuous time: stable when the real part is negative) or `z` (discrete time: stable inside the unit circle). */
  readonly domain: 's' | 'z'
  readonly poles: readonly Complex[]
  readonly zeros?: readonly Complex[]
  /** The branches of a root locus, drawn as lines. */
  readonly locus?: readonly LocusBranch[]
  /** A name for each pole (`p1`, a frequency), written beside it. */
  readonly labels?: readonly string[]
  /** What the plane is (`closed loop at gain 2`), for the caption. */
  readonly caption?: string
}

/**
 * Whether a pole is in the stable region of its domain: a negative real part in `s`, a modulus below one in `z`.
 *
 * @param domain - `s` or `z`.
 * @param pole - The pole as `[real, imaginary]`.
 * @returns `true` when stable.
 * @example
 * ```ts
 * import { isStablePole } from 'mdml-engine/exposition'
 * isStablePole('s', [-1, 2]) // => true
 * isStablePole('z', [0.6, 0.9]) // => false
 * ```
 */
export function isStablePole(domain: 's' | 'z', pole: Complex): boolean {
  return domain === 's' ? pole[0] < 0 : Math.hypot(pole[0], pole[1]) < 1
}

/**
 * A complex plane, with every point checked to be finite and the labels to match the poles.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` when a point is not finite; `shape` when the labels do not match the poles.
 * @example
 * ```ts
 * import { complexPlane } from 'mdml-engine/exposition'
 * const r = complexPlane({ domain: 's', poles: [[-1, 2], [-1, -2]], zeros: [[-3, 0]] })
 * r.poles.length // => 2
 * ```
 */
export function complexPlane(record: Omit<ComplexPlane, 'kind'>): ComplexPlane {
  const check = (name: string, points: readonly Complex[]): void => {
    for (const [i, p] of points.entries())
      if (!(Number.isFinite(p[0]) && Number.isFinite(p[1])))
        throw new MdmlError('value', `complexPlane: ${name}[${i}] is not finite`)
  }
  check('poles', record.poles)
  check('zeros', record.zeros ?? [])
  for (const [k, b] of (record.locus ?? []).entries()) check(`locus[${k}]`, b.path)
  if (record.labels !== undefined && record.labels.length !== record.poles.length)
    throw new MdmlError(
      'shape',
      `complexPlane: labels has ${record.labels.length} entries, expected ${record.poles.length}`,
    )
  return seal({ kind: 'complexPlane', ...record })
}
