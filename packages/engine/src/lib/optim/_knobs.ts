/**
 * The tolerance specs and the typed readers the solver tables of `optim` share: the gradient tolerance and the L-BFGS
 * reduction tolerance as one knob each with their defaults (the same numbers the algorithms fall back to), and the
 * readers that take a number, a flag or a choice from a mode's configuration without a cast at every knob. Private to
 * the domain.
 */

import { OmaiError } from '../base/index.ts'
import { type ModeConfig, type RealSpace, real, type SpaceValue } from '../space/index.ts'

/** The default gradient tolerance of the vector methods (scipy's gtol): absolute, on the infinity norm. */
export const defaultGradTol = 1e-5

/** The default relative reduction tolerance of L-BFGS and L-BFGS-B (scipy's `ftol`, the machine epsilon times 1e7). */
export const defaultFtol = 2.220446049250313e-9

/**
 * The gradient tolerance every gradient-based solver stops at (scipy's gtol): absolute, on the infinity norm, so an
 * objective whose gradients are everywhere below it (data in very small units) stops at its start: rescale such a
 * problem, lower the tolerance, or give the method `gradTolRel`.
 */
export const gradTolKnob: RealSpace = real(1e-12, 1e-2, {
  log: true,
  default: defaultGradTol,
  label: 'gradient tolerance (absolute, on the infinity norm)',
})

/** The relative reduction tolerance of the L-BFGS family. */
export const ftolKnob: RealSpace = real(1e-16, 1e-3, {
  log: true,
  default: defaultFtol,
  label: 'relative reduction tolerance',
})

/**
 * A number knob of a mode's configuration, absent when the configuration is partial (the algorithm's own default then
 * applies).
 *
 * @param k - The mode's knobs.
 * @param name - The knob.
 * @returns The number, or `undefined` when the knob is absent (a partial configuration, or a conditional knob of an
 *   inactive branch).
 * @throws OmaiError with code `value` when the knob is present and not a number.
 */
export function numberKnob(k: ModeConfig, name: string): number | undefined {
  const v: SpaceValue | undefined = k[name]
  if (v === undefined || typeof v === 'number') return v
  throw new OmaiError('value', `knob '${name}': expected a number, got ${typeof v}`)
}

/**
 * A flag knob of a mode's configuration.
 *
 * @param k - The mode's knobs.
 * @param name - The knob.
 * @returns The flag, or `undefined` when the knob is absent.
 * @throws OmaiError with code `value` when the knob is present and not a boolean.
 */
export function boolKnob(k: ModeConfig, name: string): boolean | undefined {
  const v: SpaceValue | undefined = k[name]
  if (v === undefined || typeof v === 'boolean') return v
  throw new OmaiError('value', `knob '${name}': expected a boolean, got ${typeof v}`)
}

/**
 * A choice knob of a mode's configuration, as the union its options name.
 *
 * @param k - The mode's knobs.
 * @param name - The knob.
 * @returns The chosen string, or `undefined` when the knob is absent.
 * @throws OmaiError with code `value` when the knob is present and not a string.
 */
export function choiceKnob<T extends string>(k: ModeConfig, name: string): T | undefined {
  const v: SpaceValue | undefined = k[name]
  if (v === undefined) return undefined
  if (typeof v === 'string') return v as T
  throw new OmaiError('value', `knob '${name}': expected a choice, got ${typeof v}`)
}

/**
 * The entries of a record that are not `undefined`, for spreading into an options object (the options of the
 * algorithms are optional, not `undefined`-valued).
 *
 * @param entries - The knobs read from a configuration.
 * @returns The record without its `undefined` entries.
 */
export function given<T extends Record<string, unknown>>(entries: T): { [K in keyof T]?: Exclude<T[K], undefined> } {
  const out: Record<string, unknown> = {}
  for (const [name, v] of Object.entries(entries)) if (v !== undefined) out[name] = v
  return out as { [K in keyof T]?: Exclude<T[K], undefined> }
}
