// Small helpers for the option tables of the graph domain: an option whose knobs are read with their defaults filling
// those not given. Not re-exported.
import { defaults, type dict, type Option, type SpaceValue } from '../space/index.ts'

/** The knobs of a configuration read with the option's defaults filling those not given. */
export type Knobs = (name: string) => number | string

/** Read the knobs of a configuration against the space of the option. */
export function knobsOf(space: ReturnType<typeof dict>, config: Readonly<Record<string, SpaceValue>>): Knobs {
  const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
  return (name) => full[name] as number | string
}

/** An option of a table: its description, properties, knob space and the builder over the knobs. */
export function knobOption<T, Props>(
  description: string,
  properties: Props,
  space: ReturnType<typeof dict>,
  build: (k: Knobs) => T,
): Option<T, Props> {
  return { description, properties, space, make: (config) => build(knobsOf(space, config)) }
}
