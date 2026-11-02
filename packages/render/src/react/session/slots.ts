/**
 * The comparison over a slot: one variant per option of a union at a path of a method's configuration (every solver,
 * every loss), each with the option's defaults, ready for `useSessions`. The rest of the configuration is the page's
 * shared options, so the slot's option is the only difference between the variants.
 */

import { MdmlError } from 'mdml-engine/base'
import { defaults, type Space, type SpaceValue } from 'mdml-engine/space'
import type { Config } from 'mdml-engine/trace'
import type { ControlPath } from '../controls/space-controls.ts'
import { settleValue, valueAt, withValueAt } from '../controls/space-controls.ts'
import type { Variant } from './useSession.ts'

/** Which options of the slot to compare, and how to label them. */
export interface SlotOptions {
  /** Only these options, in this order (default every option of the union). */
  readonly only?: readonly string[]
  /** Whether an option may be compared (a mode table's `admissible`); the others are left out. */
  readonly admissible?: (option: string) => boolean
  /** A label for an option (default its name). */
  readonly label?: (option: string) => string
  /** The knobs the page holds for each option (a settings popover each), over the option's defaults. */
  readonly knobs?: Readonly<Record<string, Readonly<Record<string, SpaceValue>>>>
}

const recordOf = (v: SpaceValue | undefined): Readonly<Record<string, SpaceValue>> =>
  typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Readonly<Record<string, SpaceValue>>) : {}

// The space at a path of a value: dict keys, and through a union the option the value names (else its default).
function spaceAt(space: Space, value: SpaceValue, path: ControlPath): Space | undefined {
  let s: Space | undefined = space
  let v: SpaceValue | undefined = value
  for (const key of path) {
    if (s?.kind === 'union') {
      const named: SpaceValue | undefined = recordOf(v)[s.tag ?? 'kind']
      const option: string =
        typeof named === 'string' && named in s.options ? named : (s.default ?? Object.keys(s.options)[0] ?? '')
      s = s.options[option]
    }
    if (s?.kind !== 'dict') return undefined
    s = s.spaces[String(key)]
    v = recordOf(v)[String(key)]
  }
  return s
}

/**
 * One variant per option of the union at `path`, for `useSessions`: the label is the option's name and the config is
 * the top-level key of the configuration holding the slot, with that option chosen (its own knobs at their defaults,
 * or as `base` has them when `base` already names it) and the slot's siblings as `base` (else the defaults) has them.
 *
 * @param space - The method's configuration space (a dict).
 * @param path - Where the union sits: `'solver'`, `'model.solver'` or `['model', 'solver']`.
 * @param base - The configuration the variants share, over the space's defaults.
 * @param options - Which options to compare and how to label them.
 * @returns The variants, in the union's order.
 * @throws MdmlError with code `value` when the path holds no union.
 * @index function Make one comparison variant per option of a slot: `variantsOfSlot(method.config, 'solver')`
 */
export function variantsOfSlot<C extends Config>(
  space: Space,
  path: string | ControlPath,
  base: Partial<C> = {},
  options: SlotOptions = {},
): Variant<C>[] {
  const at: ControlPath = typeof path === 'string' ? path.split('.') : path
  const start = settleValue(space, { ...recordOf(defaults(space)), ...(base as Readonly<Record<string, SpaceValue>>) })
  const slot = spaceAt(space, start, at)
  if (slot?.kind !== 'union')
    throw new MdmlError('value', `variantsOfSlot: '${at.join('.')}' is not a union of the space`)
  const tag = slot.tag ?? 'kind'
  const top = String(at[0])
  const names = (options.only ?? Object.keys(slot.options)).filter(
    (n) => n in slot.options && (options.admissible?.(n) ?? true),
  )
  const kept = recordOf(valueAt(start, at))
  return names.map((name): Variant<C> => {
    // The option's knobs: those the page holds for it, else the base's when it is the base's own, over the defaults.
    const own = recordOf(defaults(slot.options[name] as Space))
    const knobs = { ...own, ...(options.knobs?.[name] ?? (kept[tag] === name ? kept : {})) }
    const whole = withValueAt(space, start, at, { ...knobs, [tag]: name })
    return {
      label: options.label?.(name) ?? name,
      config: { [top]: valueAt(whole, [top]) } as unknown as Partial<C>,
    }
  })
}
