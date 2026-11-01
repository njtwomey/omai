/**
 * The pure rules that turn an engine `Space` into controls: which control each dimension calls for (log-scaled or
 * stepped number fields, segmented controls or selects, toggles) at which path of the value, and reading and updating a
 * value by path while keeping conditional dimensions in line with their conditions.
 */

import { activeDims, defaults, type Space, type SpaceValue } from 'mdml-engine/space'
import type { NumberSpec } from './number.ts'
import type { OptionInput, OptionValue } from './options.ts'

/** Where a control's value sits inside a space's value: dict keys and tuple or vector indices. */
export type ControlPath = readonly (string | number)[]

interface ControlBase {
  /** A stable React key and DOM id: the path joined by dots. */
  readonly id: string
  readonly path: ControlPath
  readonly label: string
  readonly description?: string
  readonly unit?: string
}

/** One control generated from a space: what to draw, at which path of the value. */
export type ControlSpec =
  | (ControlBase & { readonly kind: 'number'; readonly spec: NumberSpec })
  | (ControlBase & {
      readonly kind: 'choice' | 'select'
      readonly options: readonly OptionValue[]
      /** A line on each option, by its text (the option's tooltip). */
      readonly descriptions?: Readonly<Record<string, string>>
    })
  | (ControlBase & { readonly kind: 'toggle' })
  | (ControlBase & { readonly kind: 'group'; readonly children: readonly ControlSpec[] })

/** Options of {@link spaceControls}. */
export interface SpaceControlOptions {
  /** Choices with at most this many short options are segmented controls; longer ones are selects (default 4). */
  readonly segmentedUpTo?: number
}

// Segments share a half-width panel column: a few short options, and not too much text in all (four of six to eight
// letters already clip).
const isShort = (options: readonly OptionValue[], upTo: number): boolean =>
  options.length <= upTo &&
  options.every((o) => String(o).length <= 10) &&
  options.reduce<number>((n, o) => n + String(o).length + 4, 0) <= 36

function meta(space: Space, path: ControlPath, fallback: string): ControlBase {
  return {
    id: path.join('.'),
    path,
    label: space.label ?? fallback,
    ...(space.description === undefined ? {} : { description: space.description }),
    ...(space.unit === undefined ? {} : { unit: space.unit }),
  }
}

const plain = (path: ControlPath, label: string): ControlBase => ({ id: path.join('.'), path, label })

function asRecord(value: SpaceValue | undefined): Readonly<Record<string, SpaceValue>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Readonly<Record<string, SpaceValue>>)
    : {}
}

/** The keys of a dict space that are active for a value (its conditions hold), in the space's order. */
function activeKeys(space: Space & { readonly kind: 'dict' }, value: SpaceValue): Set<string> {
  return new Set(activeDims(space, value).map((p) => p.split('.')[0] as string))
}

/**
 * The controls a space calls for, for a value of it: reals and integers as number fields (log-scaled, stepped and
 * bounded as the space says), short choices as segmented controls and long ones as selects, flags as toggles, discrete
 * values as whole-number fields, boxes, multi-discrete and multi-binary values as groups of fields, and dicts and
 * tuples as groups. Conditional dict keys appear only while their condition holds.
 *
 * @param space - The space (usually a dict of named dimensions).
 * @param value - Its current value (decides which conditional keys show).
 * @param options - When choices become selects.
 * @returns The control specifications; a dict gives its children directly.
 */
export function spaceControls(space: Space, value: SpaceValue, options: SpaceControlOptions = {}): ControlSpec[] {
  const upTo = options.segmentedUpTo ?? 4
  const walk = (s: Space, v: SpaceValue | undefined, path: ControlPath, name: string): ControlSpec => {
    const base = meta(s, path, name)
    switch (s.kind) {
      case 'real':
        return {
          ...base,
          kind: 'number',
          spec: {
            min: s.low,
            max: s.high,
            ...(s.log === undefined ? {} : { log: s.log }),
            ...(s.step === undefined ? {} : { step: s.step }),
          },
        }
      case 'int':
        return {
          ...base,
          kind: 'number',
          spec: { min: s.low, max: s.high, integer: true, ...(s.log === undefined ? {} : { log: s.log }) },
        }
      case 'discrete': {
        const start = s.start ?? 0
        return { ...base, kind: 'number', spec: { min: start, max: start + s.n - 1, integer: true } }
      }
      case 'choice':
        return {
          ...base,
          kind: isShort(s.options, upTo) ? 'choice' : 'select',
          options: s.options,
          ...(s.descriptions === undefined ? {} : { descriptions: s.descriptions }),
        }
      case 'bool':
        return { ...base, kind: 'toggle' }
      case 'box': {
        const n = s.shape.reduce((a, b) => a * b, 1)
        const at = (bound: number | readonly number[], i: number): number =>
          typeof bound === 'number' ? bound : (bound[i] ?? Number.NaN)
        const children = Array.from({ length: n }, (_, i): ControlSpec => {
          const lo = at(s.low, i)
          const hi = at(s.high, i)
          return {
            ...plain([...path, i], `${base.label}[${i}]`),
            kind: 'number',
            spec: {
              ...(Number.isFinite(lo) ? { min: lo } : {}),
              ...(Number.isFinite(hi) ? { max: hi } : {}),
            },
          }
        })
        return { ...base, kind: 'group', children }
      }
      case 'multiDiscrete':
        return {
          ...base,
          kind: 'group',
          children: s.nvec.map(
            (k, i): ControlSpec => ({
              ...plain([...path, i], `${base.label}[${i}]`),
              kind: 'number',
              spec: { min: 0, max: k - 1, integer: true },
            }),
          ),
        }
      case 'multiBinary':
        return {
          ...base,
          kind: 'group',
          children: Array.from(
            { length: s.n },
            (_, i): ControlSpec => ({ ...plain([...path, i], `${base.label}[${i}]`), kind: 'toggle' }),
          ),
        }
      case 'tuple': {
        const items = Array.isArray(v) ? (v as readonly SpaceValue[]) : []
        return {
          ...base,
          kind: 'group',
          children: s.spaces.map((c, i) => walk(c, items[i], [...path, i], `${base.label}[${i}]`)),
        }
      }
      case 'dict': {
        const record = asRecord(v)
        const active = activeKeys(s, record)
        return {
          ...base,
          kind: 'group',
          children: Object.entries(s.spaces)
            .filter(([k]) => active.has(k))
            .map(([k, c]) => walk(c, record[k], [...path, k], k)),
        }
      }
      case 'union': {
        // The option as a choice at the tag's path, then the chosen option's knobs at the same level (a union's value
        // carries them beside the tag), so choosing another option swaps the knobs that follow.
        const record = asRecord(v)
        const tag = s.tag ?? 'kind'
        const names = Object.keys(s.options)
        const name = unionOption(s, record)
        // The chooser carries the union's label; its group has none, so the label is not drawn twice.
        const chooser: ControlSpec = {
          ...plain([...path, tag], base.label),
          kind: isShort(names, upTo) ? 'choice' : 'select',
          options: names,
        }
        const inner = walk(s.options[name] as Space, record, path, name)
        return {
          ...base,
          label: '',
          kind: 'group',
          children: [chooser, ...(inner.kind === 'group' ? inner.children : [inner])],
        }
      }
    }
  }
  const top = walk(space, value, [], space.label ?? '')
  return top.kind === 'group' && (space.kind === 'dict' || space.kind === 'union') ? [...top.children] : [top]
}

/** The option a union value names, else the union's default (the first option). */
function unionOption(space: Space & { readonly kind: 'union' }, value: Readonly<Record<string, SpaceValue>>): string {
  const name = value[space.tag ?? 'kind']
  return typeof name === 'string' && name in space.options
    ? name
    : (space.default ?? (Object.keys(space.options)[0] as string))
}

/**
 * Bring a dict value in line with its conditions: conditional keys that have become active get their defaults, keys
 * that are no longer active are dropped, recursively. Other spaces' values are returned as they are.
 *
 * @param space - The space.
 * @param value - A value that may have just changed.
 * @returns The settled value.
 */
export function settleValue(space: Space, value: SpaceValue): SpaceValue {
  if (space.kind === 'union') {
    // Keep the tag, settle the chosen option's knobs (the other options' knobs drop; missing ones get defaults).
    const record = asRecord(value)
    const tag = space.tag ?? 'kind'
    const name = unionOption(space, record)
    const { [tag]: _tag, ...rest } = record
    return { [tag]: name, ...asRecord(settleValue(space.options[name] as Space, rest)) }
  }
  if (space.kind !== 'dict') return value
  const out: Record<string, SpaceValue> = { ...asRecord(value) }
  const keys = Object.keys(space.spaces)
  for (let round = 0; round <= keys.length; round++) {
    let added = false
    for (const k of activeKeys(space, out)) {
      if (!(k in out)) {
        out[k] = defaults(space.spaces[k] as Space)
        added = true
      }
    }
    if (!added) break
  }
  const active = activeKeys(space, out)
  const settled: Record<string, SpaceValue> = {}
  for (const k of keys)
    if (active.has(k) && k in out) settled[k] = settleValue(space.spaces[k] as Space, out[k] as SpaceValue)
  return settled
}

/** The value at a path of a space's value (`undefined` when absent). */
export function valueAt(value: SpaceValue, path: ControlPath): SpaceValue | undefined {
  let v: SpaceValue | undefined = value
  for (const p of path) {
    if (v === undefined || typeof v !== 'object' || v === null) return undefined
    v = (v as Readonly<Record<string | number, SpaceValue>>)[p]
  }
  return v
}

/**
 * A copy of a space's value with the value at `path` replaced, then settled ({@link settleValue}), so changing a
 * choice brings its dependent dimensions in or out.
 *
 * @param space - The space.
 * @param value - The current value.
 * @param path - Where to write.
 * @param leaf - The new value there.
 * @returns The new value of the space.
 */
export function withValueAt(space: Space, value: SpaceValue, path: ControlPath, leaf: SpaceValue): SpaceValue {
  const write = (v: SpaceValue | undefined, i: number): SpaceValue => {
    if (i === path.length) return leaf
    const p = path[i] as string | number
    if (typeof p === 'number') {
      const items = Array.isArray(v) ? [...(v as readonly SpaceValue[])] : []
      items[p] = write(items[p], i + 1)
      return items
    }
    const record = { ...asRecord(v) }
    record[p] = write(record[p], i + 1)
    return record
  }
  return settleValue(space, write(value, 0))
}

/** Which options of a choice are admissible: a predicate over the control's path and the option, and why not. */
export interface Admissibility {
  /** Whether the option may be chosen given the page's other settings (the mode table's `admissible`). */
  readonly admissible: (path: ControlPath, option: OptionValue) => boolean
  /** The note on an option that may not be chosen (default a general one). */
  readonly why?: (path: ControlPath, option: OptionValue) => string | undefined
}

/**
 * The options of a choice control with the inadmissible ones disabled and annotated: a disabled option keeps its place
 * and carries the reason as its description (its title), so the reader sees what exists and why it is out of reach.
 *
 * @param options - The choice's options.
 * @param path - The control's path in the space's value.
 * @param rule - The admissibility rule (none leaves the options as they are).
 * @param descriptions - A line on each option, by its text, kept as its tooltip when the option is available.
 * @returns The options.
 */
export function greyOptions(
  options: readonly OptionValue[],
  path: ControlPath,
  rule: Admissibility | undefined,
  descriptions?: Readonly<Record<string, string>>,
): OptionInput<OptionValue>[] {
  const described = (value: OptionValue): OptionInput<OptionValue> => {
    const description = descriptions?.[String(value)]
    return description === undefined ? value : { value, description }
  }
  if (rule === undefined) return options.map(described)
  return options.map(
    (value): OptionInput<OptionValue> =>
      rule.admissible(path, value)
        ? described(value)
        : {
            value,
            disabled: true,
            description: rule.why?.(path, value) ?? 'not available with the current settings',
          },
  )
}
