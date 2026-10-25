import { activeDims, defaults, type Space, type SpaceValue } from 'whitebox-engine/space'
import type { NumberSpec } from './number.ts'
import type { OptionValue } from './options.ts'

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
  | (ControlBase & { readonly kind: 'choice' | 'select'; readonly options: readonly OptionValue[] })
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
        return { ...base, kind: isShort(s.options, upTo) ? 'choice' : 'select', options: s.options }
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
    }
  }
  const top = walk(space, value, [], space.label ?? '')
  return top.kind === 'group' && space.kind === 'dict' ? [...top.children] : [top]
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
