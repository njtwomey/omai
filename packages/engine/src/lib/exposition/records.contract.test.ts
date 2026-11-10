import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { SEAL_OPTIONS, seal } from './_check.ts'
import { curveSet } from './curves.ts'
import { isExposition } from './record.ts'
import { expositionSamples } from './samples.ts'
import { thresholdSweep } from './thresholds.ts'

type Json = null | boolean | number | string | Json[] | { [key: string]: Json }

/** The paths of the numbers of a value, with the property name each sits under. */
function leaves(
  value: unknown,
  path: (string | number)[] = [],
  key = '',
): { path: (string | number)[]; key: string }[] {
  if (typeof value === 'number') return [{ path, key }]
  if (Array.isArray(value)) return value.flatMap((v, i) => leaves(v, [...path, i], key))
  if (typeof value === 'object' && value !== null)
    return Object.entries(value).flatMap(([k, v]) => leaves(v, [...path, k], k))
  return []
}

/** A deep copy of a record with the number at `path` replaced. */
function withNumber(record: unknown, path: (string | number)[], to: number): unknown {
  const copy = JSON.parse(JSON.stringify(record)) as Json
  let at = copy as Record<string | number, Json>
  for (const step of path.slice(0, -1)) at = at[step] as Record<string | number, Json>
  at[path[path.length - 1] as string | number] = to
  return copy
}

const kinds = Object.keys(expositionSamples) as (keyof typeof expositionSamples)[]

describe('the contract of the exposition records', () => {
  it('has one sample per kind, tagged, plain data, and sealing it again changes nothing', () => {
    expect(kinds).toHaveLength(63)
    for (const kind of kinds) {
      const record = expositionSamples[kind]()
      expect([kind, record.kind, isExposition(record)]).toEqual([kind, kind, true])
      expect(JSON.parse(JSON.stringify(record)), kind).toEqual(record)
      const again = seal(record)
      expect(again, kind).toEqual(record)
      expect(again, kind).not.toBe(record)
    }
  })

  it('rejects a NaN or an infinity in any number of any kind, except in the fields a kind lists', () => {
    for (const kind of kinds) {
      const record = expositionSamples[kind]()
      const options = SEAL_OPTIONS[kind] ?? {}
      let rejected = 0
      // What a kind keeps by reference (a scene or structure, checked by its own builder) is not looked into.
      const owned = (path: (string | number)[]): boolean => !path.some((step) => options.opaque?.includes(String(step)))
      for (const { path, key } of leaves(record).filter((l) => owned(l.path))) {
        for (const [bad, allowed] of [
          [Number.NaN, options.gaps],
          [Number.POSITIVE_INFINITY, options.unbounded],
        ] as const) {
          const run = (): unknown => seal(withNumber(record, path, bad) as typeof record)
          if (allowed?.includes(key) === true) expect(run, `${kind} ${path.join('.')}`).not.toThrow()
          else {
            expect(run, `${kind} ${path.join('.')}`).toThrow(expect.objectContaining({ code: 'value' }))
            rejected++
          }
        }
      }
      if (leaves(record).some((l) => owned(l.path))) expect(rejected, kind).toBeGreaterThan(0)
    }
  })

  it('checks and copies in the constructors themselves', () => {
    const y = [1, 0.5]
    const x = [0, 1]
    const curves = curveSet({ x, xLabel: 'epoch', series: [{ name: 'a', y }] })
    y[0] = 99
    x[1] = 7
    expect([curves.series[0]?.y, curves.x]).toEqual([
      [1, 0.5],
      [0, 1],
    ])
    for (const make of [
      () => curveSet({ x: [0, Number.POSITIVE_INFINITY], xLabel: 'e', series: [{ name: 'a', y: [1, 2] }] }),
      () => curveSet({ x: [0, 1], xLabel: 'e', series: [{ name: 'a', y: [Number.NaN, 2] }] }),
      () => curveSet({ x: [0, 1], xLabel: 'e', series: [{ name: 'a', y: [1] }] }),
    ])
      expect(make).toThrow(OmaiError)
    const sweep = expositionSamples.thresholdSweep()
    expect(() => thresholdSweep({ ...sweep, thresholds: [Number.NaN] })).toThrow(OmaiError)
  })
})
