import { MdmlError } from 'mdml-engine/base'
import { type Exposition, type ExpositionKind, expositionSamples } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { stillOf, viewedKinds, viewFor } from './registry.ts'

const kinds = Object.keys(expositionSamples) as ExpositionKind[]

describe('the registry of record views, over a small valid record of every kind', () => {
  it('has a view of every kind, draws a still of each record and names an unknown kind', () => {
    expect(viewedKinds().sort()).toEqual([...kinds].sort())
    for (const kind of kinds) {
      expect(viewFor(kind).kind, kind).toBe(kind)
      const still = stillOf(expositionSamples[kind]() as Exposition)
      expect(typeof still, kind).toBe('object')
      expect('series' in still || (still as { kind?: string }).kind === 'scene', kind).toBe(true)
    }
    // An unknown kind is named, not a TypeError from inside the registry.
    for (const run of [() => viewFor('nope' as ExpositionKind), () => stillOf({ kind: 'toString' } as never)])
      expect(run).toThrow(
        expect.objectContaining({ code: 'value', message: expect.stringContaining('no standard view') }),
      )
    expect(() => viewFor('curves')).not.toThrow(MdmlError)
  })
})
