import { datasetRegistry, generatorEntry, moons, registerDatasets } from 'omai-engine/datasets'
import { describe, expect, it } from 'vitest'
import { collectionOf, entriesOf, entryKnobSpace, loadSelection, selectEntry } from './datasets.ts'

describe('dataset selections', () => {
  it('starts an entry from its demo knobs and the generator defaults', () => {
    const s = selectEntry('classification2d', 'moons', 3)
    expect(s.name).toBe('moons')
    expect(s.seed).toBe(3)
    expect(s.knobs['n']).toBe(200)
    expect(s.knobs['noise']).toBe(0.2)
  })

  it('generates the selection reproducibly', () => {
    const s = selectEntry('classification2d', 'moons')
    const a = loadSelection('classification2d', { ...s, knobs: { ...s.knobs, n: 20 } })
    const b = loadSelection('classification2d', { ...s, knobs: { ...s.knobs, n: 20 } })
    expect(a.data.kind).toBe('classification')
    expect(a.data.meta.key).toEqual(b.data.meta.key)
  })

  it('has no knobs for vendored datasets', () => {
    const iris = collectionOf('classificationTabular').entries.find((e) => e.name === 'iris')
    expect(iris === undefined ? 'missing' : entryKnobSpace(iris)).toBeUndefined()
    expect(selectEntry('classificationTabular', 'iris').knobs).toEqual({})
  })

  it('refuses unknown collections and entries', () => {
    expect(() => collectionOf('nope')).toThrow()
    expect(() => selectEntry('classification2d', 'nope')).toThrow()
    expect(() => selectEntry(datasetRegistry, 'nope')).toThrow(/the registry has no entry/)
  })

  it('offers a registry the same way: generator defaults, starting knobs, vendored loads, remote refused', () => {
    expect(entriesOf(datasetRegistry)).toBe(datasetRegistry.entries)
    expect(entriesOf('regression1d')).toBe(collectionOf('regression1d').entries)
    const s = selectEntry(datasetRegistry, 'moons', 2)
    expect(s).toEqual({ name: 'moons', seed: 2, knobs: { n: 100, noise: 0.1, shuffle: true } })
    expect(loadSelection(datasetRegistry, { ...s, knobs: { ...s.knobs, n: 8 } }).data.kind).toBe('classification')
    expect(selectEntry(datasetRegistry).name).toBe(datasetRegistry.entries[0]?.name)
    expect(selectEntry(datasetRegistry, 'iris').knobs).toEqual({})
    expect(loadSelection(datasetRegistry, selectEntry(datasetRegistry, 'iris')).data.kind).toBe('classification')
    expect(() => loadSelection(datasetRegistry, selectEntry(datasetRegistry, 'californiaHousing'))).toThrow(/remote/)
    const mine = registerDatasets(datasetRegistry, [
      generatorEntry(moons, { name: 'noisyMoons', title: 'Noisy moons', modality: 'tabular', knobs: { noise: 0.4 } }),
    ])
    expect(selectEntry(mine, 'noisyMoons').knobs).toEqual({ n: 100, noise: 0.4, shuffle: true })
    const found = mine.entries.find((e) => e.name === 'noisyMoons')
    expect(found === undefined ? undefined : entryKnobSpace(found)).toBe(moons.knobs)
  })
})
