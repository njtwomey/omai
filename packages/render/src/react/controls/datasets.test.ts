import { describe, expect, it } from 'vitest'
import { collectionOf, entryKnobSpace, loadSelection, selectEntry } from './datasets.ts'

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
  })
})
