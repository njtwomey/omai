import { describe, expect, it } from 'vitest'
import { dict } from 'whitebox-engine/space'
import { catalogue } from '../catalogue/index.ts'
import { buildCatalogue, defineEntry, type HubEntry } from './entry.ts'

const spec = (id: string, collection = 'demo'): Omit<HubEntry, 'kind'> => ({
  id,
  title: id,
  summary: 'A demo entry.',
  collection,
  tags: [],
  datasets: ['classification2d'],
  params: dict({}),
  run: () => ({ exposition: [] }),
  views: [],
})

describe('hub entries', () => {
  it('check their id, collection and datasets', () => {
    expect(defineEntry(spec('demo/a')).kind).toBe('hubEntry')
    expect(() => defineEntry(spec('other/a'))).toThrow(/not under its collection/)
    expect(() => defineEntry({ ...spec('demo/a'), datasets: ['nope'] })).toThrow(/unknown dataset collection/)
  })

  it('gather into a catalogue by id and collection, rejecting duplicate ids', () => {
    const a = defineEntry(spec('demo/a'))
    const b = defineEntry(spec('zoo/b', 'zoo'))
    const c = buildCatalogue([{ a, b, notAnEntry: 3 }, { a }])
    expect(c.entries.map((e) => e.id)).toEqual(['demo/a', 'zoo/b'])
    expect([...c.collections.keys()]).toEqual(['demo', 'zoo'])
    expect(() => buildCatalogue([{ a }, { a2: defineEntry(spec('demo/a')) }])).toThrow(/share the id/)
  })

  it('build the generated catalogue', () => {
    expect(catalogue.entries.length).toBe(catalogue.byId.size)
  })
})
