import { describe, expect, it } from 'vitest'
import { dict } from 'whitebox-engine/space'
import { catalogue } from '../catalogue/index.ts'
import { buildCatalogue, defineEntry, type HubEntry } from './entry.ts'
import { datasetState, demoState, runDemo } from './run.ts'
import { stateFromQuery, stateToQuery } from './url.ts'

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

describe('entry state in a URL', () => {
  it('round-trips through the query and ignores what does not fit', () => {
    const lasso = catalogue.byId.get('sklearn/linear-model/lasso')
    if (lasso === undefined) throw new Error('no lasso entry')
    const demo = demoState(lasso)
    expect(stateToQuery(lasso, demo)).toBe('')
    const state = { ...demo, ...datasetState('regression1d', 'sincCurve', 4), params: { ...demo.params, alpha: 0.5 } }
    const knobs = { ...state.knobs, n: 50 }
    expect(stateFromQuery(lasso, stateToQuery(lasso, { ...state, knobs }))).toEqual({ ...state, knobs })
    expect(stateFromQuery(lasso, '?alpha=1e9&data=nope/x&k.zzz=1')).toEqual(demo)
  })
})

describe('entries without a dataset', () => {
  it('run on their parameters alone and report progress', async () => {
    const counter = defineEntry({
      ...spec('demo/counter'),
      datasets: [],
      run: ({ data, progress }) => {
        for (let i = 1; i <= 4; i++) progress?.({ fraction: i / 4, label: `step ${i}` })
        return { exposition: [], sawData: data !== undefined }
      },
    })
    const seen: number[] = []
    const out = await runDemo(counter, demoState(counter), (p) => seen.push(p.fraction))
    expect(seen).toEqual([0.25, 0.5, 0.75, 1])
    expect(out.rows).toBe(0)
    expect(stateToQuery(counter, { ...demoState(counter), seed: 2 })).toBe('seed=2')
  })
})
