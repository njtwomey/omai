import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { rowCount } from '../data/index.ts'
import { key } from '../random/index.ts'
import { datasetGenerators, datasetRegistry, remoteDatasets } from './catalogue.ts'
import { moons } from './classification.ts'
import {
  DATASET_MODALITIES,
  fetchDataset,
  findDataset,
  generatorEntry,
  loadDataset,
  matchesDatasetQuery,
  registerDatasets,
  remoteEntry,
  vendoredEntry,
  whereDatasets,
} from './registry.ts'
import { californiaHousing } from './regression-real.ts'

describe('the dataset registry', () => {
  it('registers every generator of the catalogue once, with its kind, and every remote asset', () => {
    const generators = datasetRegistry.entries.filter((e) => e.form === 'generator')
    expect(generators.map((e) => e.name).sort()).toEqual(Object.keys(datasetGenerators).sort())
    for (const e of generators) {
      expect(e.kind, e.name).toBe(e.generator.kind)
      expect(e.real, e.name).toBe(false)
    }
    const remote = datasetRegistry.entries.filter((e) => e.form === 'remote')
    expect(remote.map((e) => e.name).sort()).toEqual(Object.keys(remoteDatasets).sort())
    for (const e of remote) expect(e.licence, e.name).toBeTypeOf('string')
  })

  it('tags every entry with a title, a description, a known modality and a real flag', () => {
    for (const e of datasetRegistry.entries) {
      expect(e.title.length, e.name).toBeGreaterThan(0)
      expect(e.description.length, e.name).toBeGreaterThan(10)
      expect(DATASET_MODALITIES, e.name).toContain(e.modality)
      expect(typeof e.real, e.name).toBe('boolean')
      if (e.form !== 'generator') expect(e.real, e.name).toBe(true)
    }
  })

  it('loads every vendored dataset with the kind and size it is tagged with', () => {
    for (const e of datasetRegistry.entries) {
      if (e.form !== 'vendored') continue
      const { data, truth } = e.load()
      expect(data.kind, e.name).toBe(e.kind)
      expect(truth, e.name).toBeDefined()
      if (e.modality !== 'graph') expect(rowCount(data), e.name).toBe(e.size)
    }
  })

  it('returns everything for an empty query and filters by kind, modality, real, form and text', () => {
    expect(whereDatasets(datasetRegistry)).toEqual(datasetRegistry.entries)
    expect(whereDatasets(datasetRegistry, { kind: 'image', real: true }).map((e) => e.name)).toEqual([
      'digitImages',
      'fashionMnistTest',
    ])
    const tabularReal = whereDatasets(datasetRegistry, { modality: 'tabular', real: true })
    expect(tabularReal.length).toBeGreaterThan(10)
    for (const e of tabularReal) expect(e.form).not.toBe('generator')
    expect(
      whereDatasets(datasetRegistry, { kind: ['timeseries', 'events'], form: 'vendored' }).map((e) => e.name),
    ).toEqual(['airPassengers', 'sunspots', 'coalMiningDisasters', 'coalMiningCounts'])
    expect(whereDatasets(datasetRegistry, { text: 'two MOONS' }).map((e) => e.name)).toEqual([
      'moons',
      'moonsWithOutliers',
    ])
    expect(whereDatasets(datasetRegistry, { text: 'nothing of the sort' })).toEqual([])
    expect(matchesDatasetQuery(lookup(datasetRegistry, 'iris'), { real: false })).toBe(false)
  })

  it('looks entries up by name', () => {
    expect(findDataset(datasetRegistry, 'wine')?.title).toBe('Wine')
    expect(findDataset(datasetRegistry, 'nope')).toBeUndefined()
  })

  it('generates, loads and fetches entries', async () => {
    const m = lookup(datasetRegistry, 'moons')
    const a = loadDataset(m, { key: key(1), knobs: { n: 12 } })
    const b = loadDataset(m, { key: key(1), knobs: { n: 12 } })
    expect(rowCount(a.data)).toBe(12)
    expect(a.data.meta.key).toEqual(b.data.meta.key)
    expect(a.truth.posterior).toBeDefined()
    expect(loadDataset(lookup(datasetRegistry, 'iris')).data.kind).toBe('classification')
    expect(loadDataset(lookup(datasetRegistry, 'oldFaithful')).truth.clusters).toBeDefined()
    expect(() => loadDataset(lookup(datasetRegistry, 'californiaHousing'))).toThrow(/remote/)
    expect((await fetchDataset(lookup(datasetRegistry, 'wine'))).data.kind).toBe('classification')
    // A remote entry goes through `load` with the given fetch; a wrong checksum is refused.
    const bad = async (): Promise<{ ok: boolean; status: number; arrayBuffer: () => Promise<ArrayBuffer> }> => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
    })
    await expect(fetchDataset(lookup(datasetRegistry, 'californiaHousing'), { fetch: bad })).rejects.toThrow(/checksum/)
  })

  it('builds entries from generators, loaders and references', () => {
    const e = generatorEntry(moons, {
      title: 'Noisy moons',
      name: 'noisyMoons',
      modality: 'tabular',
      knobs: { noise: 0.4 },
    })
    expect(e.form).toBe('generator')
    expect(e.name).toBe('noisyMoons')
    expect(e.kind).toBe('classification')
    expect(e.real).toBe(false)
    expect(rowCount(loadDataset(e, { key: key(0), knobs: { n: 5 } }).data)).toBe(5)
    const v = vendoredEntry('tiny', () => loadDataset(lookup(datasetRegistry, 'iris')).data, {
      title: 'Tiny',
      description: 'A copy of iris.',
      kind: 'classification',
      modality: 'tabular',
    })
    expect(v.real).toBe(true)
    expect(loadDataset(v).truth).toEqual({})
    const rr = remoteEntry(californiaHousing, {
      title: 'CA',
      description: 'California housing.',
      kind: 'regression',
      modality: 'tabular',
    })
    expect(rr.licence).toBe(californiaHousing.kind === 'remote' ? californiaHousing.licence : undefined)
  })

  it('extends into a new registry and refuses mistagged generators', () => {
    const mine = registerDatasets(datasetRegistry, [
      generatorEntry(moons, { title: 'Noisy moons', name: 'noisyMoons', modality: 'tabular', knobs: { noise: 0.4 } }),
    ])
    expect(mine.entries.length).toBe(datasetRegistry.entries.length + 1)
    expect(findDataset(datasetRegistry, 'noisyMoons')).toBeUndefined()
    expect(whereDatasets(mine, { text: 'noisy moons' }).map((e) => e.name)).toEqual(['noisyMoons'])
    const wrong = { ...generatorEntry(moons, { title: 'x', modality: 'tabular' }), kind: 'regression' as const }
    expect(() => registerDatasets(datasetRegistry, [wrong])).toThrow(/tagged regression/)
  })
})
