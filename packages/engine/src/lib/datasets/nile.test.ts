import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { datasetRegistry } from './catalogue.ts'
import { changepointSeries } from './changepoints.ts'
import { datasetCollections, loadEntry } from './collections.ts'
import { loadDataset, whereDatasets } from './registry.ts'
import { nile } from './timeseries.ts'

// Counted from Rdatasets' Nile.csv (sha256 d0452bea...): 100 yearly flows, 1871 to 1970.
describe('the Nile annual flow', () => {
  it('has the 100 flows of the published file', () => {
    const series = nile()
    const v = Array.from(series.values.data)
    expect(series.values.shape).toEqual([100, 1])
    expect(series.times?.data[0]).toBe(1871)
    expect(series.times?.data[99]).toBe(1970)
    expect(v[0]).toBe(1120)
    expect(v[99]).toBe(740)
    expect(Math.min(...v)).toBe(456)
    expect(Math.max(...v)).toBe(1370)
    expect(v.reduce((a, b) => a + b, 0) / 100).toBeCloseTo(919.35, 9)
    expect(series.meta.sha256).toBe('3456fc26a94f7afec5c9e1696349d6b5a8d2ce11ba60ad9b78c99edf3d3a5780')
  })

  it('is registered as a real time series, with the coal-mining counts and the changepoint generator', () => {
    const entry = lookup(datasetRegistry, 'nile')
    expect(entry).toMatchObject({ form: 'vendored', kind: 'timeseries', modality: 'timeSeries', real: true, size: 100 })
    expect(loadDataset(entry).data.kind).toBe('timeseries')
    const real = whereDatasets(datasetRegistry, { kind: 'timeseries', real: true }).map((e) => e.name)
    expect(real).toEqual(expect.arrayContaining(['nile', 'coalMiningCounts']))
    const generator = lookup(datasetRegistry, 'changepointSeries')
    expect(generator).toMatchObject({ form: 'generator', kind: 'timeseries', modality: 'timeSeries', real: false })
  })

  it('is in the changepoints collection beside the generator, with demo knobs for a rate change', () => {
    const entries = datasetCollections['changepoints']?.entries ?? []
    expect(entries.map((e) => e.name)).toEqual(['changepointSeries', 'rateChanges', 'coalMiningCounts', 'nile'])
    for (const e of entries) expect(loadEntry(e, key(0)).data.kind, e.name).toBe('timeseries')
    const truth = generate(changepointSeries, { key: key(0), knobs: { kind: 'rate', n: 200 } }).truth
    expect(Array.isArray(truth.changepoints)).toBe(true)
  })
})
