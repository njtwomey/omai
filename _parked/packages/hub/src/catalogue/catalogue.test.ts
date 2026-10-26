import { isExposition } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { runDemo } from '../entry/run.ts'
import { catalogue } from './index.ts'

// Every entry's demo runs on its default dataset within a budget and returns well-formed records. Demos aim at about a
// second; the budget is generous so a busy machine (several checkouts testing at once) doesn't fail the gate.
describe('the catalogue', () => {
  it('holds entries under their collections', () => {
    expect(catalogue.entries.length).toBeGreaterThan(0)
    for (const e of catalogue.entries) expect(catalogue.collections.get(e.collection)).toContain(e)
  })

  it('labels every control of the sklearn, pyod and probabilistic entries', () => {
    for (const e of catalogue.entries)
      if (/^(sklearn|pyod|probabilistic)\//.test(e.id))
        for (const [k, s] of Object.entries(e.params.spaces)) expect(s.label, `${e.id}: ${k}`).toBeTypeOf('string')
  })

  for (const entry of catalogue.entries)
    it(`runs ${entry.id}`, async () => {
      const out = await runDemo(entry)
      expect(out.exposition.length).toBeGreaterThan(0)
      for (const r of out.exposition) expect(isExposition(r)).toBe(true)
      for (const v of entry.views)
        if (v.index === undefined || v.index === 0) expect(out.exposition.map((r) => r.kind)).toContain(v.record)
    }, 20_000)
})
