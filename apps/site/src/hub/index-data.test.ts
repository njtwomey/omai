import { catalogue } from 'mdml-hub'
import { entryModules } from 'mdml-hub/lazy'
import { describe, expect, it } from 'vitest'
import { hubIndex } from '../../plugins/hub-index-data.ts'

describe('the hub index', () => {
  it('lists every catalogue entry with its card, and the module that defines it', async () => {
    const index = await hubIndex()
    expect(index.entries.map((e) => e.id)).toEqual(catalogue.entries.map((e) => e.id))
    for (const meta of index.entries) {
      const entry = catalogue.byId.get(meta.id)
      expect(meta.title, meta.id).toBe(entry?.title)
      expect(meta.summary, meta.id).toBe(entry?.summary)
      expect(meta.collection, meta.id).toBe(entry?.collection)
      const load = entryModules[meta.module]
      expect(load, meta.id).toBeDefined()
      expect(Object.values(await (load as () => Promise<Record<string, unknown>>)()), meta.id).toContain(entry)
    }
    expect(Object.keys(index.descriptions).length).toBeGreaterThan(0)
  })
})
