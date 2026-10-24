import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { datasetGenerators } from './catalogue.ts'
import { datasetCollections, loadEntry } from './collections.ts'

describe('dataset collections', () => {
  it('name only known generators, with ids matching their keys', () => {
    for (const [id, c] of Object.entries(datasetCollections)) {
      expect(c.id).toBe(id)
      for (const e of c.entries) if ('generator' in e) expect(datasetGenerators[e.generator], e.generator).toBeDefined()
    }
  })

  it('generate or load every entry with its demo knobs', () => {
    for (const c of Object.values(datasetCollections))
      for (const e of c.entries) expect(loadEntry(e, key(1)).data.kind, `${c.id}/${e.name}`).toBeTypeOf('string')
  })
})
