import { describe, expect, it } from 'vitest'
import type { Catalogue, SymbolDoc } from '../schema/catalogue.ts'
import { conceptIndex } from './concepts.ts'

const symbol = (id: string, concepts: string[]): SymbolDoc => ({
  id,
  package: 'p',
  module: 'p/m',
  name: id,
  kind: 'function',
  signature: '',
  summary: '',
  description: '',
  params: [],
  returns: '',
  throws: [],
  examples: [],
  concepts,
  see: [],
  references: [],
  aliases: [],
  members: [],
  source: { file: '', line: 1 },
})

describe('conceptIndex', () => {
  it('maps concepts to symbols and back', () => {
    const c: Catalogue = {
      version: 1,
      packages: [],
      modules: [],
      files: [],
      symbols: [symbol('b', ['x']), symbol('a', ['x', 'y']), symbol('c', [])],
    }
    const index = conceptIndex(c)
    expect(index.symbols).toEqual({ x: ['a', 'b'], y: ['a'] })
    expect(index.concepts['a']).toEqual(['x', 'y'])
    expect(index.untagged).toEqual(['c'])
  })
})
