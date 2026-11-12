// The page that lists what the compiler maps is written from the mapping tables: it must be current (`make ep-compiler-doc` writes it).
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { compilerMappingsDoc } from './_doc.ts'

describe('docs/probprog/compiler-mappings.md', () => {
  it('is the page the mapping tables give (run make ep-compiler-doc when this fails)', () => {
    const page = readFileSync(
      new URL('../../../../../../../docs/probprog/compiler-mappings.md', import.meta.url),
      'utf8',
    )
    expect(page).toBe(compilerMappingsDoc())
  })
})
