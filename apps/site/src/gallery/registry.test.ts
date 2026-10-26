import { describe, expect, it } from 'vitest'
import { ENTRIES, SECTIONS } from './registry.ts'

describe('the recipe registry', () => {
  it('lists every recipe with a title, question and section, and loads its example and source', async () => {
    expect(ENTRIES.length).toBeGreaterThan(0)
    const sections = new Set<string>(SECTIONS.map((s) => s.id))
    for (const e of ENTRIES) {
      expect(e.title.length, e.file).toBeGreaterThan(0)
      expect(e.question.endsWith('?'), e.file).toBe(true)
      expect(sections.has(e.section), e.file).toBe(true)
      const { Example, snippet } = await e.load()
      expect(typeof Example, e.file).toBe('function')
      expect(snippet.length, e.file).toBeGreaterThan(0)
      expect(snippet, e.file).not.toContain('export const recipe')
    }
  })

  it('gives every recipe its own page', () => {
    const paths = ENTRIES.map((e) => e.path)
    expect(new Set(paths).size).toBe(paths.length)
  })
})
