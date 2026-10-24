import { describe, expect, it } from 'vitest'
import { ENTRIES, SECTIONS } from './registry.ts'

describe('the recipe registry', () => {
  it('loads every recipe with a title, question, section and source', () => {
    expect(ENTRIES.length).toBeGreaterThan(0)
    const sections = new Set<string>(SECTIONS.map((s) => s.id))
    for (const e of ENTRIES) {
      expect(e.title.length, e.file).toBeGreaterThan(0)
      expect(e.question.endsWith('?'), e.file).toBe(true)
      expect(sections.has(e.section), e.file).toBe(true)
      expect(typeof e.Example, e.file).toBe('function')
      expect(e.snippet, e.file).not.toContain('export const recipe')
    }
  })

  it('gives every recipe its own page', () => {
    const paths = ENTRIES.map((e) => e.path)
    expect(new Set(paths).size).toBe(paths.length)
  })
})
