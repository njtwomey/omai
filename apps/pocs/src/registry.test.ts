import { describe, expect, it } from 'vitest'
import { type PocEntry, searchPocs } from './poc.ts'
import { mismatched, POCS, pageOf } from './registry.ts'

const entry = (slug: string, title: string, tags: string[], added: string): PocEntry => ({
  slug,
  title,
  summary: `About ${title}`,
  tags,
  added,
})

describe('searchPocs', () => {
  const all = [
    entry('a', 'Alpha detector', ['change', 'online'], '2026-01-01'),
    entry('b', 'Beta sampler', ['mcmc'], '2026-03-01'),
    entry('c', 'Gamma detector', ['change'], '2026-03-01'),
  ]
  it('keeps everything for an empty query, newest first and then by title', () => {
    expect(searchPocs(all, '  ').map((e) => e.slug)).toEqual(['b', 'c', 'a'])
  })
  it('needs every word, in the title, summary, tags or slug, ignoring case', () => {
    expect(searchPocs(all, 'DETECTOR change').map((e) => e.slug)).toEqual(['c', 'a'])
    expect(searchPocs(all, 'detector mcmc')).toEqual([])
    expect(searchPocs(all, 'about beta').map((e) => e.slug)).toEqual(['b'])
  })
})

describe('the proofs of concept', () => {
  it('each folder has both its metadata and its page', () => {
    expect(mismatched()).toEqual([])
    for (const p of POCS) expect(pageOf(p.slug)).toBeDefined()
    expect(pageOf('no-such-poc')).toBeUndefined()
  })
  it('each has a title, a summary, tags and a date', () => {
    expect(POCS.length).toBeGreaterThan(0)
    for (const p of POCS) {
      expect(p.title).not.toBe('')
      expect(p.summary).not.toBe('')
      expect(p.tags.length).toBeGreaterThan(0)
      expect(p.added).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    }
  })
})
