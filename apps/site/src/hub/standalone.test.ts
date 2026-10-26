import { catalogue } from 'mdml-hub'
import { describe, expect, it } from 'vitest'
import { MOVED } from '../gallery/moved.ts'
import { ENTRIES } from '../gallery/registry.ts'
import { PAGES, pageById } from './standalone.ts'

describe('the hub standalone pages', () => {
  it('lists every page with a title, question and collection at an id no entry uses, and loads its demo', async () => {
    expect(PAGES.length).toBeGreaterThan(0)
    for (const p of PAGES) {
      expect(p.title.length, p.file).toBeGreaterThan(0)
      expect(p.question.endsWith('?'), p.file).toBe(true)
      expect(catalogue.byId.has(p.id), p.file).toBe(false)
      const { Example, snippet } = await p.load()
      expect(typeof Example, p.file).toBe('function')
      expect(snippet, p.file).not.toContain('export const recipe')
    }
    expect(pageById.size).toBe(PAGES.length)
  })

  it('sends every moved gallery path to a page that exists', () => {
    const galleryPaths = new Set(ENTRIES.map((e) => e.path))
    for (const [from, to] of Object.entries(MOVED)) {
      expect(galleryPaths.has(from), from).toBe(false)
      const ok = to.startsWith('/hub/') ? pageById.has(to.slice(5)) : galleryPaths.has(to)
      expect(ok, `${from} -> ${to}`).toBe(true)
    }
  })
})
