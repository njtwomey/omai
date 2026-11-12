import { epDirections } from 'omai-engine/probprog/ep'
import { describe, expect, it } from 'vitest'
import { type DirNode, duplicateSlugs, findDir, pocsBelow } from './poc.ts'
import { DIR_METAS, mismatched, NOT_YET_MOVED, POCS, pageOf, TREE } from './registry.ts'

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
  it('the slug of a poc is unique across the whole tree (it is its address)', () => {
    expect(duplicateSlugs(POCS)).toEqual([])
  })
})

describe('the directory tree', () => {
  it('lists every poc once, in the directory its folder is in', () => {
    expect(pocsBelow(TREE).length).toBe(POCS.length)
    expect(TREE.count).toBe(POCS.length)
    for (const p of POCS) expect(findDir(TREE, p.dir)?.pocs.map((e) => e.slug)).toContain(p.slug)
  })
  it('every directory has a dir.ts, except the one that catches pages not yet moved', () => {
    const undescribed = (node: DirNode): string[] => [
      ...(node.path.length > 0 && !node.declared ? [node.path.join('/')] : []),
      ...node.dirs.flatMap(undescribed),
    ]
    const missing = undescribed(TREE).filter((p) => !(p === 'more' && NOT_YET_MOVED.length > 0))
    expect(missing).toEqual([])
    for (const m of Object.values(DIR_METAS)) {
      expect(m.title).not.toBe('')
      expect(m.summary).not.toBe('')
    }
  })
  it('the directories of the first level are the groups the index has always shown, in that order', () => {
    expect(TREE.dirs.map((d) => d.name).slice(0, 6)).toEqual([
      'bayesian-examples',
      'expectation-propagation',
      'probabilistic-programming',
      'structured-prediction',
      'models',
      'graphics-and-fields',
    ])
  })
  it('composed-atoms is a directory of expectation-propagation, and ep-coverage is a file of it', () => {
    expect(findDir(TREE, ['expectation-propagation', 'composed-atoms'])?.declared).toBe(true)
    expect(findDir(TREE, ['expectation-propagation'])?.pocs.map((p) => p.slug)).toContain('ep-coverage')
  })
  it('expectation propagation keeps its sections as subdirectories, each page in the one the old rules gave it', () => {
    const ep = findDir(TREE, ['expectation-propagation'])
    expect(ep?.dirs.map((d) => d.name)).toEqual(['basic', 'priors', 'learned', 'programs', 'composed-atoms'])
    const slugs = (name: string): string[] =>
      (findDir(TREE, ['expectation-propagation', name])?.pocs ?? []).map((p) => p.slug)
    for (const s of slugs('programs')) expect(s).toMatch(/^ep-program-/)
    for (const s of slugs('priors')) expect(s).toMatch(/^(ep-prior-|ep-gate-weight)/)
    for (const s of slugs('learned')) expect(s).toMatch(/-learned$/)
    for (const s of slugs('basic'))
      expect(s).not.toMatch(/-learned$|^ep-program-|^ep-prior-|^ep-gate-weight|^ep-coverage$/)
  })
  it('lists the pages not yet moved (a note, not a failure): they sit in the directory their legacy group names', () => {
    if (NOT_YET_MOVED.length > 0) console.warn(`pocs not yet moved into a directory: ${NOT_YET_MOVED.join(', ')}`)
    for (const s of NOT_YET_MOVED) expect(POCS.find((p) => p.slug === s)?.dir.length).toBe(1)
  })
})

describe('the wider suite of the EP coverage page', () => {
  it('every page it points at exists', () => {
    const slugs = new Set(POCS.map((p) => p.slug))
    const missing = epDirections.flatMap((d) => d.pages.filter((s) => !slugs.has(s)).map((s) => `${d.id}: ${s}`))
    expect(missing).toEqual([])
  })
})
