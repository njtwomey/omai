import { epDirections } from 'omai-engine/probprog/ep'
import { describe, expect, it } from 'vitest'
import { GROUPS } from './groups.ts'
import { groupPocs, type PocEntry, searchPocs, splitSubgroups } from './poc.ts'
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

describe('groups', () => {
  const g = [
    { slug: 'x', title: 'X', summary: 's' },
    { slug: 'y', title: 'Y', summary: 's' },
  ]
  const e = (slug: string, group?: string): PocEntry => ({
    slug,
    title: slug,
    summary: slug,
    tags: ['t'],
    added: '2026-01-01',
    ...(group === undefined ? {} : { group }),
  })
  it('sections follow the declared order, keep the entry order inside, and leave out empty groups', () => {
    const s = groupPocs([e('a', 'y'), e('b', 'x'), e('c', 'y'), e('d'), e('f', 'unknown')], g)
    expect(s.map((x) => [x.group?.slug ?? null, x.entries.map((p) => p.slug)])).toEqual([
      ['x', ['b']],
      ['y', ['a', 'c']],
      [null, ['d', 'f']],
    ])
    expect(groupPocs([e('a', 'y')], g).map((x) => x.group?.slug)).toEqual(['y'])
  })
  it('a search also matches the group slug', () => {
    expect(searchPocs([e('a', 'alpha'), e('b', 'beta')], 'alpha').map((p) => p.slug)).toEqual(['a'])
  })
  it('every poc names a declared group, and every group slug is unique', () => {
    const slugs = GROUPS.map((x) => x.slug)
    expect(new Set(slugs).size).toBe(slugs.length)
    for (const p of POCS) if (p.group !== undefined) expect(slugs).toContain(p.group)
  })
})

describe('splitSubgroups', () => {
  const group = {
    slug: 'g',
    title: 'G',
    summary: 's',
    subgroups: [
      { slug: 'one', title: 'One', match: ['prior-*', '*-learned'] },
      { slug: 'basic', title: 'Basic', rest: true },
      { slug: 'exact', title: 'Exact', match: ['coverage'] },
    ],
  }
  const all = ['prior-a', 'gate', 'gate-learned', 'coverage', 'more'].map((slug) => entry(slug, slug, [], '2026-01-01'))
  it('puts each poc in the first subgroup whose pattern fits its slug, the rest in the rest subgroup, in declared order', () => {
    const sections = splitSubgroups(all, group)
    expect(sections.map((s) => [s.subgroup?.slug, s.entries.map((e) => e.slug)])).toEqual([
      ['one', ['prior-a', 'gate-learned']],
      ['basic', ['gate', 'more']],
      ['exact', ['coverage']],
    ])
  })
  it('leaves out an empty subgroup and gives one untitled section for a group without subgroups', () => {
    expect(splitSubgroups([entry('gate', 'gate', [], '2026-01-01')], group).map((s) => s.subgroup?.slug)).toEqual([
      'basic',
    ])
    expect(splitSubgroups(all, { slug: 'x', title: 'X', summary: '' }).map((s) => s.subgroup)).toEqual([undefined])
    expect(splitSubgroups([], group)).toEqual([])
  })
  it('the expectation propagation group claims every one of its pocs', () => {
    const ep = GROUPS.find((g) => g.slug === 'expectation-propagation')
    const mine = POCS.filter((p) => p.group === 'expectation-propagation')
    const sections = splitSubgroups(mine, ep)
    expect(sections.flatMap((s) => s.entries).length).toBe(mine.length)
  })
})

describe('the wider suite of the EP coverage page', () => {
  it('every page it points at exists', () => {
    const slugs = new Set(POCS.map((p) => p.slug))
    const missing = epDirections.flatMap((d) => d.pages.filter((s) => !slugs.has(s)).map((s) => `${d.id}: ${s}`))
    expect(missing).toEqual([])
  })
})
