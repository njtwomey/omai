import { describe, expect, it } from 'vitest'
import {
  ancestors,
  breadcrumbs,
  buildTree,
  dirHref,
  dirOfKey,
  dirTitles,
  duplicateSlugs,
  findDir,
  locate,
  type PocDir,
  type PocEntry,
  pocsBelow,
  searchPocs,
} from './poc.ts'

const entry = (slug: string, dir: string[], added = '2026-01-01', title = slug, tags: string[] = ['t']): PocEntry => ({
  slug,
  title,
  summary: `About ${title}`,
  tags,
  added,
  dir,
})

const meta = (title: string, order?: number): PocDir => ({
  title,
  summary: `${title} summary`,
  ...(order === undefined ? {} : { order }),
})

const dirs: Record<string, PocDir> = {
  a: meta('Alpha', 20),
  'a/x': meta('Ex', 2),
  'a/y': meta('Why', 1),
  b: meta('Beta', 10),
  empty: meta('Empty', 30),
}
const entries = [
  entry('p1', ['a'], '2026-01-01', 'One'),
  entry('p2', ['a', 'x'], '2026-02-01', 'Two'),
  entry('p3', ['a', 'x'], '2026-03-01', 'Three'),
  entry('p4', ['a', 'x'], '2026-03-01', 'Apple'),
  entry('p5', ['b'], '2026-01-01', 'Five'),
  entry('p6', ['loose'], '2026-01-01', 'Six'),
]
const tree = buildTree(entries, dirs)

describe('locate', () => {
  it('reads the slug and the directory from the folders above the file', () => {
    expect(locate('./pocs/expectation-propagation/learned/ep-gaussian-learned/poc.ts')).toEqual({
      slug: 'ep-gaussian-learned',
      dir: ['expectation-propagation', 'learned'],
      flat: false,
    })
    expect(locate('./pocs/models/gam/Page.tsx').dir).toEqual(['models'])
  })
  it('puts a page still directly under pocs/ in the directory of its legacy group, or in more', () => {
    expect(locate('./pocs/ep-new-thing/poc.ts', 'expectation-propagation')).toEqual({
      slug: 'ep-new-thing',
      dir: ['expectation-propagation'],
      flat: true,
    })
    expect(locate('./pocs/composed-atoms-x/poc.ts', 'composed-atoms').dir).toEqual(['composed-atoms'])
    expect(locate('./pocs/stray/poc.ts').dir).toEqual(['more'])
  })
  it('reads the directory of a dir.ts', () => {
    expect(dirOfKey('./pocs/a/b/dir.ts')).toEqual(['a', 'b'])
  })
})

describe('buildTree', () => {
  it('has the directories in order of their `order` (those without one last), and counts recursively', () => {
    expect(tree.dirs.map((d) => d.name)).toEqual(['b', 'a', 'empty', 'loose'])
    const a = findDir(tree, ['a'])
    expect(a?.count).toBe(4)
    expect(a?.pocs.map((p) => p.slug)).toEqual(['p1'])
    expect(a?.dirs.map((d) => d.name)).toEqual(['y', 'x'])
    expect(tree.count).toBe(6)
  })
  it('lists the pocs of a directory newest first, then by title', () => {
    expect(findDir(tree, ['a', 'x'])?.pocs.map((p) => p.slug)).toEqual(['p4', 'p3', 'p2'])
  })
  it('keeps an empty directory that has a dir.ts, and makes a title for one nobody described', () => {
    const empty = findDir(tree, ['empty'])
    expect(empty?.count).toBe(0)
    expect(empty?.declared).toBe(true)
    const loose = findDir(tree, ['loose'])
    expect(loose?.declared).toBe(false)
    expect(loose?.meta.title).toBe('Loose')
    expect(findDir(buildTree([entry('q', ['graphics-and-fields'])], {}), ['graphics-and-fields'])?.meta.title).toBe(
      'Graphics and fields',
    )
  })
  it('gives every ancestor of a directory that exists only through its pocs a node', () => {
    const t = buildTree([entry('q', ['u', 'v', 'w'])], {})
    expect(ancestors(t, ['u', 'v', 'w']).map((d) => d.name)).toEqual(['u', 'v', 'w'])
    expect(t.count).toBe(1)
  })
  it('lists every poc once through pocsBelow', () => {
    expect(
      pocsBelow(tree)
        .map((p) => p.slug)
        .sort(),
    ).toEqual(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
  })
})

describe('finding and the trail', () => {
  it('finds a directory by path, the root by the empty path, and nothing for an unknown path', () => {
    expect(findDir(tree, [])).toBe(tree)
    expect(findDir(tree, ['a', 'x'])?.meta.title).toBe('Ex')
    expect(findDir(tree, ['a', 'nope'])).toBeUndefined()
    expect(findDir(tree, ['nope'])).toBeUndefined()
    expect(ancestors(tree, ['a', 'nope']).map((d) => d.name)).toEqual(['a'])
  })
  it('makes the trail of a directory (no link on the last) and of a poc (the poc last, without a link)', () => {
    expect(breadcrumbs(tree, [])).toEqual([])
    expect(breadcrumbs(tree, ['a', 'x'])).toEqual([
      { label: 'Alpha', to: '/dir/a' },
      { label: 'Ex', to: undefined },
    ])
    const p = entries[1] as PocEntry
    expect(breadcrumbs(tree, p.dir, p)).toEqual([
      { label: 'Alpha', to: '/dir/a' },
      { label: 'Ex', to: '/dir/a/x' },
      { label: 'Two', to: undefined },
    ])
    expect(dirTitles(tree, p)).toEqual(['Alpha', 'Ex'])
  })
  it('addresses a directory by /dir/<path> and the root by /', () => {
    expect(dirHref([])).toBe('/')
    expect(dirHref(['a', 'x'])).toBe('/dir/a/x')
  })
})

describe('searchPocs', () => {
  const all = [
    entry('a', ['one'], '2026-01-01', 'Alpha detector', ['change', 'online']),
    entry('b', ['two', 'deep'], '2026-03-01', 'Beta sampler', ['mcmc']),
    entry('c', ['one'], '2026-03-01', 'Gamma detector', ['change']),
  ]
  it('keeps everything for an empty query, newest first and then by title', () => {
    expect(searchPocs(all, '  ').map((e) => e.slug)).toEqual(['b', 'c', 'a'])
  })
  it('needs every word, in the title, summary, tags or slug, ignoring case', () => {
    expect(searchPocs(all, 'DETECTOR change').map((e) => e.slug)).toEqual(['c', 'a'])
    expect(searchPocs(all, 'detector mcmc')).toEqual([])
    expect(searchPocs(all, 'about beta').map((e) => e.slug)).toEqual(['b'])
  })
  it('also matches the directory path, and a result keeps its directory', () => {
    const hits = searchPocs(all, 'deep')
    expect(hits.map((e) => e.slug)).toEqual(['b'])
    expect(hits[0]?.dir).toEqual(['two', 'deep'])
    expect(searchPocs(all, 'one').map((e) => e.slug)).toEqual(['c', 'a'])
  })
})

describe('duplicateSlugs', () => {
  it('names the slugs used by more than one poc, wherever they sit', () => {
    expect(duplicateSlugs(entries)).toEqual([])
    expect(duplicateSlugs([...entries, entry('p2', ['b'])])).toEqual(['p2'])
  })
})
