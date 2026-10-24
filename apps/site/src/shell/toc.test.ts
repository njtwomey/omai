import { describe, expect, it } from 'vitest'
import { neighbours, pageOrder } from './PrevNext.tsx'
import { ancestorsOf, filterToc, innerIds, type TocNode } from './toc.ts'

const tree: TocNode[] = [
  {
    id: 'a',
    title: 'Controls',
    path: '/g/a',
    children: [
      { id: 'a1', title: 'Number field', path: '/g/a/1', keywords: 'slider' },
      { id: 'a2', title: 'Select', path: '/g/a/2' },
    ],
  },
  { id: 'b', title: 'Animation', children: [{ id: 'b1', title: 'Player', path: '/g/b/1' }] },
]

describe('contents trees', () => {
  it('filters to matches and their ancestors', () => {
    expect(filterToc(tree, 'player')).toEqual([{ ...tree[1], children: [tree[1]?.children?.[0]] }])
    expect(filterToc(tree, 'slider').flatMap((n) => n.children?.map((c) => c.id) ?? [])).toEqual(['a1'])
    expect(filterToc(tree, 'controls')[0]?.children).toHaveLength(2)
    expect(filterToc(tree, '  ')).toHaveLength(2)
    expect(filterToc(tree, 'nothing')).toEqual([])
  })

  it('finds the groups to open for a page', () => {
    expect(ancestorsOf(tree, '/g/b/1')).toEqual(['b'])
    expect(ancestorsOf(tree, '/g/a')).toEqual(['a'])
    expect(ancestorsOf(tree, '/nowhere')).toEqual([])
    expect(innerIds(tree)).toEqual(['a', 'b'])
  })

  it('orders pages depth first across groups', () => {
    const order = pageOrder(tree, { path: '/g', title: 'Gallery' }).map((p) => p.path)
    expect(order).toEqual(['/g', '/g/a', '/g/a/1', '/g/a/2', '/g/b/1'])
    expect(neighbours(pageOrder(tree), '/g/a/2')).toEqual({
      prev: { path: '/g/a/1', title: 'Number field' },
      next: { path: '/g/b/1', title: 'Player' },
    })
  })
})
