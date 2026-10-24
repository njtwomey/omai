import { describe, expect, it } from 'vitest'
import { neighbours, pageOrder } from './PrevNext.tsx'
import { ancestorsOf, filterToc, innerIds, type TocNode } from './toc.ts'

const tree: TocNode[] = [
  {
    id: 'a',
    title: 'controls',
    children: [
      { id: 'a1', title: 'Number field', path: '/g/a/1' },
      { id: 'a2', title: 'Select', path: '/g/a/2' },
    ],
  },
  { id: 'b', title: 'animation', children: [{ id: 'b1', title: 'Player', path: '/g/b/1' }] },
]

describe('contents trees', () => {
  it('filters to matching titles and their folders', () => {
    expect(filterToc(tree, 'player')).toEqual([{ ...tree[1], children: [tree[1]?.children?.[0]] }])
    expect(filterToc(tree, 'controls')[0]?.children).toHaveLength(2)
    expect(filterToc(tree, '  ')).toHaveLength(2)
    expect(filterToc(tree, 'nothing')).toEqual([])
  })

  it('finds the folders on the way to a page', () => {
    expect(ancestorsOf(tree, '/g/b/1')).toEqual(['b'])
    expect(ancestorsOf(tree, '/nowhere')).toEqual([])
    expect(innerIds(tree)).toEqual(['a', 'b'])
  })

  it('orders pages depth first across folders', () => {
    const order = pageOrder(tree, { path: '/g', title: 'Gallery' }).map((p) => p.path)
    expect(order).toEqual(['/g', '/g/a/1', '/g/a/2', '/g/b/1'])
    expect(neighbours(pageOrder(tree), '/g/a/2')).toEqual({
      prev: { path: '/g/a/1', title: 'Number field' },
      next: { path: '/g/b/1', title: 'Player' },
    })
  })
})
