import { describe, expect, it } from 'vitest'
import { neighbours, pageOrder } from './PrevNext.tsx'
import { ancestorsOf, filterToc, innerIds, locate, type TocNode } from './toc.ts'

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

  it('finds folder pages whether or not their folder links them, and pages within pages', () => {
    const hub: TocNode[] = [
      {
        id: 'hub:sk',
        title: 'sk',
        children: [
          { id: 'hub:sk/tree', title: 'tree', children: [{ id: 't1', title: 'Tree', path: '/hub/sk/tree/t1' }] },
          { id: 'hub:sk/lin', title: 'lin', children: [{ id: 'l1', title: 'Ridge', path: '/hub/sk/lin/l1' }] },
        ],
      },
      { id: 'linked', title: 'linked', path: '/g/linked', children: [{ id: 'c', title: 'c', path: '/g/linked/c' }] },
    ]
    expect(locate(hub, '/hub/sk/tree')).toEqual({ id: 'hub:sk/tree', open: ['hub:sk', 'hub:sk/tree'] })
    expect(locate(hub, '/hub/sk')).toEqual({ id: 'hub:sk', open: ['hub:sk'] })
    expect(locate(hub, '/g/linked')).toEqual({ id: 'linked', open: ['linked'] })
    expect(locate(hub, '/g/linked/c/deeper')).toEqual({ id: 'c', open: ['linked'] })
    expect(locate(hub, '/elsewhere')).toEqual({ open: [] })
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
