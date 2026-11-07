import { describe, expect, it } from 'vitest'
import { circularLayout, forceLayout, spectralLayout } from './force.ts'
import type { GraphLayout, LayoutBox, LayoutGraph } from './graph.ts'
import { bipartiteLayout, gridLayout, latticeLayout } from './grids.ts'
import { layered } from './layered.ts'
import { dendrogram, radialTree, tidyTree } from './trees.ts'

const nodes = (ids: string[], size = 20): LayoutGraph['nodes'] => ids.map((id) => ({ id, width: size, height: size }))
const at = (l: GraphLayout, id: string): LayoutBox => {
  const b = l.nodes[id]
  if (b === undefined) throw new Error(`no node ${id}`)
  return b
}
const overlaps = (a: LayoutBox, b: LayoutBox): boolean =>
  Math.abs(a.x - b.x) < (a.width + b.width) / 2 && Math.abs(a.y - b.y) < (a.height + b.height) / 2

describe('layered', () => {
  const g: LayoutGraph = {
    nodes: nodes(['a', 'b', 'c', 'd', 'e'], 30),
    edges: [
      { from: 'a', to: 'b' },
      { from: 'a', to: 'c' },
      { from: 'b', to: 'd' },
      { from: 'c', to: 'd' },
      { from: 'a', to: 'd' },
      { from: 'd', to: 'a' },
      { from: 'e', to: 'd' },
    ],
  }
  const l = layered(g)

  it('puts each edge downwards, breaking cycles, and keeps nodes apart', () => {
    expect(at(l, 'a').y).toBeLessThan(at(l, 'b').y)
    expect(at(l, 'b').y).toBe(at(l, 'c').y)
    expect(at(l, 'd').y).toBeGreaterThan(at(l, 'b').y)
    const ids = Object.keys(l.nodes)
    for (let i = 0; i < ids.length; i++)
      for (let j = i + 1; j < ids.length; j++)
        expect(overlaps(at(l, ids[i] as string), at(l, ids[j] as string))).toBe(false)
  })

  it('routes long edges through the ranks they cross, ends on the boxes', () => {
    const long = l.edges.find((e) => e.from === 'a' && e.to === 'd')
    expect(long?.points.length).toBe(3)
    const first = long?.points[0]
    expect(first?.[1]).toBeCloseTo(at(l, 'a').y + at(l, 'a').height / 2, 9)
    expect(l.edges).toHaveLength(g.edges.length)
  })

  it('lays out left to right and stays inside its size', () => {
    const r = layered(g, { direction: 'right' })
    expect(at(r, 'a').x).toBeLessThan(at(r, 'b').x)
    for (const b of Object.values(r.nodes)) {
      expect(b.x - b.width / 2).toBeGreaterThanOrEqual(0)
      expect(b.x + b.width / 2).toBeLessThanOrEqual(r.width)
    }
  })

  it('keeps groups together and boxes them', () => {
    const grouped = layered({
      nodes: [
        { id: 'r', width: 20, height: 20 },
        { id: 'x1', width: 20, height: 20, group: 'g' },
        { id: 'y', width: 20, height: 20 },
        { id: 'x2', width: 20, height: 20, group: 'g' },
      ],
      edges: ['x1', 'y', 'x2'].map((to) => ({ from: 'r', to })),
    })
    const box = grouped.groups['g']
    expect(box).toBeDefined()
    expect(Math.abs(at(grouped, 'x1').x - at(grouped, 'x2').x)).toBeLessThan(
      Math.abs(at(grouped, 'x1').x - at(grouped, 'y').x) + 1e-9,
    )
  })

  it('is deterministic', () => {
    expect(layered(g)).toEqual(l)
  })
})

describe('trees', () => {
  const g: LayoutGraph = {
    nodes: nodes(['r', 'a', 'b', 'a1', 'a2']),
    edges: [
      { from: 'r', to: 'a' },
      { from: 'r', to: 'b' },
      { from: 'a', to: 'a1' },
      { from: 'a', to: 'a2' },
    ],
  }

  it('centres parents over their children', () => {
    const l = tidyTree(g)
    expect(at(l, 'a').x).toBeCloseTo((at(l, 'a1').x + at(l, 'a2').x) / 2, 9)
    expect(at(l, 'r').x).toBeCloseTo((at(l, 'a').x + at(l, 'b').x) / 2, 9)
    expect(at(l, 'a1').y).toBeGreaterThan(at(l, 'a').y)
  })

  it('puts depths on rings and merges at their heights', () => {
    const r = radialTree(g)
    const d = (id: string): number => Math.hypot(at(r, id).x - at(r, 'r').x, at(r, id).y - at(r, 'r').y)
    expect(d('a')).toBeCloseTo(70, 6)
    expect(d('a1')).toBeCloseTo(140, 6)
    const den = dendrogram(g, { r: 3, a: 1 })
    expect(at(den, 'r').y).toBeLessThan(at(den, 'a').y)
    expect(at(den, 'a1').y).toBe(at(den, 'b').y)
    expect(den.edges[0]?.points).toHaveLength(3)
  })
})

describe('force, circular, spectral and grid layouts', () => {
  const ring: LayoutGraph = {
    nodes: nodes(['a', 'b', 'c', 'd', 'e', 'f']),
    edges: ['a', 'b', 'c', 'd', 'e', 'f'].map((v, i, all) => ({ from: v, to: all[(i + 1) % all.length] as string })),
  }

  it('settles a force layout deterministically, neighbours closer than strangers', () => {
    const l = forceLayout(ring, { seed: 3 })
    expect(forceLayout(ring, { seed: 3 })).toEqual(l)
    expect(l.iterations).toBeLessThanOrEqual(300)
    const dist = (u: string, v: string): number => Math.hypot(at(l, u).x - at(l, v).x, at(l, u).y - at(l, v).y)
    expect(dist('a', 'b')).toBeLessThan(dist('a', 'd'))
  })

  it('places circles, spectra, grids, sides and lattices', () => {
    const c = circularLayout(ring, { radius: 50 })
    expect(Math.round(at(c, 'd').y - at(c, 'a').y)).toBe(100)
    expect(Object.keys(spectralLayout(ring).nodes)).toHaveLength(6)
    const grid = gridLayout(ring, { columns: 3, cell: 40 })
    expect(at(grid, 'd').y - at(grid, 'a').y).toBe(40)
    const bi = bipartiteLayout(ring, ['a', 'b', 'c'])
    expect(at(bi, 'd').x - at(bi, 'a').x).toBe(200)
    expect(
      at(latticeLayout(ring, 2, { spacing: 30 }), 'b').x - at(latticeLayout(ring, 2, { spacing: 30 }), 'a').x,
    ).toBe(30)
  })
})
