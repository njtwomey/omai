import { describe, expect, it } from 'vitest'
import { type EdgeRoute, type LayoutBox, type LayoutGraph, manualLayout, withOrthogonalRoutes } from './graph.ts'
import { layered } from './layered.ts'
import { orthogonalRoutes, separateRoutes, sideReach } from './orthogonal.ts'
import { tidyTree } from './trees.ts'

const box = (x: number, y: number, width = 40, height = 20): LayoutBox => ({ x, y, width, height })
type Points = readonly (readonly [number, number])[]

const rectilinear = (points: Points): boolean =>
  points.slice(1).every((q, i) => {
    const p = points[i] as readonly [number, number]
    return Math.abs(p[0] - q[0]) < 1e-9 || Math.abs(p[1] - q[1]) < 1e-9
  })
/** Whether a rectilinear polyline passes through the inside of a box. */
const through = (points: Points, b: LayoutBox, pad = 0): boolean =>
  points.slice(1).some((q, i) => {
    const p = points[i] as readonly [number, number]
    const [x0, x1] = [b.x - b.width / 2 - pad, b.x + b.width / 2 + pad]
    const [y0, y1] = [b.y - b.height / 2 - pad, b.y + b.height / 2 + pad]
    const [lx, hx, ly, hy] = [Math.min(p[0], q[0]), Math.max(p[0], q[0]), Math.min(p[1], q[1]), Math.max(p[1], q[1])]
    return hx > x0 + 1e-6 && lx < x1 - 1e-6 && hy > y0 + 1e-6 && ly < y1 - 1e-6
  })
/** Whether a point lies on the boundary of a box. */
const onBoundary = (p: readonly [number, number], b: LayoutBox): boolean => {
  const [dx, dy] = [Math.abs(p[0] - b.x), Math.abs(p[1] - b.y)]
  const [hw, hh] = [b.width / 2, b.height / 2]
  return (Math.abs(dx - hw) < 1e-6 && dy <= hh + 1e-6) || (Math.abs(dy - hh) < 1e-6 && dx <= hw + 1e-6)
}
/** The collinear stretch two routes share, if any: their segments on one line that overlap for more than a pixel. */
const overlapLength = (p: Points, q: Points): number => {
  let longest = 0
  for (let i = 0; i + 1 < p.length; i++)
    for (let j = 0; j + 1 < q.length; j++) {
      const [a, b, c, d] = [p[i], p[i + 1], q[j], q[j + 1]] as [
        readonly [number, number],
        readonly [number, number],
        readonly [number, number],
        readonly [number, number],
      ]
      for (const k of [0, 1] as const) {
        const m = k === 0 ? 1 : 0
        if (Math.abs(a[m] - b[m]) > 1e-9 || Math.abs(c[m] - d[m]) > 1e-9 || Math.abs(a[m] - c[m]) > 0.5) continue
        const len =
          Math.min(Math.max(a[k], b[k]), Math.max(c[k], d[k])) - Math.max(Math.min(a[k], b[k]), Math.min(c[k], d[k]))
        longest = Math.max(longest, len)
      }
    }
  return longest
}
const route = (r: EdgeRoute | undefined): Points => {
  if (r === undefined) throw new Error('no route')
  return r.points
}

describe('orthogonalRoutes: the rules', () => {
  it('goes straight when the boxes overlap on an axis, in the middle of the overlap', () => {
    const [v] = orthogonalRoutes({ a: box(0, 0), b: box(10, 100) }, [{ from: 'a', to: 'b' }])
    expect(route(v)).toEqual([
      [5, 10],
      [5, 90],
    ])
    const [h] = orthogonalRoutes({ a: box(0, 0), b: box(100, 6) }, [{ from: 'a', to: 'b' }])
    expect(route(h)).toEqual([
      [20, 3],
      [80, 3],
    ])
    expect(v?.route).toBe('orthogonal')
  })

  it('leaves the facing side and enters the target square on, with an L when one fits', () => {
    const nodes = { a: box(0, 0), b: box(100, 80) }
    const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }])
    const p = route(r)
    expect(p).toHaveLength(3)
    expect(rectilinear(p)).toBe(true)
    expect(onBoundary(p[0] as [number, number], nodes.a)).toBe(true)
    expect(onBoundary(p[2] as [number, number], nodes.b)).toBe(true)
    // Along the flow (down): sideways out of a, down into the top of b.
    expect(p).toEqual([
      [20, 0],
      [100, 0],
      [100, 70],
    ])
    const [right] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }], { flow: 'right' })
    expect(route(right)).toEqual([
      [0, 10],
      [0, 80],
      [80, 80],
    ])
  })

  it('takes a Z with its bend in the middle of the gap when both Ls are blocked', () => {
    const nodes = { a: box(0, 0), b: box(200, 200), p: box(200, 0), q: box(0, 200) }
    const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }])
    expect(route(r)).toEqual([
      [0, 10],
      [0, 100],
      [200, 100],
      [200, 190],
    ])
  })

  it('takes the L with fewer bends over a Z, and a clear one over a blocked one', () => {
    const nodes = { a: box(0, 0), b: box(200, 200), p: box(200, 0) }
    const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }])
    // The sideways-first L would pass through p: the other L is used.
    expect(route(r)).toEqual([
      [0, 10],
      [0, 200],
      [180, 200],
    ])
  })

  it('detours around the boxes in the way when no L or Z is clear', () => {
    const nodes = { a: box(0, 0), b: box(200, 0), wall: box(100, 0, 40, 200) }
    const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }])
    const p = route(r)
    expect(rectilinear(p)).toBe(true)
    expect(through(p, nodes.wall)).toBe(false)
    expect(onBoundary(p[0] as [number, number], nodes.a)).toBe(true)
    expect(onBoundary(p[p.length - 1] as [number, number], nodes.b)).toBe(true)
    expect(p.length).toBeGreaterThan(3)
  })

  it('keeps clear of every other box', () => {
    let seed = 7
    const rnd = (): number => {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      return seed / 4294967296
    }
    for (let trial = 0; trial < 25; trial++) {
      const nodes: Record<string, LayoutBox> = {}
      for (let i = 0; i < 8; i++) nodes[`n${i}`] = box(Math.round(rnd() * 5) * 90, Math.round(rnd() * 5) * 70, 50, 24)
      const ids = Object.keys(nodes)
      // Boxes on one spot would overlap: drop the later one.
      const seen = new Set<string>()
      for (const id of ids) {
        const b = nodes[id] as LayoutBox
        const k = `${b.x},${b.y}`
        if (seen.has(k)) delete nodes[id]
        seen.add(k)
      }
      const keep = Object.keys(nodes)
      const edges = keep.slice(1).map((to, i) => ({ from: keep[i] as string, to }))
      for (const r of orthogonalRoutes(nodes, edges)) {
        expect(rectilinear(r.points), `${r.from}->${r.to} rectilinear`).toBe(true)
        for (const id of keep) {
          if (id === r.from || id === r.to) continue
          expect(through(r.points, nodes[id] as LayoutBox), `${r.from}->${r.to} through ${id}`).toBe(false)
        }
      }
    }
  })

  it('puts parallel segments of different edges on tracks of their own', () => {
    const nodes = { a1: box(0, 0), a2: box(60, 0), b1: box(100, 100), b2: box(160, 100) }
    const rs = orthogonalRoutes(nodes, [
      { from: 'a1', to: 'b1' },
      { from: 'a2', to: 'b2' },
    ])
    const [p, q] = [route(rs[0]), route(rs[1])]
    expect(overlapLength(p, q)).toBeLessThan(1)
    const ys = [(p[1] as [number, number])[1], (q[1] as [number, number])[1]]
    expect(Math.abs((ys[0] as number) - (ys[1] as number))).toBeGreaterThanOrEqual(7.99)
  })

  it('merges edges that leave the same side, and edges that enter it', () => {
    const nodes = { a: box(0, 0), b: box(100, 100), c: box(200, 100) }
    const fan = orthogonalRoutes(nodes, [
      { from: 'a', to: 'b' },
      { from: 'a', to: 'c' },
    ])
    // One port, one trunk: the first points agree and the edges share their first stretch, then part.
    expect(route(fan[0])[0]).toEqual(route(fan[1])[0])
    expect((route(fan[0])[1] as [number, number])[1]).toBe((route(fan[1])[1] as [number, number])[1])
    expect(overlapLength(route(fan[0]), route(fan[1]))).toBeGreaterThan(70)
    const join = orthogonalRoutes({ a: box(0, 0), b: box(200, 0), t: box(100, 100) }, [
      { from: 'a', to: 't' },
      { from: 'b', to: 't' },
    ])
    expect(route(join[0]).at(-1)).toEqual(route(join[1]).at(-1))
  })

  it('runs an edge and its twin side by side', () => {
    const nodes = { a: box(0, 0), b: box(0, 100) }
    const [f, r] = orthogonalRoutes(nodes, [
      { from: 'a', to: 'b' },
      { from: 'b', to: 'a' },
    ])
    expect(overlapLength(route(f), route(r))).toBe(0)
    expect(rectilinear(route(f))).toBe(true)
    expect(Math.abs((route(f)[0] as [number, number])[0] - (route(r)[1] as [number, number])[0])).toBeGreaterThan(3)
  })

  it('passes through waypoints, turning at each, and honours forced sides', () => {
    const nodes = { a: box(0, 0), b: box(200, 200) }
    const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b', via: [[200, 0]] }])
    expect(route(r)).toContainEqual([200, 0])
    expect(rectilinear(route(r))).toBe(true)
    const [forced] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b', fromSide: 'top', toSide: 'bottom' }])
    const p = route(forced)
    expect((p[0] as [number, number])[1]).toBe(-10)
    expect((p[p.length - 1] as [number, number])[1]).toBe(210)
    expect(rectilinear(p)).toBe(true)
  })

  it('is deterministic, skips self-loops and unplaced ends', () => {
    const nodes = { a: box(0, 0), b: box(120, 90), c: box(40, 200) }
    const edges = [
      { from: 'a', to: 'b' },
      { from: 'b', to: 'c' },
      { from: 'a', to: 'a' },
      { from: 'a', to: 'missing' },
      { from: 'c', to: 'a' },
    ]
    const once = orthogonalRoutes(nodes, edges)
    expect(orthogonalRoutes(nodes, edges)).toEqual(once)
    expect(once.map((r) => `${r.from}>${r.to}`)).toEqual(['a>b', 'b>c', 'c>a'])
  })

  it('has the usable reach of a side', () => {
    expect(sideReach(30)).toBe(22)
    expect(sideReach(10)).toBe(7)
    expect(sideReach(0)).toBe(0)
  })
})

describe('separateRoutes', () => {
  const z = (from: string, x0: number, x1: number): EdgeRoute => ({
    from,
    to: `${from}'`,
    points: [
      [x0, 0],
      [x0, 20],
      [x1, 20],
      [x1, 40],
    ],
  })

  it('moves overlapping middle segments onto tracks centred on the line, and leaves the ends put', () => {
    const [p, q] = separateRoutes([z('a', 0, 50), z('b', 10, 60)])
    expect((route(p)[1] as [number, number])[1]).not.toBe((route(q)[1] as [number, number])[1])
    expect(((route(p)[1] as [number, number])[1] + (route(q)[1] as [number, number])[1]) / 2).toBeCloseTo(20, 9)
    expect(route(p)[0]).toEqual([0, 0])
    expect(route(p).at(-1)).toEqual([50, 40])
  })

  it('leaves segments that do not overlap, and edges that share a port, alone', () => {
    const [p, q] = separateRoutes([z('a', 0, 20), z('b', 30, 60)])
    expect((route(p)[1] as [number, number])[1]).toBe(20)
    expect((route(q)[1] as [number, number])[1]).toBe(20)
    const fan = separateRoutes([z('a', 0, 20), { ...z('a', 0, 60), to: 'other' }])
    expect((route(fan[0])[1] as [number, number])[1]).toBe(20)
    expect((route(fan[1])[1] as [number, number])[1]).toBe(20)
  })

  it('does not move a segment into a box', () => {
    const nodes = { wall: box(30, 26, 20, 4) }
    const rs = separateRoutes([z('a', 0, 50), z('b', 10, 60)], { nodes })
    for (const r of rs) expect(through(r.points, nodes.wall)).toBe(false)
    // The track on the wall's side stays where it was; the other moved away.
    expect(rs.map((r) => (r.points[1] as [number, number])[1]).sort()).toEqual([16, 20])
  })
})

describe('layered with orthogonal routes', () => {
  const ids = ['start', 'read', 'check', 'fix', 'ok', 'log', 'end']
  const graph: LayoutGraph = {
    nodes: ids.map((id) => ({ id, width: 70, height: 30 })),
    edges: [
      ['start', 'read'],
      ['read', 'check'],
      ['check', 'fix'],
      ['check', 'ok'],
      ['fix', 'read'],
      ['ok', 'end'],
      ['fix', 'end'],
      ['start', 'log'],
      ['log', 'end'],
      ['start', 'end'],
    ].map(([from, to]) => ({ from: from as string, to: to as string })),
  }

  it('leaves the straight layout exactly as it was', () => {
    expect(layered(graph, { route: 'straight' })).toEqual(layered(graph))
    expect(layered(graph).edges.every((e) => e.route === undefined)).toBe(true)
  })

  for (const direction of ['down', 'right'] as const) {
    it(`routes every edge with right angles into the target's side, direction ${direction}`, () => {
      const l = layered(graph, { route: 'orthogonal', direction })
      expect(l.edges).toHaveLength(graph.edges.length)
      for (const e of l.edges) {
        expect(e.route).toBe('orthogonal')
        expect(rectilinear(e.points), `${e.from}->${e.to}`).toBe(true)
        const [a, b] = [l.nodes[e.from] as LayoutBox, l.nodes[e.to] as LayoutBox]
        expect(onBoundary(e.points[0] as [number, number], a), `${e.from} start`).toBe(true)
        expect(onBoundary(e.points.at(-1) as [number, number], b), `${e.from}->${e.to} end`).toBe(true)
        for (const id of ids) {
          if (id === e.from || id === e.to) continue
          expect(through(e.points, l.nodes[id] as LayoutBox), `${e.from}->${e.to} through ${id}`).toBe(false)
        }
        // The last segment runs along the flow, into the target's side facing the flow.
        const [p, q] = [e.points.at(-2) as [number, number], e.points.at(-1) as [number, number]]
        if (direction === 'down') expect(p[0]).toBeCloseTo(q[0], 9)
        else expect(p[1]).toBeCloseTo(q[1], 9)
      }
    })
  }

  it('sizes the layout by the routed segments and keeps everything inside the margin', () => {
    const l = layered(graph, { route: 'orthogonal' })
    const all = l.edges.flatMap((e) => e.points)
    expect(Math.min(...all.map((p) => p[0]))).toBeGreaterThanOrEqual(16 - 1e-9)
    expect(Math.max(...all.map((p) => p[0]))).toBeLessThanOrEqual(l.width - 16 + 1e-9)
    expect(Math.max(...all.map((p) => p[1]))).toBeLessThanOrEqual(l.height - 16 + 1e-9)
  })

  it('merges edges that leave one port and gives parallel runs in a channel their own tracks', () => {
    const fan: LayoutGraph = {
      nodes: ['a', 'b', 'c', 'd'].map((id) => ({ id, width: 40, height: 20 })),
      edges: ['b', 'c', 'd'].map((to) => ({ from: 'a', to })),
    }
    const f = layered(fan, { route: 'orthogonal', nodeSep: 60 })
    const out = (to: string): Points => route(f.edges.find((e) => e.to === to))
    // The edges that bend leave a's bottom by one port and cross the channel on one track.
    expect(out('b')[0]).toEqual(out('d')[0])
    expect(out('b').length).toBe(4)
    expect((out('b')[1] as [number, number])[1]).toBe((out('d')[1] as [number, number])[1])
    const l = layered(graph, { route: 'orthogonal' })
    // Edges that do not share a port never overdraw.
    const rs = l.edges
    for (let i = 0; i < rs.length; i++)
      for (let j = i + 1; j < rs.length; j++) {
        const [p, q] = [rs[i] as EdgeRoute, rs[j] as EdgeRoute]
        const sharePort =
          p.from === q.from ||
          p.to === q.to ||
          (p.from === q.to && p.to === q.from) ||
          p.from === q.to ||
          p.to === q.from
        if (!sharePort)
          expect(overlapLength(p.points, q.points), `${p.from}->${p.to} with ${q.from}->${q.to}`).toBeLessThan(1)
      }
  })

  it('draws a long edge down the column of each rank it skips, and a reversed edge into the node it points at', () => {
    const l = layered(graph, { route: 'orthogonal' })
    const long = route(l.edges.find((e) => e.from === 'start' && e.to === 'end'))
    // start is rank 0, end rank 5: the edge has more than a Z's worth of bends.
    expect(long.length).toBeGreaterThan(4)
    const back = route(l.edges.find((e) => e.from === 'fix' && e.to === 'read'))
    const readBox = l.nodes['read'] as LayoutBox
    // fix is below read: the edge arrives from below, at read's bottom.
    expect((back.at(-1) as [number, number])[1]).toBeCloseTo(readBox.y + readBox.height / 2, 9)
  })

  it('is deterministic', () => {
    expect(layered(graph, { route: 'orthogonal' })).toEqual(layered(graph, { route: 'orthogonal' }))
  })
})

describe('manualLayout and the other layouts with edge routes', () => {
  const g: LayoutGraph = {
    nodes: ['a', 'b', 'c'].map((id) => ({ id, width: 40, height: 20 })),
    edges: [
      { from: 'a', to: 'b', via: [[50, 60]] },
      { from: 'a', to: 'c' },
    ],
  }
  const at = { a: [0, 0], b: [100, 0], c: [100, 100] } as const

  it('draws straight edges as before, and a polyline through the waypoints of an edge that has them', () => {
    const l = manualLayout(g, at)
    expect(route(l.edges[1])).toHaveLength(2)
    const p = route(l.edges[0])
    expect(p).toHaveLength(3)
    // The waypoint, shifted with everything else into the margin.
    expect(p[1]).toEqual([86, 86])
    expect(l.edges[0]?.route).toBeUndefined()
    // The old margin argument still works.
    expect(manualLayout(g, at, 16)).toEqual(l)
  })

  it('routes through the waypoints with right angles when asked', () => {
    const l = manualLayout(g, at, { route: 'orthogonal' })
    const p = route(l.edges[0])
    expect(rectilinear(p)).toBe(true)
    expect(p).toContainEqual([86, 86])
    expect(l.edges.every((e) => e.route === 'orthogonal')).toBe(true)
    expect(l.height).toBeGreaterThan(100)
  })

  it('turns the edges of any layout orthogonal, keeping its boxes', () => {
    const tree: LayoutGraph = {
      nodes: ['r', 'a', 'b', 'c'].map((id) => ({ id, width: 40, height: 20 })),
      edges: [
        { from: 'r', to: 'a' },
        { from: 'r', to: 'b' },
        { from: 'r', to: 'c' },
      ],
    }
    const l = tidyTree(tree)
    const o = withOrthogonalRoutes(l, tree)
    expect(o.nodes).toEqual(l.nodes)
    for (const e of o.edges) expect(rectilinear(e.points)).toBe(true)
    expect(o.edges.map((e) => e.route)).toEqual(['orthogonal', 'orthogonal', 'orthogonal'])
  })
})
