import { describe as describeNet, mlp } from 'omai-engine/nn'
import { type Scene, type SceneNode, type Structure, structure } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { diagram } from './diagram.ts'

/** A scene node with its absolute position: group translations applied. */
type Placed = { readonly node: SceneNode; readonly dx: number; readonly dy: number }
function walk(nodes: readonly SceneNode[], dx = 0, dy = 0): Placed[] {
  return nodes.flatMap((n): Placed[] => {
    if (n.kind !== 'group') return [{ node: n, dx, dy }]
    const [tx, ty] = n.translate ?? [0, 0]
    return [{ node: n, dx, dy }, ...walk(n.children, dx + tx, dy + ty)]
  })
}
const all = (s: Scene): Placed[] => walk(s.children)
type Pt = readonly [number, number]
const edgePoints = (s: Scene, id: string): Pt[] => {
  const p = all(s).find((q) => q.node.kind === 'path' && q.node.id === id)
  if (p?.node.kind !== 'path') throw new Error(`no edge ${id}`)
  return p.node.points.map((q): Pt => [q[0] + p.dx, q[1] + p.dy])
}
const boxOf = (s: Scene, id: string): { x0: number; y0: number; x1: number; y1: number } => {
  const g = all(s).find((q) => q.node.kind === 'group' && q.node.id === id)
  if (g?.node.kind !== 'group') throw new Error(`no node ${id}`)
  const rs = walk(g.node.children, g.dx + (g.node.translate?.[0] ?? 0), g.dy + (g.node.translate?.[1] ?? 0))
  const shape = rs.find(
    (q) => q.node.kind === 'rect' || q.node.kind === 'circle' || q.node.kind === 'ellipse' || q.node.kind === 'path',
  )
  if (shape === undefined) throw new Error(`no shape for ${id}`)
  const n = shape.node
  if (n.kind === 'path') {
    const [xs, ys] = [n.points.map((q) => q[0] + shape.dx), n.points.map((q) => q[1] + shape.dy)]
    return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) }
  }
  if (n.kind === 'rect')
    return { x0: n.x + shape.dx, y0: n.y + shape.dy, x1: n.x + n.width + shape.dx, y1: n.y + n.height + shape.dy }
  if (n.kind === 'circle')
    return {
      x0: n.cx - n.r + shape.dx,
      y0: n.cy - n.r + shape.dy,
      x1: n.cx + n.r + shape.dx,
      y1: n.cy + n.r + shape.dy,
    }
  if (n.kind === 'ellipse')
    return {
      x0: n.cx - n.rx + shape.dx,
      y0: n.cy - n.ry + shape.dy,
      x1: n.cx + n.rx + shape.dx,
      y1: n.cy + n.ry + shape.dy,
    }
  throw new Error('shape')
}
const rectilinear = (p: readonly Pt[]): boolean =>
  p.slice(1).every((q, i) => Math.abs((p[i] as Pt)[0] - q[0]) < 1e-6 || Math.abs((p[i] as Pt)[1] - q[1]) < 1e-6)
/** Whether a rectilinear polyline crosses the inside of a rectangle. */
const crosses = (p: readonly Pt[], r: { x0: number; y0: number; x1: number; y1: number }): boolean =>
  p.slice(1).some((q, k) => {
    const a = p[k] as Pt
    return (
      Math.max(a[0], q[0]) > r.x0 &&
      Math.min(a[0], q[0]) < r.x1 &&
      Math.max(a[1], q[1]) > r.y0 &&
      Math.min(a[1], q[1]) < r.y1
    )
  })
const node = (id: string, extra: object = {}): { id: string; label: string; role: 'other' } => ({
  id,
  label: id,
  role: 'other',
  ...extra,
})

const flow = structure(
  'pipeline',
  ['start', 'parse', 'valid', 'reject', 'cache', 'compute', 'store', 'end'].map((id) =>
    node(id, id === 'valid' ? { look: { shape: 'diamond' } } : {}),
  ),
  [
    { from: 'start', to: 'parse' },
    { from: 'parse', to: 'valid' },
    { from: 'valid', to: 'reject', label: 'no' },
    { from: 'reject', to: 'parse', label: 'retry' },
    { from: 'valid', to: 'cache', label: 'yes' },
    { from: 'cache', to: 'compute' },
    { from: 'cache', to: 'end', label: 'hit' },
    { from: 'compute', to: 'store' },
    { from: 'store', to: 'end' },
    { from: 'start', to: 'end', label: 'a long way round' },
  ],
)

describe('diagram route', () => {
  it('leaves every straight diagram as it is: route straight is the default', () => {
    for (const options of [
      {},
      { layout: 'tree' as const },
      { layout: 'circular' as const },
      { direction: 'right' as const },
    ])
      expect(diagram(flow, { ...options, route: 'straight' })).toEqual(diagram(flow, options))
  })

  for (const layout of ['layered', 'tree', 'circular', 'force', 'bipartite'] as const)
    it(`draws right-angled edges under the ${layout} layout, with the head along the last segment`, () => {
      const s = diagram(flow, { layout, route: 'orthogonal' })
      flow.edges.forEach((_, i) => {
        const p = edgePoints(s, `edge:${i}`)
        expect(rectilinear(p), `edge ${i}`).toBe(true)
        const node = all(s).find((q) => q.node.kind === 'path' && q.node.id === `edge:${i}`)?.node
        expect(node).toMatchObject({ arrow: true, smooth: false })
      })
    })

  it('enters the target square on its side and leaves the source on a facing side', () => {
    const s = diagram(flow, { route: 'orthogonal' })
    flow.edges.forEach((e, i) => {
      const p = edgePoints(s, `edge:${i}`)
      const [a, b] = [boxOf(s, e.from), boxOf(s, e.to)]
      const [first, last] = [p[0] as Pt, p.at(-1) as Pt]
      const near = (v: number, w: number): boolean => Math.abs(v - w) < 1.5
      // The ends are on the outline of the shapes (a diamond's, an ordinary box's): inside the box, on a side.
      const within = (q: Pt, r: typeof a): boolean =>
        q[0] >= r.x0 - 1 && q[0] <= r.x1 + 1 && q[1] >= r.y0 - 1 && q[1] <= r.y1 + 1
      expect(within(first, a), `${e.from} start`).toBe(true)
      expect(within(last, b), `${e.to} end`).toBe(true)
      // Perpendicular entry: the last segment is along the axis of the side it meets.
      const [pen, tip] = [p.at(-2) as Pt, last]
      const vertical = near(pen[0], tip[0])
      if (vertical) expect(near(tip[1], b.y0) || near(tip[1], b.y1) || e.to === 'valid').toBe(true)
      else expect(near(tip[0], b.x0) || near(tip[0], b.x1) || e.to === 'valid').toBe(true)
    })
  })

  it('keeps every route off the nodes it does not join', () => {
    const s = diagram(flow, { route: 'orthogonal' })
    flow.edges.forEach((e, i) => {
      const p = edgePoints(s, `edge:${i}`)
      for (const n of flow.nodes) {
        if (n.id === e.from || n.id === e.to) continue
        const b = boxOf(s, n.id)
        const hit = p.slice(1).some((q, k) => {
          const a = p[k] as Pt
          return (
            Math.max(a[0], q[0]) > b.x0 + 1 &&
            Math.min(a[0], q[0]) < b.x1 - 1 &&
            Math.max(a[1], q[1]) > b.y0 + 1 &&
            Math.min(a[1], q[1]) < b.y1 - 1
          )
        })
        expect(hit, `edge ${i} (${e.from} to ${e.to}) through ${n.id}`).toBe(false)
      }
    })
  })

  it('draws the same scene twice', () => {
    expect(diagram(flow, { route: 'orthogonal' })).toEqual(diagram(flow, { route: 'orthogonal' }))
  })

  it('keeps both heads of a bidirectional edge, the second lying on the first stretch', () => {
    const s = structure(
      'x',
      [node('a'), node('b', { position: [0, 0] })],
      [{ from: 'a', to: 'b', bidirectional: true }],
    )
    const t = structure(
      'x',
      [node('a', { position: [0, 0] }), node('b', { position: [120, 80] })],
      [{ from: 'a', to: 'b', bidirectional: true }],
    )
    void s
    const sc = diagram(t, { route: 'orthogonal' })
    const edge = edgePoints(sc, 'edge:0')
    const tail = edgePoints(sc, 'edge:0:tail')
    expect(rectilinear(edge)).toBe(true)
    expect(tail[1]).toEqual(edge[0])
    // Along the first segment of the route, which is horizontal or vertical.
    expect(tail[0]?.[0] === (tail[1] as Pt)[0] || tail[0]?.[1] === (tail[1] as Pt)[1]).toBe(true)
  })

  it('runs an edge and the edge back side by side', () => {
    const t = structure(
      'x',
      ['a', 'b'].map((id, i) => node(id, { position: [0, i * 100] })),
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'a' },
      ],
    )
    const sc = diagram(t, { route: 'orthogonal' })
    const [f, r] = [edgePoints(sc, 'edge:0'), edgePoints(sc, 'edge:1')]
    expect(rectilinear(f) && rectilinear(r)).toBe(true)
    expect(Math.abs((f[0] as Pt)[0] - (r[0] as Pt)[0])).toBeGreaterThan(3)
  })

  it('puts labels on the middle of a long segment, on a plate, off the nodes', () => {
    const s = diagram(flow, { route: 'orthogonal' })
    const placed = all(s)
    const labels = placed.filter(
      (q) => q.node.kind === 'text' && ['no', 'retry', 'yes', 'hit', 'a long way round'].includes(q.node.text),
    )
    expect(labels).toHaveLength(5)
    for (const l of labels) {
      if (l.node.kind !== 'text') continue
      const w = l.node.text.length * 10 * 0.6
      const x0 = l.node.anchor === 'middle' ? l.node.x - w / 2 : l.node.x
      for (const n of flow.nodes) {
        const b = boxOf(s, n.id)
        const [lx0, lx1, ly0, ly1] = [x0 + l.dx, x0 + w + l.dx, l.node.y + l.dy - 5, l.node.y + l.dy + 5]
        expect(lx1 <= b.x0 || lx0 >= b.x1 || ly1 <= b.y0 || ly0 >= b.y1, `label ${l.node.text} clear of ${n.id}`).toBe(
          true,
        )
      }
    }
    // The plate lies under the text, above the nodes.
    const i = placed.findIndex((q) => q.node.kind === 'text' && q.node.text === 'hit')
    expect(placed[i - 1]?.node).toMatchObject({ kind: 'rect', fill: 'bg' })
    expect(i).toBeGreaterThan(placed.findIndex((q) => q.node.kind === 'group' && q.node.id === 'end'))
  })

  it('puts a label at labelAt along the route when asked', () => {
    const t = structure(
      'x',
      [node('a', { position: [0, 0] }), node('b', { position: [200, 0] })],
      [{ from: 'a', to: 'b', label: 'mid', labelAt: 0.5 }],
    )
    const s = diagram(t, { route: 'orthogonal' })
    const e = edgePoints(s, 'edge:0')
    const l = all(s).find((q) => q.node.kind === 'text' && q.node.text === 'mid')
    expect(l?.node.kind === 'text' && l.node.x + l.dx).toBeCloseTo(((e[0] as Pt)[0] + (e[1] as Pt)[0]) / 2, 6)
  })

  it('sizes the scene by the routed segments and the labels', () => {
    const s = diagram(flow, { route: 'orthogonal' })
    for (let i = 0; i < flow.edges.length; i++)
      for (const [x, y] of edgePoints(s, `edge:${i}`)) {
        expect(x).toBeGreaterThanOrEqual(0)
        expect(y).toBeGreaterThanOrEqual(0)
        expect(x).toBeLessThanOrEqual(s.width)
        expect(y).toBeLessThanOrEqual(s.height)
      }
  })

  it('leaves self-loops, groups and plates as they are', () => {
    const g = structure(
      'bayes-net',
      [node('a', { group: 'p' }), node('b', { group: 'p' }), node('c')],
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'b', label: 'loop' },
        { from: 'b', to: 'c' },
      ],
      { groups: [{ id: 'p', label: 'N', kind: 'plate', count: 'N' }] },
    )
    const s = diagram(g, { route: 'orthogonal' })
    const placed = all(s)
    expect(placed.some((q) => q.node.kind === 'rect' && q.node.id === 'group:p')).toBe(true)
    const loop = placed.find((q) => q.node.kind === 'path' && q.node.id === 'edge:1')
    expect(loop?.node).toMatchObject({ smooth: true, arrow: true })
    expect(placed.some((q) => q.node.kind === 'text' && q.node.text === 'loop')).toBe(true)
  })

  it('routes the manual layout, through the waypoints of an edge that has them', () => {
    const t = structure(
      'x',
      [node('a', { position: [0, 0] }), node('b', { position: [200, 120] })],
      [{ from: 'a', to: 'b', via: [[200, 0]] }],
    )
    for (const route of ['straight', 'orthogonal'] as const) {
      const s = diagram(t, { route })
      const p = edgePoints(s, 'edge:0')
      const a = boxOf(s, 'a')
      // Through the waypoint (200, 0) in the layout's units: the same height as a's centre.
      const cy = (a.y0 + a.y1) / 2
      expect(p.some((q) => Math.abs(q[1] - cy) < 1e-6 && q[0] > a.x1)).toBe(true)
      expect(p.length).toBeGreaterThanOrEqual(3)
      const e = all(s).find((q) => q.node.kind === 'path' && q.node.id === 'edge:0')?.node
      expect(e).toMatchObject({ smooth: false })
    }
  })
})

describe('diagram: detail lines, TeX edge labels, group looks, notes and outside labels', () => {
  const texts = (s: Scene): string[] => all(s).flatMap((q) => (q.node.kind === 'text' ? [q.node.text] : []))
  const texs = (s: Scene): string[] => all(s).flatMap((q) => (q.node.kind === 'tex' ? [q.node.tex] : []))

  it('draws a multi-line detail as muted lines and sizes the node for them', () => {
    const one = diagram(structure('x', [node('a', { detail: 'first' })], []))
    const two = diagram(structure('x', [node('a', { detail: 'first\nsecond\nthird' })], []))
    expect(texts(two)).toEqual(['a', 'first', 'second', 'third'])
    expect(boxOf(two, 'a').y1 - boxOf(two, 'a').y0).toBeGreaterThan(boxOf(one, 'a').y1 - boxOf(one, 'a').y0 + 20)
    const fills = all(two).flatMap((q) => (q.node.kind === 'text' ? [[q.node.size, q.node.fill]] : []))
    expect(fills).toEqual([
      [12, 'fg'],
      [10, 'muted'],
      [10, 'muted'],
      [10, 'muted'],
    ])
    expect(all(two).find((q) => q.node.kind === 'group' && q.node.id === 'a')?.node).toMatchObject({
      title: 'a: first second third',
    })
  })

  it('typesets a label line written $...$ as TeX, on any edge, and takes several lines', () => {
    const t = structure(
      'x',
      [node('a', { position: [0, 0] }), node('b', { position: [200, 0] }), node('c', { position: [200, 100] })],
      [
        { from: 'a', to: 'b', label: '$q(x_t \\mid x_0)$' },
        { from: 'a', to: 'c', label: 'noise\n$\\epsilon$' },
        { from: 'c', to: 'c', label: '$\\alpha$' },
      ],
    )
    const s = diagram(t)
    expect(texs(s)).toEqual(expect.arrayContaining(['q(x_t \\mid x_0)', '\\epsilon', '\\alpha']))
    expect(texts(s)).toContain('noise')
    // The plate covers both lines.
    const placed = all(s)
    const i = placed.findIndex((q) => q.node.kind === 'text' && q.node.text === 'noise')
    const plate = placed[i - 1]?.node
    const tex = placed.find((q) => q.node.kind === 'tex' && q.node.tex === '\\epsilon')?.node
    expect(plate?.kind).toBe('rect')
    if (plate?.kind !== 'rect' || tex?.kind !== 'tex') throw new Error('plate')
    expect(tex.y).toBeGreaterThan(plate.y)
    expect(tex.y).toBeLessThan(plate.y + plate.height)
  })

  it('does not change the plain single-line labels', () => {
    const t = structure(
      'x',
      [node('a', { position: [0, 0] }), node('b', { position: [200, 0] })],
      [{ from: 'a', to: 'b', label: 'rank' }],
    )
    const l = all(diagram(t)).find((q) => q.node.kind === 'text' && q.node.text === 'rank')?.node
    expect(l).toMatchObject({ size: 10, fill: 'muted', baseline: 'middle' })
  })

  it('draws a group with its own look: frame, fill, dash and label colour', () => {
    const g = (look?: object): Structure =>
      structure('bayes-net', [node('a', { group: 'p' })], [], {
        groups: [{ id: 'p', label: 'forward', kind: 'plate', count: 3, ...(look === undefined ? {} : { look }) }],
      })
    const plain = all(diagram(g())).find((q) => q.node.kind === 'rect' && q.node.id === 'group:p')?.node
    expect(plain).toMatchObject({ stroke: 'fg', fill: 'none' })
    const s = diagram(g({ stroke: 'tab4', fill: 'surface', dash: [6, 3], strokeWidth: 2, text: 'tab4' }))
    const frame = all(s).find((q) => q.node.kind === 'rect' && q.node.id === 'group:p')?.node
    expect(frame).toMatchObject({ stroke: 'tab4', fill: 'surface', dash: [6, 3], strokeWidth: 2 })
    const label = all(s).find((q) => q.node.kind === 'text' && q.node.text === 'forward')?.node
    expect(label).toMatchObject({ fill: 'tab4' })
    expect(all(s).find((q) => q.node.kind === 'text' && q.node.text === '3')?.node).toMatchObject({ fill: 'tab4' })
    expect(
      all(diagram(g({ stroke: 'tab4' }))).find((q) => q.node.kind === 'rect' && q.node.id === 'group:p')?.node,
    ).not.toHaveProperty('dash')
  })

  it('draws a note under a node, with TeX, inside the bounds and clear of the node', () => {
    const t = structure('x', [node('a', { note: 'a note\n$x_0$' }), node('b')], [{ from: 'a', to: 'b' }])
    const plain = diagram(structure('x', [node('a'), node('b')], [{ from: 'a', to: 'b' }]))
    const s = diagram(t)
    const note = all(s).find((q) => q.node.kind === 'text' && q.node.text === 'a note')
    const eq = all(s).find((q) => q.node.kind === 'tex' && q.node.tex === 'x_0')
    if (note?.node.kind !== 'text' || eq?.node.kind !== 'tex') throw new Error('note')
    const a = boxOf(s, 'a')
    // Under the node (the edge leaves a's middle, so the note takes the side of it), never on it.
    expect(note.node.y + note.dy).toBeGreaterThan(a.y1)
    expect(note.node.fill).toBe('muted')
    expect(eq.node.y + eq.dy).toBeGreaterThan(note.node.y + note.dy)
    expect(s.height).toBeGreaterThanOrEqual(plain.height)
    // It sits clear of every edge.
    const w = 'a note'.length * 10 * 0.6
    const x0 =
      note.node.anchor === 'middle' ? note.node.x - w / 2 : note.node.anchor === 'end' ? note.node.x - w : note.node.x
    const y = note.node.y + note.dy
    expect(crosses(edgePoints(s, 'edge:0'), { x0: x0 + note.dx, y0: y - 5, x1: x0 + note.dx + w, y1: y + 5 })).toBe(
      false,
    )
  })

  it('widens the scene to hold a note that reaches past the layout, and the group to hold the notes of its members', () => {
    const wide = structure('x', [node('a', { note: 'a note that is much longer than the node it is under' })], [])
    expect(diagram(wide).width).toBeGreaterThanOrEqual(
      'a note that is much longer than the node it is under'.length * 10 * 0.6,
    )
    const inGroup = structure('bayes-net', [node('a', { group: 'g', note: 'n\nm' })], [], {
      groups: [{ id: 'g', label: 'g', kind: 'plate' }],
    })
    const s = diagram(inGroup)
    const frame = all(s).find((q) => q.node.kind === 'rect' && q.node.id === 'group:g')
    const note = all(s).find((q) => q.node.kind === 'text' && q.node.text === 'm')
    if (frame?.node.kind !== 'rect' || note?.node.kind !== 'text') throw new Error('frame')
    expect(note.node.y + note.dy).toBeLessThan(frame.node.y + frame.dy + frame.node.height)
  })

  it('keeps the text of a free-standing shape none node: no note needed for a caption', () => {
    const t = structure('x', [node('cap', { label: 'caption', position: [0, 0], look: { shape: 'none' } })], [])
    expect(texts(diagram(t))).toEqual(['caption'])
  })

  it('moves the label of a factor square off the edges, and leaves it below when nothing is there', () => {
    const base = (edges: { from: string; to: string }[]): Structure =>
      structure(
        'factor-graph',
        [
          { id: 'f', label: 'f', role: 'factor' as const, position: [0, 0] as const },
          { id: 'x', label: 'x', role: 'latent' as const, position: [0, 100] as const },
          { id: 'y', label: 'y', role: 'latent' as const, position: [0, -100] as const },
        ],
        edges,
      )
    const f = (s: Scene): { x: number; y: number } => {
      const t = all(s).find((q) => q.node.kind === 'text' && q.node.text === 'f')
      if (t?.node.kind !== 'text') throw new Error('label')
      return { x: t.node.x + t.dx, y: t.node.y + t.dy }
    }
    const free = diagram(base([]))
    const sq = boxOf(free, 'f')
    expect(f(free).y).toBeGreaterThan(sq.y1)
    expect(f(free).x).toBeCloseTo((sq.x0 + sq.x1) / 2, 6)
    // An edge leaving downwards crosses the place below: the label moves off it, beside the line.
    const crossed = diagram(base([{ from: 'f', to: 'x' }]))
    const at = f(crossed)
    const line = edgePoints(crossed, 'edge:0')
    expect(crosses(line, { x0: at.x, y0: at.y - 5, x1: at.x + 7, y1: at.y + 5 })).toBe(false)
    expect(at.x).toBeGreaterThan((line[0] as Pt)[0])
    // Edges up and down: still off both.
    const both = diagram(
      base([
        { from: 'f', to: 'x' },
        { from: 'f', to: 'y' },
      ]),
    )
    const t = f(both)
    for (const id of ['edge:0', 'edge:1'])
      expect(crosses(edgePoints(both, id), { x0: t.x, y0: t.y - 5, x1: t.x + 7, y1: t.y + 5 })).toBe(false)
  })
})

describe('diagram: routing, kinds of edge, label plates, origin and corners', () => {
  const at = (id: string, x: number, y: number, extra: object = {}): ReturnType<typeof node> =>
    node(id, { position: [x, y], ...extra })
  const longest = (p: readonly Pt[], q: readonly Pt[]): number => {
    let best = 0
    for (let i = 0; i + 1 < p.length; i++)
      for (let j = 0; j + 1 < q.length; j++) {
        const [a, b, c, d] = [p[i], p[i + 1], q[j], q[j + 1]] as [Pt, Pt, Pt, Pt]
        for (const k of [0, 1] as const) {
          const m = k === 0 ? 1 : 0
          if (Math.abs(a[m] - b[m]) > 1e-6 || Math.abs(c[m] - d[m]) > 1e-6 || Math.abs(a[m] - c[m]) > 0.5) continue
          best = Math.max(
            best,
            Math.min(Math.max(a[k], b[k]), Math.max(c[k], d[k])) - Math.max(Math.min(a[k], b[k]), Math.min(c[k], d[k])),
          )
        }
      }
    return best
  }

  it('lets edges pass through an invisible node and through a node that says it is not an obstacle', () => {
    const row = (extra: object): Structure =>
      structure('x', [at('a', 0, 0), at('ghost', 120, 0, extra), at('b', 240, 0)], [{ from: 'a', to: 'b' }])
    // A box in the way: the edge goes round it.
    expect(edgePoints(diagram(row({}), { route: 'orthogonal' }), 'edge:0').length).toBeGreaterThan(2)
    // Invisible (no outline, no text): straight through, as one segment.
    const ghost = { label: '', look: { shape: 'none' } }
    expect(edgePoints(diagram(row(ghost), { route: 'orthogonal' }), 'edge:0')).toHaveLength(2)
    // Said so: a visible node that edges may cross; and invisible but said to be an obstacle.
    expect(edgePoints(diagram(row({ obstacle: false }), { route: 'orthogonal' }), 'edge:0')).toHaveLength(2)
    expect(
      edgePoints(diagram(row({ ...ghost, obstacle: true }), { route: 'orthogonal' }), 'edge:0').length,
    ).toBeGreaterThan(2)
  })

  it('still starts and ends an edge at a node that is not an obstacle, and keeps labels clear of nodes that are', () => {
    const t = structure(
      'x',
      [at('a', 0, 0), at('ph', 200, 100, { label: '', obstacle: false })],
      [{ from: 'a', to: 'ph', label: 'into the placeholder' }],
    )
    const s = diagram(t, { route: 'orthogonal' })
    const [p, ph] = [edgePoints(s, 'edge:0'), boxOf(s, 'ph')]
    const [first, last] = [p[0] as Pt, p.at(-1) as Pt]
    expect(last[0] >= ph.x0 - 1 && last[0] <= ph.x1 + 1 && last[1] >= ph.y0 - 1 && last[1] <= ph.y1 + 1).toBe(true)
    expect(first[0]).toBeLessThanOrEqual(boxOf(s, 'a').x1 + 1)
  })

  it('draws edges of different colours into one box on separate ports and keeps their own colours', () => {
    const t = structure(
      'x',
      [at('a', 0, 0), at('b', 0, 60), at('w', 200, 200, { label: 'wide box' })],
      [
        { from: 'a', to: 'w', look: { stroke: 'tab0' } },
        { from: 'b', to: 'w', look: { stroke: 'tab1' } },
      ],
    )
    const s = diagram(t, { route: 'orthogonal' })
    const [p, q] = [edgePoints(s, 'edge:0'), edgePoints(s, 'edge:1')]
    expect((p.at(-1) as Pt)[0] === (q.at(-1) as Pt)[0] && (p.at(-1) as Pt)[1] === (q.at(-1) as Pt)[1]).toBe(false)
    expect(longest(p, q)).toBeLessThan(1)
    // Alike, the same two edges merge into one port.
    const same = diagram(
      structure('x', t.nodes, [
        { from: 'a', to: 'w' },
        { from: 'b', to: 'w' },
      ]),
      { route: 'orthogonal' },
    )
    expect(edgePoints(same, 'edge:0').at(-1)).toEqual(edgePoints(same, 'edge:1').at(-1))
    // `merge: false` separates even those.
    const apart = diagram(
      structure('x', t.nodes, [
        { from: 'a', to: 'w' },
        { from: 'b', to: 'w' },
      ]),
      { route: 'orthogonal', merge: false },
    )
    expect(edgePoints(apart, 'edge:0').at(-1)).not.toEqual(edgePoints(apart, 'edge:1').at(-1))
  })

  it('honours the sides and places an edge asks for, under the manual layout', () => {
    const t = structure(
      'x',
      [at('a', 0, 0), at('b', 0, 160), at('w', 200, 80, { label: 'wide box' })],
      [
        { from: 'a', to: 'w', toSide: 'top', look: { stroke: 'tab0' } },
        { from: 'b', to: 'w', toSide: 'bottom', look: { stroke: 'tab1' } },
        { from: 'a', to: 'b', fromSide: 'left', toSide: 'left' },
        { from: 'w', to: 'a', fromSide: 'top', fromAt: 0.9, toSide: 'right', toAt: 0.1 },
      ],
    )
    const s = diagram(t, { route: 'orthogonal' })
    const w = boxOf(s, 'w')
    const [p, q, r, u] = [0, 1, 2, 3].map((i) => edgePoints(s, `edge:${i}`)) as [Pt[], Pt[], Pt[], Pt[]]
    expect((p.at(-1) as Pt)[1]).toBeCloseTo(w.y0, 6)
    expect((q.at(-1) as Pt)[1]).toBeCloseTo(w.y1, 6)
    expect((r[0] as Pt)[0]).toBeCloseTo(boxOf(s, 'a').x0, 6)
    expect((r.at(-1) as Pt)[0]).toBeCloseTo(boxOf(s, 'b').x0, 6)
    expect((u[0] as Pt)[1]).toBeCloseTo(w.y0, 6)
    // Near the right end of the top side, and near the top of a's right side.
    expect((u[0] as Pt)[0]).toBeGreaterThan((w.x0 + w.x1) / 2)
    expect((u.at(-1) as Pt)[0]).toBeCloseTo(boxOf(s, 'a').x1, 6)
    for (const e of [p, q, r, u]) expect(rectilinear(e as Pt[])).toBe(true)
  })

  it('separates three edges of different looks into one wide box', () => {
    const t = structure(
      'x',
      [
        at('a', -300, 0),
        at('b', -300, 70),
        at('c', -300, 140),
        at('w', 0, 300, { label: 'a wide box that takes all three' }),
      ],
      ['tab0', 'tab1', 'tab2'].map((stroke, i) => ({
        from: 'abc'[i] as string,
        to: 'w',
        look: { stroke: stroke as 'tab0' },
      })),
    )
    const s = diagram(t, { route: 'orthogonal' })
    const ps = [0, 1, 2].map((i) => edgePoints(s, `edge:${i}`))
    expect(new Set(ps.map((p) => (p.at(-1) as Pt)[0])).size).toBe(3)
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++) expect(longest(ps[i] as Pt[], ps[j] as Pt[])).toBeLessThan(1)
  })

  for (const layout of ['layered', 'manual'] as const)
    for (const direction of ['down', 'right'] as const)
      it(`keeps the dashed target edge of an autoencoder off the solid edges (${layout}, ${direction})`, () => {
        const net = describeNet(mlp([8, 3, 8]), [32, 8], { target: 'reconstruct' })
        // Placed in a row (or a column), as a slide would, for the manual layout.
        const placed =
          layout === 'manual'
            ? structure(
                net.type,
                net.nodes.map((n, i) => ({
                  ...n,
                  position: (direction === 'down' ? [0, i * 110] : [i * 190, 0]) as [number, number],
                })),
                net.edges,
                { title: net.title as string },
              )
            : net
        const s = diagram(placed, { route: 'orthogonal', direction, layout })
        const ps = placed.edges.map((_, i) => edgePoints(s, `edge:${i}`))
        const t = placed.edges.findIndex((e) => e.kind === 'skip')
        const target = ps[t] as Pt[]
        expect(rectilinear(target)).toBe(true)
        ps.forEach((p, i) => {
          if (i !== t) expect(longest(target, p), `target with edge ${i}`).toBeLessThan(1)
        })
        for (const n of placed.nodes) {
          if (n.id === 'input' || n.id === 'output') continue
          expect(crosses(target, boxOf(s, n.id)), `target through ${n.id}`).toBe(false)
        }
        // It does not share a port with the solid edges at its two ends.
        const solidEnds = ps.filter((_, i) => i !== t).flatMap((p) => [p[0] as Pt, p.at(-1) as Pt])
        for (const end of [target[0] as Pt, target.at(-1) as Pt])
          expect(solidEnds.some((q) => Math.hypot(q[0] - end[0], q[1] - end[1]) < 2)).toBe(false)
      })

  it('draws the label plate in the page colour by default, in a given paint, or not at all', () => {
    const t = structure('x', [at('a', 0, 0), at('b', 200, 0)], [{ from: 'a', to: 'b', label: 'rank' }])
    const plates = (s: Scene): SceneNode[] =>
      all(s).flatMap((q) => (q.node.kind === 'rect' && q.node.id === 'edge:0:plate' ? [q.node] : []))
    expect(plates(diagram(t))[0]).toMatchObject({ fill: 'bg', opacity: 0.9 })
    expect(plates(diagram(t, { labelPlate: 'page' }))).toEqual(plates(diagram(t)))
    expect(plates(diagram(t, { labelPlate: 'surface' }))[0]).toMatchObject({ fill: 'surface', opacity: 1 })
    expect(plates(diagram(t, { labelPlate: '#222222' }))[0]).toMatchObject({ fill: '#222222' })
    expect(plates(diagram(t, { labelPlate: 'none' }))).toHaveLength(0)
    expect(all(diagram(t, { labelPlate: 'none' })).some((q) => q.node.kind === 'text' && q.node.text === 'rank')).toBe(
      true,
    )
  })

  for (const route of ['straight', 'orthogonal'] as const)
    it(`keeps a label with no plate off its own edge, the nodes and the other edges (${route})`, () => {
      const t = structure(
        'x',
        [at('a', 0, 0), at('b', 200, 0), at('c', 100, 120), at('d', 100, -120)],
        [
          { from: 'a', to: 'b', label: 'no plate here' },
          { from: 'c', to: 'd', label: 'cross' },
        ],
      )
      const s = diagram(t, { route, labelPlate: 'none' })
      for (const [i, text] of ['no plate here', 'cross'].entries()) {
        const l = all(s).find((q) => q.node.kind === 'text' && q.node.text === text)
        if (l?.node.kind !== 'text') throw new Error('label')
        const w = text.length * 10 * 0.6
        const x0 = (l.node.anchor === 'middle' ? l.node.x - w / 2 : l.node.x) + l.dx
        const box = { x0: x0 - 1, x1: x0 + w + 1, y0: l.node.y + l.dy - 5, y1: l.node.y + l.dy + 5 }
        expect(crosses(edgePoints(s, `edge:${i}`), box), `${text}: on its own edge`).toBe(false)
        for (const n of t.nodes) {
          const b = boxOf(s, n.id)
          expect(box.x1 <= b.x0 || box.x0 >= b.x1 || box.y1 <= b.y0 || box.y0 >= b.y1, `${text}: on ${n.id}`).toBe(true)
        }
      }
    })

  it('gives an edge label stable ids, `edge:<i>:label` for the label and `edge:<i>:plate` for its plate', () => {
    const t = structure(
      'x',
      [at('a', 0, 0), at('b', 200, 0), at('c', 400, 0)],
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c', label: 'two\nlines' },
      ],
    )
    const s = diagram(t, { route: 'orthogonal' })
    const label = all(s).find((q) => q.node.kind === 'group' && q.node.id === 'edge:1:label')
    if (label?.node.kind !== 'group') throw new Error('no label group')
    expect(label.node.children.map((c) => c.kind)).toEqual(['rect', 'text', 'text'])
    expect(label.node.children[0]).toMatchObject({ id: 'edge:1:plate' })
    expect(all(s).some((q) => q.node.kind === 'group' && q.node.id === 'edge:0:label')).toBe(false)
    const bare = all(diagram(t, { labelPlate: 'none' })).find(
      (q) => q.node.kind === 'group' && q.node.id === 'edge:1:label',
    )
    if (bare?.node.kind !== 'group') throw new Error('no label group')
    expect(bare.node.children.map((c) => c.kind)).toEqual(['text', 'text'])
  })

  it('puts a position at the same place whichever nodes are shown, when given an origin, and sizes the scene by it', () => {
    const full = structure('x', [at('a', 0, 0), at('b', 100, 50), at('c', 260, 120)], [{ from: 'b', to: 'c' }])
    const part = structure('x', full.nodes.slice(1), full.edges)
    // The origin is the top left of the drawing: left of and above every node's box.
    const [f, p] = [diagram(full, { origin: [-30, -20] }), diagram(part, { origin: [-30, -20] })]
    expect(boxOf(p, 'b')).toEqual(boxOf(f, 'b'))
    expect(boxOf(p, 'c')).toEqual(boxOf(f, 'c'))
    expect(p.width).toBeCloseTo(260 + 20 + 30 + 2 * 16, 0)
    // Without it, the subset moves to the margin.
    expect(boxOf(diagram(part), 'b').x0).toBeLessThan(boxOf(p, 'b').x0)
    // A node left of the origin still fits in the scene.
    const left = diagram(structure('x', [at('a', -100, 0), at('b', 100, 0)], []), { origin: [0, 0] })
    expect(boxOf(left, 'a').x0).toBeGreaterThanOrEqual(0)
    expect(boxOf(left, 'b').x1).toBeLessThanOrEqual(left.width)
  })

  it('rounds the corners of orthogonal edges with a corner radius, and leaves straight ones and the default alone', () => {
    const t = structure('x', [at('a', 0, 0), at('b', 200, 120)], [{ from: 'a', to: 'b' }])
    const find = (s: Scene): SceneNode | undefined =>
      all(s).find((q) => q.node.kind === 'path' && q.node.id === 'edge:0')?.node
    expect(find(diagram(t, { route: 'orthogonal' }))).not.toHaveProperty('cornerRadius')
    expect(find(diagram(t, { route: 'orthogonal', cornerRadius: 0 }))).not.toHaveProperty('cornerRadius')
    const rounded = diagram(t, { route: 'orthogonal', cornerRadius: 8 })
    expect(find(rounded)).toMatchObject({ cornerRadius: 8, smooth: false, arrow: true })
    // The route's points are the corners, not the curve.
    expect(rectilinear(edgePoints(rounded, 'edge:0'))).toBe(true)
    expect(find(diagram(t, { cornerRadius: 8 }))).not.toHaveProperty('cornerRadius')
  })
})

/** The collinear overlap of the edges of a drawn structure, split by how the edges relate, and the crossings. */
function measure(
  st: Structure,
  s: Scene,
  direction: 'down' | 'right' = 'down',
): { unrelated: number; sameLook: number; otherLook: number; longVsShort: number; crossings: number } {
  const paths = st.edges.map((_, i) => edgePoints(s, `edge:${i}`))
  const centre = (id: string): number => {
    const b = boxOf(s, id)
    return direction === 'down' ? (b.y0 + b.y1) / 2 : (b.x0 + b.x1) / 2
  }
  const levels = [...new Set(st.nodes.map((n) => Math.round(centre(n.id))))].sort((a, b) => a - b)
  const span = (i: number): number =>
    Math.abs(
      levels.indexOf(Math.round(centre((st.edges[i] as { from: string }).from))) -
        levels.indexOf(Math.round(centre((st.edges[i] as { to: string }).to))),
    )
  const look = (i: number): string => JSON.stringify([st.edges[i]?.look ?? null, st.edges[i]?.kind ?? null])
  const m = { unrelated: 0, sameLook: 0, otherLook: 0, longVsShort: 0, crossings: 0 }
  paths.forEach((p, i) => {
    paths.forEach((q, j) => {
      if (j <= i) return
      const [ei, ej] = [st.edges[i], st.edges[j]] as [(typeof st.edges)[number], (typeof st.edges)[number]]
      const related = ei.from === ej.from || ei.to === ej.to
      for (let u = 0; u + 1 < p.length; u++)
        for (let v = 0; v + 1 < q.length; v++) {
          const [a, b, c, d] = [p[u], p[u + 1], q[v], q[v + 1]] as [Pt, Pt, Pt, Pt]
          const [ah, ch] = [Math.abs(a[1] - b[1]) < 1e-6, Math.abs(c[1] - d[1]) < 1e-6]
          if (ah === ch) {
            const [across, along]: [0 | 1, 0 | 1] = ah ? [1, 0] : [0, 1]
            if (Math.abs(a[across] - c[across]) > 0.5) continue
            const len =
              Math.min(Math.max(a[along], b[along]), Math.max(c[along], d[along])) -
              Math.max(Math.min(a[along], b[along]), Math.min(c[along], d[along]))
            if (len <= 0.5) continue
            if (!related) m.unrelated += len
            else if (look(i) !== look(j)) m.otherLook += len
            else if (span(i) > 1 !== span(j) > 1) m.longVsShort += len
            else m.sameLook += len
          } else {
            const [h0, h1, v0, v1] = ah ? [a, b, c, d] : [c, d, a, b]
            const inH = v0[0] > Math.min(h0[0], h1[0]) + 0.5 && v0[0] < Math.max(h0[0], h1[0]) - 0.5
            const inV = h0[1] > Math.min(v0[1], v1[1]) + 0.5 && h0[1] < Math.max(v0[1], v1[1]) - 0.5
            if (inH && inV) m.crossings++
          }
        }
    })
  })
  return m
}

describe('diagram: edges kept apart, and the clearance of detours', () => {
  const stage = (id: string, label: string): ReturnType<typeof node> => node(id, { label })
  const pipeline = structure(
    'pipeline',
    [
      ['src', 'source'],
      ['lint', 'lint'],
      ['types', 'typecheck'],
      ['unit', 'unit tests'],
      ['build', 'build'],
      ['linux', 'package linux'],
      ['mac', 'package macOS'],
      ['win', 'package windows'],
      ['sign', 'sign'],
      ['docs', 'docs'],
      ['publish', 'publish'],
      ['notify', 'notify'],
    ].map(([id, label]) => stage(id as string, label as string)),
    [
      ['src', 'lint'],
      ['src', 'types'],
      ['src', 'unit'],
      ['src', 'build'],
      ['lint', 'build'],
      ['types', 'build'],
      ['build', 'linux'],
      ['build', 'mac'],
      ['build', 'win'],
      ['unit', 'sign'],
      ['linux', 'sign'],
      ['mac', 'sign'],
      ['win', 'sign'],
      ['src', 'docs'],
      ['docs', 'publish'],
      ['sign', 'publish'],
      ['sign', 'notify'],
      ['publish', 'notify'],
      ['win', 'build'],
    ].map(([from, to]) => ({
      from: from as string,
      to: to as string,
      ...(from === 'win' && to === 'build' ? { look: { stroke: 'tab3' as const, dash: [5, 4] } } : {}),
    })),
  )

  for (const direction of ['down', 'right'] as const)
    it(`keeps unrelated runs apart in a dense pipeline, direction ${direction}: a long edge never shares a run with a bundle of short ones`, () => {
      const m = measure(pipeline, diagram(pipeline, { route: 'orthogonal', direction }), direction)
      expect(m.unrelated).toBe(0)
      expect(m.otherLook).toBe(0)
      expect(m.longVsShort).toBe(0)
      // Before the edges were kept apart: unrelated 0, long edges merged into bundles for 286 px, 4 crossings.
      expect(m.crossings).toBeLessThanOrEqual(direction === 'down' ? 4 : 7)
    })

  it('merges the long edge into the bundle again with separate: false (the earlier routes)', () => {
    const old = { separate: false, clearance: 6, trackGap: 8 }
    const m = measure(pipeline, diagram(pipeline, { route: 'orthogonal', ...old }))
    expect(m.longVsShort).toBeGreaterThan(100)
    const on = measure(pipeline, diagram(pipeline, { route: 'orthogonal' }))
    expect(on.longVsShort).toBeLessThan(m.longVsShort)
  })

  it('leaves a source and a target their own ports for a long edge, side by side with the bundle', () => {
    const s = diagram(pipeline, { route: 'orthogonal' })
    const [long, near] = [edgePoints(s, 'edge:3'), edgePoints(s, 'edge:4')]
    // src to build skips a rank; lint to build does not: they enter build at different ports.
    expect(Math.abs((long.at(-1) as Pt)[0] - (near.at(-1) as Pt)[0])).toBeGreaterThan(4)
    const [a, b] = [edgePoints(s, 'edge:16'), edgePoints(s, 'edge:17')]
    // sign to notify (skips publish's rank) and publish to notify enter notify at different ports.
    expect(Math.abs((a.at(-1) as Pt)[0] - (b.at(-1) as Pt)[0])).toBeGreaterThan(4)
  })

  const row = structure(
    'pipeline',
    ['a', 'b', 'c', 'd'].map((id, i) => node(id, { position: [i * 120, 0] })),
    [
      { from: 'a', to: 'b' },
      { from: 'b', to: 'c' },
      { from: 'c', to: 'd' },
      { from: 'a', to: 'd', look: { stroke: 'tab3' as const, dash: [5, 4] } },
    ],
  )
  const gapAbove = (s: Scene): number => {
    const detour = edgePoints(s, 'edge:3')
    const run = Math.min(...detour.map((p) => p[1]))
    return boxOf(s, 'b').y0 - run
  }

  it('runs a detour over a row of nodes `clearance` above them, 18 px by default', () => {
    // The route keeps at least the clearance (the stub off the port adds a pixel).
    for (const [options, gap] of [
      [{}, 18],
      [{ clearance: 30 }, 30],
      [{ clearance: 6 }, 6],
    ] as const) {
      const got = gapAbove(diagram(row, { route: 'orthogonal', ...options }))
      expect(got).toBeGreaterThanOrEqual(gap - 1e-6)
      expect(got).toBeLessThan(gap + 1.5)
    }
  })

  it('grows the scene to hold the detour, and keeps every edge inside it', () => {
    for (const clearance of [6, 18, 40]) {
      const s = diagram(row, { route: 'orthogonal', clearance })
      for (let i = 0; i < 4; i++)
        for (const [x, y] of edgePoints(s, `edge:${i}`)) {
          expect(x).toBeGreaterThanOrEqual(0)
          expect(y).toBeGreaterThanOrEqual(0)
          expect(x).toBeLessThanOrEqual(s.width)
          expect(y).toBeLessThanOrEqual(s.height)
        }
    }
    expect(diagram(row, { route: 'orthogonal', clearance: 40 }).height).toBeGreaterThan(
      diagram(row, { route: 'orthogonal' }).height,
    )
  })

  it('leaves straight diagrams as they were, whatever the options of the orthogonal router', () => {
    const plain = diagram(pipeline, {})
    expect(diagram(pipeline, { clearance: 40, trackGap: 20, separate: false })).toEqual(plain)
    expect(diagram(pipeline, { route: 'straight', clearance: 40 })).toEqual(plain)
  })

  it('puts parallel tracks a track gap apart', () => {
    // The fan-out of build and the back edge into it share a channel: their runs lie a gap apart.
    const runs = (trackGap: number): number[] => {
      const s = diagram(pipeline, { route: 'orthogonal', trackGap })
      const ys = [6, 7, 8, 18].flatMap((i) => edgePoints(s, `edge:${i}`).map((p) => p[1]))
      const bus = ys.filter((y) => y > boxOf(s, 'build').y1 && y < boxOf(s, 'linux').y0)
      return [...new Set(bus.map((y) => Math.round(y * 10) / 10))].sort((a, b) => a - b)
    }
    const [tight, wide] = [runs(8), runs(20)]
    const gapOf = (ys: number[]): number => Math.min(...ys.slice(1).map((y, i) => y - (ys[i] as number)))
    expect(tight.length).toBeGreaterThan(1)
    expect(gapOf(wide)).toBeGreaterThan(gapOf(tight))
  })
})
