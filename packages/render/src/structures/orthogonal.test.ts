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
