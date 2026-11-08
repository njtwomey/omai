import { type SceneNode, structure } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { diagram } from './diagram.ts'
import { styleFor } from './styles.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))

const model = structure(
  'bayes-net',
  [
    { id: 'z', label: 'z', role: 'latent', group: 'N' },
    { id: 'x', label: 'x', role: 'observed', group: 'N' },
    { id: 'mu', label: 'mu', role: 'latent' },
  ],
  [
    { from: 'z', to: 'x' },
    { from: 'mu', to: 'x' },
  ],
  { groups: [{ id: 'N', label: 'points', kind: 'plate', count: 'N' }] },
)

describe('structure styles', () => {
  it('draws roles in each kind’s language', () => {
    expect(styleFor('bayes-net').node('observed').fill).toBe('grid')
    expect(styleFor('bayes-net').node('latent').fill).toBe('bg')
    expect(styleFor('factor-graph').edge({ from: 'a', to: 'b' }).arrow).toBe(false)
    expect(styleFor('tree').layout).toBe('tree')
    expect(styleFor('anything else').node('input').stroke).toBe('tab0')
  })
})

describe('diagram', () => {
  it('draws nodes as groups by id, edges with arrows, and plates with their count', () => {
    const s = diagram(model)
    const all = flatten(s.children)
    expect(all.filter((n) => n.kind === 'group' && ['z', 'x', 'mu'].includes(n.id ?? ''))).toHaveLength(3)
    expect(all.some((n) => n.id === 'group:N')).toBe(true)
    expect(all.some((n) => n.kind === 'text' && n.text === 'N')).toBe(true)
    const edges = all.filter((n) => n.kind === 'path' && n.id?.startsWith('edge:'))
    expect(edges).toHaveLength(2)
    expect(edges.every((e) => e.kind === 'path' && e.arrow === true)).toBe(true)
    expect(s.width).toBeGreaterThan(0)
  })

  it('marks overlay state: active nodes and edges in the accent, badges as text', () => {
    const s = diagram(model, {
      overlay: { active: ['x'], edges: [{ from: 'z', to: 'x' }], badges: { x: { p: 0.25 } } },
    })
    const all = flatten(s.children)
    const x = all.find((n) => n.kind === 'group' && n.id === 'x')
    expect(x?.kind === 'group' && x.children.some((c) => 'stroke' in c && c.stroke === 'accent')).toBe(true)
    expect(all.some((n) => n.kind === 'path' && n.id === 'edge:0' && n.stroke === 'accent')).toBe(true)
    expect(all.some((n) => n.kind === 'text' && n.text === 'p 0.25')).toBe(true)
  })

  it('bends an edge whose twin runs the other way', () => {
    const loop = structure(
      'automaton',
      [
        { id: 'a', label: 'a', role: 'state' },
        { id: 'b', label: 'b', role: 'state' },
      ],
      [
        { from: 'a', to: 'b', label: 'x' },
        { from: 'b', to: 'a', label: 'y' },
      ],
    )
    const paths = flatten(diagram(loop).children).filter((n) => n.kind === 'path')
    expect(paths.every((p) => p.kind === 'path' && p.points.length === 3)).toBe(true)
  })
})

describe('architectures', () => {
  it('draws a network four ways, counting parameters once', async () => {
    const { describe: describeNet, mlp, TransformerEncoderLayer } = await import('omai-engine/nn')
    const { architecture, flowOrder } = await import('./architectures.ts')
    const s = describeNet(mlp([2, 4, 1]), [8, 2])
    expect(flowOrder(s).map((n) => n.label)).toEqual(['input', 'Linear', 'ReLU', 'Linear', 'output'])
    for (const v of ['diagram', 'compact', 'shapes', 'params'] as const) expect(architecture(s, v).kind).toBe('scene')
    const total = (sc: ReturnType<typeof architecture>): string =>
      flatten(sc.children).find((n) => n.kind === 'text' && n.text.endsWith('parameters'))?.kind === 'text'
        ? (flatten(sc.children).find((n) => n.kind === 'text' && n.text.endsWith('parameters')) as { text: string })
            .text
        : ''
    expect(total(architecture(s, 'params'))).toBe('17 parameters')
    const t = describeNet(new TransformerEncoderLayer(8, 2, { batchFirst: true }), [1, 3, 8], { expandParts: true })
    const tParams = t.nodes.find((n) => n.data?.['parts'] !== undefined)?.badges?.['params']
    expect(total(architecture(t, 'params'))).toBe(`${(Number(tParams) / 1e3).toPrecision(3)}k parameters`)
  })
})

describe('traversals', () => {
  it('orders searches and numbers visits frame by frame', async () => {
    const { searchOrder, traversalFrames } = await import('./overlays.ts')
    const s = structure(
      'tree',
      ['r', 'a', 'b', 'c'].map((id) => ({ id, label: id, role: 'internal' as const })),
      [
        { from: 'r', to: 'a' },
        { from: 'r', to: 'b' },
        { from: 'a', to: 'c' },
      ],
    )
    expect(searchOrder(s, 'r', 'bfs')).toEqual(['r', 'a', 'b', 'c'])
    expect(searchOrder(s, 'r', 'dfs')).toEqual(['r', 'a', 'c', 'b'])
    const frames = traversalFrames(s, ['r', 'a', 'c'])
    expect(frames[2]).toEqual({
      active: ['c'],
      edges: [{ from: 'a', to: 'c' }],
      badges: { r: { order: 1 }, a: { order: 2 }, c: { order: 3 } },
    })
  })
})

describe('diagram self-loops and stable sizes', () => {
  const dfa = structure(
    'automaton',
    ['p', 'q'].map((id) => ({ id, label: id, role: 'state' as const })),
    [
      { from: 'p', to: 'p', label: 'a' },
      { from: 'p', to: 'q', label: 'b' },
      { from: 'q', to: 'q', label: 'a, b' },
    ],
    { direction: 'right' },
  )
  const nodeBox = (all: SceneNode[], id: string): { x: number; y: number } => {
    const g = all.find((n) => n.kind === 'group' && n.id === id)
    const c = g?.kind === 'group' ? g.children.find((n) => n.kind === 'circle' || n.kind === 'rect') : undefined
    if (c?.kind === 'circle') return { x: c.cx, y: c.cy }
    if (c?.kind === 'rect') return { x: c.x, y: c.y }
    throw new Error(`no node ${id}`)
  }

  it('draws a loop per self-edge, matched to its own edge, with its label, inside the scene', () => {
    const s = diagram(dfa)
    const all = flatten(s.children)
    const loop = all.find((n) => n.kind === 'path' && n.id === 'edge:0')
    const straight = all.find((n) => n.kind === 'path' && n.id === 'edge:1')
    expect(loop?.kind === 'path' && loop.points.length).toBe(5)
    expect(straight?.kind === 'path' && straight.points.length).toBe(2)
    expect(all.some((n) => n.kind === 'text' && n.text === 'a, b')).toBe(true)
    // The loop leaves and re-enters the node: both ends lie on the circle around p.
    const p = nodeBox(all, 'p')
    if (loop?.kind === 'path') {
      const [first, last] = [loop.points[0], loop.points[4]]
      expect(Math.hypot((first?.[0] ?? 0) - p.x, (first?.[1] ?? 0) - p.y)).toBeCloseTo(
        Math.hypot((last?.[0] ?? 0) - p.x, (last?.[1] ?? 0) - p.y),
        6,
      )
    }
    // Everything sits inside the scene (the loops rise above the layout's own box).
    const translate = s.children[0]?.kind === 'group' ? (s.children[0].translate ?? [0, 0]) : [0, 0]
    for (const n of all)
      if (n.kind === 'path')
        for (const [x, y] of n.points) {
          expect(x + (translate[0] ?? 0)).toBeGreaterThanOrEqual(0)
          expect(y + (translate[1] ?? 0)).toBeGreaterThanOrEqual(0)
          expect(x + (translate[0] ?? 0)).toBeLessThanOrEqual(s.width)
        }
  })

  it('keeps every node in place across an animation when given its frames', () => {
    const frames = [{}, { badges: { p: { value: 1.2345 } } }, { badges: { p: { value: 1.2345 }, q: { value: 42 } } }]
    const at = (overlay: (typeof frames)[number]): string =>
      JSON.stringify(['p', 'q'].map((id) => nodeBox(flatten(diagram(dfa, { overlay, frames }).children), id)))
    expect(at(frames[0] ?? {})).toBe(at(frames[2] ?? {}))
    // Without the frames, badges grow the nodes and move them.
    const loose = (overlay: (typeof frames)[number]): string =>
      JSON.stringify(['p', 'q'].map((id) => nodeBox(flatten(diagram(dfa, { overlay }).children), id)))
    expect(loose(frames[0] ?? {})).not.toBe(loose(frames[2] ?? {}))
  })
})

describe('diagram: manual layout, looks, multi-line labels, bidirectional edges, labels and bounds', () => {
  const box = (all: SceneNode[], id: string): { x: number; y: number; w: number; h: number } => {
    const g = all.find((n) => n.kind === 'group' && n.id === id)
    const r = g?.kind === 'group' ? g.children.find((n) => n.kind === 'rect') : undefined
    if (r?.kind !== 'rect') throw new Error(`no rect for ${id}`)
    return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height }
  }
  const node = (id: string, extra: object = {}): { id: string; label: string; role: 'other' } => ({
    id,
    label: id,
    role: 'other',
    ...extra,
  })
  const texts = (all: SceneNode[]): string[] => all.flatMap((n) => (n.kind === 'text' ? [n.text] : []))

  it('places nodes at their positions: with layout manual, and by default when all have one', () => {
    const s = structure(
      'graph',
      [node('hub', { position: [0, 0] }), node('a', { position: [100, 0] }), node('b', { position: [0, 100] })],
      [
        { from: 'hub', to: 'a' },
        { from: 'hub', to: 'b' },
      ],
    )
    for (const options of [{}, { layout: 'manual' as const }]) {
      const all = flatten(diagram(s, { ...options, style: styleFor('general') }).children)
      expect(box(all, 'a').x - box(all, 'hub').x).toBeCloseTo(100, 9)
      expect(box(all, 'b').y - box(all, 'hub').y).toBeCloseTo(100, 9)
    }
    // An explicit layout wins over positions.
    const layered = flatten(diagram(s, { layout: 'layered', style: styleFor('general') }).children)
    expect(box(layered, 'a').x - box(layered, 'hub').x).not.toBeCloseTo(100, 3)
  })

  it('throws OmaiError value for a node without a position under layout manual', () => {
    const s = structure('graph', [node('a', { position: [0, 0] }), node('b')], [])
    expect(() => diagram(s, { layout: 'manual' })).toThrow(/'b' has no position/)
    try {
      diagram(s, { layout: 'manual' })
    } catch (e) {
      expect((e as { code?: string }).code).toBe('value')
    }
  })

  it('draws a multi-line label as lines, and sizes the node for them', () => {
    const one = flatten(diagram(structure('x', [node('a', { label: 'first' })], [])).children)
    const two = flatten(diagram(structure('x', [node('a', { label: 'first\nsecond' })], [])).children)
    expect(texts(two)).toEqual(['first', 'second'])
    expect(box(two, 'a').h).toBeGreaterThan(box(one, 'a').h)
    const sizes = two.flatMap((n) => (n.kind === 'text' ? [n.size, n.fill] : []))
    expect(sizes).toEqual([12, 'fg', 12, 'fg'])
  })

  it('applies a node look and an edge look over the style', () => {
    const s = structure(
      'x',
      [node('a', { look: { stroke: 'tab3', shape: 'circle' } }), node('b')],
      [{ from: 'a', to: 'b', look: { stroke: 'tab5', strokeWidth: 3, dash: [1, 1], arrow: false } }],
    )
    const all = flatten(diagram(s).children)
    expect(all.some((n) => n.kind === 'circle' && n.stroke === 'tab3')).toBe(true)
    const e = all.find((n) => n.kind === 'path' && n.id === 'edge:0')
    expect(e).toMatchObject({ stroke: 'tab5', strokeWidth: 3, dash: [1, 1], arrow: false })
  })

  it('draws shape none as text over an invisible box sized by the text, and stack as three boxes', () => {
    const s = structure(
      'x',
      [node('t', { label: 'caption', look: { shape: 'none' } }), node('k', { look: { shape: 'stack' } })],
      [],
    )
    const all = flatten(diagram(s).children)
    const t = all.find((n) => n.kind === 'group' && n.id === 't')
    const hit = t?.kind === 'group' ? t.children.find((n) => n.kind === 'rect') : undefined
    expect(hit).toMatchObject({ opacity: 0, stroke: 'none' })
    expect(box(all, 't').w).toBeCloseTo('caption'.length * 12 * 0.6 + 8, 9)
    const k = all.find((n) => n.kind === 'group' && n.id === 'k')
    expect(k?.kind === 'group' && k.children.filter((n) => n.kind === 'rect')).toHaveLength(3)
  })

  it('draws a bidirectional edge once with a head at both ends, without moving the ranks', () => {
    const base = structure(
      'x',
      ['a', 'b', 'c'].map((id) => node(id)),
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
      ],
    )
    const both = structure(
      'x',
      ['a', 'b', 'c'].map((id) => node(id)),
      [
        { from: 'a', to: 'b', bidirectional: true },
        { from: 'b', to: 'c' },
        { from: 'b', to: 'a' },
      ],
    )
    const all = flatten(diagram(both).children)
    expect(all.filter((n) => n.kind === 'path' && n.id === 'edge:2')).toHaveLength(0)
    const heads = all.filter((n) => n.kind === 'path' && n.arrow === true && n.id?.startsWith('edge:0'))
    expect(heads.map((n) => n.id)).toEqual(['edge:0', 'edge:0:tail'])
    const plain = flatten(diagram(base).children)
    expect(['a', 'b', 'c'].map((id) => box(all, id))).toEqual(['a', 'b', 'c'].map((id) => box(plain, id)))
    // The tail arrow lies on the start of the edge and points back at node a.
    const edge = all.find((n) => n.kind === 'path' && n.id === 'edge:0')
    const tail = all.find((n) => n.kind === 'path' && n.id === 'edge:0:tail')
    if (edge?.kind !== 'path' || tail?.kind !== 'path') throw new Error('edge paths')
    expect(tail.points[1]).toEqual(edge.points[0])
  })

  it('separates twin edges routed through other ranks', () => {
    const twins = structure(
      'x',
      ['a', 'b', 'c'].map((id) => node(id)),
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
        { from: 'a', to: 'c' },
        { from: 'c', to: 'a' },
      ],
    )
    const paths = flatten(diagram(twins).children).filter((n) => n.kind === 'path')
    const [f, r] = [paths[2], paths[3]]
    if (f?.kind !== 'path' || r?.kind !== 'path') throw new Error('edges')
    expect(f.points.length).toBeGreaterThan(2)
    expect(f.points[1]).not.toEqual(r.points[1])
  })

  it('draws edge labels above the nodes, on a plate, at labelAt when given, and off the nodes', () => {
    const s = structure(
      'x',
      ['a', 'b'].map((id) => node(id)),
      [{ from: 'a', to: 'b', label: 'rank', labelAt: 0.5 }],
      {
        direction: 'right',
      },
    )
    const sc = diagram(s)
    const top = flatten(sc.children)
    const labelAt = top.findIndex((n) => n.kind === 'text' && n.text === 'rank')
    const lastNode = top.findIndex((n) => n.kind === 'group' && n.id === 'b')
    expect(labelAt).toBeGreaterThan(lastNode)
    expect(top[labelAt - 1]).toMatchObject({ kind: 'rect', fill: 'bg' })
    const lab = top[labelAt]
    const [a, b] = [box(top, 'a'), box(top, 'b')]
    expect(lab?.kind === 'text' && lab.x).toBeCloseTo((a.x + a.w / 2 + (b.x - b.w / 2)) / 2, 6)
    // A long label whose default place is on a node is moved off every node.
    const long = structure(
      'x',
      ['a', 'b', 'c'].map((id) => node(id)),
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c', label: 'a very very long edge label indeed' },
      ],
      { direction: 'right' },
    )
    const all = flatten(diagram(long).children)
    const t = all.find((n) => n.kind === 'text' && n.text.startsWith('a very'))
    if (t?.kind !== 'text') throw new Error('label')
    const w = t.text.length * 10 * 0.6
    for (const id of ['a', 'b', 'c']) {
      const n = box(all, id)
      const apart = t.x + w < n.x - n.w / 2 || t.x > n.x + n.w / 2 || t.y + 5 < n.y - n.h / 2 || t.y - 5 > n.y + n.h / 2
      expect(apart, `label clear of ${id}`).toBe(true)
    }
  })

  it('widens a group to its label, and the scene to outside labels and notes', () => {
    const s = structure('bayes-net', [node('a', { group: 'g' })], [], {
      groups: [{ id: 'g', label: 'a group label that is much longer than its member', kind: 'plate' }],
    })
    const sc = diagram(s)
    const g = flatten(sc.children).find((n) => n.kind === 'rect' && n.id === 'group:g')
    expect(g?.kind === 'rect' && g.width).toBeGreaterThanOrEqual(
      'a group label that is much longer than its member'.length * 12 * 0.6 + 16,
    )
    const fg = structure('factor-graph', [{ id: 'f', label: 'a wide factor caption', role: 'factor' }], [])
    const out = diagram(fg)
    expect(out.width).toBeGreaterThanOrEqual('a wide factor caption'.length * 12 * 0.6)
    expect(out.height).toBeGreaterThan(14 + 13)
    const noted = structure('x', [node('a')], [], { annotations: [{ target: 'a', text: 'a note to the right of a' }] })
    expect(diagram(noted).width).toBeGreaterThan(diagram(structure('x', [node('a')], [])).width + 100)
  })

  it('spaces trees by rankSep and nodeSep', () => {
    const t = structure(
      'tree',
      ['r', 'a', 'b'].map((id) => node(id)),
      [
        { from: 'r', to: 'a' },
        { from: 'r', to: 'b' },
      ],
    )
    const tight = diagram(t)
    const loose = diagram(t, { rankSep: 120, nodeSep: 80 })
    expect(loose.height).toBeGreaterThan(tight.height + 50)
    expect(loose.width).toBeGreaterThan(tight.width + 50)
  })
})
