import { type SceneNode, structure } from 'mdml-engine/scene'
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
    const { describe: describeNet, mlp, TransformerEncoderLayer } = await import('mdml-engine/nn')
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
