import { describe, expect, it } from 'vitest'
import { type SceneNode, structure } from 'whitebox-engine/scene'
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
