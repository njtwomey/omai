import { defineModel, toStructure } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { type SceneNode, structure } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { diagram } from './diagram.ts'
import { isModelStructure, modelGraph } from './model-graph.ts'
import { styleFor } from './styles.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))
const rects = (s: ReturnType<typeof diagram>): Map<string, { x: number; y: number; width: number; height: number }> =>
  new Map(
    flatten(s.children).flatMap((n) =>
      n.kind === 'rect' && n.id?.startsWith('group:') === true ? [[n.id, n] as const] : [],
    ),
  )

const irt = defineModel('irt', (b) => {
  const s = b.range('student', { size: 40 })
  const e = b.range('exam', { size: 5 })
  const mu = b.latent('mu', d.normal(0, 1))
  const skill = b.latent('skill', d.normal(mu, 1), { dims: [s] })
  const difficulty = b.latent('difficulty', d.normal(0, 1), { dims: [e] })
  const y = b.data('y', { dims: [s, e], role: 'response' })
  b.observe('y', d.normal(skill.sub(difficulty), 1), y)
})

const reg = defineModel('reg', (b) => {
  const n = b.range('row', { size: 40 })
  const k = b.range('feature', { size: 3 })
  const x = b.data('x', { dims: [n, k], role: 'covariate' })
  const w = b.latent('w', d.normal(0, 2), { dims: [k] })
  const y = b.data('y', { dims: [n], role: 'response' })
  b.observe('y', d.normal(x.matmul(w), 1), y)
})

describe('modelGraph', () => {
  it('leaves a structure that is not a spec alone, and does not prepare twice', () => {
    const plain = structure('bayes-net', [{ id: 'a', label: 'a', role: 'latent' }], [])
    expect(isModelStructure(plain)).toBe(false)
    expect(modelGraph(plain)).toBe(plain)
    const once = modelGraph(toStructure(irt))
    expect(isModelStructure(toStructure(irt))).toBe(true)
    expect(modelGraph(once)).toBe(once)
  })

  it('writes the size of a plate as N = 40, and only the symbol when it is not known', () => {
    const s = modelGraph(toStructure(irt))
    expect(s.type).toBe('model-graph')
    expect(s.groups.filter((g) => g.kind === 'plate').map((g) => [g.id, g.count])).toEqual(
      expect.arrayContaining([
        ['plate:student', 'S = 40'],
        ['plate:exam', 'E = 5'],
      ]),
    )
    const unsized = defineModel('u', (b) => {
      b.range('row')
      b.latent('x', d.normal(0, 1), { dims: ['row'] })
    })
    expect(modelGraph(toStructure(unsized)).groups[0]?.count).toBe('R')
    expect(
      modelGraph(toStructure(irt), { symbols: { student: 'N' } }).groups.find((g) => g.id === 'plate:student')?.count,
    ).toBe('N = 40')
  })

  it('puts a node over two plates in both when the plates would otherwise draw one dimension twice', () => {
    const s = modelGraph(toStructure(irt))
    const y = s.nodes.find((n) => n.id === 'y')
    expect(y?.groups).toEqual(['plate:student', 'plate:exam'])
    expect(s.groups.find((g) => g.id === 'plate:student×exam')?.intersection).toEqual(['plate:student', 'plate:exam'])
    // Nested dimensions keep the nest: the row plate holds the row-by-feature one and nothing draws feature twice
    // beyond what the nest says.
    const r = modelGraph(toStructure(reg))
    expect(r.nodes.some((n) => n.groups !== undefined)).toBe(true)
  })

  it('draws data as dots, generated quantities dashed, and drops the dims badge the plates already say', () => {
    const s = modelGraph(toStructure(reg))
    const x = s.nodes.find((n) => n.id === 'data:x')
    expect(x?.look?.shape).toBe('dot')
    expect(x?.badges).toBeUndefined()
    expect(x?.detail).toBe('covariate')
    expect(modelGraph(toStructure(reg), { data: 'node' }).nodes.find((n) => n.id === 'data:x')?.look).toBeUndefined()
  })

  it('unrolls vector sites into elements, with the edges of the site to each', () => {
    const base = toStructure(reg)
    const s = modelGraph(base, { expand: { w: 3 } })
    expect(s.nodes.filter((n) => n.id.startsWith('w[')).map((n) => n.id)).toEqual(['w[0]', 'w[1]', 'w[2]'])
    expect(s.nodes.some((n) => n.id === 'w')).toBe(false)
    expect(s.edges.filter((e) => e.from.startsWith('w[')).length).toBe(3)
    const auto = modelGraph(base, { expand: 'auto' })
    expect(auto.nodes.filter((n) => n.id.startsWith('w[')).length).toBe(3)
    // Nothing is unrolled past the cap.
    expect(modelGraph(base, { expand: 'auto', maxElements: 2 }).nodes.some((n) => n.id === 'w')).toBe(true)
  })

  it('has its own visual language: solid arrows, and a factor view in ranks', () => {
    expect(
      styleFor('model-graph').edge({ from: 'a', to: 'b', kind: 'dependency', directed: true }).dash,
    ).toBeUndefined()
    expect(styleFor('model-graph').node('parameter').shape).toBe('dot')
    expect(styleFor('model-factor-graph').layout).toBe('layered')
    expect(modelGraph(toStructure(irt, { view: 'factor' })).type).toBe('model-factor-graph')
  })
})

describe('diagram of crossing plates', () => {
  it('draws the two plates crossing, each around its own members, with no frame for their overlap', () => {
    const r = rects(diagram(toStructure(irt)))
    expect(r.has('group:plate:student×exam')).toBe(false)
    const a = r.get('group:plate:student')
    const b = r.get('group:plate:exam')
    expect(a).toBeDefined()
    expect(b).toBeDefined()
    if (a === undefined || b === undefined) return
    const overlap = a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
    expect(overlap).toBe(true)
    // Neither contains the other: they cross.
    expect(a.x + a.width < b.x + b.width || b.x + b.width < a.x + a.width).toBe(true)
  })

  it('draws the matrix-factorisation plates (three pairwise crossing) with every node inside only its own', () => {
    const mf = defineModel('mf', (b) => {
      const u = b.range('user', { size: 20 })
      const i = b.range('item', { size: 30 })
      const k = b.range('k', { size: 4 })
      const U = b.latent('U', d.normal(0, 1), { dims: [u, k] })
      const V = b.latent('V', d.normal(0, 1), { dims: [i, k] })
      const r = b.data('r', { dims: [u, i], role: 'response' })
      b.observe('r', d.normal(U.sum('k').add(V.sum('k')), 1), r)
    })
    const r = rects(diagram(toStructure(mf)))
    expect([...r.keys()].sort()).toEqual(['group:plate:item', 'group:plate:k', 'group:plate:user'])
  })

  it('falls back to the nested chain when a frame would take in a node that is not its member', () => {
    const at = (id: string, group: string | undefined, groups: string[] | undefined, position: [number, number]) => ({
      id,
      label: id,
      role: 'latent' as const,
      position,
      ...(group === undefined ? {} : { group }),
      ...(groups === undefined ? {} : { groups }),
    })
    const s = structure(
      'graph',
      [
        at('a', 'A', undefined, [0, 0]),
        at('c', 'A×B', ['A', 'B'], [200, 0]),
        at('b', 'B', undefined, [400, 0]),
        at('stranger', undefined, undefined, [100, 0]),
      ],
      [],
      {
        groups: [
          { id: 'A', label: 'A', kind: 'plate' },
          { id: 'B', label: 'B', kind: 'plate' },
          { id: 'A×B', label: 'B', kind: 'plate', parent: 'A', intersection: ['A', 'B'] },
        ],
      },
    )
    expect(rects(diagram(s)).has('group:A×B')).toBe(true)
    // Without the stranger between them the frames cross.
    const calm = { ...s, nodes: s.nodes.filter((n) => n.id !== 'stranger') }
    expect(rects(diagram(calm)).has('group:A×B')).toBe(false)
  })
})

describe('what diagram adds for a run laid over a model', () => {
  const pair = structure(
    'model-graph',
    [
      { id: 'a', label: 'a', role: 'latent' },
      { id: 'b', label: 'b', role: 'observed' },
    ],
    [{ from: 'a', to: 'b' }],
  )
  const labels = (s: ReturnType<typeof diagram>): string[] =>
    flatten(s.children).flatMap((n) => (n.kind === 'text' ? [n.text] : []))

  it('writes the label of an active edge in the overlay on the edge, and leaves the edges that have their own', () => {
    expect(
      labels(diagram(pair, { overlay: { edges: [{ from: 'a', to: 'b', label: 'N(μ = 1, σ² = 2)' }] } })),
    ).toContain('N(μ = 1, σ² = 2)')
    const own = { ...pair, edges: [{ from: 'a', to: 'b', label: 'own' }] }
    const said = labels(diagram(own, { overlay: { edges: [{ from: 'a', to: 'b', label: 'frame' }] } }))
    expect(said).toContain('own')
    expect(said).not.toContain('frame')
  })

  it('draws what decorate returns inside the node group, with the node and its box', () => {
    const seen: string[] = []
    const s = diagram(pair, {
      decorate: (n, box) => {
        seen.push(`${n.id}:${box.width > 0}`)
        return n.id === 'a' ? [{ kind: 'circle', cx: box.x, cy: box.y, r: 1, id: 'glyph' } as SceneNode] : undefined
      },
    })
    expect(seen.sort()).toEqual(['a:true', 'b:true'])
    const a = flatten(s.children).find((n) => n.kind === 'group' && n.id === 'a')
    expect(a?.kind === 'group' && a.children.some((c) => c.id === 'glyph')).toBe(true)
    const b = flatten(s.children).find((n) => n.kind === 'group' && n.id === 'b')
    expect(b?.kind === 'group' && b.children.some((c) => c.id === 'glyph')).toBe(false)
  })
})
