import { bayesNet, plateModel, toStructure } from 'omai-engine/pgm'
import { type SceneNode, structure } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { diagram } from './diagram.ts'
import { bayesNetCards, nestPlates, uNet, uNetOrder, unetLevels } from './models.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))
const ids = (nodes: readonly SceneNode[]): string[] => flatten(nodes).flatMap((n) => (n.id === undefined ? [] : [n.id]))

describe('uNet', () => {
  const levels = unetLevels(64, [16, 32, 64, 128])

  it('halves the size per level', () => {
    expect(levels.map((l) => l.size)).toEqual([64, 32, 16, 8])
  })

  it('draws encoder, bottleneck and decoder blocks, with skips only when asked', () => {
    const s = uNet(levels)
    const all = ids(s.children)
    for (const id of ['enc:0', 'enc:2', 'mid', 'dec:0', 'dec:2', 'skip:1', 'down:2', 'up:0']) expect(all).toContain(id)
    expect(all).not.toContain('enc:3')
    expect(ids(uNet(levels, { skips: false }).children).some((id) => id.startsWith('skip:'))).toBe(false)
    expect(ids(uNet(levels, { timeEmbedding: true }).children)).toContain('time')
  })

  it('keeps the U symmetric and inside the scene, the deeper blocks shorter', () => {
    const s = uNet(levels)
    const blocks = flatten(s.children).filter((n) => n.kind === 'group')
    const box = (id: string): { x: number; w: number; h: number } => {
      const g = blocks.find((b) => b.id === id)
      const rects = g?.kind === 'group' ? g.children.filter((c) => c.kind === 'rect') : []
      const xs = rects.flatMap((r) => (r.kind === 'rect' ? [r.x, r.x + r.width] : []))
      const h = rects[0]?.kind === 'rect' ? rects[0].height : 0
      return { x: Math.min(...xs), w: Math.max(...xs) - Math.min(...xs), h }
    }
    const enc = box('enc:1')
    const dec = box('dec:1')
    expect(enc.x - 0).toBeCloseTo(s.width - (dec.x + dec.w), 6)
    expect(box('enc:0').h).toBeGreaterThan(enc.h)
    for (const n of flatten(s.children))
      if (n.kind === 'rect') {
        expect(n.x).toBeGreaterThanOrEqual(0)
        expect(n.x + n.width).toBeLessThanOrEqual(s.width)
        expect(n.y + n.height).toBeLessThanOrEqual(s.height)
      }
  })

  it('orders a forward pass and highlights the active blocks', () => {
    expect(uNetOrder(levels)).toEqual([
      'enc:0',
      'enc:1',
      'enc:2',
      'mid',
      'skip:2',
      'dec:2',
      'skip:1',
      'dec:1',
      'skip:0',
      'dec:0',
      'out',
    ])
    expect(uNetOrder(levels, { skips: false })).not.toContain('skip:0')
    const s = uNet(levels, { active: ['mid'] })
    const mid = flatten(s.children).find((n) => n.kind === 'group' && n.id === 'mid')
    expect(mid?.kind === 'group' && mid.children.every((c) => 'stroke' in c && c.stroke === 'accent')).toBe(true)
    expect(() => uNet(unetLevels(8, [4]))).toThrow(/two levels/)
  })
})

describe('bayesNetCards', () => {
  const bn = bayesNet([
    { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
    { name: 'sprinkler', states: ['off', 'on'], parents: ['rain'], cpt: [0.6, 0.4, 0.99, 0.01] },
    {
      name: 'wet',
      states: ['no', 'yes'],
      parents: ['rain', 'sprinkler'],
      cpt: [1, 0, 0.1, 0.9, 0.2, 0.8, 0.01, 0.99],
    },
  ])

  it('draws a card per variable with its CPT, rows per parent assignment', () => {
    const s = bayesNetCards(bn)
    const wet = s.children.find((n) => n.kind === 'group' && n.id === 'wet')
    const texts = wet?.kind === 'group' ? wet.children.flatMap((c) => (c.kind === 'text' ? [c.text] : [])) : []
    expect(texts).toContain('rain=yes, sprinkler=on')
    expect(texts.filter((t) => t === '0.99')).toHaveLength(1)
    expect(s.children.filter((n) => n.kind === 'path')).toHaveLength(3)
  })

  it('highlights the row the evidence selects and shows marginals', () => {
    const s = bayesNetCards(bn, { evidence: { rain: 'yes', sprinkler: 'off' }, marginals: { wet: [0.3, 0.7] } })
    const wet = s.children.find((n) => n.kind === 'group' && n.id === 'wet')
    const parts = wet?.kind === 'group' ? wet.children : []
    expect(parts.filter((c) => c.kind === 'rect' && c.fill === 'accent' && c.opacity === 0.14)).toHaveLength(1)
    expect(parts.some((c) => c.kind === 'text' && c.text === '0.70')).toBe(true)
    const rain = s.children.find((n) => n.kind === 'group' && n.id === 'rain')
    expect(rain?.kind === 'group' && rain.children.some((c) => c.kind === 'text' && c.text === '= yes')).toBe(true)
  })
})

describe('nestPlates and the bipartite layout', () => {
  it('nests the plates of a plate model inside one another', () => {
    const m = plateModel([
      { name: 'skill', states: ['lo', 'hi'], plates: ['student'], parents: [], cpt: [0.5, 0.5] },
      {
        name: 'pass',
        states: ['no', 'yes'],
        plates: ['student', 'exam'],
        parents: ['skill'],
        cpt: [0.6, 0.4, 0.1, 0.9],
      },
    ])
    const s = nestPlates(toStructure(m, ['pass']), { exam: 'E' })
    expect(s.groups.map((g) => [g.id, g.label, g.count, g.parent])).toEqual([
      ['plate:student', 'student', 'STUDENT', undefined],
      ['plate:student×exam', 'exam', 'E', 'plate:student'],
    ])
    // Drawn, the inner plate sits inside the outer one.
    const all = flatten(diagram(s).children)
    const box = (id: string): [number, number, number, number] => {
      const r = all.find((n) => n.id === id)
      return r?.kind === 'rect' ? [r.x, r.y, r.x + r.width, r.y + r.height] : [0, 0, 0, 0]
    }
    const [ox0, oy0, ox1, oy1] = box('group:plate:student')
    const [ix0, iy0, ix1, iy1] = box('group:plate:student×exam')
    expect(ix0 > ox0 && iy0 > oy0 && ix1 < ox1 && iy1 < oy1).toBe(true)
  })

  it('draws factor graphs with the factors in their own row', () => {
    const fg = structure(
      'factor-graph',
      [
        { id: 'a', label: 'a', role: 'latent' },
        { id: 'b', label: 'b', role: 'latent' },
        { id: 'f', label: 'f', role: 'factor' },
      ],
      [
        { from: 'a', to: 'f', directed: false },
        { from: 'b', to: 'f', directed: false },
      ],
    )
    const all = flatten(diagram(fg).children)
    const y = (id: string): number => {
      const g = all.find((n) => n.kind === 'group' && n.id === id)
      const shape = g?.kind === 'group' ? g.children[0] : undefined
      return shape?.kind === 'circle' ? shape.cy : shape?.kind === 'rect' ? shape.y + shape.height / 2 : Number.NaN
    }
    expect(y('a')).toBeCloseTo(y('b'), 9)
    expect(y('f')).toBeGreaterThan(y('a') + 50)
  })
})
