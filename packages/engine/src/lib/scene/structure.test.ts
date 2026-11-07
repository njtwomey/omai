import { describe, expect, it } from 'vitest'
import { isStructure, mergeStructures, structure, subStructure } from './structure.ts'

const layer = structure(
  'architecture',
  [
    {
      id: 'x',
      label: 'x',
      role: 'input',
      badges: { shape: '[4]' },
      ports: [{ id: 'out', direction: 'out' }],
    },
    { id: 'fc', label: 'Linear', role: 'operation', group: 'block', ports: [{ id: 'in', direction: 'in' }] },
    { id: 'y', label: 'y', role: 'output' },
  ],
  [
    { from: 'x', to: 'fc', fromPort: 'out', toPort: 'in' },
    { from: 'fc', to: 'y' },
  ],
  {
    groups: [{ id: 'block', label: 'block', kind: 'repeat', count: 3 }],
    annotations: [{ target: 'fc', text: 'shared weights', kind: 'note' }, { text: 'a whole-structure note' }],
    title: 'net',
    direction: 'right',
  },
)

describe('structure', () => {
  it('keeps ports, groups and annotations', () => {
    expect(isStructure(layer)).toBe(true)
    expect(isStructure({ kind: 'scene' })).toBe(false)
    expect(layer.edges[0]).toMatchObject({ fromPort: 'out', toPort: 'in' })
    expect(layer.annotations).toHaveLength(2)
  })

  it('subStructure follows edges and keeps the annotations that still apply', () => {
    const up = subStructure(layer, ['fc'], 'ancestors')
    expect(up.nodes.map((n) => n.id)).toEqual(['x', 'fc'])
    expect(up.annotations).toHaveLength(2)
    const down = subStructure(layer, ['y'], 'descendants')
    expect(down.nodes.map((n) => n.id)).toEqual(['y'])
    expect(down.annotations?.map((a) => a.text)).toEqual(['a whole-structure note'])
    // A long chain with its edges listed far end first: reached in one pass (a sweep until nothing changes took 325 ms
    // for 4,000 nodes), and cycles and branches are visited once.
    const n = 4000
    const node = (i: number): { id: string; label: string; role: 'other' } => ({ id: `n${i}`, label: `n${i}`, role: 'other' })
    const edges = Array.from({ length: n - 1 }, (_, i) => ({ from: `n${n - 2 - i}`, to: `n${n - 1 - i}` }))
    const chain = structure('chain', Array.from({ length: n }, (_, i) => node(i)), [...edges, { from: `n${n - 1}`, to: 'n0' }])
    expect(subStructure(chain, ['n0'], 'descendants').nodes).toHaveLength(n)
    expect(subStructure(chain, ['n3999'], 'ancestors').nodes).toHaveLength(n)
  })

  it('mergeStructures identifies shared ids and merges their fields', () => {
    const extra = structure(
      'architecture',
      [
        {
          id: 'fc',
          label: 'Linear(4, 8)',
          role: 'operation',
          badges: { params: 40 },
          ports: [{ id: 'w', direction: 'in' }],
        },
        { id: 'loss', label: 'loss', role: 'output' },
      ],
      [
        { from: 'fc', to: 'y' },
        { from: 'y', to: 'loss' },
      ],
      { groups: [{ id: 'block', label: 'repeated block', kind: 'repeat' }] },
    )
    const m = mergeStructures([layer, extra], { title: 'merged' })
    expect(m.nodes.map((n) => n.id)).toEqual(['x', 'fc', 'y', 'loss'])
    const fc = m.nodes.find((n) => n.id === 'fc')
    expect(fc?.label).toBe('Linear(4, 8)')
    expect(fc?.ports?.map((p) => p.id)).toEqual(['in', 'w'])
    expect(fc?.badges).toEqual({ params: 40 })
    expect(m.edges).toHaveLength(3)
    expect(m.groups).toEqual([{ id: 'block', label: 'repeated block', kind: 'repeat', count: 3 }])
    expect(m).toMatchObject({ type: 'architecture', title: 'merged', direction: 'right' })
  })

  it('mergeStructures with prefixes keeps parts apart', () => {
    const m = mergeStructures([layer, layer], { prefixes: ['a', 'b'], type: 'pair' })
    expect(m.nodes).toHaveLength(6)
    expect(m.nodes[1]).toMatchObject({ id: 'a/fc', group: 'a/block' })
    expect(m.edges[0]).toMatchObject({ from: 'a/x', to: 'a/fc' })
    expect(m.groups.map((g) => g.id)).toEqual(['a/block', 'b/block'])
    expect(m.annotations?.map((a) => a.target)).toEqual(['a/fc', undefined, 'b/fc', undefined])
    expect(m.type).toBe('pair')
    expect(() => mergeStructures([layer], { prefixes: [] })).toThrow()
    expect(mergeStructures([]).nodes).toEqual([])
  })
})
