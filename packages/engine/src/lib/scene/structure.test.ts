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
