import { describe, expect, it } from 'vitest'
import { add, bind1, definePrimitive, mul, sin, sum, tensor, type Value } from '../tensor/index.ts'
import { computationGraph, graphStats } from './graph.ts'

const GRAD = 'grad'

describe('computationGraph', () => {
  it('badges nodes with values and the gradient of a scalar output', () => {
    const g = computationGraph((x: Value) => add(mul(x, x), sin(x)), tensor(2))
    const badge = (label: string, key: string): unknown => g.nodes.find((n) => n.label === label)?.badges?.[key]
    expect(badge('x', 'grad')).toBeCloseTo(4 + Math.cos(2), 5)
    expect(badge('add', 'grad')).toBe(1)
    expect(badge('mul', 'grad')).toBe(1)
    expect(badge('sin', 'value')).toBeCloseTo(Math.sin(2), 5)
  })

  it('reports gradient norms for tensors and can leave gradients out', () => {
    const f = (p: { w: Value }): Value => sum(mul(p.w, p.w))
    const g = computationGraph(f, { w: tensor([3, 4]) })
    expect(g.nodes[0]?.badges?.['|grad|']).toBe(10)
    const plain = computationGraph(f, { w: tensor([3, 4]) }, { gradients: false })
    expect(plain.nodes[0]?.badges?.['|grad|']).toBeUndefined()
    const multi = computationGraph((x: Value) => [x, mul(x, 2)], tensor(1))
    expect(multi.nodes.every((n) => n.badges?.[GRAD] === undefined)).toBe(true)
  })

  it('still draws graphs that cannot be differentiated', () => {
    const opaque = definePrimitive<null>({
      name: 'opaque',
      abstractEval: ([v]) => [v as NonNullable<typeof v>],
      impl: ([v]) => [v as NonNullable<typeof v>],
    })
    const g = computationGraph((x: Value) => sum(bind1(opaque, [x], null)), tensor([1, 3, 2]))
    expect(g.nodes.map((n) => n.label)).toEqual(['x', 'opaque', 'reduce_sum'])
    expect(g.nodes.every((n) => n.badges?.[GRAD] === undefined && n.badges?.['|grad|'] === undefined)).toBe(true)
  })
})

describe('graphStats', () => {
  it('counts nodes, operations and depth', () => {
    const g = computationGraph((x: Value) => add(mul(add(x, 1), x), sin(x)), tensor(0.5))
    expect(graphStats(g)).toEqual({
      nodes: 6,
      edges: 7,
      inputs: 1,
      constants: 1,
      operations: 4,
      depth: 3,
      operationCounts: { add: 2, mul: 1, sin: 1 },
    })
  })
})
