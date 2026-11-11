// The graph builder of the model compiler: the gate under assembly and the helpers that emit graph structure.
import { describe, expect, it } from 'vitest'
import { defaultOperators } from '../../../message-passing/index.ts'
import { defaultOperators as modelOperators } from '../operators.ts'
import { discreteRowGate, GateBuilder, GraphBuilder } from './_builder.ts'

describe('GateBuilder', () => {
  it('numbers the shared variables by position, adding each once', () => {
    const gate = new GateBuilder('z')
    expect([gate.slot('a'), gate.slot('b'), gate.slot('a')]).toEqual([0, 1, 0])
    expect(gate.variables).toEqual(['a', 'b'])
  })

  it('counts the output the gate defines as position 0 when it starts the list', () => {
    const gate = new GateBuilder('z', ['out'])
    expect([gate.slot('a'), gate.slot('out'), gate.slot('b')]).toEqual([1, 0, 2])
  })

  it('assembles the gate factor: selector then variables as arguments, arity the number of variables, one branch per state', () => {
    const gate = new GateBuilder('z', ['out'])
    const mean = gate.slot('m')
    gate.add({ operator: 'gaussian', args: [0, mean], params: { precision: 4 } })
    gate.add({ operator: 'gaussian', args: [0, { const: 1 }], params: { precision: 1 } })
    expect(gate.factor()).toEqual({
      operator: 'gate',
      args: ['z', 'out', 'm'],
      params: {
        arity: 2,
        branches: [
          { operator: 'gaussian', args: [0, 1], params: { precision: 4 } },
          { operator: 'gaussian', args: [0, { const: 1 }], params: { precision: 1 } },
        ],
      },
    })
  })

  it('does not share the list it was given', () => {
    const given = ['out']
    new GateBuilder('z', given).slot('a')
    expect(given).toEqual(['out'])
  })
})

describe('discreteRowGate', () => {
  it('is a gate whose branch j is the discrete factor on row j, the output being position 0', () => {
    expect(discreteRowGate('prev', 'next', ['row0', 'row1', 'row2'])).toEqual({
      operator: 'gate',
      args: ['prev', 'next', 'row0', 'row1', 'row2'],
      params: {
        arity: 4,
        branches: [
          { operator: 'discrete', args: [0, 1], params: {} },
          { operator: 'discrete', args: [0, 2], params: {} },
          { operator: 'discrete', args: [0, 3], params: {} },
        ],
      },
    })
  })
})

describe('GraphBuilder.sizeOf', () => {
  const builder = (): GraphBuilder => new GraphBuilder(defaultOperators(), modelOperators())

  it('is the declared number of states of a discrete variable', () => {
    const g = builder()
    g.addVariable({ name: 'z', family: { kind: 'discrete', size: 5 } })
    expect(g.sizeOf('z')).toBe(5)
  })

  it('reads a variable with no declared size, and a name that is not a variable, as 2', () => {
    const g = builder()
    g.addVariable({ name: 'b', family: { kind: 'bernoulli' } })
    expect(g.sizeOf('b')).toBe(2)
    expect(g.sizeOf('nothing')).toBe(2)
  })
})
