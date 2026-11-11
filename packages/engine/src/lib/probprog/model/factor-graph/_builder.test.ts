// The graph builder of the model compiler: the gate under assembly and the helpers that emit graph structure.
import { describe, expect, it } from 'vitest'
import { defaultOperators } from '../../../message-passing/index.ts'
import { defaultOperators as modelOperators } from '../operators.ts'
import { discreteRowGate, GateBuilder, GraphBuilder, precisionBranch, precisionFactor } from './_builder.ts'
import { affOf, combine, constCell } from './_cells.ts'
import { Refusal } from './_refusal.ts'

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

describe('GraphBuilder.portOf', () => {
  const builder = (): GraphBuilder => new GraphBuilder(defaultOperators(), modelOperators())

  it('is { const } for a known cell', () => {
    expect(builder().portOf(constCell(2.5), 'y', undefined)).toEqual({ const: 2.5 })
  })

  it('is the variable itself for a cell that is one variable', () => {
    const g = builder()
    g.addVariable({ name: 'x', family: { kind: 'gaussian' } })
    expect(g.portOf(g.varAtom('x'), 'y', undefined)).toBe('x')
    expect(g.variables.map((v) => v.name)).toEqual(['x'])
  })

  it('makes an auxiliary variable for an affine form of several, named after the site and the index', () => {
    const g = builder()
    g.addVariable({ name: 'a', family: { kind: 'gaussian' } })
    g.addVariable({ name: 'b', family: { kind: 'gaussian' } })
    const a = affOf(g.varAtom('a'))
    const b = affOf(g.varAtom('b'))
    const port = g.portOf({ kind: 'aff', aff: combine(a, b, 1) }, 'y', 3)
    expect(port).toBe('y#0[3]')
    expect(g.factors.map((f) => f.operator)).toEqual(['plus'])
    expect(g.auxiliary).toEqual(['y#0[3]'])
  })

  it('refuses a cell that is not an affine form', () => {
    expect(() => builder().portOf({ kind: 'lookup', variable: 'z', values: [1, 2] }, 'y', undefined)).toThrow(Refusal)
  })
})

describe('the gaussian-gamma-precision helpers', () => {
  it('omits the scale and the offset of a factor when they are 1 and 0', () => {
    expect(precisionFactor('y', { const: 2 }, { variable: 't', scale: 1, offset: 0 })).toEqual({
      operator: 'gaussian-gamma-precision',
      args: ['y', { const: 2 }, 't'],
    })
    expect(precisionFactor('y', 'mu', { variable: 't', scale: 3, offset: 0.5 })).toEqual({
      operator: 'gaussian-gamma-precision',
      args: ['y', 'mu', 't'],
      params: { scale: 3, offset: 0.5 },
    })
  })

  it('always names both parameters of a gate branch, whatever their values', () => {
    expect(precisionBranch([0, { const: 0 }, 1], { scale: 1, offset: 0 })).toEqual({
      operator: 'gaussian-gamma-precision',
      args: [0, { const: 0 }, 1],
      params: { scale: 1, offset: 0 },
    })
  })
})
