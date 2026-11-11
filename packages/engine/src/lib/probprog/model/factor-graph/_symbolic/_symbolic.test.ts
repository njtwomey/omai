// The evaluator of the model compiler: an expression as a tensor of affine forms, with the graph structure that random values need.
import { describe, expect, it } from 'vitest'
import { defaultOperators as factorOperators } from '../../../../message-passing/index.ts'
import * as d from '../../dists/index.ts'
import { defineModel } from '../../index.ts'
import { defaultOperators } from '../../operators.ts'
import type { ExprSpec } from '../../spec.ts'
import { isConstCell } from '../_cells.ts'
import { Context } from '../_context.ts'
import { evalSym } from './_affine.ts'

const model = defineModel('evaluator', (m) => {
  const row = m.range('row')
  m.data('x', { dims: [row] })
  m.latent('a', d.normal(0, 1))
  m.latent('b', d.normal(0, 1))
  m.latent('z', d.bernoulli({ probs: 0.5 }))
  m.observe('y', d.normal(0, 1), m.data('y', { dims: [row] }))
})

/** A context with the Gaussian variables `a` and `b` and the Bernoulli variable `z` declared, as the walk would leave them. */
function context(): Context {
  const cx = new Context(model.bind({ x: [1, 2], y: [0, 0] }), defaultOperators(), factorOperators(), {
    vectorise: true,
    packScalars: false,
    denseScales: true,
  })
  for (const name of ['a', 'b']) {
    cx.g.addVariable({ name, family: { kind: 'gaussian' } })
    cx.syms.set(name, { dims: [], cells: [cx.g.varAtom(name)] })
  }
  cx.g.addVariable({ name: 'z', family: { kind: 'bernoulli' } })
  cx.syms.set('z', { dims: [], cells: [{ kind: 'rv', variable: 'z', family: 'bernoulli' }] })
  cx.current = 'y'
  return cx
}
const ref = (name: string): ExprSpec => ({ kind: 'ref', name })
const num = (value: number): ExprSpec => ({ kind: 'const', value })
const op = (name: string, ...args: ExprSpec[]): ExprSpec => ({ kind: 'op', op: name, args })

describe('evalSym', () => {
  it('folds known values to constant cells over their dimensions, making no graph structure', () => {
    const cx = context()
    const sym = evalSym(cx, op('mul', ref('x'), num(2)))
    expect(sym.dims).toEqual(['row'])
    expect(sym.cells.map((c) => isConstCell(c) && c.aff.offset)).toEqual([2, 4])
    expect(cx.g.factors).toEqual([])
  })

  it('keeps arithmetic on random Gaussian values as one affine form of their variables, with no factor', () => {
    const cx = context()
    const cell = evalSym(cx, op('add', ref('a'), op('mul', num(2), ref('b')))).cells[0]
    expect(cell?.kind).toBe('aff')
    const terms = cell?.kind === 'aff' ? [...cell.aff.terms.values()] : []
    expect(terms.sort()).toEqual([1, 2])
    expect(cx.g.factors).toEqual([])
  })

  it('makes a variable and a `product` factor for the product of two random values, named after the current site', () => {
    const cx = context()
    evalSym(cx, op('mul', ref('a'), ref('b')))
    expect(cx.g.factors.map((f) => f.operator)).toEqual(['product'])
    expect(cx.g.variables.map((v) => v.name)).toEqual(['a', 'b', 'z', 'y#0[0]'])
  })

  it('makes an `apply` factor for a function of a random value and remembers the chain behind its output', () => {
    const cx = context()
    evalSym(cx, op('exp', ref('a')))
    expect(cx.g.factors.map((f) => [f.operator, f.params])).toEqual([['apply', { fn: 'exp' }]])
    expect([...cx.applied.values()]).toEqual([{ input: 'a', steps: [['exp']] }])
  })

  it('refuses a random value it has no rule for, with the operator named', () => {
    expect(() => evalSym(context(), op('exp', op('pow', ref('a'), num(3))))).toThrow(/pow/)
  })
})

describe('EvalMode', () => {
  const where = op('where', ref('z'), ref('a'), ref('b'))

  it('plain: a value chosen by a Bernoulli variable among random values is a `select` variable', () => {
    const cx = context()
    const cell = evalSym(cx, where).cells[0]
    expect(cell?.kind).toBe('aff')
    expect(cx.g.factors.map((f) => f.operator)).toEqual(['select'])
  })

  it('lookup: the same value stays a lookup over the variable (its forms the states), and no factor is made', () => {
    const cx = context()
    const cell = evalSym(cx, where, 'lookup').cells[0]
    expect(cell?.kind).toBe('lookup')
    expect(cx.g.factors).toEqual([])
  })
})
