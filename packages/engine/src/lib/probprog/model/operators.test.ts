// The operator registry: shape rules over named dimensions, evaluation against the tensor ops, alignment by name.
import { describe, expect, it } from 'vitest'
import { ndtr } from '../../numerics/special/index.ts'
import { tensor, toArray } from '../../tensor/index.ts'
import {
  alignTo,
  apply,
  constant,
  defaultOperators,
  joinShapes,
  type NamedValue,
  operatorShape,
  ops,
  Var,
} from './index.ts'

const table = defaultOperators()
const val = (dims: string[], v: unknown): NamedValue => ({ dims, value: tensor(v as number[]) })
const run = (op: string, args: NamedValue[], params = {}): NamedValue =>
  (table[op] as NonNullable<(typeof table)[string]>).evaluate(args, params)
const num = (x: NamedValue): unknown => toArray(x.value)

describe('the registry', () => {
  it('is a closed, documented set of 31 operators', () => {
    const names = Object.keys(table)
    expect(names).toHaveLength(31)
    for (const n of names) expect(table[n]?.description.length).toBeGreaterThan(3)
    expect(Object.keys(defaultOperators())).toEqual(names)
    expect(defaultOperators()).not.toBe(table)
  })

  it('evaluates the elementwise functions like the tensor ops', () => {
    const x = 0.7
    const cases: Record<string, number> = {
      neg: -x,
      exp: Math.exp(x),
      log: Math.log(x),
      sqrt: Math.sqrt(x),
      square: x * x,
      abs: x,
      tanh: Math.tanh(x),
      cos: Math.cos(x),
      sigmoid: 1 / (1 + Math.exp(-x)),
      softplus: Math.log1p(Math.exp(x)),
    }
    for (const [op, want] of Object.entries(cases))
      expect(num(run(op, [val([], x)])) as number, op).toBeCloseTo(want, 14)
    expect(num(run('probit', [val([], x)]))).toBeCloseTo(0.758036347776927, 14)
    expect(num(run('probit', [val([], x)]))).toBe(toArray(ndtr(x)))
  })

  it('keeps the probit inside (0, 1), so a Bernoulli on it has a finite logit in the tails', () => {
    const tails = num(run('probit', [val([], [-60, 0, 60])])) as number[]
    expect(tails[0]).toBeGreaterThan(0)
    expect(tails[1]).toBe(0.5)
    expect(tails[2]).toBeLessThan(1)
    expect(Math.log(tails[2] as number) - Math.log1p(-(tails[2] as number))).toBeLessThan(40)
  })

  it('evaluates arithmetic, comparisons, where and clip', () => {
    expect(num(run('add', [val([], 2), val([], 3)]))).toBe(5)
    expect(num(run('sub', [val([], 2), val([], 3)]))).toBe(-1)
    expect(num(run('mul', [val([], 2), val([], 3)]))).toBe(6)
    expect(num(run('div', [val([], 3), val([], 2)]))).toBe(1.5)
    expect(num(run('pow', [val([], 3), val([], 2)]))).toBe(9)
    expect(num(run('maximum', [val([], 3), val([], 2)]))).toBe(3)
    expect(num(run('minimum', [val([], 3), val([], 2)]))).toBe(2)
    expect(num(run('gt', [val([], 3), val([], 2)]))).toBe(1)
    expect(num(run('lt', [val([], 3), val([], 2)]))).toBe(0)
    expect(num(run('eq', [val([], 3), val([], 3)]))).toBe(1)
    const c = run('gt', [val(['i'], [1, 5, 2]), val([], 1.5)])
    expect(num(run('where', [c, val(['i'], [10, 20, 30]), val([], 0)]))).toEqual([0, 20, 30])
    expect(num(run('clip', [val(['i'], [-2, 0.5, 9]), val([], 0), val([], 1)]))).toEqual([0, 0.5, 1])
  })

  it('reduces over a named dimension, or over all of them', () => {
    const m = val(
      ['row', 'col'],
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
    )
    expect(run('sum', [m], { dim: 'row' }).dims).toEqual(['col'])
    expect(num(run('sum', [m], { dim: 'row' }))).toEqual([5, 7, 9])
    expect(num(run('sum', [m], { dim: 'col' }))).toEqual([6, 15])
    expect(num(run('sum', [m]))).toBe(21)
    expect(num(run('mean', [m], { dim: 'col' }))).toEqual([2, 5])
    expect(num(run('mean', [m]))).toBe(3.5)
  })

  it('contracts the last dimension of the left with the first of the right in matmul', () => {
    const a = val(
      ['row', 'k'],
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    )
    const v = run('matmul', [a, val(['k'], [1, 10])])
    expect(v.dims).toEqual(['row'])
    expect(num(v)).toEqual([21, 43, 65])
    const w = run('matmul', [
      a,
      val(
        ['k', 'out'],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ])
    expect(w.dims).toEqual(['row', 'out'])
    expect(num(w)).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    expect(num(run('matmul', [val(['k'], [1, 2]), val(['k'], [3, 4])]))).toBe(11)
  })

  it('gathers with at: the index dimensions replace the first dimension', () => {
    const table2 = val(['group'], [10, 20, 30])
    const g = run('at', [table2, val(['row'], [2, 0, 0, 1])])
    expect(g.dims).toEqual(['row'])
    expect(num(g)).toEqual([30, 10, 10, 20])
    const rows = run('at', [
      val(
        ['group', 'k'],
        [
          [1, 2],
          [3, 4],
        ],
      ),
      val(['row'], [1, 1, 0]),
    ])
    expect(rows.dims).toEqual(['row', 'k'])
    expect(num(rows)).toEqual([
      [3, 4],
      [3, 4],
      [1, 2],
    ])
  })
})

describe('named dimensions', () => {
  it('align operands by name, whatever order they declared', () => {
    const a = val(['row'], [1, 2, 3])
    const b = val(['feature'], [10, 20])
    const out = run('add', [a, b])
    expect(out.dims).toEqual(['row', 'feature'])
    expect(num(out)).toEqual([
      [11, 21],
      [12, 22],
      [13, 23],
    ])
    const swapped = run('add', [
      val(
        ['feature', 'row'],
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ),
      a,
    ])
    expect(swapped.dims).toEqual(['feature', 'row'])
    expect(num(swapped)).toEqual([
      [2, 4, 6],
      [5, 7, 9],
    ])
    const rev = run('mul', [
      val(
        ['row', 'feature'],
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      ),
      val(
        ['feature', 'row'],
        [
          [1, 1, 1],
          [2, 2, 2],
        ],
      ),
    ])
    expect(rev.dims).toEqual(['row', 'feature'])
    expect(num(rev)).toEqual([
      [1, 4],
      [3, 8],
      [5, 12],
    ])
  })

  it('alignTo permutes, inserts size-one axes and pads event axes', () => {
    expect(alignTo(val(['b'], [1, 2, 3]), ['a', 'b']).shape).toEqual([1, 3])
    expect(
      alignTo(
        val(
          ['b', 'a'],
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
        ),
        ['a', 'b'],
      ).shape,
    ).toEqual([2, 3])
    expect(alignTo(val(['a'], [1, 2]), ['a', 'b'], 1).shape).toEqual([2, 1, 1])
    expect(() => alignTo(val(['c'], [1]), ['a'])).toThrow(/does not fit/)
  })

  it('the joint shape lists dimensions by first appearance and takes the largest event rank', () => {
    expect(
      joinShapes([
        { dims: ['b', 'a'], eventRank: 0 },
        { dims: ['c', 'a'], eventRank: 1 },
      ]),
    ).toEqual({
      dims: ['b', 'a', 'c'],
      eventRank: 1,
    })
  })
})

describe('shape rules', () => {
  const s = (...dims: string[]) => ({ dims, eventRank: 0 })

  it('reject a matmul whose contracted dimensions differ, and say which', () => {
    expect(() => operatorShape(table, 'matmul', [s('row', 'feature'), s('other')])).toThrow(/'feature' and 'other'/)
    expect(() => operatorShape(table, 'matmul', [s(), s('k')])).toThrow(/need/)
    expect(() => operatorShape(table, 'matmul', [{ dims: ['a'], eventRank: 1 }, s('a')])).toThrow(/event axes/)
  })

  it('reject a reduction over a dimension the operand lacks', () => {
    expect(() => operatorShape(table, 'sum', [s('row')], { dim: 'col' })).toThrow(/'col' is not one of \[row\]/)
  })

  it('reject unknown operators and the wrong arity', () => {
    expect(() => operatorShape(table, 'cube', [s()])).toThrow(/unknown operator 'cube'/)
    expect(() => operatorShape(table, 'add', [s()])).toThrow(/takes 2 argument/)
  })

  it('reject an at whose result would repeat a dimension', () => {
    expect(() => operatorShape(table, 'at', [s('g', 'row'), s('row')])).toThrow(/repeats a dimension/)
  })
})

describe('Var', () => {
  it('builds checked expression trees with dimensions, never values', () => {
    const X = new Var({ kind: 'ref', name: 'X' }, ['row', 'feature'], 0)
    const w = new Var({ kind: 'ref', name: 'w' }, ['feature'], 0)
    const eta = ops.probit(X.matmul(w).add(0.5))
    expect(eta.dims).toEqual(['row'])
    expect(eta.expr.kind).toBe('op')
    expect(JSON.parse(JSON.stringify(eta.expr))).toEqual(eta.expr)
    expect(() => w.matmul(X)).toThrow(/contracted dimensions differ/)
    expect(X.sum('feature').dims).toEqual(['row'])
    expect(X.mean().dims).toEqual([])
  })

  it('takes constants: numbers, arrays with named leading dimensions and event axes', () => {
    expect(constant(2).dims).toEqual([])
    const v = constant([1, 2, 3], ['feature'])
    expect([v.dims, v.eventRank]).toEqual([['feature'], 0])
    expect(constant([1, 2, 3]).eventRank).toBe(1)
    expect(() => constant([1, 2], ['a', 'b'])).toThrow(/rank 1/)
    expect(() => constant(Number.NaN)).toThrow(/finite/)
    expect(apply('add', [1, 2]).dims).toEqual([])
  })

  it('takes a tensor as a constant: read once as the nested array it holds, its leading axes named', () => {
    const v = constant(
      tensor([
        [1, 2],
        [3, 4],
      ]),
      ['row'],
    )
    expect([v.dims, v.eventRank]).toEqual([['row'], 1])
    expect(constant(tensor([1, 2, 3])).expr).toEqual({ kind: 'const', value: [1, 2, 3] })
    expect(apply('add', [1, tensor([1, 2])]).eventRank).toBe(1)
  })
})
