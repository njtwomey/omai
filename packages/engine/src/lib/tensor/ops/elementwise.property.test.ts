import fc from 'fast-check'
import { describe, it } from 'vitest'
import {
  awayFromZero,
  binaryShapes,
  expectConformant,
  RUNS,
  tensorOf,
  unaryShapes,
} from '../../../testing/arbitrary.ts'
import type { Tensor, Value } from '../index.ts'
import {
  abs,
  add,
  arccos,
  arccosh,
  arcsin,
  arcsinh,
  arctan,
  arctan2,
  arctanh,
  clip,
  cos,
  cosh,
  div,
  exp,
  expm1,
  log,
  log1p,
  log2,
  log10,
  maximum,
  minimum,
  mod,
  mul,
  neg,
  pow,
  reciprocal,
  relu,
  sigmoid,
  sin,
  sinh,
  softplus,
  sqrt,
  square,
  sub,
  tan,
  tanh,
  where,
} from '../index.ts'

type Gen = (shape: readonly number[]) => fc.Arbitrary<Tensor<'float64'>>

const unary: readonly (readonly [string, (x: Value) => Value, Gen])[] = [
  ['neg', neg, (s) => tensorOf(s)],
  ['abs', abs, (s) => awayFromZero(s)],
  ['square', square, (s) => tensorOf(s)],
  ['reciprocal', reciprocal, (s) => awayFromZero(s, 0.3)],
  ['exp', exp, (s) => tensorOf(s, -3, 3)],
  ['expm1', expm1, (s) => tensorOf(s, -3, 3)],
  ['log', log, (s) => tensorOf(s, 0.1, 5)],
  ['log1p', log1p, (s) => tensorOf(s, -0.8, 5)],
  ['log2', log2, (s) => tensorOf(s, 0.1, 5)],
  ['log10', log10, (s) => tensorOf(s, 0.1, 5)],
  ['sqrt', sqrt, (s) => tensorOf(s, 0.1, 5)],
  ['sin', sin, (s) => tensorOf(s, -4, 4)],
  ['cos', cos, (s) => tensorOf(s, -4, 4)],
  ['tan', tan, (s) => tensorOf(s, -1.2, 1.2)],
  ['arcsin', arcsin, (s) => tensorOf(s, -0.9, 0.9)],
  ['arccos', arccos, (s) => tensorOf(s, -0.9, 0.9)],
  ['arctan', arctan, (s) => tensorOf(s, -3, 3)],
  ['sinh', sinh, (s) => tensorOf(s, -2, 2)],
  ['cosh', cosh, (s) => tensorOf(s, -2, 2)],
  ['tanh', tanh, (s) => tensorOf(s, -2, 2)],
  ['arcsinh', arcsinh, (s) => tensorOf(s, -3, 3)],
  ['arccosh', arccosh, (s) => tensorOf(s, 1.2, 4)],
  ['arctanh', arctanh, (s) => tensorOf(s, -0.9, 0.9)],
  ['sigmoid', sigmoid, (s) => tensorOf(s, -5, 5)],
  ['softplus', softplus, (s) => awayFromZero(s, 0.05, 5)],
  ['relu', relu, (s) => awayFromZero(s)],
  ['clip', (x) => clip(x, -1, 1), (s) => tensorOf(s, -0.9, 0.9)],
]

describe('elementwise unary primitives conform', () => {
  for (const [name, f, gen] of unary) {
    for (const shape of unaryShapes) {
      it(`${name} on [${shape.join(', ')}]`, () => {
        fc.assert(
          fc.property(gen(shape), (x) => expectConformant(f, [x])),
          { numRuns: RUNS },
        )
      })
    }
  }
})

const binary: readonly (readonly [string, (a: Value, b: Value) => Value, Gen, Gen])[] = [
  ['add', add, (s) => tensorOf(s), (s) => tensorOf(s)],
  ['sub', sub, (s) => tensorOf(s), (s) => tensorOf(s)],
  ['mul', mul, (s) => tensorOf(s), (s) => tensorOf(s)],
  ['div', div, (s) => tensorOf(s), (s) => awayFromZero(s, 0.3)],
  ['pow', pow, (s) => tensorOf(s, 0.2, 3), (s) => tensorOf(s, -2, 2)],
  ['maximum', maximum, (s) => tensorOf(s, -1, 0), (s) => tensorOf(s, 0.1, 1)],
  ['minimum', minimum, (s) => tensorOf(s, 0.1, 1), (s) => tensorOf(s, -1, 0)],
  ['mod', mod, (s) => tensorOf(s, 0.2, 0.8), (s) => tensorOf(s, 1, 2)],
  ['arctan2', arctan2, (s) => tensorOf(s, -2, 2), (s) => tensorOf(s, 0.3, 2)],
]

describe('elementwise binary primitives conform, with broadcasting', () => {
  for (const [name, f, genA, genB] of binary) {
    for (const [sa, sb] of binaryShapes) {
      it(`${name} on [${sa.join(', ')}] and [${sb.join(', ')}]`, () => {
        fc.assert(
          fc.property(genA(sa), genB(sb), (a, b) => expectConformant(f, [a, b])),
          { numRuns: RUNS },
        )
      })
    }
  }

  it('where routes gradients to the chosen branch', () => {
    const cond = (x: Value) => (x.shape.length === 0 ? true : [true, false, true])
    fc.assert(
      fc.property(tensorOf([3]), tensorOf([3]), (a, b) => expectConformant((x, y) => where(cond(x), x, y), [a, b])),
      { numRuns: RUNS },
    )
  })
})
