import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import * as P from './index.ts'

const NAMES = ['a', 'b', 'c', 'd', 'e']
const vec = (x: unknown): number[] => toArray(decode(x) as Tensor) as number[]
const mat = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]
const flat = (m: P.GaussianMoments, logMass = false): number[] => [
  ...m.mean,
  ...m.covariance.flat(),
  ...(logMass ? [m.logMass] : []),
]

describe('Gaussian factors against numpy and scipy', () => {
  for (const c of loadFixture('pgm/gaussian_factor').cases) {
    it(c.name, () => {
      const [m1, s1, third, fourth] = c.inputs
      const f = P.gaussianFromMoments(NAMES, vec(m1), mat(s1))
      switch (c.fn) {
        case 'marginal': {
          const keep = c.params['keep'] as string[]
          const rest = NAMES.filter((v) => !keep.includes(v))
          const m = P.gaussianToMoments(P.permuteGaussian(P.gaussianMarginalise(f, rest), keep))
          expectMatches(tensor(flat(m)), c.output, c.rtol, c.atol, c.name)
          expect(m.logMass).toBeCloseTo(0, 12)
          return
        }
        case 'conditional': {
          const keep = c.params['keep'] as string[]
          const observed = c.params['observed'] as string[]
          const y = vec(third)
          const reduced = P.gaussianReduce(f, Object.fromEntries(observed.map((v, i) => [v, y[i] ?? 0])))
          const m = P.gaussianToMoments(P.permuteGaussian(reduced, keep))
          expectMatches(tensor(flat(m, true)), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'product': {
          const g = P.gaussianFromMoments(NAMES, vec(third), mat(fourth))
          const m = P.gaussianToMoments(P.gaussianProduct(f, g))
          expectMatches(tensor(flat(m, true)), c.output, c.rtol, c.atol, c.name)
          // Division undoes the product.
          const back = P.gaussianToMoments(P.gaussianDivide(P.gaussianProduct(f, g), g))
          expect(back.mean[0]).toBeCloseTo(vec(m1)[0] ?? 0, 10)
          return
        }
        case 'logpdf': {
          const points = mat(third)
          const values = points.map((x) =>
            P.gaussianLogValue(f, Object.fromEntries(NAMES.map((v, i) => [v, x[i] ?? 0]))),
          )
          expectMatches(tensor(values), c.output, c.rtol, c.atol, c.name)
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('Gaussian factor algebra', () => {
  it('linear-Gaussian conditionals integrate to one over the child', () => {
    const f = P.linearGaussianFactor('y', ['x'], [2], 1, 0.5)
    const conditional = P.gaussianMarginalise(P.gaussianReduce(f, { x: 0.3 }), [])
    expect(P.gaussianLogMass(conditional)).toBeCloseTo(0, 12)
    expect(P.gaussianToMoments(P.gaussianReduce(f, { x: 0.3 })).mean[0]).toBeCloseTo(1.6, 12)
  })

  it('validate their inputs', () => {
    expect(() =>
      P.gaussianFactor(
        ['x', 'x'],
        [
          [1, 0],
          [0, 1],
        ],
        [0, 0],
      ),
    ).toThrow(/distinct/)
    expect(() => P.gaussianFactor(['x'], [[1, 0]], [0])).toThrow(/matrix/)
    expect(() => P.gaussianFromMoments(['x'], [0], [[-1]])).toThrow(/positive definite/)
    expect(() => P.linearGaussianFactor('y', [], [], 0, 0)).toThrow(/variance/)
    expect(() => P.gaussianMarginalise(P.gaussianFactor(['x'], [[1]], [0]), ['y'])).toThrow(/not in the factor/)
  })
})
