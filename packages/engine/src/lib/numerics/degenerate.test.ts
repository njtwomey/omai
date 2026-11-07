import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { sum, toArray, type Value } from '../tensor/index.ts'
import * as A from './approximation/index.ts'
import * as D from './differentiate/index.ts'
import * as G from './geometry/index.ts'
import * as I from './inverse/index.ts'
import * as Q from './quadrature/index.ts'

// Degenerate inputs of the less-used routines: each gave NaN, a wrong zero, or an unrelated error message before.
const values = (v: unknown): unknown => (typeof v === 'number' || Array.isArray(v) ? v : toArray(v as Value))

describe('degenerate inputs give the defined value', () => {
  it.each([
    ['ulp of the largest double (numpy.spacing)', () => A.ulp(Number.MAX_VALUE), 2 ** 971],
    ['shanks of a constant sequence', () => A.shanks([1, 1, 1, 1, 1]), [1, 1, 1]],
    ['wynnEpsilon of a constant sequence', () => A.wynnEpsilon([1, 1, 1, 1]).value, 1],
    ['the Tikhonov solution for a zero matrix', () => I.tikhonov(I.spectralData([[0, 0], [0, 0]], [1, 1]), 1), [0, 0]],
    ['the Tikhonov solution at lambda 0 with a zero singular value', () => I.tikhonov(I.spectralData([[2, 0], [0, 0]], [2, 3]), 0), [1, 0]],
    ['the truncated SVD solution with a zero singular value kept', () => I.tsvd(I.spectralData([[2, 0], [0, 0]], [2, 3]), 2), [1, 0]],
    ['the filter factors at lambda 0 with a zero singular value', () => I.filterFactors([2, 0], 0), [1, 0]],
    ['the derivative of x^2 at 1e16 (the default step 0.5 is lost in x)', () => D.derivative((x) => x * x, 1e16).df, 2e16],
  ])('%s', (_name, call, expected) => {
    const v = values(call())
    if (typeof expected === 'number') expect(Math.abs((v as number) / expected - 1)).toBeLessThan(1e-9)
    else expect(v).toEqual(expected)
  })

  it.each([
    ['lambdaGrid of all-zero data', () => I.lambdaGrid(I.spectralData([[0, 0], [0, 0]], [1, 1]))],
    ['finiteDifferenceWeights with a repeated node', () => D.finiteDifferenceWeights(1, [0, 0, 1])],
    ['monteCarlo with n = 0', () => Q.monteCarlo((p) => sum(p, -1), [[0, 1], [0, 1]], key(0), 0)],
    ['homography of collinear points', () => G.homography([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 2], [2, 4], [3, 6]])],
    ['convexHull with a NaN coordinate', () => G.convexHull([[0, 0], [1, 0], [Number.NaN, 1], [0, 1]])],
  ])('%s throws a value error', (_name, call) => {
    expect(call).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('keeps the default step 0.5 of derivative at an ordinary point', () => {
    expect(D.derivative(Math.exp, 1).table[0]?.[0]).toBe((Math.exp(1.5) - Math.exp(0.5)) / 1)
  })
})
