import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { concat, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

describe('transform pairs', () => {
  it('every Laplace pair inverts back to its time function', () => {
    for (const pair of S.laplacePairs) {
      if (pair.name === 'impulse') continue
      const { num, den } = pair.rational()
      const inverse = S.inverseLaplace(num, den)
      for (const t of [0.3, 1.1, 2.5]) expect(inverse.at(t)).toBeCloseTo(pair.f(t), 9)
    }
    const params = { a: 0.7, n: 2 }
    const exp = S.laplacePairs.find((q) => q.name === 'exponential')
    if (exp === undefined) throw new Error('missing pair')
    const r = exp.rational(params)
    expect(S.inverseLaplace(r.num, r.den).at(1.3)).toBeCloseTo(exp.f(1.3, params), 9)
    expect(S.inverseLaplace([1, 0, 0], [1, 1]).direct).toEqual([1, -1])
  })

  it('every z pair inverts back to its sequence', () => {
    for (const pair of S.zTransformPairs) {
      const params = pair.name === 'sine' || pair.name === 'cosine' ? { a: 0.9, omega: 0.7 } : { a: 0.6, k: 2 }
      const { num, den } = pair.rational(params)
      // To z^-1 coefficients: pad the numerator in front to the denominator's length.
      const b = [...new Array<number>(den.length - num.length).fill(0), ...num]
      const inverse = S.inverseZ(b, den)
      for (const n of [0, 1, 4, 7]) expect(inverse.at(n)).toBeCloseTo(pair.f(n, params), 9)
      expect(pair.time.length).toBeGreaterThan(0)
    }
  })
})

describe('equations', () => {
  it('difference equations round-trip through transfer functions', () => {
    const g = S.differenceToTf([0.5, 0.25], [1, -0.8, 0.15], { dt: 0.1 })
    const back = S.tfToDifference(g)
    expect(toFlat(back.a as Tensor)).toEqual([1, -0.8, 0.15])
    expect(toFlat(back.b as Tensor)).toEqual([0.5, 0.25, 0])
    expect(back.latex).toBe('y[n] = 0.8 y[n-1] - 0.15 y[n-2] + 0.5 x[n] + 0.25 x[n-1]')
    expect(() => S.tfToDifference(S.tf([1], [1, 1]))).toThrow()
  })

  it('differential equations read back', () => {
    const e = S.tfToDifferential(S.differentialToTf([1, 4], [1, 0, 2, -3, 1, 0.5]))
    expect(e.latex).toBe("y^{(5)} + 2 y''' - 3 y'' + y' + 0.5 y = u' + 4 u")
    expect(() => S.tfToDifferential(S.tf([1], [1, 1], { dt: 1 }))).toThrow()
    const step = S.laplacePairs.find((q) => q.name === 'step')
    if (step === undefined) throw new Error('missing pair')
    expect(S.isDiscrete(S.pairToTf(step))).toBe(false)
  })

  it('differenceToTf is differentiable in the coefficients', () => {
    const f = (b: Value, a: Value): Value => {
      const g = S.differenceToTf(b, a)
      return concat([g.num, g.den], 0)
    }
    expect(checkPrimitive(f, [tensor([0.5, 0.2]), tensor([2, -0.8, 0.1])], { vmap: false }).failures).toEqual([])
  })
})
