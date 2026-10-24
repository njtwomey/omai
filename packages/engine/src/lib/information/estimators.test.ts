import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { checkPrimitive } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { add, item, mul, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as I from './index.ts'

const n = 400
const x = normal(key(1), [n])
const noise = normal(key(2), [n])
const rho = 0.8
// Y = rho X + sqrt(1 - rho^2) N: a standard bivariate normal with correlation rho.
const y = add(mul(x, rho), mul(noise, Math.sqrt(1 - rho * rho)))
const trueMi = -0.5 * Math.log(1 - rho * rho)

describe('estimators from samples', () => {
  it('plug-in entropy and mutual information, with the Miller-Madow correction', () => {
    expect(I.plugInEntropy([1, 1, 1])).toBe(0)
    expect(I.plugInEntropy(['a', 'b', 'c', 'd'], { base: 2 })).toBeCloseTo(2, 14)
    expect(I.plugInEntropy(tensor([0, 1, 0, 1]), { base: 2 })).toBeCloseTo(1, 14)
    const mm = I.plugInEntropy(['a', 'b', 'c', 'd'], { correction: 'millerMadow' })
    expect(mm).toBeCloseTo(Math.log(4) + 3 / 8, 14)
    expect(I.plugInMutualInformation([0, 1, 0, 1], [0, 0, 1, 1])).toBeCloseTo(0, 14)
    expect(() => I.plugInMutualInformation([0, 1], [0])).toThrow()
    expect(() => I.plugInEntropy([])).toThrow()
  })

  it('kernel estimators, with and without leaving samples out', () => {
    const h = 0.5 * Math.log(2 * Math.PI * Math.E)
    const resub = I.kdeEntropy(x)
    const loo = I.kdeEntropy(x, { leaveOneOut: true })
    expect(Math.abs(resub - h)).toBeLessThan(0.1)
    expect(loo).toBeGreaterThan(resub)
    expect(Math.abs(I.kdeEntropy(x, { bandwidth: 'silverman' }) - h)).toBeLessThan(0.1)
    expect(Math.abs(I.kdeEntropy(x, { bandwidth: 0.3 }) - h)).toBeLessThan(0.1)
    expect(Math.abs(I.kdeMutualInformation(x, y) - trueMi)).toBeLessThan(0.1)
    expect(() => I.kdeEntropy([1])).toThrow()
  })

  it('transfer entropy is directional, for discrete and continuous series', () => {
    const steps = 300
    const e1 = toFlat(normal(key(6), [steps]))
    const e2 = toFlat(normal(key(7), [steps]))
    const src = [0]
    const tgt = [0]
    for (let t = 1; t < steps; t++) {
      src.push(0.5 * (src[t - 1] ?? 0) + (e1[t] ?? 0))
      tgt.push(0.3 * (tgt[t - 1] ?? 0) + 0.8 * (src[t - 1] ?? 0) + 0.3 * (e2[t] ?? 0))
    }
    const forward = I.transferEntropy(src, tgt, { estimator: 'ksg' })
    const backward = I.transferEntropy(tgt, src, { estimator: 'ksg' })
    expect(forward).toBeGreaterThan(0.3)
    expect(backward).toBeLessThan(0.1)
    const bits = src.map((v) => (v > 0 ? 1 : 0))
    const copy = [0, ...bits.slice(0, -1)]
    expect(I.transferEntropy(bits, copy, { history: 2, sourceHistory: 2 })).toBeGreaterThan(0.5)
    expect(() => I.transferEntropy([0, 1], [0])).toThrow()
    expect(() => I.transferEntropy([0, 1], [0, 1], { history: 2 })).toThrow()
  })

  it('variational bounds: InfoNCE <= log n, DV >= NWJ, and their derivatives conform', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 16, maxLength: 16 }), (v) => {
        const s = tensor(Array.from({ length: 4 }, (_, i) => v.slice(4 * i, 4 * i + 4)))
        expect(item(I.infoNce(s))).toBeLessThanOrEqual(Math.log(4) + 1e-12)
        expect(item(I.mineBound(s))).toBeGreaterThanOrEqual(item(I.nwjBound(s)) - 1e-12)
      }),
      { numRuns: RUNS },
    )
    const s = tensor([
      [2, 0.1, -0.3],
      [0.4, 1.5, 0.2],
      [-0.2, 0.3, 2.5],
    ])
    for (const f of [I.infoNce, I.nwjBound, I.mineBound]) {
      expect(checkPrimitive(f as (v: Value) => Value, [s], { vmap: false }).failures).toEqual([])
    }
    expect(() => I.infoNce([[1, 2]])).toThrow()
  })
})
