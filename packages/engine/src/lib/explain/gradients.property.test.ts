import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { add, fromStorage, matmul, type Tensor, tanh, toFlat, type Value } from '../tensor/index.ts'
import { deepLift, expectedGradients, integratedGradients, smoothGrad } from './gradients.ts'

const weights = fc.array(
  fc.integer({ min: -20, max: 20 }).map((v) => v / 10),
  { minLength: 12, maxLength: 12 },
)
const inputs = fc.array(
  fc.integer({ min: -20, max: 20 }).map((v) => v / 10),
  { minLength: 6, maxLength: 6 },
)

function net(w: readonly number[]) {
  const w1 = fromStorage(Float64Array.from(w.slice(0, 6)), [2, 3], 'float64')
  const w2 = fromStorage(Float64Array.from(w.slice(6, 9)), [3, 1], 'float64')
  const b = fromStorage(Float64Array.from(w.slice(9, 12)), [3], 'float64')
  const layers = [
    { fn: (v: Value) => add(matmul(v, w1), b) },
    { fn: tanh, nonlinear: true },
    { fn: (v: Value) => matmul(v, w2) },
  ]
  return { layers, f: (v: Value): Value => layers.reduce((h, l) => l.fn(h), v) }
}

describe('attribution axioms', () => {
  it('DeepLIFT and integrated gradients are complete', () => {
    fc.assert(
      fc.property(weights, inputs, (w, xs) => {
        const { layers, f } = net(w)
        const x = fromStorage(Float64Array.from(xs), [3, 2], 'float64')
        const fx = toFlat(f(x))
        const f0 = toFlat(f(fromStorage(new Float64Array(6), [3, 2], 'float64')))
        const dl = toFlat(deepLift(layers, x))
        for (let i = 0; i < 3; i++) {
          expect((dl[2 * i] ?? 0) + (dl[2 * i + 1] ?? 0)).toBeCloseTo((fx[i] ?? 0) - (f0[i] ?? 0), 9)
        }
        const ig = integratedGradients(f, x, { nSteps: 64 })
        for (const d of ig.delta) expect(Math.abs(d)).toBeLessThan(1e-8)
      }),
      { numRuns: 30 },
    )
  })

  it('noisy estimators are reproducible from their key and exact for linear models', () => {
    const lin = (v: Value) => matmul(v, fromStorage(Float64Array.from([1.5, -0.5]), [2, 1], 'float64'))
    const x = fromStorage(Float64Array.from([1, 2, -1, 0.5]), [2, 2], 'float64')
    toFlat(smoothGrad(lin, x, { key: key(3), nSamples: 7 })).forEach((v, i) => {
      expect(v).toBeCloseTo([1.5, -0.5][i % 2] ?? 0, 12)
    })
    const bg = fromStorage(Float64Array.from([0, 0, 2, 2, -1, 1]), [3, 2], 'float64')
    const a = expectedGradients(lin, x, bg, { key: key(4), nSamples: 20 }) as Tensor
    expect(toFlat(a)).toEqual(toFlat(expectedGradients(lin, x, bg, { key: key(4), nSamples: 20 })))
  })
})
