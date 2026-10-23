import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad } from '../autodiff/index.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { fromStorage, mul, sum, toArray, toFlat, type Value } from '../tensor/index.ts'
import { Sequential } from './module/containers.ts'
import { Linear } from './module/layers.ts'
import {
  awq,
  dequantise,
  fakeQuantise,
  gptq,
  outputError,
  quantisationError,
  quantise,
  quantiseModule,
  quantParams,
} from './quantise.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('quantisation', () => {
  it('round trips within half a step, symmetric and affine, per tensor and per channel', () => {
    const x = n(0, [4, 6])
    for (const options of [
      { bits: 8 },
      { bits: 4, symmetric: false },
      { bits: 3, axis: 0 },
      { bits: 5, symmetric: false, axis: 1 },
    ]) {
      const qp = quantParams(x, options)
      const q = quantise(x, qp)
      expect(toFlat(q).every((v) => v >= qp.qmin && v <= qp.qmax)).toBe(true)
      const back = toFlat(dequantise(q, qp))
      const groups = options.axis === undefined ? 1 : (x.shape[options.axis] ?? 1)
      for (const [i, v] of toFlat(x).entries()) {
        const c = options.axis === 0 ? Math.floor(i / 6) : options.axis === 1 ? i % 6 : 0
        expect(groups).toBeGreaterThan(0)
        expect(Math.abs(v - (back[i] ?? 0))).toBeLessThanOrEqual((qp.scale[c] ?? 0) / 2 + 1e-12)
      }
    }
    expect(() => quantParams(x, { bits: 1 })).toThrow()
  })

  it('per-channel scales beat a per-tensor scale on rows of different magnitude', () => {
    const rows = fromStorage(
      Float64Array.from(toFlat(n(1, [3, 16])), (v, i) => v * 10 ** Math.floor(i / 16)),
      [3, 16],
      'float64',
    )
    const perTensor = quantParams(rows, { bits: 4 })
    const perChannel = quantParams(rows, { bits: 4, axis: 0 })
    const e1 = quantisationError(rows, dequantise(quantise(rows, perTensor), perTensor))
    const e2 = quantisationError(rows, dequantise(quantise(rows, perChannel), perChannel))
    expect(e2.mse).toBeLessThan(e1.mse)
    expect(e2.sqnr).toBeGreaterThan(e1.sqnr)
    expect(e2.maxAbs).toBeLessThanOrEqual(e1.maxAbs)
  })

  it('fake quantisation has the straight-through gradient', () => {
    const qp = quantParams([-1, 1], { bits: 4 })
    const g = grad((v: Value) => sum(mul(fakeQuantise(v, qp), 3)))(
      fromStorage(new Float64Array([0.3, -0.7, 2]), [3], 'float64'),
    )
    expect(toArray(g)).toEqual([3, 3, 0])
    expect(toArray(fakeQuantise([0.3, 1], quantParams([0.3, 1], { bits: 2 })))).toEqual([0, 1])
    // The rule is linear, so jvp and vjp agree (the forward is piecewise constant, so finite differences do not apply).
    const report = checkPrimitive(
      (v) => fakeQuantise(v, qp),
      [fromStorage(new Float64Array([0.31, -0.77]), [2], 'float64')],
      {
        eps: 1e-9,
      },
    )
    expect(report.failures.filter((f) => f.startsWith('dot'))).toEqual([])
  })

  it('GPTQ beats round-to-nearest on its calibration data; act-order works too', () => {
    const w = n(2, [6, 8])
    // Correlated inputs: the regime where error compensation helps.
    const z = toFlat(n(3, [64, 8]))
    const x = fromStorage(
      Float64Array.from(z, (v, i) => v + (z[i - (i % 8)] ?? 0)),
      [64, 8],
      'float64',
    )
    const rtnParams = quantParams(w, { bits: 3, symmetric: false, axis: 0 })
    const rtn = dequantise(quantise(w, rtnParams), rtnParams)
    const g = gptq(w, x, { bits: 3 })
    expect(outputError(w, g.weight, x)).toBeLessThanOrEqual(outputError(w, rtn, x))
    const ordered = gptq(w, x, { bits: 3, actOrder: true })
    expect(outputError(w, ordered.weight, x)).toBeLessThanOrEqual(outputError(w, rtn, x))
    expect(toArray(dequantise(g.codes, g.params))).toEqual(toArray(g.weight))
    expect(() => gptq(w, n(4, [5, 3]))).toThrow()
  })

  it('AWQ is never worse than plain round-to-nearest (alpha = 0)', () => {
    const w = n(5, [4, 6])
    const scale = [1, 1, 8, 1, 0.2, 1]
    const x = fromStorage(
      Float64Array.from(toFlat(n(6, [32, 6])), (v, i) => v * (scale[i % 6] ?? 1)),
      [32, 6],
      'float64',
    )
    const r = awq(w, x, { bits: 3, gridSize: 10 })
    const plain = awq(w, x, { bits: 3, gridSize: 1 })
    const rtnParams = quantParams(w, { bits: 3, symmetric: false, axis: 0 })
    const rtn = outputError(w, dequantise(quantise(w, rtnParams), rtnParams), x)
    expect(r.error).toBeLessThanOrEqual(rtn + 1e-12)
    expect(r.error).toBeLessThanOrEqual(plain.error)
    expect(r.alpha >= 0 && r.alpha <= 1).toBe(true)
  })

  it('quantises the weights of a module tree', () => {
    const net = new Sequential(
      new Linear(4, 3, { key: foldIn(key(7), 0) }),
      new Linear(3, 2, { key: foldIn(key(7), 1) }),
    )
    const q = quantiseModule(net.params, { bits: 4 })
    expect(Object.keys(q.report)).toEqual(['0.weight', '1.weight'])
    expect(toFlat((q.params['0'] as Record<string, Value>)['bias'] as Value)).toEqual(
      toFlat((net.params['0'] as Record<string, Value>)['bias'] as Value),
    )
    expect(q.report['0.weight']?.sqnr ?? 0).toBeGreaterThan(10)
  })
})
