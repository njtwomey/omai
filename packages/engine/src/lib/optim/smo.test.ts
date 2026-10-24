import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

const decision = (K: number[][], coef: readonly number[], bias: number): number[] =>
  K.map((row) => row.reduce((s, k, i) => s + k * (coef[i] ?? 0), bias))

describe('SMO against libsvm (scikit-learn)', () => {
  for (const c of loadFixture('optim/smo').cases) {
    const out = c.output as Out
    for (const options of [
      { selection: 'second-order' as const, shrinking: true },
      { selection: 'second-order' as const, shrinking: false },
      { selection: 'first-order' as const, shrinking: true },
    ]) {
      it(`${c.name} (${options.selection}${options.shrinking ? ', shrinking' : ''})`, () => {
        const K = toArray(decode(c.inputs[0]) as Tensor) as number[][]
        const r =
          c.fn === 'svcDual'
            ? O.svcDual(K, decode(c.inputs[1]) as Tensor, decode(c.inputs[2]) as Tensor, { tol: 1e-10, ...options })
            : O.svrDual(K, decode(c.inputs[1]) as Tensor, c.params['C'] as number, c.params['epsilon'] as number, {
                tol: 1e-10,
                ...options,
              })
        expect(r.converged).toBe(true)
        expectMatches(r.coef, out['coef'], c.rtol, c.atol, `${c.name} coef`)
        expectMatches(r.bias, out['intercept'], c.rtol, c.atol, `${c.name} intercept`)
        const f = decision(K, toFlat(r.coef), r.bias)
        expectMatches(tensor(f), out['decision'], c.rtol, c.atol, `${c.name} decision`)
      })
    }
  }
})

describe('SMO', () => {
  const K = [
    [1, 0.5, 0.2, 0.1],
    [0.5, 1, 0.3, 0.2],
    [0.2, 0.3, 1, 0.6],
    [0.1, 0.2, 0.6, 1],
  ]
  const y = [1, 1, -1, -1]

  it('satisfies the KKT conditions of the dual and records violations and working pairs', () => {
    const rec = recorder()
    const r = O.svcDual(K, y, 1, { tol: 1e-12, tracer: rec, policy: tracePolicy({ messages: true }) })
    const a = toFlat(r.alpha)
    expect(a.reduce((s, v, i) => s + v * (y[i] ?? 0), 0)).toBeCloseTo(0, 12)
    for (const v of a) expect(v >= 0 && v <= 1).toBe(true)
    // KKT: y_i f(x_i) >= 1 when alpha = 0, = 1 when free, <= 1 at C.
    const f = decision(K, toFlat(r.coef), r.bias)
    for (const [i, v] of a.entries()) {
      const margin = (y[i] ?? 0) * (f[i] ?? 0)
      if (v <= 0) expect(margin).toBeGreaterThanOrEqual(1 - 1e-9)
      else if (v >= 1) expect(margin).toBeLessThanOrEqual(1 + 1e-9)
      else expect(margin).toBeCloseTo(1, 9)
    }
    const run = rec.run()
    expect(messages(run, 'workingSet').length).toBe(r.iterations)
    expect(metricSeries(run, 'violation').values.at(-1)).toBeLessThan(1e-12)
  })

  it('starts from a feasible alpha and keeps the equality', () => {
    const r = O.smo({
      Q: [
        [2, 0],
        [0, 2],
      ],
      p: [0, 0],
      y: [1, 1],
      C: 3,
      alpha0: [2, 0],
    })
    expect(toFlat(r.alpha).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([1, 1])
    expect(() => O.smo({ Q: [[1]], p: [0], y: [2], C: 1 })).toThrow()
  })
})
