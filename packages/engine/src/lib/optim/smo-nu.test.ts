import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

const K = [
  [1, 0.6, 0.2, 0.1, 0.0, 0.1],
  [0.6, 1, 0.3, 0.1, 0.1, 0.0],
  [0.2, 0.3, 1, 0.2, 0.1, 0.1],
  [0.1, 0.1, 0.2, 1, 0.5, 0.3],
  [0.0, 0.1, 0.1, 0.5, 1, 0.4],
  [0.1, 0.0, 0.1, 0.3, 0.4, 1],
]
const y = [1, 1, 1, -1, -1, -1]

describe('the nu solver', () => {
  it('keeps both equalities: y^T a = 0 and the class sums equal nu n / 2, throughout the run', () => {
    const Q = K.map((row, i) => row.map((v, j) => v * (y[i] as number) * (y[j] as number)))
    const alg = O.nuSmoAlgorithm(
      { Q, p: new Array<number>(6).fill(0), y, C: 1, alpha0: O.nuSvcStart(y, 0.5) },
      { tol: 1e-10 },
    )
    let s = alg.init({} as never)
    for (let t = 0; t < 60 && alg.done?.(s, t) === undefined; t++) {
      s = alg.step(s, {} as never)
      const pos = [0, 1, 2].reduce((c, i) => c + (s.alpha[i] ?? 0), 0)
      const neg = [3, 4, 5].reduce((c, i) => c + (s.alpha[i] ?? 0), 0)
      expect(pos).toBeCloseTo(1.5, 10)
      expect(neg).toBeCloseTo(1.5, 10)
      expect(s.alpha.every((v) => v >= -1e-12 && v <= 1 + 1e-12)).toBe(true)
    }
  })

  it('has nu as a bound: the fraction of support vectors is at least nu and margin errors at most nu', () => {
    for (const nu of [0.3, 0.6, 0.9]) {
      const r = O.nuSvcDual(K, y, nu, { tol: 1e-10 })
      expect(r.converged).toBe(true)
      expect(r.support.length / 6).toBeGreaterThanOrEqual(nu - 1e-9)
      const a = toFlat(r.alpha)
      // Margin errors sit at the bound C = 1 / r after scaling.
      expect(a.filter((v) => v >= 1 / r.margin - 1e-9).length / 6).toBeLessThanOrEqual(nu + 1e-9)
    }
  })

  it('rejects an infeasible nu', () => {
    expect(() => O.nuSvcStart([1, -1, -1, -1], 0.9)).toThrow(/infeasible/)
    expect(() => O.nuSvcStart([1, -1], 0)).toThrow(/nu/)
    expect(() => O.nuSvrStart(3, 1, 2)).toThrow(/nu/)
  })

  it('records the dual objective falling and the working pairs within a class', () => {
    const Q = K.map((row, i) => row.map((v, j) => v * (y[i] as number) * (y[j] as number)))
    const rec = recorder()
    runAlgorithm(O.nuSmoAlgorithm({ Q, p: new Array<number>(6).fill(0), y, C: 1, alpha0: O.nuSvcStart(y, 0.5) }), {
      maxSteps: 50,
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    const objective = metricSeries(rec.run(), 'objective').values
    for (let t = 1; t < objective.length; t++)
      expect(objective[t] as number).toBeLessThanOrEqual((objective[t - 1] as number) + 1e-12)
    const pairs = rec
      .run()
      .events.filter((e) => e.type === 'message')
      .map((e) => (e as unknown as { payload: [number, number] }).payload)
    for (const [i, j] of pairs) expect(y[i]).toBe(y[j])
  })

  it('solves nu-SVR with the tube width as an output', () => {
    const r = O.nuSvrDual(K, [0, 1, 0.5, -1, -0.5, 0], 5, 0.5, { tol: 1e-10 })
    expect(r.converged).toBe(true)
    expect(r.margin).toBeGreaterThan(-1e-8)
    expect(toFlat(r.coef).reduce((s, v) => s + v, 0)).toBeCloseTo(0, 9)
  })
})
