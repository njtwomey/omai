import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { piecewiseLinear } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  mixtureOfExpertsAlgorithm,
  moeAssignments,
  moeExpertMeans,
  moeGate,
  moePosterior,
  moePredict,
} from './experts.ts'

// One interior knot at 0.44: slope 5.5 before it and 0.76 after, plus noise of 0.1.
const { data, truth } = generate(piecewiseLinear, { knobs: { n: 120, knots: 1 }, key: key(1) })
const extra = truth.extra as { readonly knots: number[]; readonly slopes: number[] }
const y = toFlat(data.y ?? [])
const xs = toFlat(data.x)

describe('mixture of experts', () => {
  it('recovers a two-piece piecewise-linear function with a gate that switches at the knot', () => {
    const out = runAlgorithm(mixtureOfExpertsAlgorithm(data.x, y, { nExperts: 2 }, key(0)), { maxSteps: 200 })
    expect(out.status).toBe('converged')
    const slopes = out.state.experts.map((e) => e[1] ?? 0).sort((a, b) => a - b)
    const truthSlopes = [...extra.slopes].sort((a, b) => a - b)
    expect(slopes[0]).toBeCloseTo(truthSlopes[0] ?? 0, 0)
    expect(slopes[1]).toBeCloseTo(truthSlopes[1] ?? 0, 0)
    // The expert with the steeper line owns the rows left of the knot; the gate hands over near the knot.
    const knot = extra.knots[0] ?? 0
    const labels = moeAssignments(data.x, y, out.state)
    const steep = out.state.experts.findIndex((e) => Math.abs(e[1] ?? 0) === Math.max(...slopes.map(Math.abs)))
    let wrong = 0
    labels.forEach((l, i) => {
      if (Math.abs((xs[i] ?? 0) - knot) > 0.05 && (l === steep) !== (xs[i] ?? 0) < knot) wrong++
    })
    expect(wrong).toBeLessThanOrEqual(3)
    // A mixture mean close to the truth: residual spread near the noise.
    const fit = moePredict(data.x, out.state)
    const rmse = Math.sqrt(fit.reduce((s, v, i) => s + (v - (y[i] ?? 0)) ** 2, 0) / fit.length)
    expect(rmse).toBeLessThan(0.15)
  })

  it('never lowers the log likelihood (plain maximum likelihood), and the responsibilities sum to one', () => {
    for (const gateRidge of [0, 0.01]) {
      const alg = mixtureOfExpertsAlgorithm(data.x, y, { nExperts: 3, gateRidge, tol: 0 }, key(2))
      let s = alg.init(key(0))
      let last = Number.NEGATIVE_INFINITY
      for (let t = 0; t < 40; t++) {
        s = alg.step(s, undefined as never)
        if (t > 0) {
          // With the ridge, EM raises the penalised likelihood: the plain one may dip by the penalty's change at most.
          expect(s.logLikelihood).toBeGreaterThanOrEqual(last - (gateRidge === 0 ? 1e-9 : 1e-3))
        }
        last = s.logLikelihood
        for (const r of s.resp) expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
      }
    }
  })

  it('has a plain state that round-trips through JSON and a posterior that favours the nearer line', () => {
    const alg = mixtureOfExpertsAlgorithm(data.x, y, { nExperts: 2 }, key(0))
    const s = alg.step(alg.init(key(0)), undefined as never)
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(s))))).toEqual(s)
    const p = {
      gate: [
        [0, 0],
        [0, 0],
      ],
      experts: [
        [0, 1],
        [0, -1],
      ],
      variances: [0.1, 0.1],
    }
    expect(moePosterior([[2], [2]], [2, -2], p).resp.map((r) => r.map(Math.round))).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(moeExpertMeans([[3]], p.experts)).toEqual([[3, -3]])
    expect(moePredict([[3]], p)).toEqual([0])
  })

  it('rejects mismatched shapes', () => {
    expect(() => moeGate([[1, 2]], [[0, 0]])).toThrow()
    expect(() => mixtureOfExpertsAlgorithm(data.x, y, { nExperts: 0 }, key(0))).toThrow()
  })
})
