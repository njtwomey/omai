// A linear dynamical system with a vector state (a position and a velocity, noisy position readings), written as a `scan` whose
// state is a multivariate normal and whose step is `matvec(F, previous)`: compiled to one Gaussian vector per step, a dense
// `gaussian-transition` between neighbours and a rank-one `linear-regression-row` per reading. On a chain of linear Gaussian factors EP
// is exact, so the reference is the Kalman filter with the Rauch-Tung-Striebel smoother of `dynamics/filtering`.
import { describe, expect, it } from 'vitest'
import { filtering } from '../../../dynamics/index.ts'
import { toArray, toFlat } from '../../../tensor/index.ts'
import { session } from '../../../trace/index.ts'
import { epSpecMethod } from '../../inference/index.ts'
import { analysisOf } from '../../inference/model-input.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../index.ts'

const DT = 0.5
const F = [
  [1, DT],
  [0, 1],
]
const Q = [
  [0.02, 0.01],
  [0.01, 0.04],
]
const P0 = [
  [4, 0],
  [0, 1],
]
const READ_SD = 0.5
const Y = [0.2, 0.9, 1.1, 1.8, 2.4, 2.1, 3.3, 3.9, 4.8, 4.9, 5.7, 6.4]

const model = defineModel('tracker-2d', (m) => {
  const time = m.range('time')
  const y = m.data('y', { dims: [time] })
  const transition = m.const(F)
  const position = m.const([1, 0])
  m.scan('x', {
    over: 'time',
    init: d.multivariateNormal([0, 0], { covariance: P0 }),
    step: ({ prev }) => d.multivariateNormal(ops.matvec(transition, prev), { covariance: Q }),
    emit: ({ state, observe }) => observe('y', d.normal(ops.dot(position, state), READ_SD), y),
  })
})

const at = (m: readonly (readonly number[])[], i: number, j: number): number => m[i]?.[j] as number

/**
 * The exact smoothed means and covariances by the Kalman filter and the RTS smoother, and the log likelihood of the readings. The
 * library's filter predicts before every update, so it starts one step before the first state: x_0 = F^-1 0 and the covariance that
 * the first prediction turns into the prior N(0, P0), `F^-1 (P0 - Q) F^-T`.
 */
function smoothed(y: readonly number[]) {
  const det = at(F, 0, 0) * at(F, 1, 1) - at(F, 0, 1) * at(F, 1, 0)
  const inv = [
    [at(F, 1, 1) / det, -at(F, 0, 1) / det],
    [-at(F, 1, 0) / det, at(F, 0, 0) / det],
  ]
  const gap = [0, 1].map((i) => [0, 1].map((j) => at(P0, i, j) - at(Q, i, j)))
  const left = [0, 1].map((i) => [0, 1].map((j) => at(inv, i, 0) * at(gap, 0, j) + at(inv, i, 1) * at(gap, 1, j)))
  const start = [0, 1].map((i) => [0, 1].map((j) => at(left, i, 0) * at(inv, j, 0) + at(left, i, 1) * at(inv, j, 1)))
  const model = { F, H: [[1, 0]], Q, R: [[READ_SD ** 2]], x0: [0, 0], P0: start }
  const readings = y.map((v) => [v])
  const filtered = filtering.kalmanFilter(model, readings)
  const s = filtering.rtsSmoother(model, filtered)
  return {
    means: toArray(s.means) as number[][],
    covariances: toArray(s.covariances) as number[][][],
    logLik: Number(toFlat(filtering.kalmanLogLikelihood(model, readings))[0]),
  }
}

describe('a vector-state scan under EP, against the Kalman smoother', () => {
  const bound = model.bind({ y: Y })

  it('compiles to a Gaussian vector per step with a dense transition between neighbours and a row per reading', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    const ops_ = c.graph?.factors.map((f) => f.operator) ?? []
    expect(ops_.filter((o) => o === 'gaussian-transition').length).toBe(Y.length - 1)
    expect(ops_.filter((o) => o === 'linear-regression-row').length).toBe(Y.length)
  })

  it('has the smoothed means and covariances of every step to within rounding', () => {
    const method = epSpecMethod(bound)
    expect(method.check?.(analysisOf(bound)).applies).toBe(true)
    const run = session(method, { steps: 200, mode: 'eager', config: { tol: 1e-12 } })
    const post = method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    expect(post.status).toBe('converged')
    const ref = smoothed(Y)
    for (let t = 0; t < Y.length; t++) {
      const mean = Array.from(toFlat(post.mean(`x[${t}]`)))
      const sds = Array.from(toFlat(post.stddev(`x[${t}]`)))
      for (let k = 0; k < 2; k++) {
        const sd = Math.sqrt(ref.covariances[t]?.[k]?.[k] as number)
        expect(Math.abs((mean[k] as number) - (ref.means[t]?.[k] as number)) / sd).toBeLessThan(1e-6)
        expect(Math.abs((sds[k] as number) / sd - 1)).toBeLessThan(1e-6)
      }
    }
    // The EP log evidence of a chain is exact: the Kalman filter's log likelihood of the readings.
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - ref.logLik)).toBeLessThan(1e-8)
  })
})
