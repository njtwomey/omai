/**
 * Tests for the online kernel learners: KLMS against the linear LMS filter it generalises, the quantised dictionary
 * bound, the perceptron's mistake bound and the passive-aggressive loss bound on separable data with a known margin, the
 * Projectron's dictionary bounded by the dimension for a linear kernel, and the budgets.
 */

import { describe, expect, it } from 'vitest'
import { key as makeKey } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { blobs, uniformRows } from './_testdata.ts'
import {
  kernelPassiveAggressive,
  kernelPassiveAggressiveAlgorithm,
  kernelPassiveAggressiveDefaults,
  kernelPerceptron,
  kernelPerceptronAlgorithm,
  kernelPerceptronDefaults,
  klms,
  klmsAlgorithm,
  klmsDefaults,
  qklms,
} from './index.ts'

describe('KLMS and quantised KLMS', () => {
  const x = uniformRows(80, 2, 71, -1, 1)
  const w = [1.5, -0.7]
  const y = x.map((r) => (w[0] ?? 0) * (r[0] ?? 0) + (w[1] ?? 0) * (r[1] ?? 0))

  it('is the LMS filter for the linear kernel', () => {
    const eta = 0.2
    const wt = [0, 0]
    for (let i = 0; i < 80; i++) {
      const r = x[i] as number[]
      const e = (y[i] ?? 0) - ((wt[0] ?? 0) * (r[0] ?? 0) + (wt[1] ?? 0) * (r[1] ?? 0))
      wt[0] = (wt[0] ?? 0) + eta * e * (r[0] ?? 0)
      wt[1] = (wt[1] ?? 0) + eta * e * (r[1] ?? 0)
    }
    const fit = klms({ kernel: 'linear', gamma: 1, eta }).fit({ x, y })
    const q = [
      [0.3, -0.4],
      [1, 1],
    ]
    const got = toFlat(fit.predict(q))
    for (let i = 0; i < q.length; i++) {
      const r = q[i] as number[]
      expect(got[i]).toBeCloseTo((wt[0] ?? 0) * (r[0] ?? 0) + (wt[1] ?? 0) * (r[1] ?? 0), 10)
    }
  })

  it('learns a planted linear map with a falling squared error', () => {
    const fit = klms({ kernel: 'linear', gamma: 1, eta: 0.3, epochs: 5 }).fit({ x, y })
    const loss = metricSeries(fit.run, 'solver/loss').values
    expect(loss.slice(-20).reduce((a, b) => a + b, 0) / 20).toBeLessThan(
      loss.slice(0, 20).reduce((a, b) => a + b, 0) / 20 / 50,
    )
  })

  it('keeps quantised centres more than the radius apart, so at most ceil(range / radius) of them', () => {
    const line = Array.from({ length: 200 }, (_, i) => [(i * 0.6180339887) % 1])
    const radius = 0.2
    const fit = qklms({ gamma: 5, quantisation: radius, eta: 0.3 }).fit({
      x: line,
      y: line.map((r) => Math.sin(4 * (r[0] ?? 0))),
    })
    const c = fit.params.centres.map((r) => r[0] ?? 0).sort((a, b) => a - b)
    for (let i = 1; i < c.length; i++) expect((c[i] ?? 0) - (c[i - 1] ?? 0)).toBeGreaterThan(radius)
    expect(c.length).toBeLessThanOrEqual(Math.ceil(1 / radius) + 1)
    const plain = klms({ gamma: 5, eta: 0.3 }).fit({ x: line, y: line.map((r) => Math.sin(4 * (r[0] ?? 0))) })
    expect(plain.dictionarySize).toBe(200)
  })

  it('has a quantised learner equal to KLMS when no input is within the radius', () => {
    const a = klms({ gamma: 1, eta: 0.3 }).fit({ x, y })
    const b = qklms({ gamma: 1, eta: 0.3, quantisation: 1e-9 }).fit({ x, y })
    expect(toFlat(b.predict(x.slice(0, 5)))).toEqual(toFlat(a.predict(x.slice(0, 5))))
  })

  it('bounds the dictionary by a budget, and serialises its state', () => {
    const fit = klms({ gamma: 1, budget: 10 }).fit({ x, y })
    expect(fit.dictionarySize).toBe(10)
    const alg = klmsAlgorithm({ x, y }, { ...klmsDefaults, budget: 10 })
    const state = runAlgorithm(alg, { maxSteps: 30 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('refuses a bad step size', () => {
    expect(() => klms({ eta: 0 }).fit({ x, y })).toThrow()
  })
})

describe('the online classifiers on separable data', () => {
  // Points with |x0| >= 0.5 labelled by the sign of x0: the unit vector (1, 0) separates them with margin 0.5.
  const n = 120
  const raw = uniformRows(n, 2, 81, -1, 1)
  const x = raw.map((r) => [(r[0] ?? 0) >= 0 ? 0.5 + 1.5 * (r[0] ?? 0) : -0.5 + 1.5 * (r[0] ?? 0), r[1] ?? 0])
  const y = x.map((r) => ((r[0] ?? 0) > 0 ? 1 : 0))
  const margin = Math.min(...x.map((r) => Math.abs(r[0] ?? 0)))
  const radius = Math.max(...x.map((r) => Math.hypot(r[0] ?? 0, r[1] ?? 0)))
  const linear = { kernel: 'linear', gamma: 1 } as const

  it('makes at most (R / gamma)^2 mistakes (Novikoff) and separates the data', () => {
    const run = runAlgorithm(
      kernelPerceptronAlgorithm({ x, y }, { ...kernelPerceptronDefaults, ...linear, epochs: 50 }),
      { maxSteps: n * 50 },
    )
    expect(run.state.mistakes).toBeGreaterThan(0)
    expect(run.state.mistakes).toBeLessThanOrEqual((radius / margin) ** 2)
    const fit = kernelPerceptron({ ...linear, epochs: 50 }).fit({ x, y })
    expect(toFlat(fit.predict(x))).toEqual(y)
  })

  it('has a passive-aggressive cumulative squared loss of at most R^2 |u|^2 for the unit-margin comparator', () => {
    const fit = kernelPassiveAggressive({ ...linear, variant: 'pa', epochs: 1 }).fit({ x, y })
    const loss = metricSeries(fit.run, 'solver/loss').values.slice(1)
    expect(loss.length).toBe(n)
    const squared = loss.reduce((a, b) => a + b * b, 0)
    expect(squared).toBeLessThanOrEqual((radius / margin) ** 2)
    expect(squared).toBeGreaterThan(0)
  })

  it('classifies with the aggressive variants too', () => {
    for (const variant of ['pa1', 'pa2'] as const) {
      const fit = kernelPassiveAggressive({ ...linear, variant, C: 1, epochs: 5 }).fit({ x, y })
      expect(fit.score({ x, y })).toBeGreaterThan(0.95)
    }
    const alg = kernelPassiveAggressiveAlgorithm({ x, y }, { ...kernelPassiveAggressiveDefaults, ...linear })
    expect(runAlgorithm(alg, { maxSteps: n }).steps).toBe(n)
  })

  it('has a Projectron dictionary no larger than the dimension for a linear kernel, whatever the mistakes', () => {
    const run = runAlgorithm(
      kernelPerceptronAlgorithm(
        { x, y },
        { ...kernelPerceptronDefaults, ...linear, budget: 'projectron', eta: 1e-6, epochs: 50 },
      ),
      { maxSteps: n * 50 },
    )
    expect(run.state.centres.length).toBeLessThanOrEqual(2)
    expect(run.state.mistakes).toBeGreaterThan(0)
  })

  it('has a Projectron dictionary smaller than the perceptron’s with an RBF kernel on overlapping classes', () => {
    const hard = blobs(60, 1.5, 93)
    const rbf = { kernel: 'rbf', gamma: 2, epochs: 3 } as const
    const plain = kernelPerceptron(rbf).fit(hard)
    const projected = kernelPerceptron({ ...rbf, budget: 'projectron', eta: 0.2 }).fit(hard)
    expect(projected.params.centres.length).toBeLessThan(plain.params.centres.length * 0.75)
    expect(projected.score(hard)).toBeGreaterThan(plain.score(hard) - 0.1)
  })

  for (const budget of ['random-removal', 'oldest', 'forgetron'] as const) {
    it(`keeps at most size support vectors with the ${budget} budget`, () => {
      const noisy = blobs(40, 4, 91)
      const fit = kernelPerceptron({ kernel: 'rbf', gamma: 0.5, budget, size: 8, epochs: 3 }).fit({
        x: noisy.x,
        y: noisy.y,
      })
      expect(fit.params.centres.length).toBeLessThanOrEqual(8)
      const counts = metricSeries(fit.run, 'solver/supportVectors').values
      expect(Math.max(...counts)).toBeLessThanOrEqual(8)
      expect(fit.score({ x: noisy.x, y: noisy.y })).toBeGreaterThan(0.8)
    })
  }

  it('refuses more than two classes and a bad size', () => {
    expect(() => kernelPerceptron().fit({ x: [[0], [1], [2]], y: [0, 1, 2] })).toThrow()
    expect(() => kernelPerceptron({ size: 0, budget: 'oldest' }).fit({ x, y })).toThrow()
  })

  it('gives the classifier after t examples as fittedAt(t)', () => {
    const est = kernelPerceptron({ ...linear, epochs: 1 })
    const s = est.session({ x, y }, { mode: 'eager', key: makeKey(1) })
    expect(s.fittedAt(0).params.centres.length).toBe(0)
    expect(s.fittedAt(s.t).params.centres.length).toBeGreaterThan(0)
  })
})
