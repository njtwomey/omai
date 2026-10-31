/**
 * Tests for kernel ridge regression (the dual solve, the leave-one-out shortcuts, the estimator) and the online kernel
 * learners (KRLS with its three dictionary rules, NORMA).
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { gpRegression } from '../gp/regression.ts'
import { effectiveDimension, linearKernel, maternKernel, pairwiseKernels, rbfKernel } from '../kernels/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  createKrlsState,
  hatDiagonal,
  kernelRidge,
  krls,
  krlsAlgorithm,
  krlsDefaults,
  krlsPredict,
  krlsStep,
  looPredictions,
  looPress,
  norma,
  normaAlgorithm,
  normaDefaults,
  solveKernelRidge,
} from './index.ts'

describe('kernel ridge regression', () => {
  const x = [
    [-1.5, 0.2],
    [-0.5, -0.4],
    [0.3, 0.8],
    [1.1, -0.1],
    [2.0, 0.5],
  ]
  const y = [0.8, -0.3, 1.2, 0.4, -0.7]

  it('has the predictions of the Gaussian process posterior mean with noise alpha', () => {
    const kernel = rbfKernel({ lengthScale: 1.1 })
    const gp = gpRegression.with({ kernel, alpha: 0.25, optimise: false }).fit({ x, y })
    const krr = kernelRidge.with({ kernel, alpha: 0.25 }).fit({ x, y })
    const query = [
      [-0.2, 0.1],
      [0.9, 0.3],
    ]
    const a = toFlat(krr.predict(query))
    const b = toFlat(gp.predict(query))
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 8)
  })

  it('solves the dual system exactly', () => {
    const gram = pairwiseKernels(x, undefined, 'rbf', { gamma: 0.5 })
    const c = toFlat(solveKernelRidge(gram, y, 0.3))
    const k = toFlat(gram)
    for (let i = 0; i < 5; i++) {
      let r = 0.3 * (c[i] ?? 0)
      for (let j = 0; j < 5; j++) r += (k[i * 5 + j] ?? 0) * (c[j] ?? 0)
      expect(r).toBeCloseTo(y[i] ?? 0, 10)
    }
  })

  it('has a hat diagonal in (0, 1) that sums to the effective dimension', () => {
    const gram = pairwiseKernels(x, undefined, 'rbf', { gamma: 0.8 })
    const h = toFlat(hatDiagonal(gram, 0.35))
    for (const v of h) {
      expect(v).toBeGreaterThan(0)
      expect(v).toBeLessThan(1)
    }
    expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(item(effectiveDimension(gram, 0.35)), 10)
  })

  it('has leave-one-out predictions that equal refitting n times', () => {
    const xs = [[-2.0], [-1.0], [-0.2], [0.5], [1.3], [2.1]]
    const ys = [1.4, 0.2, -0.5, 0.7, 1.8, 0.9]
    const hyper = { kernel: 'rbf', gamma: 0.6, alpha: 0.2 } as const
    const gram = pairwiseKernels(xs, undefined, 'rbf', { gamma: 0.6 })
    const fast = toFlat(looPredictions(gram, ys, 0.2))
    let press = 0
    for (let i = 0; i < xs.length; i++) {
      const fit = kernelRidge.with(hyper).fit({ x: xs.filter((_, j) => j !== i), y: ys.filter((_, j) => j !== i) })
      const pred = toFlat(fit.predict([xs[i] as number[]]))[0] ?? 0
      expect(fast[i]).toBeCloseTo(pred, 10)
      press += ((ys[i] ?? 0) - pred) ** 2
    }
    expect(looPress(gram, ys, 0.2)).toBeCloseTo(press, 10)
    const full = kernelRidge.with(hyper).fit({ x: xs, y: ys })
    expect(full.looPress()).toBeCloseTo(press, 10)
    expect(full.params.effectiveDf).toBeCloseTo(
      toFlat(hatDiagonal(gram, 0.2)).reduce((a, b) => a + b, 0),
      10,
    )
    const loo = toFlat(full.looPredictions())
    for (let i = 0; i < xs.length; i++) expect(loo[i]).toBeCloseTo(fast[i] ?? 0, 9)
  })

  it('handles several outputs and a custom kernel, and revives from its parameters', () => {
    const xs = [
      [0, 1],
      [1, 0],
      [1, 1],
      [2, 1],
    ]
    const ys = [
      [1, -1],
      [2, 0],
      [3, -1],
      [4, 1],
    ]
    const est = kernelRidge.with({ kernel: maternKernel({ nu: 1.5, lengthScale: 1.2 }), alpha: 0.1 })
    const fitted = est.fit({ x: xs, y: ys })
    expect(fitted.predict([[0.5, 0.5]]).shape).toEqual([1, 2])
    expect(fitted.looPredictions().shape).toEqual([4, 2])
    expect(fitted.predictDistribution([[0.5, 0.5]]).mean().shape).toEqual([1, 2])
    const revived = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    const a = toFlat(fitted.predict([[0.5, 0.5]]))
    const b = toFlat(revived.predict([[0.5, 0.5]]))
    for (let i = 0; i < 2; i++) expect(b[i]).toBeCloseTo(a[i] ?? 0, 12)
    // Each output is the single-output fit on its own column.
    const first = kernelRidge.with({ kernel: maternKernel({ nu: 1.5, lengthScale: 1.2 }), alpha: 0.1 }).fit({
      x: xs,
      y: ys.map((r) => r[0] ?? 0),
    })
    expect(toFlat(first.predict([[0.5, 0.5]]))[0]).toBeCloseTo(a[0] ?? 0, 10)
  })

  it('refuses a non-positive alpha, sample weights and mismatched rows', () => {
    expect(() => kernelRidge.with({ alpha: 0 }).fit({ x, y })).toThrow()
    expect(() => kernelRidge.fit({ x, y, weights: [1, 1, 1, 1, 1] })).toThrow()
    expect(() => kernelRidge.fit({ x, y: [1, 2] })).toThrow()
    expect(() =>
      solveKernelRidge(
        [
          [1, 0],
          [0, 1],
        ],
        [1, 2, 3],
      ),
    ).toThrow()
  })

  it('has a leave-one-out shortcut equal to refitting on random data', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 4, maxLength: 6 }),
        fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 6, maxLength: 6 }),
        fc.double({ min: 0.05, max: 2, noNaN: true }),
        (xs, ys, alpha) => {
          const n = xs.length
          const rows = xs.map((v, i) => [v + i * 0.05])
          const t = ys.slice(0, n)
          const shortcut = toFlat(looPredictions(pairwiseKernels(rows, undefined, 'rbf', { gamma: 0.5 }), t, alpha))
          for (let i = 0; i < n; i++) {
            const fit = kernelRidge
              .with({ kernel: 'rbf', gamma: 0.5, alpha })
              .fit({ x: rows.filter((_, j) => j !== i), y: t.filter((_, j) => j !== i) })
            expect(shortcut[i]).toBeCloseTo(toFlat(fit.predict([rows[i] as number[]]))[0] ?? 0, 6)
          }
        },
      ),
      { numRuns: 15 },
    )
  })
})

const hyperOf = (changes: object): typeof krlsDefaults => ({ ...krlsDefaults, ...changes })

describe('kernel recursive least squares', () => {
  const circle = Array.from({ length: 40 }, (_, i) => {
    const t = (i / 39) * Math.PI * 2
    return { x: [Math.sin(t), Math.cos(t)], y: Math.sin(t) + 0.5 * Math.cos(2 * t) }
  })

  it('interpolates when every observation enters the dictionary', () => {
    const xs = [[0], [0.7], [1.5], [2.2], [3.1]]
    const ys = [0.3, -0.4, 1.1, 0.2, -0.8]
    const h = hyperOf({ kernel: 'rbf', gamma: 1, aldThreshold: 1e-12, budget: 10 })
    let s = createKrlsState(1)
    xs.forEach((row, i) => {
      s = krlsStep(s, row, ys[i] ?? 0, h)
    })
    expect(s.dictionary.length).toBe(5)
    const pred = toFlat(krlsPredict(s, xs, h))
    for (let i = 0; i < 5; i++) expect(pred[i]).toBeCloseTo(ys[i] ?? 0, 6)
  })

  it('keeps a small dictionary on a smooth function and still fits it', () => {
    const fitted = krls
      .with({ criterion: 'ald', budget: 15, aldThreshold: 0.02, kernel: 'rbf', gamma: 0.8 })
      .fit({ x: circle.map((r) => r.x), y: circle.map((r) => r.y) })
    expect(fitted.dictionarySize).toBeGreaterThan(2)
    expect(fitted.dictionarySize).toBeLessThan(40)
    expect(fitted.score({ x: circle.map((r) => r.x), y: circle.map((r) => r.y) })).toBeGreaterThan(0.9)
  })

  it('has a sliding window that is the exact ridge solution on the last observations', () => {
    const xs = [[0], [0.5], [1.0], [1.5], [2.0], [2.5]]
    const ys = [0.1, 0.6, 1.1, 0.9, 0.2, -0.4]
    const sliding = krls
      .with({ criterion: 'sliding-window', budget: 4, regularisation: 0.05, kernel: 'rbf', gamma: 0.7 })
      .fit({ x: xs, y: ys })
    const batch = kernelRidge.with({ kernel: 'rbf', gamma: 0.7, alpha: 0.05 }).fit({ x: xs.slice(-4), y: ys.slice(-4) })
    expect(sliding.dictionarySize).toBe(4)
    const query = [[0.8], [1.8], [2.2]]
    const a = toFlat(sliding.predict(query))
    const b = toFlat(batch.predict(query))
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 9)
  })

  it('has a coherence rule that stays within the budget and skips near duplicates', () => {
    const xs = [[0], [0.01], [1.0], [1.01], [2.0], [2.01]]
    const ys = [0, 0.01, 1.0, 1.01, 0.0, 0.01]
    const fitted = krls
      .with({ criterion: 'coherence', coherence: 0.9, budget: 3, kernel: 'rbf', gamma: 1 })
      .fit({ x: xs, y: ys })
    expect(fitted.dictionarySize).toBeLessThanOrEqual(3)
    expect(fitted.dictionarySize).toBeGreaterThanOrEqual(2)
  })

  it('runs as an algorithm with a serialisable state and a session that reaches every step', () => {
    const data = { x: circle.map((r) => r.x), y: circle.map((r) => r.y) }
    const h = hyperOf({ aldThreshold: 0.05, budget: 12, gamma: 0.8 })
    const alg = krlsAlgorithm(data, h)
    const result = runAlgorithm(alg, { maxSteps: 100 })
    expect(result.steps).toBe(40)
    expect(result.status).toBe('converged')
    const round = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(result.state))))
    expect(round).toEqual(result.state)
    const session = krls.with({ aldThreshold: 0.05, budget: 12, gamma: 0.8 }).session(data)
    session.advance({ steps: 10 })
    expect(session.fittedAt(5).params.nObservations).toBe(5)
    expect(session.fittedAt(5).dictionarySize).toBeLessThanOrEqual(5)
  })

  it('refuses a wrong feature count and invalid settings', () => {
    expect(() => krlsStep(createKrlsState(2), [1], 0, krlsDefaults)).toThrow()
    expect(() => krlsStep(createKrlsState(1), [1], 0, hyperOf({ budget: 0 }))).toThrow()
    expect(() => createKrlsState(0)).toThrow()
  })

  it('accepts a Kernel object', () => {
    const fitted = krls.with({ kernel: linearKernel(), aldThreshold: 1e-6 }).fit({ x: [[1], [2], [3]], y: [2, 4, 6] })
    expect(fitted.dictionarySize).toBe(1)
    expect(toFlat(fitted.predict([[5]]))[0]).toBeCloseTo(10, 8)
  })
})

describe('NORMA', () => {
  it('fits a smooth function online and loses accuracy when the expansion is truncated hard', () => {
    const xs = Array.from({ length: 30 }, (_, i) => [(i / 29) * 2 * Math.PI])
    const ys = xs.map((r) => Math.sin(r[0] ?? 0))
    const base = { kernel: 'rbf', gamma: 1, eta: 0.5, lambda: 1e-3, epochs: 20 } as const
    const wide = norma.with({ ...base, budget: 600 }).fit({ x: xs, y: ys })
    const narrow = norma.with({ ...base, budget: 3 }).fit({ x: xs, y: ys })
    expect(wide.score({ x: xs, y: ys })).toBeGreaterThan(0.9)
    expect(narrow.dictionarySize).toBe(3)
    expect(narrow.score({ x: xs, y: ys })).toBeLessThan(wide.score({ x: xs, y: ys }))
  })

  it('adds no centre while the error is inside the epsilon tube', () => {
    const alg = normaAlgorithm(
      { x: [[0], [1], [2]], y: [0.05, -0.05, 0.02] },
      { ...normaDefaults, loss: 'epsilon-insensitive', epsilon: 0.1 },
    )
    const result = runAlgorithm(alg, { maxSteps: 10 })
    expect(result.state.centres.length).toBe(0)
    const tight = normaAlgorithm(
      { x: [[0], [1], [2]], y: [1, -1, 2] },
      { ...normaDefaults, loss: 'epsilon-insensitive', epsilon: 0.1 },
    )
    expect(runAlgorithm(tight, { maxSteps: 10 }).state.centres.length).toBeGreaterThan(0)
  })

  it('decays the old coefficients by 1 - eta_t lambda each step', () => {
    const alg = normaAlgorithm(
      { x: [[0], [5]], y: [1, 0] },
      { ...normaDefaults, kernel: 'rbf', gamma: 1, eta: 1, lambda: 0.5 },
    )
    const one = runAlgorithm(alg, { maxSteps: 1 }).state
    // Step 1: f = 0, error = -1, coefficient eta_1 * 1 = 1.
    expect(one.coef[0]).toBeCloseTo(1, 12)
    const two = runAlgorithm(alg, { maxSteps: 2 }).state
    // Step 2: eta_2 = 1 / sqrt(2); the first coefficient is shrunk by 1 - eta_2 * 0.5.
    expect(two.coef[0]).toBeCloseTo(1 - 0.5 / Math.sqrt(2), 12)
  })
})
