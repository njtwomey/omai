import { describe, expect, it } from 'vitest'
import { gaussianNB } from '../../testing/learn.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { requireVerb } from './estimator.ts'
import { adjustPriors, blackBoxShiftEstimate, emPriorAdjustment, emPriorAdjustmentAlgorithm } from './shift.ts'

/** n rows of a one-dimensional two-class problem, class means 0 and 2, unit variance, with the given class-1 share. */
function blobs(seed: number, n: number, share: number): { x: number[][]; y: number[] } {
  const u = toFlat(uniform(key(seed), [n]))
  const z = toFlat(normal(foldIn(key(seed), 1), [n]))
  const y = Array.from(u, (v) => (v < share ? 1 : 0))
  return { x: y.map((c, i) => [c * 2 + (z[i] ?? 0)]), y }
}

describe('adjustPriors', () => {
  it('multiplies by the prior ratio and renormalises (Bayes under label shift)', () => {
    const out = toFlat(
      adjustPriors(
        [
          [0.5, 0.5],
          [0.9, 0.1],
        ],
        [0.5, 0.5],
        [0.2, 0.8],
      ),
    )
    expect(out[0]).toBeCloseTo(0.2, 12)
    expect(out[1]).toBeCloseTo(0.8, 12)
    // 0.9 * 0.2 = 0.18 and 0.1 * 0.8 = 0.08.
    expect(out[2]).toBeCloseTo(0.18 / 0.26, 12)
    expect(out[3]).toBeCloseTo(0.08 / 0.26, 12)
  })
})

describe('blackBoxShiftEstimate', () => {
  it('inverts the confusion matrix exactly when the target histogram is C w', () => {
    // Joint confusion C[yhat, y] of a three-class classifier on the source, and weights w = pi_T / pi_S.
    const yS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    const pS = [0, 0, 0, 1, 1, 1, 2, 1, 2, 2, 0, 2]
    const piS = [1 / 3, 1 / 3, 1 / 3]
    const w = [2, 0.5, 0.5]
    // The predicted-label histogram of the target, built from C w.
    const C = [
      [3, 0, 1],
      [1, 3, 0],
      [0, 1, 3],
    ].map((row) => row.map((v) => v / 12))
    const muT = C.map((row) => row.reduce((a, c, y) => a + c * (w[y] ?? 0), 0))
    const counts = muT.map((m) => Math.round(m * 1000))
    const pT = counts.flatMap((c, label) => Array.from({ length: c }, () => label))
    const r = blackBoxShiftEstimate(yS, pS, pT)
    expect(r.sourcePriors).toEqual(piS.map((v) => expect.closeTo(v, 12)))
    for (const [c, v] of r.weights.entries()) expect(v, `weight ${c}`).toBeCloseTo(w[c] ?? 0, 2)
    expect(r.targetPriors.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('estimates planted target priors of a trained classifier without target labels', () => {
    const source = blobs(1, 2000, 0.5)
    const target = blobs(2, 2000, 0.2)
    const model = gaussianNB().fit({ x: source.x.slice(0, 1000), y: source.y.slice(0, 1000) })
    const hold = source.x.slice(1000)
    const r = blackBoxShiftEstimate(source.y.slice(1000), model.predict?.(hold) ?? [], model.predict?.(target.x) ?? [])
    expect(r.targetPriors[1]).toBeGreaterThan(0.12)
    expect(r.targetPriors[1]).toBeLessThan(0.28)
    // The soft version, from probabilities, is as close.
    const soft = blackBoxShiftEstimate(
      source.y.slice(1000),
      model.predictProba?.(hold) ?? [],
      model.predictProba?.(target.x) ?? [],
    )
    expect(Math.abs((soft.targetPriors[1] ?? 0) - 0.2)).toBeLessThan(0.08)
  })
})

describe('emPriorAdjustment', () => {
  it('recovers planted target priors and its log-likelihood never falls', () => {
    const source = blobs(3, 1500, 0.5)
    const target = blobs(4, 1500, 0.15)
    const model = gaussianNB().fit(source)
    const proba = requireVerb(model, 'predictProba', 'test')(target.x)
    const alg = emPriorAdjustmentAlgorithm([0.5, 0.5], proba, { tol: 1e-10 })
    const run = runAlgorithm(alg, { maxSteps: 200 })
    expect(run.status).toBe('converged')
    const r = emPriorAdjustment([0.5, 0.5], proba)
    expect(r.converged).toBe(true)
    expect(Math.abs((r.targetPriors[1] ?? 0) - 0.15)).toBeLessThan(0.05)
    // The EM fixed point: the priors are the mean of the adjusted posteriors.
    const post = toFlat(r.posteriors)
    const rows = post.length / 2
    const mean1 = Array.from({ length: rows }, (_, i) => post[2 * i + 1] ?? 0).reduce((a, b) => a + b, 0) / rows
    expect(mean1).toBeCloseTo(r.targetPriors[1] ?? 0, 6)

    let previous = Number.NEGATIVE_INFINITY
    for (let steps = 1; steps <= 15; steps++) {
      const ll = runAlgorithm(emPriorAdjustmentAlgorithm([0.5, 0.5], proba), { maxSteps: steps }).state.logLikelihood
      expect(ll).toBeGreaterThanOrEqual(previous - 1e-12)
      previous = ll
    }

    // Adjusting the posteriors to the estimated priors improves the accuracy on the shifted target.
    const hit = (probs: ArrayLike<number>): number => {
      let c = 0
      for (let i = 0; i < target.y.length; i++) c += ((probs[2 * i + 1] ?? 0) > 0.5 ? 1 : 0) === target.y[i] ? 1 : 0
      return c / target.y.length
    }
    expect(hit(toFlat(adjustPriors(proba, r.sourcePriors, r.targetPriors)))).toBeGreaterThan(hit(toFlat(proba)))
  })
})
