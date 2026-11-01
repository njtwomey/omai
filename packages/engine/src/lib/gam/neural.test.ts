/**
 * Tests for the additive networks: against `gam` on planted additive data (same objective, the same fit), neural
 * additive models against the planted components, learnable against fixed knots on a function with a jump, the
 * interaction terms, the distributional and classification variants, and the session's paths.
 */

import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { additiveInteraction, additiveModel, heteroscedastic, moons, stepCurve } from '../datasets/index.ts'
import { interactionTerm, netTerm, splineTerm } from '../nn/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { Log, metricSeries } from '../trace/index.ts'
import { gam } from './estimators.ts'
import { additiveClassifier, additiveDistribution, additiveRegressor } from './neural.ts'
import { s } from './terms.ts'

const correlation = (a: readonly number[], b: readonly number[]): number => {
  const mean = (v: readonly number[]): number => v.reduce((x, y) => x + y, 0) / v.length
  const [ma, mb] = [mean(a), mean(b)]
  let sab = 0
  let saa = 0
  let sbb = 0
  a.forEach((v, i) => {
    sab += (v - ma) * ((b[i] as number) - mb)
    saa += (v - ma) ** 2
    sbb += ((b[i] as number) - mb) ** 2
  })
  return sab / Math.sqrt(saa * sbb)
}
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)
const column = (flat: readonly number[], d: number, j: number): number[] => flat.filter((_, i) => i % d === j)
const centred = (v: readonly number[]): number[] => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return v.map((u) => u - m)
}
const cosine = (steps: number) => ({ kind: 'cosine', steps, endRatio: 0.001 })
const predict = (fit: { predict?: (x: never) => Tensor }, x: unknown): number[] =>
  toFlat((fit.predict as (x: unknown) => Tensor)(x))

describe('against lib/gam on planted additive data', () => {
  const g = generate(additiveModel, { knobs: { n: 150, nFeatures: 2, nInformative: 2, noise: 0.3 }, key: key(1) })
  const data = { x: g.data.x, y: g.data.y }
  const lambda = 5
  const reference = gam
    .with({ terms: [s(0, { basis: 'ps', k: 8 }), s(1, { basis: 'ps', k: 8 })], method: 'fixed', sp: [lambda, lambda] })
    .fit(data)
  const ours = additiveRegressor
    .with({
      interiorKnots: 4,
      boundary: 'extended',
      lambda,
      epochs: 800,
      learningRate: 0.05,
      schedule: cosine(800),
    })
    .fit(data, { key: key(2) })

  it('a fixed-knot P-spline model trained to convergence has the penalised fit of gam, fitted values to 1e-6', () => {
    const want = toFlat(reference.predict(data.x))
    const got = predict(ours, data.x)
    // Measured 1.3e-9 at 800 steps.
    expect(maxAbs(got, want)).toBeLessThan(1e-6)
  })

  it('has the term curves of gam (each centred) to 1e-6, and the effective degrees of freedom', () => {
    const wantTerms = toFlat(reference.predictTerms(data.x))
    const gotTerms = toFlat(ours.predictTerms(data.x))
    for (let j = 0; j < 2; j++) {
      const want = centred(column(wantTerms, 2, j))
      const got = centred(column(gotTerms, 2, j))
      expect(maxAbs(got, want)).toBeLessThan(1e-6)
    }
    expect(ours.edfTerms['s(x0)']).toBeCloseTo(reference.edfTerms['s(x0)'] as number, 5)
    expect(ours.edfTerms['s(x1)']).toBeCloseTo(reference.edfTerms['s(x1)'] as number, 5)
  })

  it('recovers the planted components: the fitted curves correlate with the truth above 0.9 (measured 0.94 and above)', () => {
    const comps = g.truth.extra?.['components'] as { f: (v: Value) => Value }[]
    const terms = toFlat(ours.predictTerms(data.x))
    const rs = [0, 1].map((j) =>
      correlation(column(terms, 2, j), toFlat((comps[j] as { f: (v: Value) => Value }).f(g.data.x))),
    )
    expect(Math.min(...rs)).toBeGreaterThan(0.9)
  })
})

describe('neural additive models', { timeout: 30000 }, () => {
  const g = generate(additiveModel, { knobs: { n: 200, nFeatures: 3, nInformative: 3, noise: 0.3 }, key: key(1) })
  const data = { x: g.data.x, y: g.data.y }
  const comps = g.truth.extra?.['components'] as { f: (v: Value) => Value }[]
  const recovered = (unit: 'relu' | 'exu', width: number, depth: number, lr: number, initMean?: number): number[] => {
    const fit = additiveRegressor
      .with({
        terms: [0, 1, 2].map((j) =>
          netTerm(j, { width, depth, unit, outputPenalty: 0.01, ...(initMean === undefined ? {} : { initMean }) }),
        ),
        epochs: 300,
        learningRate: lr,
      })
      .fit(data, { key: key(2) })
    expect(fit.score?.(data)).toBeGreaterThan(0.9)
    const terms = toFlat(fit.predictTerms(data.x))
    return [0, 1, 2].map((j) =>
      correlation(column(terms, 3, j), toFlat((comps[j] as { f: (v: Value) => Value }).f(g.data.x))),
    )
  }

  it('feature nets with ReLU units recover the planted terms (correlations above 0.95; measured 0.998, 0.999, 0.975)', () => {
    expect(Math.min(...recovered('relu', 8, 2, 0.02))).toBeGreaterThan(0.95)
  })

  it('feature nets with ExU units recover them too (correlations above 0.8; measured 0.971, 0.999, 0.843)', () => {
    expect(Math.min(...recovered('exu', 16, 1, 0.01, 0.5))).toBeGreaterThan(0.8)
  })
})

describe('learnable knots', { timeout: 30000 }, () => {
  it('reduce the training loss on a function with a jump against evenly spaced knots, on two data seeds', () => {
    let better = 0
    const ratios: number[] = []
    for (const seed of [3, 4]) {
      const g = generate(stepCurve, { knobs: { n: 150, noise: 0.05, steps: 1 }, key: key(seed) })
      const data = { x: g.data.x, y: g.data.y }
      const final = (knots: { kind: string }): number => {
        const fit = additiveRegressor
          .with({ knots, interiorKnots: 6, epochs: 300, learningRate: 0.05, schedule: cosine(300) })
          .fit(data, { key: key(2) })
        return metricSeries(fit.run, 'train/loss').values.at(-1) as number
      }
      const fixed = final({ kind: 'uniform' })
      const free = final({ kind: 'learnable' })
      ratios.push(free / fixed)
      if (free < fixed) better++
    }
    expect(better).toBe(2)
    // Measured ratios about 0.1.
    expect(ratios.reduce((a, b) => a + b, 0) / 2).toBeLessThan(0.5)
  })

  it('move towards the jump, and the session can probe the knots at any step', () => {
    const g = generate(stepCurve, { knobs: { n: 150, noise: 0.05, steps: 1 }, key: key(3) })
    const data = { x: g.data.x, y: g.data.y }
    const est = additiveRegressor.with({
      knots: { kind: 'learnable' },
      interiorKnots: 6,
      epochs: 400,
      learningRate: 0.05,
      schedule: cosine(400),
    })
    const session = est.session(data, { key: key(2) })
    session.advance()
    const knotsAt = (t: number): number[] =>
      (session.probe(t, 'additive/knots') as { terms: { knots: number[] }[] }[])[0]?.terms[0]?.knots ?? []
    const start = knotsAt(0)
    const end = knotsAt(400)
    // Evenly spaced to begin with.
    const gaps = start.slice(1).map((v, i) => v - (start[i] as number))
    expect(
      maxAbs(
        gaps,
        gaps.map(() => gaps[0] as number),
      ),
    ).toBeLessThan(1e-9)
    const nearest = (ks: number[]): number => Math.min(...ks.map((k) => Math.abs(k - 0.5)))
    expect(nearest(end)).toBeLessThan(0.03)
    expect(nearest(end)).toBeLessThan(nearest(start))
  })
})

describe('interactions', { timeout: 30000 }, () => {
  const g = generate(additiveInteraction, { knobs: { n: 250, strength: 3, noise: 0.2 }, key: key(6) })
  const data = { x: g.data.x, y: g.data.y }
  const base = additiveRegressor.with({ epochs: 250, learningRate: 0.05, interiorKnots: 5, schedule: cosine(250) })
  const mains = [0, 1, 2].map((j) => splineTerm(j, { interiorKnots: 5 }))

  it('a tensor-product term (strong heredity) and a pair net both capture the planted interaction that main effects miss (R2 about 0.87 without)', () => {
    const without = base.with({ terms: mains }).fit(data, { key: key(2) })
    const spline = base
      .with({ terms: [...mains, interactionTerm([0, 1], { interiorKnots: 4 })], heredity: 'strong' })
      .fit(data, { key: key(2) })
    const net = base
      .with({
        terms: [...mains, interactionTerm([0, 1], { form: 'net', width: 8, depth: 2 })],
        heredity: 'strong',
        epochs: 300,
        learningRate: 0.02,
        schedule: cosine(300),
      })
      .fit(data, { key: key(2) })
    const base0 = without.score?.(data) ?? 0
    expect(spline.score?.(data)).toBeGreaterThan(base0 + 0.08)
    expect(spline.score?.(data)).toBeGreaterThan(0.95)
    expect(net.score?.(data)).toBeGreaterThan(base0 + 0.05)
    // The terms are only identified together (an unpenalised tensor product can absorb a main effect), so the planted
    // surface f0(x0) + f1(x1) + interaction(x0, x1) is compared with the sum of the three fitted terms.
    const truth = g.truth.extra ?? {}
    const parts = [
      ...(truth['components'] as { feature: number; f: (v: Value) => Value }[]).slice(0, 2),
      ...(truth['interactions'] as { f: (v: Value) => Value }[]),
    ]
    const planted = parts.map((p) => toFlat(p.f(g.data.x))).reduce((a, b) => a.map((v, i) => v + (b[i] as number)))
    const terms = toFlat(spline.predictTerms(data.x))
    const fitted = planted.map(
      (_, i) => (terms[i * 4] as number) + (terms[i * 4 + 1] as number) + (terms[i * 4 + 3] as number),
    )
    expect(correlation(fitted, planted)).toBeGreaterThan(0.99)
    expect(spline.termLabels[0]).toEqual(['s(x0)', 's(x1)', 's(x2)', 'te(x0,x1)'])
  })
})

describe('distributional and classification variants', () => {
  it('a normal GAMLSS network recovers a standard deviation that grows with x', () => {
    const g = generate(heteroscedastic, { knobs: { n: 300 }, key: key(4) })
    const data = { x: g.data.x, y: g.data.y }
    const fit = additiveDistribution
      .with({ epochs: 500, learningRate: 0.03, interiorKnots: 4, lambda: 1 })
      .fit(data, { key: key(2) })
    const parameters = toFlat(fit.predictParameters?.([[0.1], [0.5], [0.9]]) as Tensor)
    const sigma = [parameters[1], parameters[3], parameters[5]] as number[]
    // The truth is 0.1 + 0.5 x.
    expect(maxAbs(sigma, [0.15, 0.35, 0.55])).toBeLessThan(0.1)
    expect(sigma[0] as number).toBeLessThan(sigma[1] as number)
    expect(sigma[1] as number).toBeLessThan(sigma[2] as number)
    const distribution = fit.predictDistribution?.([[0.5]])
    expect(toFlat(distribution?.stddev() as Tensor)[0]).toBeCloseTo(sigma[1] as number, 10)
    expect(fit.termLabels).toHaveLength(2)
  })

  it('an additive classifier separates the moons, with class probabilities that sum to one', () => {
    const g = generate(moons, { knobs: { n: 200, noise: 0.15 }, key: key(5) })
    const data = { x: g.data.x, y: g.data.y }
    const fit = additiveClassifier
      .with({ epochs: 300, learningRate: 0.05, interiorKnots: 5 })
      .fit(data, { key: key(2) })
    expect(fit.score?.(data)).toBeGreaterThan(0.9)
    expect(fit.classes).toEqual([0, 1])
    const p = toFlat(fit.predictProba?.(data.x) as Tensor)
    expect((p[0] as number) + (p[1] as number)).toBeCloseTo(1, 12)
  })
})

describe('the session', () => {
  const g = generate(additiveModel, { knobs: { n: 120, nFeatures: 2, nInformative: 2, noise: 0.3 }, key: key(1) })
  const data = { x: g.data.x, y: g.data.y }
  const est = additiveRegressor.with({ interiorKnots: 5, lambda: 2, epochs: 200, learningRate: 0.05 })

  it('logs the penalty and the shape functions at the steps asked for, and can probe any step', () => {
    const session = est.session(data, {
      key: key(2),
      logs: {
        'train/loss': Log(),
        'additive/penalty': Log({ every: 50 }),
        'predictor/0/term/0/curve': Log({ every: 100 }),
      },
    })
    session.advance()
    expect(session.status).toBe('completed')
    const penalty = metricSeries(session.run(), 'additive/penalty')
    expect(penalty.steps).toEqual([0, 50, 100, 150, 200])
    expect(Math.min(...penalty.values)).toBeGreaterThan(0)
    const early = session.probe(10, 'predictor/0/term/0/curve') as { grid: number[]; values: number[] }
    const late = session.probe(200, 'predictor/0/term/0/curve') as { grid: number[]; values: number[] }
    expect(early.grid).toHaveLength(60)
    expect(maxAbs(early.values, late.values)).toBeGreaterThan(0.1)
    const shapes = session.probe(200, 'additive/shape-functions') as {
      kind: string
      method: string
      curves: unknown[]
    }[]
    expect(shapes[0]?.kind).toBe('dependence')
    expect(shapes[0]?.method).toBe('shape function')
    expect(shapes[0]?.curves).toHaveLength(2)
    const edf = session.probe(200, 'additive/edf') as { terms: { label: string; edf: number }[] }[]
    expect(edf[0]?.terms.map((t) => t.label)).toEqual(['s(x0)', 's(x1)'])
  })

  it('keeps the penalty inside the loss (a larger lambda gives a larger training loss) and revives from its parameters', () => {
    const loss = (lambda: number): number =>
      metricSeries(est.with({ lambda }).fit(data, { key: key(2) }).run, 'train/loss').values.at(-1) as number
    expect(loss(50)).toBeGreaterThan(loss(0))
    const fit = est.fit(data, { key: key(2) })
    const revived = est.revive(JSON.parse(JSON.stringify(fit.params)))
    expect(maxAbs(predict(revived, data.x), predict(fit, data.x))).toBeLessThan(1e-12)
    expect(revived.termLabels).toEqual(fit.termLabels)
  })
})
