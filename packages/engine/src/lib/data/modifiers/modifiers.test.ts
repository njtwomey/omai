import { describe, expect, it } from 'vitest'
import { maxOf, minOf } from '../../base/index.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import {
  add,
  exp,
  fromStorage,
  log,
  mul,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  tensor,
  toArray,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import {
  featureTransform,
  noiseTransition,
  semiSupervisedSplit,
  shiftedSplit,
  withClassImbalance,
  withConceptDrift,
  withCorrelatedFeatures,
  withCovariateShift,
  withLabelNoise,
  withLabelShift,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withPrevalence,
  withTransform,
} from '../index.ts'
import { rowCount } from '../schema/common.ts'
import type { ClassificationData, RegressionData } from '../schema/tabular.ts'
import { xy, xyClasses } from '../tensors.ts'
import { posteriorFromJoint } from '../truth.ts'

const SQRT2PI = Math.log(2 * Math.PI)

/** Two unit Gaussian classes at (-1, 0) and (+1, 0) with equal priors; exact joint and posterior. */
function gaussians(n: number, seed: number): Generated<ClassificationData> {
  const z = normal(key(seed), [n, 2]).data
  const y = Int32Array.from({ length: n }, (_, i) => i % 2)
  const x = Float64Array.from(z, (v, i) => (i % 2 === 0 ? v + ((y[Math.floor(i / 2)] ?? 0) === 1 ? 1 : -1) : v))
  const logJoint = (v: Value, labels: Value): Value => {
    const m = add(mul(labels, 2), -1)
    const d0 = sub(reshape(sliceAxis(v, 1, 0, 1), [-1]), m)
    const d1 = reshape(sliceAxis(v, 1, 1, 2), [-1])
    return add(mul(add(mul(d0, d0), mul(d1, d1)), -0.5), Math.log(0.5) - SQRT2PI)
  }
  return {
    data: xyClasses(fromStorage(x, [n, 2], 'float64'), fromStorage(y, [n], 'int32')),
    truth: { logProb: logJoint as never, posterior: posteriorFromJoint(logJoint, 2), bayesError: 0.158655 },
  }
}

/** y = 3 x0 + noise, x ~ N(0, I); the joint log density is exact. */
function linear(n: number, seed: number, d = 2): Generated<RegressionData> {
  const x = normal(key(seed), [n, d])
  const f = (v: Value): Value => mul(reshape(sliceAxis(v, 1, 0, 1), [-1]), 3)
  const noise = normal(foldIn(key(seed), 1), [n]).data
  const y = Float64Array.from(toFlat(f(x)), (v, i) => v + (noise[i] ?? 0))
  const logProb = (v: Value, t?: Value): Value => {
    const px = mul(add(sum(mul(v, v), 1), d * SQRT2PI), -0.5)
    if (t === undefined) return px
    const r = sub(t, f(v))
    return add(px, mul(add(mul(r, r), SQRT2PI), -0.5))
  }
  return { data: xy(x, y), truth: { regressionFunction: f, noiseScale: 1, bayesError: 1, logProb } }
}

const flat = (v: Value | undefined): number[] => (v === undefined ? [] : toFlat(v))
const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('label modifiers', () => {
  it('flip labels at the asked rate and transform the posterior', () => {
    const g = gaussians(4000, 0)
    const noisy = withLabelNoise(g, { rate: 0.2, key: key(1) })
    const flipped = flat(noisy.truth.latents?.['flipped'] as Tensor)
    expect(Math.abs(mean(flipped) - 0.2)).toBeLessThan(4 * Math.sqrt(0.16 / 4000))
    const p = flat(noisy.truth.posterior?.(tensor([[0, 0]])))
    expect(p[0]).toBeCloseTo(0.5, 12)
    const far = flat(noisy.truth.posterior?.(tensor([[-10, 0]])))
    expect(far[0]).toBeCloseTo(0.8, 6)
    // Bayes error of the noisy labels: 0.2 + 0.6 * Phi(-1).
    expect(Math.abs((noisy.truth.bayesError ?? 0) - (0.2 + 0.6 * 0.158655))).toBeLessThan(0.02)
    expect(toArray(noiseTransition(3, 0.3, 'pairflip'))).toEqual([
      [0.7, 0.3, 0],
      [0, 0.7, 0.3],
      [0.3, 0, 0.7],
    ])
    expect(toFlat(withLabelNoise(g, { transition: noiseTransition(2, 0.2), key: key(5) }).data.y)).toEqual(
      toFlat(
        withLabelNoise(g, {
          transition: [
            [0.8, 0.2],
            [0.2, 0.8],
          ],
          key: key(5),
        }).data.y,
      ),
    )
    expect(() => withLabelNoise(g, { transition: [[1]], key: key(0) })).toThrow(/transition/)
  })
  it('reach exact prevalence and shift the posterior by the prior ratio', () => {
    const g = gaussians(1000, 2)
    const out = withPrevalence(g, { prevalence: 0.2, key: key(3) })
    const y = flat(out.data.y)
    expect(mean(y)).toBeCloseTo(0.2, 12)
    expect(y.length).toBe(625)
    const p = flat(out.truth.posterior?.(tensor([[0, 0]])))
    expect(p[1]).toBeCloseTo(0.2, 2)
  })
  it('label shift resamples to the target priors', () => {
    const out = withLabelShift(gaussians(1000, 4), { targetPriors: [0.9, 0.1], n: 5000, key: key(5) })
    expect(Math.abs(mean(flat(out.data.y)) - 0.1)).toBeLessThan(4 * Math.sqrt(0.09 / 5000))
    const random = withLabelShift(gaussians(100, 4), { alpha: 0.5, key: key(6) })
    const priors = random.truth.extra?.['priors'] as number[]
    expect(priors.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('imbalance classes', () => {
    const out = withClassImbalance(gaussians(400, 7), { ratio: 10, key: key(8) })
    expect(flat(out.data.y).filter((c) => c === 1).length).toBe(20)
    const step = withClassImbalance(gaussians(400, 7), { ratio: 4, profile: 'step', key: key(8) })
    expect(flat(step.data.y).filter((c) => c === 1).length).toBe(50)
  })
})

describe('feature modifiers', () => {
  it('MCAR removes entries at the rate, independent of the values', () => {
    const g = linear(5000, 1)
    const out = withMissing(g, { rate: 0.3, key: key(2) })
    const mask = flat(out.truth.latents?.['missing'] as Tensor)
    expect(Math.abs(mean(mask) - 0.3)).toBeLessThan(4 * Math.sqrt(0.21 / 10000))
    const x = flat(out.data.x)
    expect(mask.every((m, i) => (m === 1) === Number.isNaN(x[i] ?? 0))).toBe(true)
  })
  it('MAR depends on the driver column and MNAR on the value itself', () => {
    const g = linear(6000, 3)
    const mar = withMissing(g, { mechanism: 'MAR', rate: 0.3, key: key(4) })
    const x = flat(g.data.x)
    const m = flat(mar.truth.latents?.['missing'] as Tensor)
    const col1 = m.filter((_, i) => i % 2 === 1)
    expect(m.filter((_, i) => i % 2 === 0).every((v) => v === 0)).toBe(true)
    expect(Math.abs(mean(col1) - 0.3)).toBeLessThan(0.03)
    // Missingness in column 1 rises with column 0 (the driver).
    const hi = col1.filter((_, i) => (x[2 * i] ?? 0) > 0)
    const lo = col1.filter((_, i) => (x[2 * i] ?? 0) <= 0)
    expect(mean(hi)).toBeGreaterThan(mean(lo) + 0.2)
    const mnar = withMissing(g, { mechanism: 'MNAR', rate: 0.3, key: key(5) })
    const mm = flat(mnar.truth.latents?.['missing'] as Tensor)
    const highOwn = mm.filter((_, i) => (x[i] ?? 0) > 0)
    const lowOwn = mm.filter((_, i) => (x[i] ?? 0) <= 0)
    expect(mean(highOwn)).toBeGreaterThan(mean(lowOwn) + 0.2)
    expect(() => withMissing(g, { mechanism: 'MAR', columns: [0, 1], key: key(0) })).toThrow(/driver/)
  })
  it('outliers contaminate the asked fraction', () => {
    const g = linear(200, 6)
    const out = withOutliers(g, { fraction: 0.1, target: 'both', key: key(7) })
    const flags = flat(out.truth.latents?.['outliers'] as Tensor)
    expect(flags.filter((v) => v === 1).length).toBe(20)
    const dy = flat(out.data.y).map((v, i) => Math.abs(v - (flat(g.data.y)[i] ?? 0)))
    expect(dy.filter((v) => v > 0).length).toBe(20)
    expect(() => withOutliers(gaussians(10, 0), { target: 'targets', key: key(0) })).toThrow(/regression/)
  })
  it('nuisance features leave the truth consistent', () => {
    const g = linear(50, 8)
    const out = withNuisanceFeatures(g, { count: 3, key: key(9) })
    expect(out.data.x.shape).toEqual([50, 5])
    expect(flat(out.truth.regressionFunction?.(out.data.x))).toEqual(flat(g.truth.regressionFunction?.(g.data.x)))
    // The joint density gains three standard normal factors.
    const extra = sliceAxis(out.data.x, 1, 2, 5)
    const expected = add(
      g.truth.logProb?.(g.data.x, g.data.y) as Value,
      mul(add(sum(mul(extra, extra), 1), 3 * SQRT2PI), -0.5),
    )
    const got = out.truth.logProb?.(out.data.x, out.data.y)
    flat(got).forEach((v, i) => {
      expect(v).toBeCloseTo(flat(expected)[i] ?? 0, 10)
    })
    const shadow = withNuisanceFeatures(g, { count: 2, kind: 'permuted', key: key(1) })
    expect(shadow.truth.logProb).toBeUndefined()
    expect(flat(sliceAxis(shadow.data.x, 1, 2, 3)).sort()).toEqual(flat(sliceAxis(g.data.x, 1, 0, 1)).sort())
  })
  it('transforms keep the density exact (change of variables)', () => {
    const g = linear(400, 10)
    const out = withTransform(g, { transform: 'exp', columns: [1] })
    expect(
      flat(out.data.x)
        .filter((_, i) => i % 2 === 1)
        .every((v) => v > 0),
    ).toBe(true)
    expect(flat(out.truth.regressionFunction?.(out.data.x))).toEqual(flat(g.truth.regressionFunction?.(g.data.x)))
    // The transformed density integrates to one along the transformed axis (trapezoid on a log grid).
    const grid = Array.from({ length: 4001 }, (_, i) => Math.exp(-8 + (16 * i) / 4000))
    const pts = tensor(grid.map((v) => [0, v]))
    const dens = flat(exp(out.truth.logProb?.(pts) as Value))
    let area = 0
    for (let i = 1; i < grid.length; i++)
      area += 0.5 * ((dens[i] ?? 0) + (dens[i - 1] ?? 0)) * ((grid[i] ?? 0) - (grid[i - 1] ?? 0))
    expect(area).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 4)
    for (const name of ['exp', 'log', 'sigmoid', 'logit', 'sinh', 'arcsinh', 'cube'] as const) {
      const t = featureTransform(name)
      const v = tensor([0.2, 0.7])
      const back = name === 'log' || name === 'logit' ? t.forward(t.inverse(v)) : t.inverse(t.forward(v))
      expect(flat(back).map((u) => Math.round(u * 1e10) / 1e10)).toEqual([0.2, 0.7])
      // logAbsDetInverse is the log derivative of the inverse, checked by central differences.
      const y = name === 'log' || name === 'arcsinh' ? tensor([0.3]) : t.forward(tensor([0.3]))
      const h = 1e-6
      const deriv = (flat(t.inverse(add(y, h)))[0] ?? 0) - (flat(t.inverse(sub(y, h)))[0] ?? 0)
      expect(flat(t.logAbsDetInverse(y))[0]).toBeCloseTo(Math.log(Math.abs(deriv / (2 * h))), 6)
    }
  })
  it('correlated features reach the target correlation with an exact density', () => {
    const g = linear(20000, 11, 3)
    for (const structure of ['ar1', 'equicorrelated'] as const) {
      const out = withCorrelatedFeatures(g, { rho: 0.6, structure })
      const x = flat(out.data.x)
      const col = (j: number) => x.filter((_, i) => i % 3 === j)
      const corr = (a: number[], b: number[]) => {
        const ma = mean(a)
        const mb = mean(b)
        const cov = mean(a.map((v, i) => (v - ma) * ((b[i] ?? 0) - mb)))
        return cov / Math.sqrt(mean(a.map((v) => (v - ma) ** 2)) * mean(b.map((v) => (v - mb) ** 2)))
      }
      expect(corr(col(0), col(1))).toBeCloseTo(0.6, 1)
      expect(corr(col(0), col(2))).toBeCloseTo(structure === 'ar1' ? 0.36 : 0.6, 1)
      // Density of the mixed rows: the multivariate normal with that correlation (checked at one point).
      const r = structure === 'ar1' ? 0.36 : 0.6
      const sigma = [
        [1, 0.6, r],
        [0.6, 1, 0.6],
        [r, 0.6, 1],
      ]
      const det = (sigma[0]?.[0] ?? 0) * (1 - 0.36) - 0.6 * (0.6 - 0.6 * r) + r * (0.36 - r)
      const lp = flat(out.truth.logProb?.(tensor([[0, 0, 0]])))[0] ?? 0
      expect(lp).toBeCloseTo(-1.5 * SQRT2PI - 0.5 * Math.log(det), 10)
    }
    expect(() => withCorrelatedFeatures(g, { rho: 1 })).toThrow(/rho/)
  })
})

describe('shift modifiers', () => {
  it('covariate shift tilts p(x) and keeps p(y | x)', () => {
    const g = linear(4000, 12)
    const out = withCovariateShift(g, { strength: 1, key: key(13) })
    const x0 = flat(out.data.x).filter((_, i) => i % 2 === 0)
    // Tilting N(0, 1) by exp(z) gives N(1, 1).
    expect(mean(x0)).toBeCloseTo(1, 1)
    expect(out.truth.regressionFunction).toBe(g.truth.regressionFunction)
    const ratio = out.truth.extra?.['densityRatio'] as (v: Value) => Value
    expect(flat(ratio(tensor([[1, 0]])))[0] ?? 0).toBeGreaterThan(flat(ratio(tensor([[0, 0]])))[0] ?? 0)
    const lp = out.truth.logProb?.(tensor([[0, 0]])) as Value
    expect(Number.isFinite(flat(log(exp(lp)))[0])).toBe(true)
  })
  it('concept drift changes p(y | x) after the drift point', () => {
    const g = gaussians(100, 14)
    const out = withConceptDrift(g, { at: [60], key: key(0) })
    expect(out.truth.changepoints).toEqual([60])
    const before = flat(out.data.y).slice(0, 60)
    const after = flat(out.data.y).slice(60)
    expect(before).toEqual(flat(g.data.y).slice(0, 60))
    expect(after).toEqual(
      flat(g.data.y)
        .slice(60)
        .map((c) => 1 - c),
    )
    const r = withConceptDrift(linear(10, 1), { at: [0.5], kind: 'recurring', key: key(0) })
    expect(flat(r.truth.latents?.['concepts'] as Tensor)).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    const gradual = withConceptDrift(gaussians(1000, 1), { at: [500], kind: 'gradual', width: 200, key: key(2) })
    const c = flat(gradual.truth.latents?.['concepts'] as Tensor)
    expect(mean(c.slice(0, 400))).toBe(0)
    expect(mean(c.slice(600))).toBe(1)
  })
  it('shifted and semi-supervised splits', () => {
    const g = linear(100, 15)
    const s = shiftedSplit(g.data, { trainFraction: 0.7 })
    expect(rowCount(s.train)).toBe(70)
    const maxTrain = maxOf(flat(s.train.x).filter((_, i) => i % 2 === 0))
    const minTest = minOf(flat(s.test.x).filter((_, i) => i % 2 === 0))
    expect(maxTrain).toBeLessThanOrEqual(minTest)
    const soft = shiftedSplit(g.data, { mode: 'soft', key: key(1) })
    expect(rowCount(soft.train) + rowCount(soft.test)).toBe(100)
    const semi = semiSupervisedSplit(gaussians(100, 2).data, { labelled: 0.1, key: key(3) })
    expect(rowCount(semi.labelled)).toBe(10)
    expect(flat(semi.labelled.y).filter((c) => c === 1).length).toBe(5)
    expect(flat(semi.unlabelled.y).every((c) => c === -1)).toBe(true)
    const reg = semiSupervisedSplit(g.data, { labelled: 5, key: key(4) })
    expect(flat(reg.unlabelled.y).every(Number.isNaN)).toBe(true)
    expect(rowCount(reg.labelled)).toBe(5)
  })
})
