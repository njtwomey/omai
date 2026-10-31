import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { treatmentEffects } from '../datasets/index.ts'
import { randomForestRegressor } from '../ensembles/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { solveSmooth } from './_solve.ts'
import {
  aipw,
  balanceTable,
  bootstrapEffect,
  doubleMachineLearning,
  doubleMachineLearningAlgorithm,
  effectiveSampleSize,
  ipw,
  ipwWeights,
  overlap,
  propensityScores,
  regressionAdjustment,
  splitTreatment,
  stratification,
  tmle,
  tmleFluctuationProblem,
} from './index.ts'

const study = (knobs: Record<string, number | string>, seed = 1) => {
  const g = generate(treatmentEffects, { key: key(seed), knobs: { n: 4000, ...knobs } })
  const data = splitTreatment(g.data)
  const truth = g.truth.extra?.['sampleAte'] as number
  return { data, truth, g }
}

const naive = (t: ArrayLike<number>, y: ArrayLike<number>): number => {
  let a = 0
  let b = 0
  let n1 = 0
  let n0 = 0
  for (let i = 0; i < t.length; i++) {
    if (t[i] === 1) {
      a += y[i] ?? 0
      n1++
    } else {
      b += y[i] ?? 0
      n0++
    }
  }
  return a / n1 - b / n0
}

describe('estimators recover the planted effect where the naive difference does not', () => {
  const { data, truth } = study({ ate: 2, confounding: 1.5, heterogeneity: 0.5 })
  const t = toFlat(data.treatment as never)
  const y = toFlat(data.outcome as never)

  it('the naive difference of means is biased by the confounding', () => {
    expect(Math.abs(naive(t, y) - truth)).toBeGreaterThan(0.4)
  })

  it('regression adjustment (linear, interacted, g-computation) recovers the ATE', () => {
    for (const mode of ['linear', 'interacted'] as const) {
      const r = regressionAdjustment(data, { mode })
      expect(Math.abs(r.estimate.value - truth)).toBeLessThan(0.15)
      expect(r.estimate.interval?.[0]).toBeLessThan(r.estimate.interval?.[1] ?? 0)
    }
    const g = regressionAdjustment(data, { mode: 'g-computation', bootstrap: 10 })
    expect(Math.abs(g.estimate.value - truth)).toBeLessThan(0.15)
  })

  it('inverse probability weighting recovers the ATE and balances the covariates', () => {
    const r = ipw(data)
    expect(Math.abs(r.estimate.value - truth)).toBeLessThan(0.3)
    const before = balanceTable(data.x, data.treatment)
    const after = balanceTable(data.x, data.treatment, r.weights)
    expect(Math.abs(before[0]?.before ?? 0)).toBeGreaterThan(0.3)
    expect(Math.abs(after[0]?.after ?? 1)).toBeLessThan(0.1)
    expect(r.ess).toBeGreaterThan(500)
    expect(r.ess).toBeLessThan(4000)
  })

  it('stratification, AIPW, TMLE and double machine learning recover it too', () => {
    expect(Math.abs(stratification(data, { strata: 10 }).estimate.value - truth)).toBeLessThan(0.35)
    const a = aipw(data)
    expect(Math.abs(a.estimate.value - truth)).toBeLessThan(0.15)
    expect(Math.abs(aipw(data, { folds: 4 }).estimate.value - truth)).toBeLessThan(0.15)
    const m = tmle(data)
    expect(Math.abs(m.estimate.value - truth)).toBeLessThan(0.15)
    const d = doubleMachineLearning(data, { folds: 4 })
    expect(Math.abs(d.estimate.value - truth)).toBeLessThan(0.15)
    const irm = doubleMachineLearning(data, { folds: 4, model: 'irm' })
    expect(Math.abs(irm.estimate.value - truth)).toBeLessThan(0.15)
  })

  it('the ATT is recovered by the weighted, doubly robust and regression estimators', () => {
    const att = (study({ ate: 2, confounding: 1.5, heterogeneity: 1 }).g.truth.effects?.att ?? 0) as number
    const s = study({ ate: 2, confounding: 1.5, heterogeneity: 1 })
    for (const r of [
      ipw(s.data, { estimand: 'att' }),
      aipw(s.data, { estimand: 'att' }),
      regressionAdjustment(s.data, { mode: 'interacted', estimand: 'att' }),
    ])
      expect(Math.abs(r.estimate.value - att)).toBeLessThan(0.3)
  })
})

describe('double robustness', () => {
  // The nonlinear outcome breaks the linear outcome model; the propensity stays logistic.
  const { data, truth } = study({ ate: 2, confounding: 1.5, setting: 'nonlinear', heterogeneity: 0.5 }, 3)
  it('linear regression adjustment is biased, AIPW and TMLE are not', () => {
    const ra = regressionAdjustment(data, { mode: 'linear' })
    const dr = aipw(data)
    const t = tmle(data)
    expect(Math.abs(dr.estimate.value - truth)).toBeLessThan(0.25)
    expect(Math.abs(t.estimate.value - truth)).toBeLessThan(0.25)
    expect(Math.abs(ra.estimate.value - truth)).toBeGreaterThan(Math.abs(dr.estimate.value - truth))
  })

  it('a flexible outcome learner (a forest) fixes the outcome model and cross-fitting keeps the estimate honest', () => {
    const r = aipw(data, { outcome: randomForestRegressor({ nEstimators: 30, minSamplesLeaf: 5 }), folds: 3 })
    expect(Math.abs(r.estimate.value - truth)).toBeLessThan(0.3)
  })
})

describe('propensity tools', () => {
  it('weights follow the estimands and stabilised weights average about one', () => {
    expect(Array.from(ipwWeights([1, 0, 0], [0.5, 0.5, 0.25], { estimand: 'att' }))).toEqual([1, 1, 1 / 3])
    expect(Array.from(ipwWeights([1, 0], [0.5, 0.25], { estimand: 'atc' }))).toEqual([1, 1])
    const { data } = study({ confounding: 1.5 })
    const e = propensityScores(data)
    const t = Array.from(toFlat(data.treatment as never))
    const sw = ipwWeights(t, e, { stabilised: true })
    expect(sw.reduce((a, b) => a + b, 0) / sw.length).toBeGreaterThan(0.9)
    expect(sw.reduce((a, b) => a + b, 0) / sw.length).toBeLessThan(1.1)
    const o = overlap(e, t)
    expect(o.share).toBeGreaterThan(0.8)
    expect(effectiveSampleSize(ipwWeights(t, e))).toBeLessThan(t.length)
  })

  it('trimming removes units with extreme scores and a known score gives the analytic Hajek estimate', () => {
    const x = [[0], [0], [0], [0], [1], [1], [1], [1]]
    const r = ipw(
      { x, treatment: [1, 0, 0, 0, 1, 1, 1, 0], outcome: [3, 1, 1, 1, 5, 5, 5, 3] },
      { scores: [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75] },
    )
    expect(r.estimate.value).toBeCloseTo(2, 12)
    const trimmed = ipw(
      { x, treatment: [1, 0, 0, 0, 1, 1, 1, 0], outcome: [3, 1, 1, 1, 5, 5, 5, 3] },
      { scores: [0.02, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75], trim: 0.1 },
    )
    expect(trimmed.kept).toBe(7)
    expect(trimmed.weights[0]).toBe(0)
  })
})

describe('cross-fitting and the solver slot', () => {
  const { data } = study({ confounding: 1 })
  it('double machine learning takes one fold per step, converges, and its state serialises', () => {
    const alg = doubleMachineLearningAlgorithm(data, { folds: 4 })
    const run = runAlgorithm(alg, { maxSteps: 10, key: key(0) })
    expect(run.steps).toBe(4)
    expect(run.status).toBe('converged')
    expect(run.state.filled.length).toBe(4000)
    const back = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(back?.estimate).toBeCloseTo(run.state.estimate, 12)
  })

  it('the TMLE fluctuation reaches the same epsilon under every solver of the slot', () => {
    const n = 200
    const h = Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1.8 : -1.6))
    const offset = Array.from({ length: n }, (_, i) => 0.3 * Math.sin(i))
    const y = Array.from({ length: n }, (_, i) => 0.5 + 0.2 * Math.cos(i) + (i % 2 === 0 ? 0.1 : -0.05))
    const problem = tmleFluctuationProblem({ h, offset, y })
    const eps = ['newton', 'lbfgs', 'bfgs', 'conjugateGradient'].map(
      (kind) =>
        solveSmooth(
          problem,
          smoothSolverConfig({ kind, gradTol: 1e-9, ...(kind === 'lbfgs' ? { ftol: 1e-16 } : {}) }),
          key(0),
          500,
        ).x[0] ?? 0,
    )
    for (const e of eps) expect(e).toBeCloseTo(eps[0] ?? 0, 5)
    const m = tmle(data, { solver: smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-9, ftol: 1e-16 }) })
    expect(m.epsilon).toBeCloseTo(tmle(data).epsilon, 4)
  })

  it('the bootstrap spread of the naive difference is a plausible standard error', () => {
    const e = bootstrapEffect(data, (d) => naive(toFlat(d.treatment as never), toFlat(d.outcome as never)), {
      resamples: 30,
    })
    expect(e.standardError).toBeGreaterThan(0.01)
    expect(e.standardError).toBeLessThan(0.2)
  })
})

describe('standard errors', () => {
  it('the AIPW and TMLE standard errors match the spread of their errors over replications', () => {
    for (const estimator of [aipw, tmle]) {
      const errors: number[] = []
      const ses: number[] = []
      for (let s = 0; s < 24; s++) {
        const { data, truth } = study({ n: 800, ate: 1, confounding: 1, heterogeneity: 0 }, 200 + s)
        const r = estimator(data)
        errors.push(r.estimate.value - truth)
        ses.push(r.estimate.standardError ?? 0)
      }
      const m = errors.reduce((a, b) => a + b, 0) / errors.length
      const sd = Math.sqrt(errors.reduce((a, b) => a + (b - m) ** 2, 0) / (errors.length - 1))
      const se = ses.reduce((a, b) => a + b, 0) / ses.length
      expect(se / sd).toBeGreaterThan(0.6)
      expect(se / sd).toBeLessThan(1.6)
    }
  })
})
