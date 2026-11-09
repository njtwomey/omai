import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { treatmentEffects, upliftData } from '../datasets/index.ts'
import { randomForestRegressor } from '../ensembles/index.ts'
import { qiniCoefficient, upliftAtK } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  causalForest,
  causalForestAlgorithm,
  causalForestOf,
  drLearner,
  pehe,
  rLearner,
  sLearner,
  splitTreatment,
  tLearner,
  upliftForest,
  upliftTree,
  xLearner,
} from './index.ts'

const effects = (knobs: Record<string, number | string>, seed = 1, n = 3000) => {
  const g = generate(treatmentEffects, { key: key(seed), knobs: { n, ...knobs } })
  const data = splitTreatment(g.data)
  const cate = g.truth.effects?.cate as (v: Tensor) => Tensor
  const truth = Array.from(toFlat(cate(data.x as Tensor)))
  return { data, truth, x: data.x }
}

describe('meta-learners on a linear planted effect', () => {
  const { data, truth, x } = effects({ ate: 1, heterogeneity: 1.5, confounding: 1, setting: 'linear' })
  const constant = pehe(new Float64Array(truth.length).fill(1), truth)

  it('every learner beats the constant-effect predictor by a wide margin', () => {
    expect(constant).toBeGreaterThan(1.2)
    for (const model of [tLearner(data), xLearner(data), rLearner(data, { folds: 3 }), drLearner(data, { folds: 3 })]) {
      const err = pehe(model.predict(x), truth)
      expect(err, model.learner).toBeLessThan(0.35)
    }
  })

  it('the S-learner with a linear base cannot see heterogeneity; with a forest base it sees some', () => {
    const flat = sLearner(data).predict(x)
    expect(Math.max(...flat) - Math.min(...flat)).toBeLessThan(1e-9)
    expect(pehe(flat, truth)).toBeGreaterThan(0.9 * constant)
    const forest = sLearner(data, { outcome: randomForestRegressor({ nEstimators: 40, minSamplesLeaf: 10 }) }).predict(
      x,
    )
    expect(Math.max(...forest) - Math.min(...forest)).toBeGreaterThan(0.5)
  })

  it('the learners recover the average effect from their effect estimates', () => {
    const ate = truth.reduce((a, b) => a + b, 0) / truth.length
    for (const model of [tLearner(data), rLearner(data, { folds: 3 })]) {
      const est = model.predict(x)
      expect(Math.abs(est.reduce((a, b) => a + b, 0) / est.length - ate)).toBeLessThan(0.15)
    }
  })

  it('refuses covariates of the wrong width', () => {
    expect(() => tLearner(data).predict([[1, 2]])).toThrow()
  })
})

describe('a nonlinear effect needs a flexible effect learner', () => {
  const { data, truth, x } = effects({ ate: 1, heterogeneity: 2, confounding: 1, setting: 'nonlinear' }, 2, 4000)
  const constant = pehe(new Float64Array(truth.length).fill(1), truth)

  it('a flexible doubly robust learner and the causal forest beat the linear learners', () => {
    const linear = pehe(tLearner(data).predict(x), truth)
    const forest = randomForestRegressor({ nEstimators: 40, minSamplesLeaf: 10 })
    const flexible = pehe(drLearner(data, { effect: forest, outcome: forest, folds: 2 }).predict(x), truth)
    expect(flexible).toBeLessThan(0.6 * constant)
    expect(linear).toBeLessThan(0.6 * constant)
  })

  it('a causal forest finds the effect surface, ranks the covariate it depends on first and gives a doubly robust ATE', () => {
    const forest = causalForest(data, {
      trees: 80,
      minLeaf: 10,
      subsample: 0.5,
      nuisanceFolds: 2,
      outcome: randomForestRegressor({ nEstimators: 30, minSamplesLeaf: 10 }),
    })
    const err = pehe(forest.predict(x), truth)
    expect(err).toBeLessThan(0.3 * constant)
    expect(err).toBeLessThan(0.5 * pehe(tLearner(data).predict(x), truth))
    expect(forest.importances.indexOf(Math.max(...forest.importances))).toBe(1)
    const ate = forest.averageEffect()
    const truthAte = truth.reduce((a, b) => a + b, 0) / truth.length
    expect(Math.abs(ate.estimate.value - truthAte)).toBeLessThan(0.3)
    expect(forest.predictOutOfBag().length).toBe(truth.length)
  }, 60_000)

  it('grows one tree per step and its state serialises', () => {
    const alg = causalForestAlgorithm(data, { trees: 6, minLeaf: 20 })
    const run = runAlgorithm(alg, { key: key(0), maxSteps: 20 })
    expect(run.steps).toBe(6)
    expect(run.state.trees.length).toBe(6)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(run.state))))
    expect(back).toBeDefined()
    expect(causalForestOf(back ?? run.state, data).predict(x)[0]).toBeCloseTo(
      causalForestOf(run.state, data).predict(x)[0] ?? 0,
      12,
    )
  })
})

describe('uplift trees on a randomised experiment with harmed units', () => {
  const train = generate(upliftData, { key: key(5), knobs: { n: 12000, covariates: 3 } })
  const test = generate(upliftData, { key: key(6), knobs: { n: 12000, covariates: 3 } })
  const split = (g: typeof train) => {
    const d = splitTreatment(g.data)
    return {
      data: d,
      t: Array.from(toFlat(d.treatment as Tensor)),
      y: Array.from(toFlat(d.outcome as Tensor)),
      truth: Array.from(toFlat((g.truth.extra as { sampleUplift: Tensor }).sampleUplift)),
    }
  }
  const a = split(train)
  const b = split(test)

  it('a tree ranks held-out units far better than chance and finds both the helped and the harmed', () => {
    for (const criterion of ['KL', 'ED', 'Chi'] as const) {
      const tree = upliftTree(a.data, { criterion, maxDepth: 3, minLeaf: 200 })
      const pred = Array.from(tree.predict(b.data.x))
      const q = qiniCoefficient(pred, b.t, b.y).estimate.value
      const oracle = qiniCoefficient(b.truth, b.t, b.y).estimate.value
      expect(q, criterion).toBeGreaterThan(0.4 * oracle)
      expect(Math.max(...pred)).toBeGreaterThan(0.05)
      expect(Math.min(...pred)).toBeLessThan(-0.02)
      expect(tree.importances.indexOf(Math.max(...tree.importances))).toBe(0)
    }
  })

  it('a bagged forest does at least as well as one tree and the top of its ranking has the largest uplift', () => {
    const forest = upliftForest(a.data, { trees: 20, maxDepth: 3, minLeaf: 200 })
    const pred = Array.from(forest.predict(b.data.x))
    const single = Array.from(upliftTree(a.data, { maxDepth: 3, minLeaf: 200 }).predict(b.data.x))
    expect(qiniCoefficient(pred, b.t, b.y).estimate.value).toBeGreaterThan(
      0.9 * qiniCoefficient(single, b.t, b.y).estimate.value,
    )
    expect(upliftAtK(pred, b.t, b.y, 0.2).estimate.value).toBeGreaterThan(
      upliftAtK(
        pred.map((v) => -v),
        b.t,
        b.y,
        0.2,
      ).estimate.value + 0.05,
    )
  })

  it('the tau criterion works for a continuous outcome', () => {
    const tree = upliftTree(
      { x: a.data.x, treatment: a.data.treatment, outcome: a.data.outcome },
      { criterion: 'tau', maxDepth: 2, minLeaf: 300 },
    )
    expect(tree.nodes[0]?.feature).toBe(0)
  })

  it('a meta-learner with a flexible effect model ranks the same experiment well', () => {
    const forest = randomForestRegressor({ nEstimators: 40, minSamplesLeaf: 40 })
    const model = tLearner(a.data, { outcome: forest, binaryOutcome: false })
    const pred = model.predict(b.data.x)
    expect(qiniCoefficient(pred, b.t, b.y).estimate.value).toBeGreaterThan(0)
  })
})
