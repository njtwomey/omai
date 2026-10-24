import { describe, expect, it } from 'vitest'
import { cartPole } from '../../methods/rl/index.ts'
import { xy, xyClasses } from '../data/index.ts'
import { accuracyScore, meanSquaredError } from '../metrics/index.ts'
import { make, policyAgent } from '../rl/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import {
  agentObjective,
  estimatorFor,
  estimatorObjective,
  functionObjective,
  gridSampler,
  holdoutObjective,
  medianPruner,
  multiObjective,
  programObjective,
  type TunableEstimator,
  trainingObjective,
  tune,
} from './index.ts'

/** Predicts a constant c; scores well on targets near c. */
function constant(p: unknown): TunableEstimator {
  const c = (p as { c: number }).c
  return {
    name: 'constant',
    fit: () => ({ predict: (x: TensorLike) => new Array<number>(asTensor(x).shape[0] ?? 0).fill(c) }),
  }
}

/** Classifies by a threshold on the first feature. */
function stump(p: unknown): TunableEstimator {
  const t = (p as { t: number }).t
  const predict = (x: TensorLike): number[] => {
    const v = asTensor(x)
    const d = v.shape[1] ?? 1
    return toFlat(v).flatMap((value, i) => (i % d === 0 ? [value > t ? 1 : 0] : []))
  }
  return {
    name: 'stump',
    fit: () => ({
      predict,
      predictProba: (x: TensorLike) => predict(x).map((y) => [1 - y, y]),
    }),
  }
}

describe('estimator objectives', () => {
  const data = xy(
    Array.from({ length: 20 }, (_, i) => [i]),
    new Array<number>(20).fill(3),
  )
  it('cross-validate with the metric direction and record fold scores', () => {
    const objective = estimatorObjective(constant, data, { metric: meanSquaredError, cv: 4 })
    expect(objective.directions).toEqual(['minimize'])
    const r = tune(objective, dict({ c: choice([0, 3, 5]) }), { sampler: gridSampler() })
    expect(r.best?.params).toEqual({ c: 3 })
    expect(r.study.trials[0]?.attrs?.['foldScores']).toEqual([9, 9, 9, 9])
    expect(r.study.trials[0]?.intermediate.length).toBe(4)
  })
  it('stratify classification folds and maximise accuracy', () => {
    const x = Array.from({ length: 30 }, (_, i) => [i / 30])
    const y = x.map(([v]) => ((v ?? 0) > 0.4 ? 1 : 0))
    const objective = estimatorObjective(stump, xyClasses(x, y), { metric: accuracyScore, cv: 3 })
    expect(objective.directions).toEqual(['maximize'])
    const r = tune(objective, dict({ t: real(0, 1) }), { sampler: gridSampler({ points: 31 }) })
    expect(r.best?.values[0]).toBe(1)
  })
  it('let a pruner stop a configuration between folds', () => {
    const objective = estimatorObjective(constant, data, { metric: meanSquaredError, cv: 5 })
    const r = tune(objective, dict({ c: choice([3, 2.9, 10, 20, 30]) }), {
      sampler: gridSampler(),
      pruner: medianPruner({ nStartupTrials: 2 }),
    })
    expect(r.study.trials.filter((t) => t.state === 'pruned').length).toBeGreaterThan(0)
  })
  it('score on a validation set', () => {
    const train = xy([[0], [1]], [1, 1])
    const validation = xy([[2], [3]], [2, 2])
    const objective = holdoutObjective(constant, train, validation, { metric: meanSquaredError })
    expect(tune(objective, dict({ c: choice([1, 2]) }), { sampler: gridSampler() }).best?.params).toEqual({ c: 2 })
  })
  it('make estimators from sources', () => {
    const est: TunableEstimator = { name: 'fixed', fit: () => ({ predict: () => [0] }) }
    expect(() => estimatorFor(est, {})).toThrow(/withHyper/)
    expect(estimatorFor({ ...est, withHyper: () => ({ ...est, name: 'again' }) }, {}).name).toBe('again')
  })
})

describe('training, agent, program and combined objectives', () => {
  it('run a training algorithm as a child of the trial and stop it when pruned', () => {
    const steps: number[] = []
    const gd = (lr: unknown): Algorithm<number> => ({
      name: 'gd',
      config: {},
      init: () => 10,
      step: (w, ctx) => {
        if (ctx.t === 0) steps.push(0)
        else steps[steps.length - 1] = ctx.t
        return w - (lr as number) * 2 * w
      },
    })
    const objective = trainingObjective(gd, { value: (w: number) => w * w, maxSteps: 30, reportEvery: 5 })
    const r = tune(objective, real(1e-3, 0.5, { log: true }), {
      nTrials: 12,
      seed: 2,
      pruner: medianPruner({ nStartupTrials: 3 }),
    })
    const pruned = r.study.trials.filter((t) => t.state === 'pruned')
    expect(pruned.length).toBeGreaterThan(0)
    for (const t of pruned) expect(steps[t.number]).toBeLessThan(29)
    expect(r.best?.values[0]).toBeLessThan(1e-3)
    expect(() => trainingObjective(gd, { value: (w: number) => w, reportEvery: 0 })).toThrow()
  })
  it('use a schedule budget as the number of training steps', () => {
    const objective = trainingObjective(
      (): Algorithm<number> => ({ name: 'count', config: {}, init: () => 0, step: (s) => s + 1 }),
      { value: (s: number) => -s },
    )
    const r = tune(objective, real(0, 1), {
      schedule: { kind: 'asha', reductionFactor: 3, minResource: 1, maxResource: 9, rungs: [1, 3, 9] },
      nTrials: 10,
    })
    for (const t of r.study.trials) expect(t.values[0]).toBe(-(t.budget ?? 0))
  })
  it('train and evaluate agents', () => {
    const env = make(cartPole)
    // Push right when the pole leans right by more than the threshold.
    const objective = agentObjective(
      env,
      (p) => policyAgent((o: readonly number[]) => ((o[2] ?? 0) + (o[3] ?? 0) * (p as number) > 0 ? 1 : 0)),
      { episodes: 2, evalEpisodes: 2, reportEvery: 1 },
    )
    const r = tune(objective, real(-1, 1), { nTrials: 4, seed: 0 })
    expect(r.best?.values[0]).toBeGreaterThan(20)
    expect(r.study.trials[0]?.intermediate.length).toBe(2)
  })
  it('score programs by their evidence', () => {
    const objective = programObjective(() => ({ elbo: -3 }), { criterion: 'elbo' })
    expect(tune(objective, real(0, 1), { nTrials: 2 }).best?.values[0]).toBe(-3)
    const missing = programObjective(() => ({}))
    expect(tune(missing, real(0, 1), { nTrials: 1 }).study.trials[0]?.state).toBe('failed')
  })
  it('combine objectives into several values', () => {
    const both = multiObjective([
      functionObjective((x, t) => {
        t.report(0, x as number)
        return (x as number) ** 2
      }),
      functionObjective((x) => [1 - (x as number)], { direction: 'minimize' }),
    ])
    const r = tune(both, real(0, 1), { nTrials: 20, seed: 0 })
    expect(r.study.trials[0]?.values.length).toBe(2)
    expect(r.bestTrials.length).toBeGreaterThan(1)
    expect(() => multiObjective([])).toThrow()
  })
})
