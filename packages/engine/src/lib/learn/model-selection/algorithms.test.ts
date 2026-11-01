import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { full, item, mean, toFlat } from '../../tensor/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { defineEstimator, type LearnData } from '../estimator.ts'
import { rowsOf } from '../validation.ts'
import { bootstrap632, bootstrap632Algorithm } from './criteria.ts'
import {
  type CrossValidateOptions,
  crossValidate,
  crossValidateAlgorithm,
  crossValPredict,
  crossValPredictAlgorithm,
  type LearningCurveOptions,
  learningCurve,
  learningCurveAlgorithm,
  type PermutationTestOptions,
  permutationTestAlgorithm,
  permutationTestScore,
} from './cross-validation.ts'
import {
  evaluateCandidates,
  evaluateCandidatesAlgorithm,
  type NestedOptions,
  nestedCrossValidation,
  nestedCrossValidationAlgorithm,
  parameterGrid,
  type SearchOptions,
} from './search.ts'
import { searchDesigns } from './search-options.ts'
import { splitterSchemes } from './splitter-options.ts'
import { stratifiedKFold } from './splitters.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

const meanModel = defineEstimator(
  {
    name: 'mean',
    kind: 'regressor',
    fit: (d) => ({ m: item(mean(d.y as never)) }),
    methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }),
  },
  {},
)
const constant = defineEstimator(
  {
    name: 'constant',
    kind: 'regressor',
    fit: () => ({}),
    methods: (_, h) => ({ predict: (x) => full([rowsOf(x)], h.c) }),
  },
  { c: 0 },
)
const ramp: LearnData = { x: [[0], [1], [2], [3], [4], [5], [6], [7]], y: [0, 1, 2, 3, 4, 5, 6, 7] }
const flat: LearnData = { x: [[0], [1], [2], [3], [4], [5]], y: [1, 1, 1, 1, 1, 1] }

/** Run to the end, then again from a JSON checkpoint taken after `cut` steps; both ends are returned. */
function resumed<S>(alg: Algorithm<S>, cut: number): { whole: S; rest: S; steps: [number, number] } {
  const whole = runAlgorithm(alg, { maxSteps: 200 })
  const half = runAlgorithm(alg, { maxSteps: cut }).state
  const stored = viaJson(alg.serialise?.(half) ?? null)
  const rest = runAlgorithm(alg, { maxSteps: 200, start: { state: stored, step: cut, serialised: true } })
  return { whole: whole.state, rest: rest.state, steps: [whole.steps, rest.steps] }
}

const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const],
  detail: 'full' as const,
}

describe('crossValidateAlgorithm', () => {
  it('has the scores crossValidate gives, one step per fold, and logs them by path', () => {
    const options: CrossValidateOptions = { cv: 4, scoring: 'neg_mean_squared_error', returnTrainScore: true }
    const rec = recorder()
    const run = runAlgorithm(crossValidateAlgorithm(meanModel, ramp, options), { maxSteps: 10, tracer: rec, policy })
    const r = crossValidate(meanModel, ramp, options)
    expect(run.steps).toBe(4)
    expect(run.status).toBe('completed')
    expect(run.state.test).toEqual(r.testScores)
    expect(run.state.train).toEqual(r.trainScores)
    expect(metricsOf(rec, 'cv/fold')).toEqual([0, 1, 2, 3])
    expect(metricsOf(rec, 'cv/testScore')).toEqual(r.testScores['neg_mean_squared_error'])
    expect(metricsOf(rec, 'cv/test/neg_mean_squared_error')).toEqual(r.testScores['neg_mean_squared_error'])
    expect(metricsOf(rec, 'cv/trainSize')).toEqual([6, 6, 6, 6])
  })

  it('resumes from a JSON checkpoint, including a failed fold scored NaN', () => {
    const flaky = defineEstimator(
      {
        name: 'flaky',
        kind: 'regressor',
        fit: (d) => {
          if (rowsOf(d.x) === 6 && d.y !== undefined && toFlat(d.y as never)[0] === 0) throw new Error('boom')
          return {}
        },
        methods: () => ({ predict: (x) => full([rowsOf(x)], 0) }),
      },
      {},
    )
    const alg = crossValidateAlgorithm(flaky, ramp, {
      cv: 4,
      scoring: 'neg_mean_squared_error',
      errorScore: Number.NaN,
    })
    const { whole, rest, steps } = resumed(alg, 2)
    expect(steps[1]).toBe(steps[0])
    expect(rest.test['neg_mean_squared_error']?.map(Number.isNaN)).toEqual(
      whole.test['neg_mean_squared_error']?.map(Number.isNaN),
    )
    expect(whole.test['neg_mean_squared_error']?.some(Number.isNaN)).toBe(true)
  })
})

describe('crossValPredictAlgorithm', () => {
  it('fills in the predictions fold by fold and resumes to the same end', () => {
    const alg = crossValPredictAlgorithm(meanModel, ramp, { cv: 4 })
    const { whole, rest } = resumed(alg, 2)
    expect(whole.done).toBe(4)
    expect(rest.predictions).toEqual(whole.predictions)
    expect(whole.predictions).toEqual(Array.from(toFlat(crossValPredict(meanModel, ramp, { cv: 4 }))))
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(metricsOf(rec, 'cv/predicted')).toEqual([2, 4, 6, 8])
  })

  it('refuses a splitter that does not partition', () => {
    expect(() => crossValPredictAlgorithm(meanModel, ramp, { cv: splitterSchemes.make({ kind: 'shuffle' }) })).toThrow(
      /partition/,
    )
  })
})

describe('learningCurveAlgorithm', () => {
  it('has the scores learningCurve gives and resumes', () => {
    const options: LearningCurveOptions = { cv: 3, trainSizes: [0.5, 1], scoring: 'neg_mean_squared_error' }
    const alg = learningCurveAlgorithm(meanModel, ramp, options)
    const { whole, rest } = resumed(alg, 3)
    expect(whole.done).toBe(6)
    expect(rest).toEqual(whole)
    const curve = learningCurve(meanModel, ramp, options)
    expect(curve.testScores.map((row) => row.length)).toEqual([3, 3])
    expect(curve.testScores[0]?.[1]).toBeCloseTo(whole.test[2] ?? Number.NaN, 12)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(new Set(metricsOf(rec, 'curve/size'))).toEqual(new Set(curve.values))
  })
})

describe('permutationTestAlgorithm', () => {
  it('scores the real targets first, then one permutation per step, with a running p-value', () => {
    const options: PermutationTestOptions = { cv: 2, nPermutations: 6, scoring: 'neg_mean_absolute_error', key: key(3) }
    const alg = permutationTestAlgorithm(meanModel, ramp, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy })
    expect(run.steps).toBe(7)
    const r = permutationTestScore(meanModel, ramp, options)
    expect(run.state.real).toBeCloseTo(r.score, 12)
    expect(run.state.scores).toHaveLength(6)
    for (const [i, v] of r.permutationScores.entries()) expect(v).toBeCloseTo(run.state.scores[i] ?? Number.NaN, 12)
    const p = metricsOf(rec, 'perm/pValue')
    expect(p).toHaveLength(6)
    expect(p.at(-1)).toBeCloseTo(r.pValue, 12)
    expect(metricsOf(rec, 'perm/index')).toEqual([-1, 0, 1, 2, 3, 4, 5])
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = permutationTestAlgorithm(meanModel, ramp, {
      cv: 2,
      nPermutations: 5,
      scoring: 'neg_mean_absolute_error',
    })
    const { whole, rest } = resumed(alg, 3)
    expect(rest.real).toBeCloseTo(whole.real ?? Number.NaN, 12)
    expect(rest.scores).toEqual(whole.scores)
  })
})

describe('evaluateCandidatesAlgorithm', () => {
  const candidates = parameterGrid({ c: [0, 1, 2] })
  const options: SearchOptions = { cv: 2, scoring: 'neg_mean_absolute_error' }

  it('cross-validates one candidate per step and logs the best so far', () => {
    const alg = evaluateCandidatesAlgorithm(constant, candidates, flat, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(3)
    expect(metricsOf(rec, 'search/candidate')).toEqual([0, 1, 2])
    expect(metricsOf(rec, 'search/score').map((v) => Math.abs(v))).toEqual([1, 0, 1])
    const best = metricsOf(rec, 'search/best')
    expect(best.every((v, i) => i === 0 || v >= (best[i - 1] ?? 0))).toBe(true)
    expect(metricsOf(rec, 'search/bestIndex')).toEqual([0, 1, 1])
    expect(evaluateCandidates(constant, candidates, flat, options).bestParams).toEqual({ c: 1 })
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = evaluateCandidatesAlgorithm(constant, candidates, flat, { ...options, returnTrainScore: true })
    const { whole, rest } = resumed(alg, 1)
    expect(rest).toEqual(whole)
    expect(whole.rows[0]?.train).toBeDefined()
  })

  it('takes its candidates from a design table', () => {
    expect(searchDesigns.make({ kind: 'grid' })({ a: [1, 2], b: [3] }, key(0))).toEqual([
      { a: 1, b: 3 },
      { a: 2, b: 3 },
    ])
    expect(searchDesigns.make({ kind: 'random', nIter: 2 })({ a: [1, 2, 3] }, key(0))).toHaveLength(2)
    expect(searchDesigns.properties({ kind: 'grid' }).exhaustive).toBe(true)
  })
})

describe('nestedCrossValidationAlgorithm', () => {
  it('has the choices nestedCrossValidation makes and resumes', () => {
    const options: NestedOptions = { outerCv: 2, innerCv: 3, scoring: 'neg_mean_absolute_error' }
    const alg = nestedCrossValidationAlgorithm(constant, { c: [0, 1] }, flat, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(2)
    expect(run.state.chosen).toEqual([{ c: 1 }, { c: 1 }])
    expect(metricsOf(rec, 'nested/chosen')).toEqual([1, 1])
    expect(nestedCrossValidation(constant, { c: [0, 1] }, flat, options).chosen).toEqual(run.state.chosen)
    const { whole, rest } = resumed(alg, 1)
    expect(rest.chosen).toEqual(whole.chosen)
    for (const [i, v] of rest.scores.entries()) expect(v).toBeCloseTo(whole.scores[i] ?? Number.NaN, 12)
  })
})

describe('bootstrap632Algorithm', () => {
  it('has the out-of-bag error bootstrap632 uses, one replicate per step, and resumes', () => {
    const options = { nBootstraps: 12, key: key(5) }
    const alg = bootstrap632Algorithm(meanModel, ramp, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 30, tracer: rec, policy })
    expect(run.steps).toBe(12)
    expect(metricsOf(rec, 'bootstrap/replicate')).toEqual(Array.from({ length: 12 }, (_, i) => i))
    const seen = metricsOf(rec, 'bootstrap/seen')
    expect(seen.every((v, i) => i === 0 || v >= (seen[i - 1] ?? 0))).toBe(true)
    const r = bootstrap632(meanModel, ramp, options)
    const last = metricsOf(rec, 'bootstrap/outOfBag').at(-1) ?? Number.NaN
    expect(r.outOfBagError).toBeCloseTo(last, 12)
    const { whole, rest } = resumed(alg, 5)
    expect(rest.count).toEqual(whole.count)
    for (const [i, v] of rest.sum.entries()) expect(v).toBeCloseTo(whole.sum[i] ?? Number.NaN, 12)
  })
})

describe('splitterSchemes', () => {
  it('names the schemes and builds each splitter from its knobs', () => {
    expect(splitterSchemes.names).toEqual([
      'kfold',
      'stratified',
      'repeated',
      'repeatedStratified',
      'shuffle',
      'leaveOneOut',
    ])
    expect(splitterSchemes.make({ kind: 'kfold', folds: 3 }).nSplits({ n: 9 })).toBe(3)
    expect(splitterSchemes.make({ kind: 'repeated', folds: 3, repeats: 2 }).nSplits({ n: 12 })).toBe(6)
    expect(splitterSchemes.make({ kind: 'leaveOneOut' }).nSplits({ n: 7 })).toBe(7)
    expect(splitterSchemes.make({ kind: 'stratified' }).name).toBe(stratifiedKFold(5).name)
    expect(splitterSchemes.where((p) => p.partition)).toEqual(['kfold', 'stratified', 'leaveOneOut'])
  })

  it('drives a cross-validation as its splitter', () => {
    const r = crossValidate(meanModel, ramp, { cv: splitterSchemes.make({ kind: 'kfold', folds: 2 }) })
    expect(r.folds).toHaveLength(2)
  })
})
