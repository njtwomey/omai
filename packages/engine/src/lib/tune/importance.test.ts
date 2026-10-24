import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { choice, dict, real } from '../space/index.ts'
import {
  correlationImportance,
  createStudy,
  fanovaImportance,
  functionObjective,
  pedAnovaImportance,
  permutationImportance,
  sobolSampler,
  tune,
} from './index.ts'

// f = 3 a + b on the unit square: main-effect variances 9/12 and 1/12, so a explains 0.9 of the variance.
const additive = tune(
  functionObjective((p) => {
    const { a, b } = p as { a: number; b: number }
    return 3 * a + b
  }),
  dict({ a: real(0, 1), b: real(0, 1) }),
  { sampler: sobolSampler(), nTrials: 32 },
).study

describe('parameter importance', () => {
  it('fANOVA on a GP surrogate recovers the variance shares of an additive function', () => {
    const imp = fanovaImportance(additive, { key: key(0) })
    expect(imp['a']).toBeCloseTo(0.9, 1)
    expect(imp['b']).toBeCloseTo(0.1, 1)
  })
  it('permutation and correlation importances rank the strong parameter first', () => {
    expect(Object.keys(permutationImportance(additive))[0]).toBe('a')
    const corr = correlationImportance(additive)
    expect(Object.keys(corr)[0]).toBe('a')
    expect(Object.values(corr).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
    expect(Object.keys(correlationImportance(additive, { method: 'pearson' }))[0]).toBe('a')
  })
  it('handles options and too few trials', () => {
    const study = tune(
      functionObjective((p) => ((p as { c: string }).c === 'x' ? 0 : 1)),
      dict({ c: choice(['x', 'y', 'z']), d: real(0, 1) }),
      { nTrials: 15, seed: 0, constantTrials: Infinity },
    ).study
    expect(Object.keys(fanovaImportance(study))[0]).toBe('c')
    expect(Object.keys(pedAnovaImportance(study, { targetQuantile: 0.3 }))[0]).toBe('c')
    expect(() => permutationImportance(createStudy(real(0, 1)))).toThrow()
    expect(pedAnovaImportance(createStudy(dict({ a: real(0, 1) })))).toEqual({ a: 0 })
    expect(() => pedAnovaImportance(study, { targetQuantile: 0.5, regionQuantile: 0.4 })).toThrow()
  })
})
