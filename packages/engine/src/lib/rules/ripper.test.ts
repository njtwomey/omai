import { describe, expect, it } from 'vitest'
import { plantedRules } from '../../testing/rules.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { fitCovering, fitRipper, ripperAlgorithm, ripperClassifier, ruleSetDescriptionLength } from './index.ts'

describe('description length', () => {
  it('a rule that explains its rows is cheaper than leaving them as exceptions', () => {
    const empty = ruleSetDescriptionLength([], 20, 0, 0, 40, 10)
    const one = ruleSetDescriptionLength([2], 20, 10, 0, 30, 0)
    expect(one).toBeLessThan(empty)
  })

  it('exceptions cost bits: more false positives, longer description', () => {
    const a = ruleSetDescriptionLength([2], 20, 10, 1, 30, 0)
    const b = ruleSetDescriptionLength([2], 20, 10, 4, 30, 0)
    expect(b).toBeGreaterThan(a)
  })

  it('a longer rule costs more to send', () => {
    expect(ruleSetDescriptionLength([4], 20, 10, 0, 30, 0)).toBeGreaterThan(
      ruleSetDescriptionLength([1], 20, 10, 0, 30, 0),
    )
  })
})

describe('RIPPER', () => {
  const train = plantedRules(600, 0.05, 2)
  const test = plantedRules(600, 0, 3)

  it('learns the planted list from noisy rows and generalises', () => {
    const model = fitRipper(train.data, { nominal: train.nominal, seed: 1 })
    const fitted = ripperClassifier({ nominal: train.nominal }).fit(train.data, { key: key(1) })
    expect(fitted.score(test.data)).toBeGreaterThan(0.9)
    expect(model.rules.length).toBeGreaterThan(0)
    // The least frequent class is learned first, the most frequent is the default.
    expect(model.rules.every((r) => r.label === model.rules[0]?.label)).toBe(true)
  })

  it('is reproducible from its key', () => {
    const a = fitRipper(train.data, { nominal: train.nominal, seed: 4 })
    const b = fitRipper(train.data, { nominal: train.nominal, seed: 4 })
    expect(a).toEqual(b)
  })

  it('the MDL rule stops it adding rules for labels that are pure noise, which unpruned covering would', () => {
    const noise = plantedRules(400, 0.5, 5)
    const ripper = fitRipper(noise.data, { nominal: noise.nominal, seed: 2 })
    const covering = fitCovering(noise.data, { nominal: noise.nominal, alpha: 1, minCovered: 2 })
    expect(ripper.rules.length).toBeLessThanOrEqual(2)
    expect(covering.rules.length).toBeGreaterThan(ripper.rules.length)
  })

  it('IREP learns without optimisation', () => {
    const irep = fitRipper(train.data, { nominal: train.nominal, mode: 'irep', seed: 1 })
    expect(irep.rules.length).toBeGreaterThan(0)
    const alg = ripperAlgorithm(train.data, { nominal: train.nominal, mode: 'irep' })
    const { state } = runAlgorithm(alg, { maxSteps: 200, seed: 1 })
    expect(state.phase).toBe('done')
  })

  it('takes steps that add or revisit one rule, ends done, and its state serialises', () => {
    const alg = ripperAlgorithm(train.data, { nominal: train.nominal })
    const { state, steps } = runAlgorithm(alg, { maxSteps: 300, seed: 1 })
    expect(state.phase).toBe('done')
    expect(steps).toBeGreaterThan(state.rules.length)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))
    expect(back).toEqual(state)
  })

  it('rejects a grow fraction outside (0, 1)', () => {
    expect(() => ripperAlgorithm(train.data, { growFraction: 1 })).toThrow()
  })
})
