import { describe, expect, it } from 'vitest'
import { branin, hartmann3, hartmann6 } from '../../testing/optim-problems.ts'
import { key } from '../random/index.ts'
import { box, choice, dict, int, real } from '../space/index.ts'
import { asTensor, item } from '../tensor/index.ts'
import {
  cmaesSampler,
  createStudy,
  evolutionarySampler,
  functionObjective,
  gpSampler,
  gridSampler,
  latinHypercubeSampler,
  multiObjective,
  nelderMeadSampler,
  randomSampler,
  type Sampler,
  sobolSampler,
  studyHypervolume,
  tpeSampler,
  tune,
} from './index.ts'

const braninObjective = functionObjective((p) => {
  const { x, y } = p as { x: number; y: number }
  return item(branin(asTensor([x, y])))
})
const braninSpace = dict({ x: real(-5, 10), y: real(0, 15) })
const hartmann = (f: typeof hartmann3) => functionObjective((p) => item(f(asTensor(p as number[]))))

describe('samplers on Branin (minimum 0.397887) within a budget', () => {
  const cases: [string, Sampler, number, number][] = [
    ['random', randomSampler(), 100, 2],
    ['sobol', sobolSampler(), 64, 3],
    ['latin hypercube', latinHypercubeSampler({ n: 64 }), 64, 1],
    ['grid', gridSampler({ points: 9 }), 81, 1.5],
    ['tpe', tpeSampler(), 100, 0.7],
    ['gp', gpSampler(), 25, 0.6],
    ['cmaes', cmaesSampler(), 200, 0.4],
    ['nelder-mead', nelderMeadSampler(), 150, 0.4],
    ['evolutionary', evolutionarySampler(), 300, 0.45],
  ]
  for (const [name, sampler, n, bound] of cases)
    it(name, () => {
      const best = tune(braninObjective, braninSpace, { sampler, nTrials: n, key: key(0) }).best
      expect(best?.values[0]).toBeLessThan(bound)
    })
})

describe('samplers on Hartmann functions within a budget', () => {
  it('Hartmann-3 (minimum -3.86278)', () => {
    const f = hartmann(hartmann3)
    const space = box(0, 1, [3])
    expect(tune(f, space, { sampler: tpeSampler(), nTrials: 100, key: key(0) }).best?.values[0]).toBeLessThan(-3.7)
    expect(tune(f, space, { sampler: cmaesSampler(), nTrials: 200, key: key(0) }).best?.values[0]).toBeLessThan(-3.8)
    expect(tune(f, space, { sampler: gpSampler(), nTrials: 30, key: key(0) }).best?.values[0]).toBeLessThan(-3.85)
  })
  it('Hartmann-6 (minimum -3.32237)', () => {
    const f = hartmann(hartmann6)
    const space = box(0, 1, [6])
    expect(tune(f, space, { sampler: cmaesSampler(), nTrials: 400, key: key(0) }).best?.values[0]).toBeLessThan(-3.1)
    expect(tune(f, space, { sampler: tpeSampler(), nTrials: 150, key: key(0) }).best?.values[0]).toBeLessThan(-3.0)
    const evo = evolutionarySampler()
    expect(tune(f, space, { sampler: evo, nTrials: 400, key: key(0) }).best?.values[0]).toBeLessThan(-2.9)
  })
})

describe('sampler behaviour', () => {
  it('grid stops the study after its last configuration, and shuffles with a key', () => {
    const space = dict({ a: choice([1, 2, 3]), b: choice(['x', 'y']) })
    const r = tune(
      functionObjective(() => 0),
      space,
      { sampler: gridSampler(), nTrials: 100, constantTrials: Infinity },
    )
    expect(r.study.trials.length).toBe(6)
    expect(r.status).toBe('completed')
    const shuffled = tune(
      functionObjective(() => 0),
      space,
      {
        sampler: gridSampler({ shuffle: key(3) }),
        nTrials: 100,
        constantTrials: Infinity,
      },
    ).study.trials.map((t) => JSON.stringify(t.params))
    expect(new Set(shuffled).size).toBe(6)
  })
  it('random search is a pure function of the trial key', () => {
    const a = tune(braninObjective, braninSpace, { nTrials: 5, seed: 4 }).study.trials.map((t) => t.params)
    const b = tune(braninObjective, braninSpace, { nTrials: 5, seed: 4 }).study.trials.map((t) => t.params)
    expect(a).toEqual(b)
  })
  it('scrambled Sobol differs from the plain sequence and stays in the space', () => {
    const plain = sobolSampler().ask(createStudy(braninSpace), key(0))
    const scrambled = sobolSampler({ scramble: key(1) }).ask(createStudy(braninSpace), key(0))
    expect(plain).toEqual({ x: -5, y: 0 })
    expect(scrambled).not.toEqual(plain)
  })
  it('latin hypercube batches put one point in each stratum of every axis', () => {
    const study = tune(
      functionObjective(() => 0),
      dict({ a: real(0, 1), k: int(0, 3) }),
      {
        sampler: latinHypercubeSampler({ n: 4, key: key(2) }),
        nTrials: 4,
        constantTrials: Infinity,
      },
    ).study
    const ks = study.trials.map((t) => (t.params as { k: number }).k).sort()
    expect(ks).toEqual([0, 1, 2, 3])
  })
  it('optimiser-backed samplers refuse several objectives', () => {
    const two = multiObjective([functionObjective(() => 0), functionObjective(() => 1)])
    expect(() => tune(two, braninSpace, { sampler: cmaesSampler(), nTrials: 2 })).toThrow(/one objective/)
    expect(() => tune(two, braninSpace, { sampler: gpSampler(), nTrials: 2 })).toThrow(/one objective/)
  })
  it('CMA-ES and Nelder-Mead count failed trials as worse than every value', () => {
    let calls = 0
    const flaky = functionObjective((p) => {
      calls += 1
      return calls % 3 === 0 ? Number.NaN : ((p as number) - 0.25) ** 2
    })
    for (const sampler of [cmaesSampler(), nelderMeadSampler()]) {
      const r = tune(flaky, real(0, 1), { sampler, nTrials: 60, maxFailures: 100 })
      expect(r.best?.values[0]).toBeLessThan(1e-3)
    }
  })
  it('NSGA-II style evolution spreads a two-objective front', () => {
    const zdt = multiObjective([
      functionObjective((p) => (p as number[])[0] as number),
      functionObjective((p) => {
        const x = p as number[]
        const g = 1 + (9 * x.slice(1).reduce((a, b) => a + b, 0)) / (x.length - 1)
        return g * (1 - Math.sqrt((x[0] as number) / g))
      }),
    ])
    const evolved = tune(zdt, box(0, 1, [4]), {
      sampler: evolutionarySampler({ populationSize: 20 }),
      nTrials: 400,
      key: key(1),
    })
    const random = tune(zdt, box(0, 1, [4]), { nTrials: 400, key: key(1) })
    expect(evolved.bestTrials.length).toBeGreaterThan(5)
    // The evolved front dominates more of the objective space than random search with the same budget.
    expect(studyHypervolume(evolved.study, [1.1, 11])).toBeGreaterThan(studyHypervolume(random.study, [1.1, 11]))
  })
})
