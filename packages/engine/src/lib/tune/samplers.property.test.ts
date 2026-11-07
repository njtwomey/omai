import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  bool,
  box,
  choice,
  contains,
  dict,
  discrete,
  int,
  multiBinary,
  multiDiscrete,
  real,
  type Space,
  type SpaceValue,
  tuple,
} from '../space/index.ts'
import {
  cmaesSampler,
  evolutionarySampler,
  functionObjective,
  gpSampler,
  gridSampler,
  haltonSampler,
  latinHypercubeSampler,
  nelderMeadSampler,
  randomSampler,
  type StudySampler,
  sobolSampler,
  tpeSampler,
  tune,
} from './index.ts'

const SPACES: readonly Space[] = [
  dict({ x: real(-1, 3), lr: real(1e-5, 1, { log: true }), s: real(0, 1, { step: 0.25 }) }),
  dict({ k: int(2, 9), kl: int(1, 1000, { log: true }), act: choice(['a', 'b', 'c']), flag: bool() }),
  box([-1, 0], [1, 5], [2]),
  tuple([discrete(4, { start: 1 }), real(0, 1)]),
  dict({ v: multiDiscrete([2, 3]), b: multiBinary(3) }),
  dict({
    kind: choice(['linear', 'tree']),
    alpha: real(0, 1, { when: { key: 'kind', in: ['linear'] } }),
    depth: int(1, 8, { when: { key: 'kind', in: ['tree'] } }),
    leaf: int(1, 4, { when: { key: 'depth', in: [5, 6, 7, 8] } }),
  }),
]

const SAMPLERS: readonly (() => StudySampler)[] = [
  () => randomSampler(),
  () => sobolSampler(),
  () => sobolSampler({ scramble: key(1) }),
  () => haltonSampler(),
  () => latinHypercubeSampler({ n: 5 }),
  () => cmaesSampler(),
  () => nelderMeadSampler(),
  () => evolutionarySampler({ populationSize: 4 }),
  () => tpeSampler({ nStartupTrials: 3 }),
]

/** A rugged objective of any configuration, so model-based samplers see varied values. */
function rugged(p: SpaceValue): number {
  const text = JSON.stringify(p)
  let h = 0
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) % 1000003
  return Math.sin(h) + h / 1e6
}

describe('every sampler proposes members of the space', () => {
  it('for random spaces, samplers and seeds', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: SPACES.length - 1 }),
        fc.integer({ min: 0, max: SAMPLERS.length - 1 }),
        fc.integer({ min: 0, max: 1000 }),
        (s, m, seed) => {
          const space = SPACES[s] as Space
          const sampler = (SAMPLERS[m] as () => StudySampler)()
          const study = tune(functionObjective(rugged), space, {
            sampler,
            nTrials: 12,
            seed,
            constantTrials: Infinity,
          }).study
          for (const t of study.trials)
            expect(contains(space, t.params), `${sampler.name}: ${JSON.stringify(t.params)}`).toBe(true)
        },
      ),
      { numRuns: 40 },
    )
  })
  it('for the grid and Gaussian-process samplers', () => {
    for (const space of SPACES.filter((s) => s.kind === 'dict' && JSON.stringify(s).indexOf('multi') < 0)) {
      for (const sampler of [
        gridSampler({ points: 3 }),
        gpSampler({ nStartupTrials: 3, search: { nCandidates: 32, nLocal: 1 } }),
      ]) {
        const study = tune(functionObjective(rugged), space, {
          sampler,
          nTrials: 6,
          seed: 3,
          constantTrials: Infinity,
        }).study
        for (const t of study.trials) expect(contains(space, t.params)).toBe(true)
      }
    }
  })
})
