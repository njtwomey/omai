import { describe, expect, it } from 'vitest'
import { expectMatches, type GoldenCase, loadFixture } from '../../testing/golden.ts'
import { choice, dict, flatten, int, real, type Space, type SpaceValue } from '../space/index.ts'
import { asTensor } from '../tensor/index.ts'
import {
  createStudy,
  defaultGamma,
  defaultWeights,
  leavesOfSpace,
  parzenEstimator,
  parzenLogPdf,
  type StudyDirection,
  splitTrials,
  type Trial,
} from './index.ts'

const fixture = loadFixture('tune/tpe')

const SPACES: Readonly<Record<string, Space>> = {
  x: real(-5, 5),
  lr: real(1e-4, 1, { log: true }),
  k: int(1, 1000),
  kl: int(1, 100000, { log: true }),
  s: real(0, 50, { step: 0.5 }),
  act: choice(['a', 'b', 'c']),
}

function trialsOf(json: unknown): Trial[] {
  return (json as Omit<Trial, 'source'>[]).map((t) => ({ ...t, source: 'fixture' }))
}

function closeTo(got: readonly number[], want: readonly number[], c: GoldenCase): void {
  expect(got.length).toBe(want.length)
  for (const [i, w] of want.entries()) {
    const g = got[i] as number
    expect(Math.abs(g - w) <= c.atol + c.rtol * Math.abs(w), `${c.name}[${i}]: ${g} vs ${w}`).toBe(true)
  }
}

describe('tpe against optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'defaultGamma':
          expect((c.inputs[0] as number[]).map(defaultGamma)).toEqual(c.output)
          return
        case 'defaultWeights':
          expectMatches(asTensor(defaultWeights(Number(c.params['n']))), c.output, 1e-12, 1e-12, c.name)
          return
        case 'parzen': {
          const names = c.params['names'] as string[]
          const space = dict(Object.fromEntries(Object.entries(SPACES).filter(([k]) => names.includes(k))))
          const full = dict(SPACES)
          const study = createStudy(full, { trials: trialsOf(c.inputs[0]) })
          const out = c.output as { below: number[]; logPdfBelow: number[]; logPdfAbove: number[] }
          const split = splitTrials(study, study.trials, defaultGamma(study.trials.length))
          expect(split.below.map((t) => t.number)).toEqual(out.below)
          const pick = (t: Trial): number[] => {
            const p = t.params as Readonly<Record<string, SpaceValue>>
            return flatten(space, Object.fromEntries(names.map((n) => [n, p[n] as SpaceValue])))
          }
          const options = {
            priorWeight: Number(c.params['priorWeight']),
            considerMagicClip: c.params['considerMagicClip'] === true,
            considerEndpoints: c.params['considerEndpoints'] === true,
            ...(c.params['flatWeights'] === true ? { weights: (n: number) => new Array<number>(n).fill(1) } : {}),
          }
          const leaves = leavesOfSpace(space)
          const l = parzenEstimator(leaves, split.below.map(pick), options)
          const g = parzenEstimator(leaves, split.above.map(pick), options)
          const at = (c.inputs[1] as Record<string, SpaceValue>[]).map((p) => flatten(space, p))
          closeTo(parzenLogPdf(l, at), out.logPdfBelow, c)
          closeTo(parzenLogPdf(g, at), out.logPdfAbove, c)
          return
        }
        case 'splitMultiObjective': {
          const directions = c.params['directions'] as StudyDirection[]
          const study = createStudy(dict(SPACES), { directions, trials: trialsOf(c.inputs[0]) })
          const below = splitTrials(study, study.trials, Number(c.params['nBelow'])).below
          expect(below.map((t) => t.number)).toEqual(c.output)
          return
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
