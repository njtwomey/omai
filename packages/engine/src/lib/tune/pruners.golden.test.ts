import { describe, expect, it } from 'vitest'
import { decode, type GoldenCase, loadFixture } from '../../testing/golden.ts'
import { real } from '../space/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  functionObjective,
  hyperbandBracketOf,
  hyperbandPruner,
  medianPruner,
  type Pruner,
  patientPruner,
  percentilePruner,
  type StudyDirection,
  successiveHalvingPruner,
  thresholdPruner,
  tune,
} from './index.ts'

const fixture = loadFixture('tune/pruners')

function num(c: GoldenCase, name: string): number | undefined {
  const v = c.params[name]
  return v === undefined ? undefined : Number(v)
}

function pruner(c: GoldenCase): Pruner {
  const config = c.params['config'] as Readonly<Record<string, unknown>>
  const p = (name: string): number | undefined => (config[name] === undefined ? undefined : Number(config[name]))
  const startup = p('nStartupTrials')
  const percentileOptions = {
    ...(startup === undefined ? {} : { nStartupTrials: startup }),
    ...(p('nWarmupSteps') === undefined ? {} : { nWarmupSteps: p('nWarmupSteps') as number }),
    ...(p('intervalSteps') === undefined ? {} : { intervalSteps: p('intervalSteps') as number }),
    ...(p('nMinTrials') === undefined ? {} : { nMinTrials: p('nMinTrials') as number }),
  }
  switch (config['kind']) {
    case 'median':
      return medianPruner(percentileOptions)
    case 'percentile':
      return percentilePruner(p('percentile') as number, percentileOptions)
    case 'threshold':
      return thresholdPruner({
        lower: p('lower') as number,
        upper: p('upper') as number,
        nWarmupSteps: p('nWarmupSteps') as number,
      })
    case 'patient':
      return patientPruner(undefined, p('patience') as number, p('minDelta') as number)
    case 'patientMedian':
      return patientPruner(medianPruner({ nStartupTrials: startup as number }), p('patience') as number)
    case 'halving':
      return successiveHalvingPruner({
        minResource: config['minResource'] === 'auto' ? 'auto' : (p('minResource') as number),
        ...(p('reductionFactor') === undefined ? {} : { reductionFactor: p('reductionFactor') as number }),
        ...(p('minEarlyStoppingRate') === undefined
          ? {}
          : { minEarlyStoppingRate: p('minEarlyStoppingRate') as number }),
        ...(p('bootstrapCount') === undefined ? {} : { bootstrapCount: p('bootstrapCount') as number }),
      })
    case 'hyperband':
      return hyperbandPruner({
        ...(p('minResource') === undefined ? {} : { minResource: p('minResource') as number }),
        maxResource: config['maxResource'] === 'auto' ? 'auto' : (p('maxResource') as number),
        ...(p('reductionFactor') === undefined ? {} : { reductionFactor: p('reductionFactor') as number }),
      })
    default:
      throw new Error(`unknown pruner ${String(config['kind'])}`)
  }
}

describe('pruners against optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      if (c.fn === 'brackets') {
        const out = c.output as { budgets: number[]; brackets: number[] }
        const got = out.brackets.map((_, n) => hyperbandBracketOf(String(c.params['studyName']), n, out.budgets))
        expect(got).toEqual(out.brackets)
        return
      }
      const values = decode(c.inputs[0]) as Tensor
      const [n = 0, steps = 0] = values.shape
      const flat = toFlat(values)
      const stride = num(c, 'stride') ?? 1
      const objective = functionObjective(
        (_p, trial) => {
          const row = flat.slice(trial.number * steps, (trial.number + 1) * steps)
          for (const [i, v] of row.entries()) if (trial.report(i * stride, v)) return v
          return row.at(-1) as number
        },
        { direction: c.params['direction'] as StudyDirection },
      )
      const study = tune(objective, real(0, 1), {
        nTrials: n,
        pruner: pruner(c),
        name: String(c.params['studyName']),
        maxFailures: Number.POSITIVE_INFINITY,
        constantTrials: Number.POSITIVE_INFINITY,
      }).study
      const out = c.output as { prunedAt: number[]; states: string[] }
      expect(study.trials.map((t) => t.prunedAt ?? -1)).toEqual(out.prunedAt)
      expect(study.trials.map((t) => t.state)).toEqual(out.states)
    })
  }
})
