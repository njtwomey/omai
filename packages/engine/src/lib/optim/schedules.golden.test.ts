import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as O from './index.ts'

type P = Readonly<Record<string, unknown>>
const n = (p: P, k: string): number => p[k] as number

const FACTORIES: Readonly<Record<string, (p: P) => O.LearningRateSchedule>> = {
  constantFactor: (p) => O.constantFactor(n(p, 'init'), n(p, 'factor'), n(p, 'steps')),
  stepDecay: (p) => O.stepDecay(n(p, 'init'), n(p, 'stepSize'), n(p, 'gamma')),
  multiStepDecay: (p) => O.multiStepDecay(n(p, 'init'), p['milestones'] as number[], n(p, 'gamma')),
  exponentialDecay: (p) =>
    O.exponentialDecay(n(p, 'init'), n(p, 'decayRate'), n(p, 'steps'), p['staircase'] === true, {
      ...(p['transitionBegin'] === undefined ? {} : { transitionBegin: n(p, 'transitionBegin') }),
      ...(p['endValue'] === undefined ? {} : { endValue: n(p, 'endValue') }),
    }),
  linearSchedule: (p) => O.linearSchedule(n(p, 'init'), n(p, 'end'), n(p, 'steps'), (p['begin'] as number) ?? 0),
  polynomialDecay: (p) =>
    O.polynomialDecay(n(p, 'init'), n(p, 'end'), n(p, 'power'), n(p, 'steps'), (p['begin'] as number) ?? 0),
  cosineDecay: (p) => O.cosineDecay(n(p, 'init'), n(p, 'decaySteps'), n(p, 'alpha'), (p['exponent'] as number) ?? 1),
  cosineWarmRestarts: (p) => O.cosineWarmRestarts(n(p, 'init'), n(p, 't0'), n(p, 'tMult'), n(p, 'etaMin')),
  oneCycle: (p) => O.oneCycle(p as unknown as O.OneCycleOptions),
  cyclical: (p) => O.cyclical(p as unknown as O.CyclicalOptions),
  warmupCosineDecay: (p) => O.warmupCosineDecay(p as never),
  warmupExponentialDecay: (p) => O.warmupExponentialDecay(p as never),
  warmupConstant: (p) => O.warmupConstant(n(p, 'start'), n(p, 'peak'), n(p, 'warmupSteps')),
  piecewiseInterpolate: (p) =>
    O.piecewiseInterpolate(p['kind'] as 'linear', n(p, 'init'), p['pairs'] as [number, number][]),
  sgdr: (p) =>
    O.sgdr(
      (p['cycles'] as number[][]).map(([start = 0, peak = 0, warmupSteps = 0, totalSteps = 0, end = 0]) => ({
        start,
        peak,
        warmupSteps,
        totalSteps,
        end,
      })),
    ),
}

describe('schedules against torch.optim.lr_scheduler and optax', () => {
  for (const c of loadFixture('optim/schedules').cases) {
    it(c.name, () => {
      if (c.fn === 'reduceOnPlateau') {
        const plateau = O.reduceOnPlateau(c.params as O.PlateauOptions)
        let s = plateau.init(n(c.params, 'init'))
        const rates: number[] = []
        for (const m of toFlat(decode(c.inputs[0]) as Tensor)) {
          s = plateau.step(s, m)
          rates.push(s.lr)
        }
        expectMatches(tensor(rates), c.output, c.rtol, c.atol, c.name)
        return
      }
      const schedule = (FACTORIES[c.fn] as (p: P) => O.LearningRateSchedule)(c.params)
      const values = Array.from({ length: n(c.params, 'count') }, (_, t) => schedule(t))
      expectMatches(tensor(values), c.output, c.rtol, c.atol, c.name)
    })
  }

  it('covers every schedule factory', () => {
    const used = new Set(loadFixture('optim/schedules').cases.map((c) => c.fn))
    for (const name of Object.keys(FACTORIES)) expect(used.has(name)).toBe(true)
  })
})
