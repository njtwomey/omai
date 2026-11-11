import { type DictSpace, defaults, type UnionSpace } from 'omai-engine/space'
import { session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import {
  classifierExample,
  classifierKinds,
  classifierMethod,
  classifierProblem,
  classifierSteps,
} from './classifier-method.ts'

const options = (classifierMethod.config.spaces['inference'] as UnionSpace).options
const small = (kind: string, extra: Record<string, number | string> = {}) => ({
  ...(defaults(classifierMethod.config) as Record<string, unknown>),
  rows: 80,
  inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Record<string, number>), ...extra },
})

describe('classifierMethod', () => {
  it('offers every method that runs on the model and gives the library reason for each that does not', () => {
    const offer = classifierExample.offer(defaults(classifierMethod.config) as never)
    expect(classifierKinds).toEqual(offer.applicable)
    for (const kind of ['nuts', 'hmc', 'svi', 'laplace', 'pathfinder']) expect(offer.applicable).toContain(kind)
    expect(offer.unavailable['enumeration']).toMatch(/\w/)
    expect(Object.keys(offer.space.options)).toEqual(expect.arrayContaining([...offer.applicable, 'enumeration']))
  })

  it('rebuilds the problem from its knobs, the same every time', () => {
    const config = {
      dataset: 'circles',
      rows: 100,
      noise: 0.2,
      dataSeed: 3,
      link: 'logistic',
      degree: 3,
      priorScale: 2,
    }
    const a = classifierProblem(config)
    const b = classifierProblem(config)
    expect(a.train.x.length + a.test.x.length).toBe(100)
    expect(a.map.size).toBe(10)
    expect(a.train.x).toEqual(b.train.x)
  })

  for (const kind of ['nuts', 'hmc', 'mala', 'svi', 'laplace', 'ep', 'pathfinder']) {
    it(`${kind}: runs and every probe reads its posterior`, () => {
      const config = small(
        kind,
        kind === 'nuts' || kind === 'hmc' || kind === 'mala' ? { numChains: 2, numWarmup: 40, numDraws: 60 } : {},
      )
      const s = session(classifierMethod, {
        config: config as never,
        steps: classifierSteps(config as never),
        mode: 'eager',
      })
      expect(s.status).not.toBeUndefined()
      const surface = s.probe(s.t, 'classify/surface', { size: 7 }) as { kind: string; probabilities: number[][][] }
      expect(surface.kind).toBe('decisionSurface')
      expect(surface.probabilities[1]?.length).toBe(7)
      const scores = s.probe(s.t, 'classify/scores') as { testAccuracy: number; mean: number[] }
      expect(scores.testAccuracy).toBeGreaterThan(0.5)
      expect(scores.mean.every(Number.isFinite)).toBe(true)
      const pairs = s.probe(s.t, 'classify/pairs') as { kind: string; coordinates: unknown[] }
      expect(pairs.kind).toBe('posteriorPairs')
      const diag = s.probe(s.t, 'classify/diagnostics')
      expect(diag === null).toBe(!['nuts', 'hmc', 'mala'].includes(kind))
    })
  }

  it('reads the posterior at an earlier step too (the run is replayable)', () => {
    const config = small('nuts', { numChains: 2, numWarmup: 20, numDraws: 40 })
    const s = session(classifierMethod, {
      config: config as never,
      steps: classifierSteps(config as never),
      mode: 'eager',
    })
    const early = s.probe(40, 'classify/surface', { size: 5 }) as { kind: string } | null
    expect(early === null || early.kind === 'decisionSurface').toBe(true)
  })

  it('checks a method against a reference run of the same problem', () => {
    const run = (inference: Record<string, unknown>) => {
      const config = { ...small('laplace'), inference }
      return session(classifierMethod, {
        config: config as never,
        steps: classifierSteps(config as never),
        mode: 'eager',
      })
    }
    const reference = run(small('nuts', { numChains: 2, numWarmup: 60, numDraws: 100 }).inference)
    const given = {
      scores: reference.probe(reference.t, 'classify/scores'),
      surface: reference.probe(reference.t, 'classify/surface', { size: 9 }),
    }
    const laplace = run(small('laplace').inference)
    const check = laplace.probe(laplace.t, 'classify/check', given as never) as {
      maxMeanError: number
      maxSurfaceDifference: number
      referenceDraws: number
    }
    expect(check.referenceDraws).toBe(200)
    expect(check.maxMeanError).toBeLessThan(3)
    expect(check.maxSurfaceDifference).toBeLessThan(0.5)
  })
})
