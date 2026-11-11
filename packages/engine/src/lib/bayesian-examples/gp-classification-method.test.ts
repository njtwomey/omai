import { describe, expect, it } from 'vitest'
import { defaults } from '../space/index.ts'
import { type Config, session } from '../trace/index.ts'
import { gpClassificationExample, gpClassificationMethod, gpClassificationProblem } from './gp-classification-method.ts'

const base = { ...(defaults(gpClassificationMethod.config) as Config), rows: 24 }

function run(over: Config, inference: Config): ReturnType<typeof session> {
  const offer = gpClassificationExample.offer({ ...base, ...over })
  const config = {
    ...base,
    ...over,
    inference: { ...(defaults(offer.space.options[inference['method'] as string] as never) as Config), ...inference },
  }
  return session(gpClassificationMethod, {
    config,
    steps: gpClassificationExample.steps(config),
    mode: 'eager',
  }) as never
}

describe('Gaussian process classification', () => {
  it('has a known truth on the grid for the synthetic data sets and none for the library ones', () => {
    for (const dataset of ['gp-draw', 'sinusoid', 'step', 'circle', 'xor']) {
      const p = gpClassificationProblem({ ...base, dataset })
      expect(p.truthProbability?.length).toBe(p.grid.length)
      expect(p.x.length).toBe(24)
    }
    const moons = gpClassificationProblem({ ...base, dataset: 'moons' })
    expect(moons.truthProbability).toBeNull()
    expect(moons.dim).toBe(2)
    expect(moons.grid.length).toBe(441)
  })

  it('offers message passing with fixed hyperparameters and says why not with inferred ones', () => {
    const fixed = gpClassificationExample.offer(base)
    expect(fixed.applicable).toEqual(expect.arrayContaining(['nuts', 'ep', 'vmp', 'laplace', 'svi']))
    const inferred = gpClassificationExample.offer({ ...base, hyper: 'inferred' })
    expect(inferred.applicable).not.toContain('ep')
    expect(inferred.unavailable['ep']).toMatch(/opaque/)
  })

  it('EP reads a predictive curve close to the Laplace fit of the gp domain', () => {
    const s = run({}, { method: 'ep' })
    const fit = s.probe(s.t, 'gpc/fit') as { probability: { mean: number[]; lo: number[]; hi: number[] }; dim: number }
    expect(fit.dim).toBe(1)
    expect(fit.probability.mean.every((v) => v > 0 && v < 1)).toBe(true)
    expect(fit.probability.lo.every((v, i) => v <= (fit.probability.mean[i] ?? 0) + 1e-9)).toBe(true)
    const ref = run({}, { method: 'nuts', numChains: 2, numWarmup: 80, numDraws: 150 })
    const check = s.probe(s.t, 'gpc/check', { reference: ref.probe(ref.t, 'gpc/draws') } as never) as {
      maxMeanError: number
      extras: { name: string; vsReference: number }[]
    }
    expect(check.maxMeanError).toBeLessThan(0.6)
    expect(check.extras.map((e) => e.name)).toEqual(['Laplace (gp domain)'])
  }, 60000)

  it('infers the hyperparameters and reads their draws (a variational run)', () => {
    const s = run({ hyper: 'inferred' }, { method: 'svi' })
    const scores = s.probe(s.t, 'gpc/scores') as { ell: { mean: number } | null; variance: { mean: number } | null }
    expect(scores.ell?.mean).toBeGreaterThan(0)
    expect(scores.variance?.mean).toBeGreaterThan(0)
    const pairs = s.probe(s.t, 'gpc/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
  })

  it('draws a surface for two inputs and the probit link also offers EP from the gp domain', () => {
    const s = run({ dataset: 'circle', link: 'probit' }, { method: 'laplace' })
    const fit = s.probe(s.t, 'gpc/fit') as { dim: number; probability: { mean: number[] } }
    expect(fit.dim).toBe(2)
    expect(fit.probability.mean.length).toBe(441)
    const ref = run(
      { dataset: 'circle', link: 'probit' },
      { method: 'nuts', numChains: 2, numWarmup: 60, numDraws: 80 },
    )
    const check = s.probe(s.t, 'gpc/check', { reference: ref.probe(ref.t, 'gpc/draws') } as never) as {
      extras: { name: string }[]
    }
    expect(check.extras.map((e) => e.name)).toEqual(['Laplace (gp domain)', 'EP (gp domain)'])
  }, 60000)
})
