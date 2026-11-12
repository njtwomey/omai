// Tied groups of latents (`findGroups`) against exact references: the floor effect and the county intercepts of a hierarchy, and the
// skills of a tournament (a vector that each game reads at two places), packed as one joint Gaussian vector so that the likelihood
// sees how they move together. Before packing their spreads were 0.71 and 0.76 to 0.85 of the exact ones.
import { describe, expect, it } from 'vitest'
import { session } from '../../../trace/index.ts'
import { runEp } from '../../ep/index.ts'
import { epSpecMethod } from '../../inference/index.ts'
import { benchmarkModel } from '../../models/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops } from '../index.ts'

/** The sd ratio and the distance in reference sds of every marginal of a benchmark model run under EP. */
function againstTruth(name: string): {
  ratios: Record<string, number>
  zs: Record<string, number>
  converged: boolean
} {
  const model = benchmarkModel(name)
  const run = runEp(model.bound(), { light: true })
  if (run.status !== 'ran') throw new Error(`${name}: ${run.status}`)
  const truth = model.truth(undefined, 'fast')
  const ratios: Record<string, number> = {}
  const zs: Record<string, number> = {}
  for (const [site, oracle] of Object.entries(truth.marginals)) {
    const belief = run.beliefs[site]
    if (belief === undefined) continue
    ratios[site] = belief.sd / oracle.sd
    zs[site] = Math.abs(belief.mean - oracle.mean) / oracle.sd
  }
  return { ratios, zs, converged: run.converged }
}

describe('the hierarchical radon regression (county intercepts and the floor effect tied to one vector)', () => {
  const { ratios, zs, converged } = againstTruth('radon')

  it('converges', () => {
    expect(converged).toBe(true)
  })

  it('has the spread of the floor effect within 10% of the exact one (it was 0.71 of it) and the intercepts within 15%', () => {
    expect(ratios['beta']).toBeGreaterThan(0.88)
    expect(ratios['beta']).toBeLessThan(1.1)
    for (let j = 0; j < 6; j++) {
      expect(ratios[`alpha[${j}]`]).toBeGreaterThan(0.85)
      expect(ratios[`alpha[${j}]`]).toBeLessThan(1.1)
    }
  })

  it('keeps every mean within a quarter of an exact sd', () => {
    for (const z of Object.values(zs)) expect(z).toBeLessThan(0.25)
  })
})

describe('the tournament (the skills are one vector that each game reads at two places)', () => {
  it('has the spreads of the exact posterior of three players (0.74 to 0.85 of it before) and settles in a few sweeps', () => {
    const { ratios, zs } = againstTruth('trueskill-three-players')
    for (const site of ['skill[0]', 'skill[1]', 'skill[2]']) {
      expect(ratios[site]).toBeGreaterThan(0.97)
      expect(ratios[site]).toBeLessThan(1.03)
      expect(zs[site]).toBeLessThan(0.02)
    }
  })

  it('has an evidence for a tournament whose spread is learned (its own, or that of the model without the ties)', () => {
    const tournament = defineModel('tournament', (m) => {
      const player = m.range('player', { size: 4 })
      const game = m.range('game')
      const a = m.data('a', { dims: [game], support: 'index', indexes: player })
      const b = m.data('b', { dims: [game], support: 'index', indexes: player })
      const tau = m.latent('tau', d.gamma(2, 2))
      const skill = m.latent('skill', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [player] })
      const gap = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
      m.observe('win', d.bernoulli({ probs: ops.probit(gap) }), m.data('win', { dims: [game] }))
    })
    const bound = tournament.bind({ a: [0, 1, 2, 3, 0, 2], b: [1, 2, 3, 0, 2, 1], win: [1, 1, 0, 1, 1, 0] })
    const method = epSpecMethod(bound)
    const s = session(method, { seed: 1, steps: 300, mode: 'eager', config: { tol: 1e-8 } })
    const posterior = method.posterior?.(s.state, s.config)
    expect(posterior?.status).toBe('converged')
    expect(Number.isFinite(posterior?.logEvidence?.value ?? Number.NaN)).toBe(true)
    expect(posterior?.logEvidence?.method).toMatch(/^ep-energy/)
  })
})
