// The variational plans: every plan names a kind the method table has, the campaigns are named per model, and at a small
// budget the exact methods still agree with the truth of the conjugate models (the rest is the periodic campaign's business).
import { describe, expect, it } from 'vitest'
import { methodKinds } from '../inference/index.ts'
import { benchmarkAgreement } from './agreement.ts'
import { gammaPoisson, normalNormal } from './conjugate.ts'
import { sviCampaigns, sviPlansFor } from './svi-plans.ts'

describe('variational plans', () => {
  it('name method kinds that exist, for every model they cover', () => {
    for (const name of ['normal-normal', 'gamma-poisson', 'logistic-regression', 'banana', 'bimodal']) {
      const plans = sviPlansFor(name, 1)
      expect(plans.length, name).toBeGreaterThan(0)
      for (const p of plans) expect(Object.keys(methodKinds), `${name}: ${p.kind}`).toContain(p.kind)
    }
    expect(sviPlansFor('no-such-model', 1)).toEqual([])
    expect(sviCampaigns().map((c) => c.name)).toEqual([
      'svi/normal-normal',
      'svi/gamma-poisson',
      'svi/logistic-regression',
      'svi/banana',
      'svi/bimodal',
    ])
  })

  it('conjugate natural-gradient SVI and Pathfinder agree with the exact posterior of the conjugate models', () => {
    for (const model of [normalNormal, gammaPoisson]) {
      const wrapped = { ...model, plans: (scale = 1) => sviPlansFor(model.name, scale) }
      const a = benchmarkAgreement(wrapped, { scale: 0.05, only: ['svi-conjugate-natural', 'pathfinder'] })
      expect(a.passed, model.name).toBe(true)
      expect(a.runs.map((r) => r.plan.kind).sort()).toEqual(['pathfinder', 'svi-conjugate-natural'])
    }
  })
})
