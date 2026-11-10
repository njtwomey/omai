// The campaigns of the kernel methods run at a tiny budget: they are named and ordered, the Geweke test passes the adjusted kernels and
// catches the unadjusted one, and the calibration, agreement and two-mode campaigns run and write their tables. The committed tables
// come from `make probprog-validate` at the full budget.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { runCampaigns } from '../validation/index.ts'
import { benchmarkAgreement } from './agreement.ts'
import { logisticRegression } from './glm.ts'
import { kernelCampaigns, kernelPlans } from './kernel-campaigns.ts'

const campaigns = kernelCampaigns()

describe('kernelCampaigns', () => {
  it('lists the Geweke test, the calibration of every kernel method, the agreement tables and the two-mode target, uniquely named', () => {
    const names = campaigns.map((c) => c.name)
    expect(new Set(names).size).toBe(names.length)
    expect(names[0]).toBe('kernels/geweke')
    expect(names.filter((n) => n.startsWith('kernels/sbc-')).length).toBeGreaterThanOrEqual(12)
    expect(names).toContain('kernels/agreement-normal-normal')
    expect(names.at(-1)).toBe('kernels/bimodal')
    for (const c of campaigns) expect(c.description.length, c.name).toBeGreaterThan(30)
  })

  it('passes the adjusted kernels and catches the unadjusted one in the Geweke test', () => {
    const [run] = runCampaigns(campaigns, { only: ['kernels/geweke'], scale: 0.3 })
    expect(run?.passed, run?.verdict).toBe(true)
    expect(run?.markdown).toMatch(/ULA without the Metropolis step/)
  })

  it('runs a calibration, an agreement and the two-mode campaign and writes their tables', { timeout: 120_000 }, () => {
    const runs = runCampaigns(campaigns, {
      only: ['kernels/sbc-rwm', 'kernels/agreement-normal-normal', 'kernels/bimodal'],
      scale: 0.1,
      seed: 1,
    })
    expect(runs.map((r) => r.name)).toEqual(['kernels/sbc-rwm', 'kernels/agreement-normal-normal', 'kernels/bimodal'])
    for (const r of runs) {
      expect(r.markdown.length, r.name).toBeGreaterThan(50)
      expect(r.verdict.length, r.name).toBeGreaterThan(10)
    }
    expect(key(0)).toBeDefined()
  })
})

describe('kernelPlans', () => {
  it('reports on a hierarchy with a funnel the samplers that mix too slowly there, and requires the rest', () => {
    const reported = (hard: boolean): string[] =>
      kernelPlans(1, [], hard)
        .filter((p) => p.expect === 'report')
        .map((p) => p.kind)
    expect(reported(true)).toEqual(
      expect.arrayContaining(['rwm', 'mala', 'barker', 'aies', 'ensemble-slice', 'mclmc', 'sgld', 'sghmc']),
    )
    expect(reported(false)).not.toContain('rwm')
    expect(reported(false)).not.toContain('mala')
    for (const kind of ['hmc', 'slice', 'chees', 'parallel-tempering']) expect(reported(true), kind).not.toContain(kind)
  })

  // Random-walk Metropolis learns the shape of its proposal from the second half of the warm-up and keeps it: from the 200 draws
  // of a 400-step warm-up it failed the agreement on a three-dimensional logistic regression for seven seeds in eight (an effective
  // sample size of 5 to 100 in 3000 draws, and no better in 30000); with the warm-up of the plan it agrees.
  it('agrees with the quadrature truth of the logistic regression with random-walk Metropolis for every seed tried', {
    timeout: 60_000,
  }, () => {
    const model = {
      ...logisticRegression,
      plans: (scale: number) => kernelPlans(scale).filter((p) => p.kind === 'rwm'),
    }
    const truth = model.truth()
    for (const seed of [1, 3, 4]) {
      const a = benchmarkAgreement(model, { key: key(seed), truth })
      expect(
        a.failures.map((f) => `${f.row.quantity}: ${f.failed.join(', ')}`),
        `seed ${seed}`,
      ).toEqual([])
    }
  })
})
