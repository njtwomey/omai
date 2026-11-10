// The campaigns of the kernel methods run at a tiny budget: they are named and ordered, the Geweke test passes the adjusted kernels and
// catches the unadjusted one, and the calibration, agreement and two-mode campaigns run and write their tables. The committed tables
// come from `make probprog-validate` at the full budget.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { runCampaigns } from '../validation/index.ts'
import { kernelCampaigns } from './kernel-campaigns.ts'

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
