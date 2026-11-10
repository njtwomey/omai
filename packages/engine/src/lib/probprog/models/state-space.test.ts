// The AR(1) state space as a scan: every planned method applies to its spec and runs, and the plans that must agree with the
// Kalman truth do.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { benchmarkAgreement, runPlan } from './agreement.ts'
import { methodVerdicts } from './benchmark.ts'
import { ar1StateSpace } from './state-space.ts'

describe('the AR(1) scan benchmark', () => {
  const verdicts = methodVerdicts(ar1StateSpace)
  it('every planned method applies to the spec, and runs', () => {
    for (const plan of ar1StateSpace.plans(0.05)) {
      expect(verdicts.find((v) => v.kind === plan.kind)?.verdict.applies, plan.kind).toBe(true)
      const run = runPlan(ar1StateSpace, plan, { key: key(1) })
      expect(run.status, plan.kind).not.toBe('failed')
      expect(run.posterior.sites.length, plan.kind).toBeGreaterThan(0)
    }
  })
  it('the Kalman smoother and the Laplace fit agree with the truth', () => {
    const a = benchmarkAgreement(ar1StateSpace, { only: ['gaussian', 'laplace'], scale: 0.3 })
    expect(a.passed).toBe(true)
  })
})
