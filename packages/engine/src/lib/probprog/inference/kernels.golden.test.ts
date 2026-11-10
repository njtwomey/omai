// The kernel methods against the samplers of NumPyro (AIES, ESS) and BlackJAX (MALA, Barker, MCLMC, SGLD, SGHMC) on the real-valued
// programs `mvn` and `logistic`: the means and standard deviations of long reference chains, compared within the Monte Carlo errors of
// both (a fixture stores the summary and the errors, never draws). The stochastic-gradient samplers are biased by their step size
// in both libraries, so they also get a slack in posterior standard deviations.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { MODELS } from '../../../testing/probprog.ts'
import { ess, parameterDraws } from '../../sampling/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Config, session } from '../../trace/index.ts'
import { aiesMethod, ensembleSliceMethod } from './ensemble-methods.ts'
import { barkerMethod, malaMethod, mclmcMethod, sghmcMethod, sgldMethod } from './mcmc-methods.ts'
import type { AnyInferenceMethod } from './models.ts'

const WARM = 300
const RUNS: Record<string, { make: (m: never) => AnyInferenceMethod; config: Config; slack: number }> = {
  aies: { make: aiesMethod, config: { numWalkers: 12, numWarmup: WARM, numDraws: 800 }, slack: 0 },
  ess: { make: ensembleSliceMethod, config: { numWalkers: 12, numWarmup: WARM, numDraws: 300 }, slack: 0 },
  mala: { make: malaMethod, config: { numChains: 4, numWarmup: WARM, numDraws: 1500 }, slack: 0 },
  barker: { make: barkerMethod, config: { numChains: 4, numWarmup: WARM, numDraws: 1500 }, slack: 0 },
  mclmc: { make: mclmcMethod, config: { numChains: 4, numWarmup: 600, numDraws: 1500 }, slack: 0.05 },
  sgld: {
    make: sgldMethod,
    config: { numChains: 4, numWarmup: WARM, numDraws: 3000, stepSize: 0.01, trackDensity: false },
    slack: 0.12,
  },
  sghmc: {
    make: sghmcMethod,
    config: {
      numChains: 2,
      numWarmup: WARM,
      numDraws: 1500,
      stepSize: 0.05,
      numSteps: 10,
      friction: 0.5,
      trackDensity: false,
    },
    slack: 0.12,
  },
}

describe('the kernel methods against NumPyro and BlackJAX', () => {
  for (const c of loadFixture('probprog/inference/kernels').cases) {
    const sampler = c.params['sampler'] as string
    it(c.name, { timeout: 120_000 }, () => {
      const run = RUNS[sampler]
      if (run === undefined) throw new Error(`no run for ${sampler}`)
      const model = MODELS[c.params['model'] as string] as never
      const method = run.make(model)
      const s = session(method, { config: run.config, steps: 100_000, mode: 'eager', seed: 11 })
      const chains = method.posterior?.(s.state, s.config).chains
      expect(chains).toBeDefined()
      if (chains === undefined) return
      const ref = toFlat(c.output as never) as ArrayLike<number>
      const d = chains.dim
      for (let i = 0; i < d; i++) {
        const x = parameterDraws(chains, i)
        const n = x.data.length
        const mean = x.data.reduce((a, v) => a + v, 0) / n
        const sd = Math.sqrt(x.data.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1))
        const neff = Math.max(5, ess(x))
        const [rm, rs, , rme, rse] = [0, 1, 2, 3, 4].map((r) => ref[r * d + i] ?? Number.NaN)
        const seMean = Math.hypot(sd / Math.sqrt(neff), rme as number)
        const seSd = Math.hypot(sd / Math.sqrt(2 * neff), rse as number)
        expect(Math.abs(mean - (rm as number)), `${c.name} mean of ${chains.names[i]}: ${mean} vs ${rm}`).toBeLessThan(
          4 * seMean + run.slack * (rs as number),
        )
        expect(Math.abs(sd - (rs as number)), `${c.name} sd of ${chains.names[i]}: ${sd} vs ${rs}`).toBeLessThan(
          4 * seSd + run.slack * (rs as number),
        )
      }
    })
  }
})
