// The evidence estimators as methods, each against the closed-form log evidence of the conjugate benchmark models (a normal
// mean, a gamma-Poisson rate; the beta-Bernoulli one is in the periodic campaign) within its stated error; each resumes from a serialised state,
// snapshots on the cadence of a fake clock and cancels between steps; `check` refuses discrete models; and the lookup lists
// the harmonic mean as refused with the reason.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { DISCRETE, MODELS } from '../../../testing/probprog.ts'
import { normal } from '../../dists/index.ts'
import { type Config, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { gammaPoisson, normalNormal } from '../models/conjugate.ts'
import { observe, sample } from '../primitives.ts'
import {
  bridgeSamplingMethod,
  evidenceEstimators,
  importanceEvidenceMethod,
  steppingStoneMethod,
  thermodynamicIntegrationMethod,
} from './evidence-methods.ts'
import type { InferenceMethod } from './method.ts'
import type { ModelAnalysis } from './model-input.ts'
import { nestedMethod } from './nested-method.ts'

type Any = InferenceMethod<never, Config, string, ModelAnalysis>

const cases: readonly {
  name: string
  make: (m: never) => Any
  config: Partial<Config>
  steps: number
  slack: number
}[] = [
  {
    name: 'stepping-stone',
    make: steppingStoneMethod as never,
    config: { numParticles: 250, targetEss: 0.95 },
    steps: 300,
    slack: 0.04,
  },
  {
    name: 'thermodynamic-integration',
    make: thermodynamicIntegrationMethod as never,
    config: { numParticles: 250, targetEss: 0.95 },
    steps: 300,
    slack: 0.06,
  },
  { name: 'bridge-sampling', make: bridgeSamplingMethod as never, config: { numDraws: 400 }, steps: 2000, slack: 0.03 },
  { name: 'importance-evidence', make: importanceEvidenceMethod as never, config: {}, steps: 100, slack: 0.03 },
  { name: 'nested', make: nestedMethod as never, config: { numLive: 80, innerSteps: 3 }, steps: 20000, slack: 0.03 },
]

describe('the evidence estimators against the closed-form evidence of the conjugate benchmarks', () => {
  for (const benchmark of [normalNormal, gammaPoisson]) {
    it(`${benchmark.name}: every estimator is within its stated error of the exact log evidence`, () => {
      const truth = benchmark.truth().logEvidence ?? Number.NaN
      for (const c of cases) {
        const method = c.make(benchmark.methodModel() as never)
        const s = session(method as never, { steps: c.steps, mode: 'eager', seed: 3, config: c.config })
        expect(['converged', 'completed'], `${c.name} on ${benchmark.name}: ${s.status}`).toContain(s.status)
        const z = (method as InferenceMethod<unknown, Config, string, ModelAnalysis>).posterior?.(
          s.state,
          s.config,
        ).logEvidence
        const se = z?.standardError ?? 0
        expect(Math.abs((z?.value ?? Number.NaN) - truth), `${c.name} on ${benchmark.name}`).toBeLessThan(
          4 * (Number.isFinite(se) ? se : 0) + c.slack,
        )
      }
    }, 60_000)
  }
})

describe('the evidence methods as methods', () => {
  const model = normalNormal.methodModel() as never
  const makers: readonly [string, (m: never) => Any, Partial<Config>][] = [
    ['stepping-stone', steppingStoneMethod as never, { numParticles: 60, targetEss: 0.97 }],
    ['thermodynamic-integration', thermodynamicIntegrationMethod as never, { numParticles: 60, targetEss: 0.97 }],
    ['bridge-sampling', bridgeSamplingMethod as never, { numWarmup: 5, fitDraws: 10, numDraws: 30 }],
    ['importance-evidence', importanceEvidenceMethod as never, { batchSize: 20, pilotBatches: 2, numBatches: 6 }],
  ]
  for (const [name, make, config] of makers) {
    it(`${name} resumes from a serialised state, snapshots on the cadence of a clock and cancels between steps`, () => {
      const method = make(model) as never
      expectResumeEqual(method, { config, steps: 60, at: 3 })
      expectSnapshotCadence(method, { config, steps: 60, budgetMs: 300, everyMs: 100, tick: 10 })
      expectCancelAndContinue(method, { config, steps: 60, cancelAfterSlices: 2 })
    })
  }

  it('log the estimate and its error on the tape, and the estimate exists at every step', () => {
    for (const [name, make, config] of makers) {
      const method = make(model)
      const tape = session(method as never, { steps: 40, mode: 'eager', config, logs: 'all' }).run()
      const names = new Set(tape.events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
      expect(names.has('evidence/estimate'), name).toBe(true)
    }
  })

  it('check refuses a model with discrete latent sites, and a run throws that reason', () => {
    const mixture = DISCRETE['mixture'] as never
    for (const [name, make] of makers) {
      const method = make(mixture)
      expect(method.check?.(analyse(mixture, undefined) as never)?.applies, name).toBe(false)
      expect(() => session(method as never, { steps: 1, mode: 'eager' }), name).toThrow(/discrete/)
    }
    expect(
      bridgeSamplingMethod(MODELS['normalMean'] as never).check?.(analyse(MODELS['normalMean'] as never, undefined))
        ?.applies,
    ).toBe(true)
  })

  it('importance sampling reports a Pareto k-hat, and plain prior sampling on a sharp likelihood is flagged by it', () => {
    // m ~ N(0, 10^2), y = 3 observed with sd 0.05: the posterior is a sliver of the prior.
    const sharp = () => {
      const m = sample('m', normal(0, 10))
      observe('y', normal(m, 0.05), 3)
    }
    const exact = -0.5 * Math.log(2 * Math.PI * (100 + 0.0025)) - 9 / (2 * (100 + 0.0025))
    const run = (proposal: 'prior' | 'adaptive-normal') => {
      const method = importanceEvidenceMethod(sharp)
      const s = session(method, { steps: 100, mode: 'eager', seed: 1, config: { proposal, batchSize: 200 } })
      const post = method.posterior?.(s.state, s.config)
      return {
        z: post?.logEvidence?.value ?? Number.NaN,
        k: post?.diagnostics['paretoK'] as number,
        ess: post?.diagnostics['ess'] as number,
      }
    }
    const plain = run('prior')
    const fitted = run('adaptive-normal')
    expect(fitted.k).toBeLessThan(0.7)
    expect(Math.abs(fitted.z - exact)).toBeLessThan(0.1)
    expect(plain.k).toBeGreaterThan(fitted.k)
    expect(plain.ess).toBeLessThan(fitted.ess / 5)
  })
})

describe('the lookup of evidence estimators', () => {
  it('lists every estimator with a way to make it, and the harmonic mean as refused with the reason', () => {
    expect(Object.keys(evidenceEstimators)).toEqual(
      expect.arrayContaining([
        'bridge-sampling',
        'stepping-stone',
        'thermodynamic-integration',
        'importance-evidence',
        'nested',
        'tempered-smc',
        'harmonic-mean',
      ]),
    )
    for (const [name, e] of Object.entries(evidenceEstimators)) {
      if (name === 'harmonic-mean') {
        expect(e.make).toBeUndefined()
        expect(e.refused).toMatch(/infinite variance/)
      } else expect(typeof e.make, name).toBe('function')
    }
  })
})
