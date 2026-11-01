import { describe, expect, it } from 'vitest'
import { bernoulli, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { add, asTensor, mul, tensor, toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { findInitialParams, initContract, initialParamsAlgorithm, potentialEnergy } from '../density.ts'
import { factor, observe, sample } from '../primitives.ts'
import { initStrategies, mcmcKernels } from './mcmc-options.ts'
import { inferDiscrete, inferDiscreteAlgorithm, mcmc, mcmcAlgorithm, mcmcContract, mcmcPosterior } from './nuts.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const logPolicy = { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' } as const
const model = (): void => {
  const mu = sample('mu', normal(0, 1))
  observe('y', normal(mu, 1), 1)
}
const options = { key: key(4), numChains: 2, numWarmup: 60, numDraws: 30 }

describe('mcmcAlgorithm', () => {
  it('finds a start per chain, then samples, and gives the posterior of mcmc', () => {
    const run = runAlgorithm(mcmcAlgorithm(model, undefined, options), { maxSteps: 20 })
    expect([run.status, run.steps]).toEqual(['converged', 3])
    expect(run.state.starts).toHaveLength(2)
    expect(run.state.chains?.numDraws).toBe(30)
    const post = mcmcPosterior(model, undefined, options, run.state)
    const direct = mcmc(model, undefined, options)
    expect(post.status).toBe('completed')
    expect(Array.from(toFlat(post.draws?.['mu'] as never))).toEqual(Array.from(toFlat(direct.draws?.['mu'] as never)))
    expect(post.chains?.numChains).toBe(2)
  })

  it('serialises the chains and resumes after the starting points', () => {
    const alg = mcmcAlgorithm(model, undefined, options)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    expect(part.stage).toBe('sample')
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(Array.from(toFlat(rest.state.chains?.draws as never))).toEqual(
      Array.from(toFlat(whole.state.chains?.draws as never)),
    )
    // The finished state, chains included, survives a JSON round trip too.
    const stored = viaJson(alg.serialise?.(whole.state) ?? null)
    const back = alg.deserialise?.(stored)
    expect(back?.chains?.numChains).toBe(2)
    expect(Array.from(toFlat(back?.chains?.draws as never))).toEqual(
      Array.from(toFlat(whole.state.chains?.draws as never)),
    )
    expect(stateToJson(whole.state.starts)).toHaveLength(2)
  })

  it('logs the starting searches and the per-chain statistics', () => {
    const rec = recorder()
    runAlgorithm(mcmcAlgorithm(model, undefined, options), { maxSteps: 20, tracer: rec, policy: logPolicy })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('mcmc/chain')).toEqual([0, 1])
    const accept = rec.events.filter((e) => e.type === 'tensor' && e.name === 'mcmc/accept')
    expect(accept).toHaveLength(1)
    expect(Object.keys(mcmcContract)).toHaveLength(8)
  })

  it('fails without a valid starting point, and the posterior says why', () => {
    const broken = (): void => {
      sample('x', normal(0, 1))
      factor('impossible', Number.NEGATIVE_INFINITY)
    }
    const run = runAlgorithm(
      mcmcAlgorithm(broken, undefined, { key: key(0), numChains: 1, numWarmup: 5, numDraws: 5 }),
      { maxSteps: 20 },
    )
    expect(run.status).toBe('failed')
    const post = mcmcPosterior(broken, undefined, { key: key(0) }, run.state)
    expect(post.status).toBe('failed')
    expect(String(post.diagnostics['reason'])).toMatch(/starting point/)
  })

  it('recovers enumerated discrete sites one draw at a time', () => {
    const mixture = (): void => {
      const mu = sample('mu', normal(0, 1))
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(add(mu, mul(c, 2)), 1), 1.4)
    }
    const o = { key: key(2), numChains: 1, numWarmup: 30, numDraws: 8 }
    const run = runAlgorithm(mcmcAlgorithm(mixture, undefined, o), { maxSteps: 40 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(1 + 1 + 8)
    const post = mcmcPosterior(mixture, undefined, o, run.state)
    expect(post.draws?.['c']?.shape).toEqual([8])
    expect(toFlat(post.draws?.['c'] as never).every((v) => v === 0 || v === 1)).toBe(true)
  })
})

describe('initialParamsAlgorithm and inferDiscreteAlgorithm', () => {
  it('tries starting points one per step and equals findInitialParams', () => {
    const um = potentialEnergy(model, undefined)
    const alg = initialParamsAlgorithm(um, model, undefined, { key: key(3) })
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.status).toBe('converged')
    const direct = findInitialParams(um, model, undefined, { key: key(3) })
    expect(Array.from(run.state.last?.unconstrained ?? [])).toEqual(Array.from(direct.unconstrained))
    expect(direct.tries).toBe(run.state.tries)
    const back = alg.deserialise?.(viaJson(alg.serialise?.(run.state) ?? null))
    expect(back?.last?.potential).toBeCloseTo(direct.potential, 12)
    expect(Object.keys(initContract)).toHaveLength(3)
    expect(initStrategies.names).toEqual(['uniform', 'prior'])
  })

  it('draws the enumerated sites one per step, as inferDiscrete does', () => {
    const m = (): void => {
      const mu = sample('mu', normal(0, 1))
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(add(mu, mul(c, 2)), 1), 1.4)
    }
    const values = { mu: tensor(0.3) }
    const run = runAlgorithm(inferDiscreteAlgorithm(m, undefined, values, key(1)), { maxSteps: 10 })
    expect([run.status, run.steps]).toEqual(['converged', 2])
    expect(Object.keys(run.state.fixed).sort()).toEqual(['c', 'mu'])
    expect(toFlat(asTensor(run.state.fixed['c'] as never))).toEqual(
      toFlat(asTensor(inferDiscrete(m, undefined, values, key(1))['c'] as never)),
    )
  })
})

describe('mcmcKernels', () => {
  it('names the kernels and builds options', () => {
    expect(mcmcKernels.names).toEqual(['nuts', 'hmc'])
    expect(mcmcKernels.make({ kind: 'hmc', numSteps: 5 })).toMatchObject({ algorithm: 'hmc', numSteps: 5 })
    expect(mcmcKernels.make({ kind: 'nuts', numDraws: 100 })).toMatchObject({ algorithm: 'nuts', numDraws: 100 })
  })
})
