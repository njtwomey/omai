// The chain of an unrolled scan as one factor: the stochastic volatility model with the persistence fixed and the level and the
// innovation precision unknown compiles to one `gaussian-chain` factor on the states, the level and the precision (a `linear` factor and a
// `gaussian-gamma-precision` factor per step without `packScalars`), and EP on it has the posterior of the level and the precision that a
// point-mass filter gives (the likelihood of each (level, precision) on a grid with the prior, 120 states over -6.5 to 4.5), at a
// persistence of 0.69, of 0.1 and of 0.9. The scalar form's level is twice as wide as the exact one and its precision does not settle at
// a low persistence (the case that was open): the numbers of both are here.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../../message-passing/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import { bind } from '../bind.ts'
import { doSpec } from '../compose.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../index.ts'

const T = 60
const TRUTH = { mu: -1, phi: 0.9, tau: 11.1 }
const unit = (i: number, s: number): number => {
  const x = Math.sin(i * 12.9898 + s * 78.233) * 43758.5453
  return (x - Math.floor(x)) * 0.999998 + 1e-6
}
const gauss = (i: number, s: number): number =>
  Math.sqrt(-2 * Math.log(unit(i, s))) * Math.cos(2 * Math.PI * unit(i + 7919, s * 3 + 1))

/** Returns of a log-variance that follows an AR(1) around a level (seed 1; the return at step 33 is 0.004: a likelihood with a flat tail). */
function simulate(seed: number): number[] {
  const sigma = TRUTH.tau ** -0.5
  let h = TRUTH.mu + (sigma / Math.sqrt(1 - TRUTH.phi ** 2)) * gauss(0, seed)
  const y: number[] = []
  for (let t = 0; t < T; t++) {
    if (t > 0) h = TRUTH.mu + TRUTH.phi * (h - TRUTH.mu) + sigma * gauss(2 * t, seed)
    y.push(Math.exp(h / 2) * gauss(2 * t + 1, seed))
  }
  return y
}

const model = defineModel('sv-learned', (b) => {
  const y = b.data('y', { dims: ['time'] })
  const mu = b.latent('mu', d.normal(-1, 1))
  const phi = b.latent('phi', d.normal(0.8, 0.3))
  const tau = b.latent('tau', d.gamma(3, 0.3))
  b.scan('h', {
    over: 'time',
    init: d.normal(-1, 1),
    step: ({ prev }) => d.normal(prev.sub(mu).mul(phi).add(mu), tau.pow(-0.5)),
    emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(state.mul(0.5))), y),
  })
})

/** The posterior of the level and the precision at a fixed persistence by a point-mass filter (seed 1, scratch script). */
const EXACT: Record<string, { mu: [number, number]; tau: [number, number] }> = {
  '0.69': { mu: [-0.886, 0.251], tau: [9.896, 5.337] },
  '0.1': { mu: [-0.8, 0.199], tau: [10.389, 5.689] },
  '0.9': { mu: [-0.963, 0.433], tau: [12.048, 5.696] },
}

/** EP on the model with the persistence fixed: the level's and precision's mean and sd, and the sweeps (a run that did not settle has none). */
function run(phi: number, packScalars: boolean) {
  const bound = model.bind({ y: simulate(1) })
  const conditioned = bind({ spec: doSpec(bound.spec, { phi }), customs: bound.customs }, bound.data, bound.options)
  const c = toFactorGraph(conditioned, { packScalars })
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-4, damping: 0.2, maxSweeps: 150, improper: 'skip' }), {
    maxSteps: 152,
  })
  const m = marginalsOf(g, r.state as MpState) as Record<string, Record<string, number>>
  const mu = m['mu'] as { mean: number; variance: number }
  const tau = m['tau'] as { mean: number; variance: number }
  return {
    factors: (c.graph?.factors ?? []).map((f) => f.operator),
    status: r.status,
    mu: [mu.mean, Math.sqrt(mu.variance)] as const,
    tau: [tau.mean, Math.sqrt(tau.variance)] as const,
  }
}

describe('the chain of a scan with a shared precision and a shared level', () => {
  it('compiles to one gaussian-chain factor on the states, the level and the precision, and to a pair of factors a step without packScalars', () => {
    const bound = model.bind({ y: simulate(1) })
    const conditioned = bind(
      { spec: doSpec(bound.spec, { phi: 0.69 }), customs: bound.customs },
      bound.data,
      bound.options,
    )
    const packed = toFactorGraph(conditioned, { packScalars: true })
    expect(packed.issues).toEqual([])
    const chain = packed.graph?.factors.filter((f) => f.operator === 'gaussian-chain') ?? []
    expect(chain).toHaveLength(1)
    expect(packed.graph?.factors.some((f) => f.operator === 'gaussian-gamma-precision')).toBe(false)
    const [factor] = chain
    expect(factor?.args).toEqual(['tau', ...Array.from({ length: T }, (_, t) => `h[${t}]`), 'mu'])
    expect(factor?.params?.['length']).toBe(T)
    expect(factor?.params?.['coefficient'] as number).toBeCloseTo(0.69, 12)
    expect(((factor?.params?.['drive'] ?? []) as number[])[0]).toBeCloseTo(0.31, 12)
    // it comes before the readings of the states it links
    const position = packed.graph?.factors.indexOf(factor as never) as number
    expect(
      packed.graph?.factors.findIndex((f) => f.operator === 'gaussian-log-scale' && f.args[0] === 'y[5]'),
    ).toBeGreaterThan(position)
    // the unpacked form is the scalar steps
    const scalar = toFactorGraph(conditioned)
    expect(scalar.graph?.factors.filter((f) => f.operator === 'gaussian-gamma-precision')).toHaveLength(T - 1)
    expect(scalar.graph?.factors.some((f) => f.operator === 'gaussian-chain')).toBe(false)
  })

  it('has the exact level and precision at a persistence of 0.69, 0.1 and 0.9 (the means within a fifth of a sd, the spreads within a tenth)', () => {
    for (const phi of [0.69, 0.1, 0.9]) {
      const exact = EXACT[String(phi)] as (typeof EXACT)[string]
      const ep = run(phi, true)
      expect(ep.status, `phi ${phi}`).toBe('converged')
      expect(Math.abs(ep.mu[0] - exact.mu[0]), `mu mean ${phi}`).toBeLessThan(0.2 * exact.mu[1])
      expect(ep.mu[1] / exact.mu[1], `mu sd ${phi}`).toBeGreaterThan(0.9)
      expect(ep.mu[1] / exact.mu[1], `mu sd ${phi}`).toBeLessThan(1.1)
      expect(Math.abs(ep.tau[0] - exact.tau[0]), `tau mean ${phi}`).toBeLessThan(0.2 * exact.tau[1])
      expect(ep.tau[1] / exact.tau[1], `tau sd ${phi}`).toBeGreaterThan(0.9)
      expect(ep.tau[1] / exact.tau[1], `tau sd ${phi}`).toBeLessThan(1.1)
    }
  }, 60_000)

  it('is what the scalar steps could not do: their level is about twice as wide as the exact one (0.47 against 0.25 at 0.69)', () => {
    const scalar = run(0.69, false)
    expect(scalar.factors).toContain('gaussian-gamma-precision')
    const exact = EXACT['0.69'] as (typeof EXACT)[string]
    expect(scalar.mu[1] / exact.mu[1]).toBeGreaterThan(1.6)
  }, 60_000)
})
