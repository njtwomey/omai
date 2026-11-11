// Priors with no message family (half-normal, half-Cauchy, log-normal, uniform, truncated normal) as the exact numeric `prior-site` (with nothing else on the variable its message is the Gaussian projection
// of the density), positive scales carried on their log, a latent with a random scale in its non-centred form, and an observed value with a random
// scale (`gaussian-scale`): compiled to factor graphs and run under EP, against the exact posterior on a grid.
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../message-passing/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { eightSchoolsMarginal } from '../models/schools.ts'
import { gridPosterior } from '../validation/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph, toProgram } from './index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

type Builder = Parameters<Parameters<typeof defineModel>[1]>[0]

function ep(def: ReturnType<typeof defineModel>, data: Record<string, unknown>, damping = 0.5) {
  const bound = def.bind(data as never)
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = graphOf(c.graph)
  expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-8, damping }), { maxSteps: 800 })
  expect(r.status).toBe('converged')
  const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
  const read = (name: string): { mean: number; sd: number } => ({
    mean: toFlat(post.mean(name))[0] as number,
    sd: post.summary().find((s) => s.site === name)?.sd as number,
  })
  return { bound, read, graph: c.graph as FactorGraphSpec }
}

describe('priors without a message family, an exact numeric site', () => {
  const prior = (family: () => ReturnType<typeof d.halfNormal>): ReturnType<typeof defineModel> =>
    defineModel('p', (m: Builder) => {
      m.latent('t', family() as never)
    })

  it('a half-normal scale is a Gaussian on its log (the moments of log |Z|) and exp of it as the value', () => {
    const { read, graph } = ep(
      prior(() => d.halfNormal(2)),
      {},
    )
    expect(graph.factors.map((f) => f.operator)).toEqual(['prior-site', 'apply'])
    // log tau ~ N(log 2 - (gamma + log 2) / 2, pi^2 / 8), so tau has the log-normal moments of that.
    const m = Math.log(2) - (0.5772156649015329 + Math.log(2)) / 2
    const v = Math.PI ** 2 / 8
    const t = read('t')
    expect(t.mean).toBeCloseTo(Math.exp(m + v / 2), 4)
    expect(t.sd).toBeCloseTo(Math.exp(m + v / 2) * Math.sqrt(Math.exp(v) - 1), 3)
  })

  it('a log-normal is exact on the log scale', () => {
    const { read } = ep(
      prior(() => d.logNormal(0.3, 0.7)),
      {},
    )
    expect(read('t').mean).toBeCloseTo(Math.exp(0.3 + 0.49 / 2), 5)
  })

  it('a uniform on (-1, 1) is a Gaussian with its mean and variance on the value (the bound is what is lost)', () => {
    const { read } = ep(
      prior(() => d.uniform(-1, 1)),
      {},
    )
    expect(read('t').mean).toBeCloseTo(0, 5)
    expect(read('t').sd).toBeCloseTo(Math.sqrt(1 / 3), 4)
  })

  it('a normal truncated to the positive half line is carried on its log', () => {
    const { graph } = ep(
      prior(() => d.truncatedNormal(1, 2, 0, 20)),
      {},
    )
    expect(graph.factors.map((f) => f.operator)).toEqual(['prior-site', 'apply'])
  })
})

describe('a positive scale in a hierarchy', () => {
  it('eight schools, centred and non-centred, with a half-Cauchy spread: mu and tau against the exact marginal posterior', () => {
    const y = [28, 8, -3, 7, -1, 1, 18, 12]
    const sigma = [15, 10, 16, 11, 9, 11, 10, 18]
    const exact = gridPosterior(toProgram(eightSchoolsMarginal.bind({ y, sigma })), undefined, {
      points: 61,
      maxRefinements: 1,
    })
    for (const centred of [true, false]) {
      const def = defineModel(`schools-${centred}`, (m: Builder) => {
        const school = m.range('school')
        const mu = m.latent('mu', d.normal(0, 5))
        const tau = m.latent('tau', d.halfCauchy(5))
        const sd = m.data('sigma', { dims: [school], support: 'positive' })
        const obs = m.data('y', { dims: [school] })
        if (centred) {
          const theta = m.latent('theta', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), { dims: [school] })
          m.observe('y', d.normal(theta, sd), obs)
        } else {
          const eta = m.latent('eta', d.normal(0, 1), { dims: [school] })
          m.observe('y', d.normal(mu.add(tau.mul(eta)), sd), obs)
        }
      })
      const out = ep(def, { y, sigma })
      for (const name of ['mu', 'tau']) {
        const ref = exact.marginal(name)
        expect(
          Math.abs(out.read(name).mean - ref.mean) / ref.sd,
          `${centred ? 'centred' : 'non-centred'} mean of ${name}`,
        ).toBeLessThan(0.3)
        expect(out.read(name).sd / ref.sd, `${centred ? 'centred' : 'non-centred'} sd of ${name}`).toBeGreaterThan(0.75)
        expect(out.read(name).sd / ref.sd).toBeLessThan(1.3)
      }
    }
  })

  it('a latent with an exp(v / 2) scale: the centred funnel has the exact sd exp(9 / 4) of its coordinates', () => {
    const def = defineModel('funnel', (m: Builder) => {
      const v = m.latent('v', d.normal(0, 3))
      m.latent('x', d.normal(0, ops.exp(v.mul(0.5))), { dims: [m.range('coord', { size: 4 })] })
    })
    const { read } = ep(def, {})
    expect(read('v').sd).toBeCloseTo(3, 4)
    // x_i given v is N(0, e^v): the Gaussian belief of its product has variance E[e^v] = exp(9 / 2).
    expect(read('x').sd).toBeCloseTo(Math.exp(9 / 4), 3)
  })

  it('an observed value with a half-normal scale and an unknown mean agrees with the exact two-dimensional posterior', () => {
    const y = [0.8, -0.4, 1.9, 1.2, 0.1, 2.6, -1.1, 0.9]
    const def = defineModel('scale', (m: Builder) => {
      const row = m.range('row')
      const mu = m.latent('mu', d.normal(0, 3))
      const sigma = m.latent('sigma', d.halfNormal(2))
      m.observe('y', d.normal(mu, sigma), m.data('y', { dims: [row] }))
    })
    const out = ep(def, { y })
    const exact = gridPosterior(toProgram(out.bound), undefined, { points: 101, maxRefinements: 1 })
    for (const name of ['mu', 'sigma']) {
      const ref = exact.marginal(name)
      expect(Math.abs(out.read(name).mean - ref.mean) / ref.sd, `mean of ${name}`).toBeLessThan(0.3)
      expect(out.read(name).sd / ref.sd, `sd of ${name}`).toBeGreaterThan(0.75)
      expect(out.read(name).sd / ref.sd).toBeLessThan(1.3)
    }
  })

  it('a scale that is not a log-scale latent keeps its refusal with the precision idiom named', () => {
    const def = defineModel('gamma-scale', (m: Builder) => {
      const s = m.latent('s', d.gamma(2, 2))
      m.observe('y', d.normal(0, s), m.data('y', { dims: ['row'] }))
    })
    const c = toFactorGraph(def.bind({ y: [1, 2] }))
    expect(c.issues[0]?.reason).toMatch(/precision idiom|1 \/ sqrt/)
  })
})
