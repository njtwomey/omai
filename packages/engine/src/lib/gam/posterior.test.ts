/**
 * The posterior samplers against the Gaussian approximation: on a Poisson additive model with moderate counts the
 * Hamiltonian and Metropolis chains centre on the fitted coefficients with the spread of the approximation; the same
 * for an extended (beta regression) model; the chain is reproducible and its state serialises and resumes.
 */

import { describe, expect, it } from 'vitest'
import { families } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  extendedModel,
  extendedPosteriorTarget,
  fitContext,
  fitExtended,
  fitGam,
  GAM_DEFAULTS,
  gamPosteriorTarget,
  gamProblem,
  posteriorKernel,
  posteriorSampler,
  s,
  samplePosterior,
} from './index.ts'

const n = 120
const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
const counts = (level: number): number[] =>
  x.map((r, i) => Math.round(Math.exp(level + 0.8 * Math.sin(2 * Math.PI * (r[0] ?? 0))) + 0.7 * Math.sin(i * 12.9898)))

function setup(level: number) {
  const hyper = { ...GAM_DEFAULTS, family: families.poisson(), terms: [s(0, { k: 7 })] }
  const problem = gamProblem(
    x,
    counts(level).map((v) => Math.max(v, 0)),
    hyper,
  )
  const state = fitGam(problem, hyper).state
  const p = problem.design.p
  const covariance = fitContext(problem, state).vp
  return { problem, state, p, covariance, target: gamPosteriorTarget(problem, state) }
}

describe('the posterior target', () => {
  it('has a gradient equal to differences of its log density, and its mode is the fit', () => {
    const { state, target, p } = setup(1.2)
    const beta = Float64Array.from(state.beta)
    const g = target.valueAndGrad(beta).grad
    expect(Math.max(...Array.from(g, Math.abs))).toBeLessThan(1e-4 * (1 + Math.abs(target.logDensity(beta))))
    const moved = Float64Array.from(beta, (v, i) => v + 0.01 * Math.sin(i + 1))
    const gm = target.valueAndGrad(moved).grad
    for (let j = 0; j < p; j++) {
      const h = 1e-6
      const up = Float64Array.from(moved)
      const dn = Float64Array.from(moved)
      up[j] = (up[j] ?? 0) + h
      dn[j] = (dn[j] ?? 0) - h
      expect(gm[j] ?? 0).toBeCloseTo((target.logDensity(up) - target.logDensity(dn)) / (2 * h), 4)
    }
  })
})

describe('the posterior chains on a Poisson additive model', () => {
  const { state, target, p, covariance } = setup(1.2)
  const beta = Float64Array.from(state.beta)
  const sd = Array.from({ length: p }, (_, j) => Math.sqrt(covariance[j * p + j] ?? 0))

  for (const method of ['hmc', 'metropolis'] as const) {
    it(`${method}: centres on the fit with the spread of the Gaussian approximation`, () => {
      const out = samplePosterior(target, beta, {
        covariance,
        method,
        key: key(11),
        numDraws: method === 'hmc' ? 1500 : 6000,
        numWarmup: 200,
      })
      expect(out.acceptance).toBeGreaterThan(method === 'hmc' ? 0.6 : 0.15)
      for (let j = 0; j < p; j++) {
        let mean = 0
        for (let i = 0; i < out.count; i++) mean += out.draws[i * p + j] ?? 0
        mean /= out.count
        let v = 0
        for (let i = 0; i < out.count; i++) v += ((out.draws[i * p + j] ?? 0) - mean) ** 2
        const sdDraw = Math.sqrt(v / (out.count - 1))
        expect(Math.abs(mean - (beta[j] ?? 0)) / (sd[j] ?? 1), `mean ${j}`).toBeLessThan(0.3)
        expect(sdDraw / (sd[j] ?? 1), `sd ${j}`).toBeGreaterThan(0.8)
        expect(sdDraw / (sd[j] ?? 1), `sd ${j}`).toBeLessThan(1.25)
      }
    }, 120000)
  }

  it('is reproducible from the key, and the chain as an algorithm serialises and resumes', () => {
    const a = samplePosterior(target, beta, { covariance, key: key(3), numDraws: 30, numWarmup: 5 })
    const b = samplePosterior(target, beta, { covariance, key: key(3), numDraws: 30, numWarmup: 5 })
    expect(Array.from(a.draws)).toEqual(Array.from(b.draws))
    const algorithm = posteriorSampler(target, beta, { covariance })
    const straight = runAlgorithm(algorithm, { key: key(4), maxSteps: 12 })
    const head = runAlgorithm(algorithm, { key: key(4), maxSteps: 5 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(4),
      maxSteps: 12 - head.steps,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(Array.from(resumed.state.state.position)).toEqual(Array.from(straight.state.state.position))
  })

  it('refuses a covariance of the wrong size', () => {
    expect(() => posteriorKernel(target, { covariance: [1] })).toThrow(/covariance/)
  })
})

describe('the posterior of an extended model', () => {
  it('has HMC draws centred on the penalised maximum for beta regression', () => {
    const m = 100
    const xs = Array.from({ length: m }, (_, i) => [(i + 0.5) / m])
    const y = xs.map((r, i) =>
      Math.min(0.95, Math.max(0.05, 0.5 + 0.3 * Math.sin(5 * (r[0] ?? 0)) + 0.08 * Math.sin(i * 7.3))),
    )
    const model = extendedModel(xs, y, families.betaRegression(), [[s(0, { k: 6 })]])
    const fit = fitExtended(model)
    const target = extendedPosteriorTarget(model, fit.state.logLambda, fit.state.aux)
    const P = model.likelihood.p
    const out = samplePosterior(target, fit.state.beta, {
      covariance: fit.covariance,
      key: key(5),
      numDraws: 1500,
      numWarmup: 200,
    })
    expect(out.acceptance).toBeGreaterThan(0.6)
    for (let j = 0; j < P; j++) {
      let mean = 0
      for (let i = 0; i < out.count; i++) mean += out.draws[i * P + j] ?? 0
      mean /= out.count
      expect(Math.abs(mean - (fit.state.beta[j] ?? 0)) / Math.sqrt(fit.covariance[j * P + j] ?? 1)).toBeLessThan(0.35)
    }
  }, 120000)
})
