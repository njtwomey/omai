import { describe, expect, it } from 'vitest'
import { defaults } from '../../space/index.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { methodConfig } from '../inference/method-space.ts'
import { distributionFamilies } from '../model/dists/index.ts'
import {
  bayesianGam,
  bayesianGamAgreement,
  type bayesianGamFit,
  bayesianGamMethod,
  bayesianGamMethods,
  bayesianGamReference,
} from './bayesian-gam.ts'

const x = Array.from({ length: 60 }, (_, i) => i / 59)
const noise = (i: number): number => 0.25 * Math.sin(i * 12.9898) * Math.cos(i * 78.233)
const y = x.map((v, i) => Math.sin(2 * Math.PI * v) + noise(i))
const model = bayesianGam(x, y, { k: 10 })
const reference = bayesianGamReference(model, { gridSize: 20, points: 28 })

describe('bayesianGam', () => {
  it('splits the penalty into smoothed directions and a wide-prior line, and standardises the response', () => {
    expect(model.size).toBe(10)
    expect(model.penalised).toBe(8)
    expect(model.data['y']).toHaveLength(60)
    const [low, high] = [model.precision(0), model.precision(2)]
    // The smoothed directions scale with the smoothing parameter; the intercept and the line keep the wide prior.
    expect(low.filter((p, j) => p !== high[j])).toHaveLength(model.penalised)
    expect(high.filter((p) => Math.abs(p - 1 / 25) < 1e-12)).toHaveLength(model.size - model.penalised)
    expect(() => bayesianGam([1, 2], [1, 2])).toThrow()
  })

  it('has an exact reference that finds the smoothing the data call for and a band that widens at the ends', () => {
    expect(reference.hyper.logSigma[0]).toBeCloseTo(Math.log(0.125 / model.standardisation.sd), 0)
    const widths = reference.upper.map((u, i) => u - (reference.lower[i] ?? 0))
    expect(Math.min(...widths)).toBeGreaterThan(0)
    expect(widths[0]).toBeGreaterThan(widths[10] ?? 0)
    const truth = reference.grid.map((v) => Math.sin(2 * Math.PI * v))
    const worst = Math.max(...truth.map((t, i) => Math.abs(t - (reference.mean[i] ?? 0))))
    expect(worst).toBeLessThan(0.3)
  })
})

describe('bayesianGamMethod', () => {
  it('puts the curve and band of the posterior on the tape as a probe, and HMC agrees with the exact reference', () => {
    const method = bayesianGamMethod('hmc', model, { fit: { gridSize: 20 } })
    const s = session(method, { steps: 2000, mode: 'eager', config: { numChains: 2, numWarmup: 150, numDraws: 200 } })
    const fit = s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    expect(fit.mean).toHaveLength(20)
    expect(fit.curves.length).toBeGreaterThan(0)
    const agreement = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(agreement.meanError).toBeLessThan(0.05)
    expect(agreement.bandRatio).toBeGreaterThan(0.75)
    expect(agreement.bandRatio).toBeLessThan(1.3)
    expect(Object.keys(fit.draws)).toEqual(['logLambda', 'logSigma', 'f(25%)', 'f(75%)'])
  })

  it('offers every method of the library that runs on the model, with its own knobs, and says why the others do not', () => {
    const offer = bayesianGamMethods()
    expect(offer.applicable).toEqual(
      expect.arrayContaining(['hmc', 'nuts', 'mclmc', 'mala', 'pathfinder', 'svi', 'laplace']),
    )
    expect(offer.applicable.length).toBeGreaterThan(15)
    // Expectation propagation and variational message passing run on the log-normal form too (the non-centred scale is the dense
    // `gaussian-mv-function-scale` factor).
    expect(offer.applicable).toContain('ep')
    expect(offer.applicable).toContain('vmp')
    expect(Object.keys(offer.space.options)).toEqual(
      expect.arrayContaining([...offer.applicable, ...Object.keys(offer.unavailable)]),
    )
    const value = defaults(offer.space) as Record<string, unknown>
    expect(methodConfig(value).kind).toBe('hmc')
    expect(Object.keys(offer.space.options['nuts']?.spaces ?? {})).not.toContain('budgetMs')
    expect(methodConfig(defaults(offer.space) as Record<string, unknown>).config).toMatchObject({ numWarmup: 150 })
    for (const kind of ['pathfinder', 'mala']) {
      const s = session(bayesianGamMethod(kind, model, { fit: { gridSize: 10 } }), { steps: 3000, mode: 'eager' })
      expect((s.probe(s.t, 'gam/fit') as { mean: number[] }).mean).toHaveLength(10)
    }
  })
})

describe('the spike and slab prior', () => {
  it('is a two-component scale mixture of normals, the indicator summed out', () => {
    const family = distributionFamilies()['normalSpikeSlab']
    const dist = family?.make({ inclusion: 0.3, slab: 2, spike: 0.1 } as never)
    const normal = (v: number, sd: number): number => Math.exp(-0.5 * (v / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
    for (const v of [0, 0.05, 1.5]) {
      const exact = Math.log(0.3 * normal(v, 2) + 0.7 * normal(v, 0.1))
      expect(item(dist?.logProb(v) as never)).toBeCloseTo(exact, 10)
    }
  })

  it('switches off the bases the data do not need, and the weighted bases add up to the posterior mean curve', () => {
    const xs = Array.from({ length: 90 }, (_, i) => i / 89)
    // Structure on the left half only.
    const ys = xs.map((v, i) => (v < 0.5 ? Math.sin(4 * Math.PI * v) : 0) + 0.3 * noise(i))
    const g = bayesianGam(xs, ys, { prior: 'spikeSlab', k: 12 })
    expect(g.prior).toBe('spikeSlab')
    const method = bayesianGamMethod('hmc', g, { fit: { gridSize: 30 } })
    const s = session(method, { steps: 2000, mode: 'eager', config: { numChains: 2, numWarmup: 150, numDraws: 150 } })
    const fit = s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    const inclusion = fit.inclusion ?? []
    expect(inclusion).toHaveLength(g.size)
    // Bases carrying the wave on the left are on (a basis where the wave crosses zero may be off); those far to the right are off.
    const left = inclusion.filter((_, j) => (fit.bases.centres[j] ?? 1) < 0.4)
    const right = inclusion.filter((_, j) => (fit.bases.centres[j] ?? 0) > 0.7)
    expect(Math.max(...left)).toBeGreaterThan(0.9)
    expect(left.reduce((a, b) => a + b, 0) / left.length).toBeGreaterThan(
      right.reduce((a, b) => a + b, 0) / right.length + 0.2,
    )
    expect(Math.max(...right)).toBeLessThan(0.6)
    for (let i = 0; i < fit.grid.length; i++) {
      const sum = fit.bases.mean.reduce((a, b) => a + (b[i] ?? 0), 0)
      expect(sum).toBeCloseTo(fit.mean[i] ?? 0, 8)
    }
    expect(() => bayesianGamReference(g)).toThrow()
  })

  it('draws the curve of the penalty prior as the sum of its weighted bases too', () => {
    const method = bayesianGamMethod('pathfinder', model, { fit: { gridSize: 15 } })
    const s = session(method, { steps: 200, mode: 'eager' })
    const fit = s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    for (let i = 0; i < 15; i++)
      expect(fit.bases.mean.reduce((a, b) => a + (b[i] ?? 0), 0)).toBeCloseTo(fit.mean[i] ?? 0, 8)
    expect(fit.bases.curves.length).toBeGreaterThan(0)
  })
})

describe('the ARD prior', () => {
  const xs = Array.from({ length: 90 }, (_, i) => i / 89)
  const ys = xs.map((v, i) => (v < 0.5 ? Math.sin(4 * Math.PI * v) : 0) + 0.3 * noise(i))

  it('gives every basis its own precision and reports how much each is in use', () => {
    const g = bayesianGam(xs, ys, { prior: 'ard', k: 12 })
    expect(g.prior).toBe('ard')
    const method = bayesianGamMethod('hmc', g, { fit: { gridSize: 30 } })
    const s = session(method, { steps: 2000, mode: 'eager', config: { numChains: 2, numWarmup: 150, numDraws: 150 } })
    const fit = s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    const relevance = fit.inclusion ?? []
    expect(relevance).toHaveLength(g.size)
    expect(Math.max(...relevance)).toBeGreaterThan(0.5)
    expect(Math.min(...relevance)).toBeLessThan(Math.max(...relevance))
    for (let i = 0; i < fit.grid.length; i++)
      expect(fit.bases.mean.reduce((a, b) => a + (b[i] ?? 0), 0)).toBeCloseTo(fit.mean[i] ?? 0, 8)
  })

  it('with no sparsity is the unit ridge: every basis in use', () => {
    const g = bayesianGam(xs, ys, { prior: 'ard', k: 8, sparsity: 0 })
    const s = session(bayesianGamMethod('pathfinder', g, { fit: { gridSize: 10 } }), { steps: 200, mode: 'eager' })
    const fit = s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    expect(Math.min(...(fit.inclusion ?? [0]))).toBeCloseTo(1, 8)
  })
})

describe('the weighted bases of the penalty prior', () => {
  const fitOf = (basis: 'ps' | 'bs' | 'cr') => {
    const g = bayesianGam(x, y, { k: 10, basis })
    const s = session(bayesianGamMethod('pathfinder', g, { fit: { gridSize: 101 } }), { steps: 200, mode: 'eager' })
    return { g, fit: s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit> }
  }

  it('are local humps that add up to the posterior mean curve and vanish away from their knots', () => {
    for (const basis of ['ps', 'bs'] as const) {
      const { g, fit } = fitOf(basis)
      expect(g.local).toBe(true)
      expect(fit.bases.local).toBe(true)
      // A partition of unity: the raw basis sums to one, and there is no separate intercept column.
      const B = g.parts(fit.grid)
      for (const row of B) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
      expect(Math.min(...B.flat())).toBeGreaterThanOrEqual(-1e-12)
      for (let i = 0; i < fit.grid.length; i++)
        expect(fit.bases.mean.reduce((a, b) => a + (b[i] ?? 0), 0)).toBeCloseTo(fit.mean[i] ?? 0, 8)
      // Each weighted basis is exactly zero outside its support (four spans of a cubic), the same for every sampled curve.
      const support = (4 / 7) * 1.05
      for (const parts of [fit.bases.mean, ...fit.bases.curves])
        for (const part of parts)
          expect(part.filter((v) => Math.abs(v) > 1e-12).length).toBeLessThanOrEqual(support * fit.grid.length)
    }
  })

  it('are not local for a global basis, and the fit says so', () => {
    const { g, fit } = fitOf('cr')
    expect(g.local).toBe(false)
    expect(fit.bases.local).toBe(false)
    for (let i = 0; i < fit.grid.length; i++)
      expect(fit.bases.mean.reduce((a, b) => a + (b[i] ?? 0), 0)).toBeCloseTo(fit.mean[i] ?? 0, 8)
  })
})

describe('the message-passing forms of the penalty prior', () => {
  const exact = (hyper: 'gammaNoise' | 'fixed') => {
    const g = bayesianGam(x, y, { k: 10, hyper, lambda: 13, noiseSd: 0.18 })
    return { g, reference: bayesianGamReference(g, { gridSize: 20, points: 28 }) }
  }
  const run = (kind: string, g: ReturnType<typeof bayesianGam>, config = {}) => {
    const s = session(bayesianGamMethod(kind, g, { fit: { gridSize: 20 } }), {
      steps: 3000,
      mode: 'eager',
      config: config as never,
    })
    return s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
  }

  it('say which methods apply: the log-normal form EP and VMP, the gamma-noise form VMP and EP, the fixed form too, ARD EP and VMP', () => {
    expect(bayesianGamMethods('penalty').applicable).toEqual(expect.arrayContaining(['ep', 'vmp']))
    expect(bayesianGamMethods('ard').applicable).toEqual(expect.arrayContaining(['ep', 'vmp']))
    const gamma = bayesianGamMethods('penalty', 'gammaNoise')
    expect(gamma.applicable).toEqual(expect.arrayContaining(['vmp', 'ep']))
    const fixed = bayesianGamMethods('penalty', 'fixed')
    expect(fixed.applicable).toEqual(expect.arrayContaining(['ep', 'vmp', 'gaussian']))
  })

  it('give VMP and EP on the gamma-noise form and EP, VMP and exact Gaussian inference on the fixed form the exact posterior', () => {
    const noisy = exact('gammaNoise')
    for (const kind of ['vmp', 'ep']) {
      const a = bayesianGamAgreement(run(kind, noisy.g), noisy.reference, noisy.g.standardisation.sd)
      expect(a.meanError, kind).toBeLessThan(0.03)
      expect(a.bandRatio, kind).toBeGreaterThan(0.85)
      expect(a.bandRatio, kind).toBeLessThan(1.15)
    }
    const known = exact('fixed')
    for (const kind of ['ep', 'vmp', 'gaussian']) {
      const b = bayesianGamAgreement(run(kind, known.g), known.reference, known.g.standardisation.sd)
      expect(b.meanError, kind).toBeLessThan(0.03)
      expect(b.bandRatio, kind).toBeGreaterThan(0.85)
      expect(b.bandRatio, kind).toBeLessThan(1.15)
    }
  })

  it('have an exact reference that agrees with a sampler', () => {
    const noisy = exact('gammaNoise')
    const a = bayesianGamAgreement(
      run('hmc', noisy.g, { numChains: 2, numWarmup: 150, numDraws: 200 }),
      noisy.reference,
      noisy.g.standardisation.sd,
    )
    expect(a.meanError).toBeLessThan(0.05)
  })
})
