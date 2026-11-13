// EP inside a grid over several hyperparameters of a Gaussian process: a length scale, a kernel variance and a noise level (three
// families, each a scalar latent a custom function reads), and an ARD kernel (a plated vector of length scales). With a Gaussian
// likelihood EP is exact at every node, so the grid is the whole error: the reference is a brute-force tensor grid over the same
// hyperparameters of the marginal likelihood written in plain arithmetic (`gpPosteriorGivenHyperparameters`).
import { describe, expect, it } from 'vitest'
import { gram, rbfKernel, scaleKernel } from '../../kernels/index.ts'
import { defaults } from '../../space/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { atomRun, epAtom } from '../ep/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { gpPosteriorGivenHyperparameters } from '../models/index.ts'
import { gridMarginals } from '../validation/index.ts'
import { epGridKnobsFor, epGridMethod } from './ep-grid.ts'
import { analysisOf } from './model-input.ts'

const X1 = [-2.4, -1.8, -1.1, -0.5, 0.1, 0.7, 1.2, 1.9, 2.5]
const Y1 = [-0.8, -1.2, -0.7, -0.1, 0.5, 0.9, 1.1, 0.4, -0.2]
const NOISE_PRIOR = { mean: Math.log(0.3), sd: 0.5 }

/** The log density of a normal on the log scale of a log-normal hyperparameter (the density of its log). */
const logNormalOfLog = (u: number, mean: number, sd: number): number =>
  -0.5 * ((u - mean) / sd) ** 2 - Math.log(sd * Math.sqrt(2 * Math.PI))

/** The posterior of the function values at the grid of hyperparameters: the mixture of the Gaussian conditionals (law of total variance). */
interface Reference {
  readonly logEvidence: number
  /** The mean and sd of the hyperparameter `k` on its own scale (the exponential of the grid coordinate). */
  readonly hyper: (k: number) => { readonly mean: number; readonly sd: number }
  readonly fn: (i: number) => { readonly mean: number; readonly sd: number }
}

/** The brute-force reference over `axes` log-scale hyperparameters whose conditional GP is given by `at`. */
function reference(
  axes: readonly { readonly mean: number; readonly sd: number }[],
  points: number,
  conditional: (u: readonly number[]) => ReturnType<typeof gpPosteriorGivenHyperparameters>,
): Reference {
  const cache = new Map<string, ReturnType<typeof gpPosteriorGivenHyperparameters>>()
  const at = (u: readonly number[]): ReturnType<typeof gpPosteriorGivenHyperparameters> => {
    const key = u.join(',')
    let hit = cache.get(key)
    if (hit === undefined) {
      hit = conditional(u)
      cache.set(key, hit)
    }
    return hit
  }
  const box = axes.map((a) => [a.mean - 4.5 * a.sd, a.mean + 4.5 * a.sd] as [number, number])
  const g = gridMarginals(
    (u) =>
      at(u).logMarginalLikelihood +
      u.reduce((s, v, k) => s + logNormalOfLog(v, axes[k]?.mean ?? 0, axes[k]?.sd ?? 1), 0),
    box,
    points,
  )
  return {
    logEvidence: g.logEvidence,
    hyper: (k) => {
      const mean = g.expect((u) => Math.exp(u[k] ?? 0))
      return { mean, sd: Math.sqrt(g.expect((u) => (Math.exp(u[k] ?? 0) - mean) ** 2)) }
    },
    fn: (i) => {
      const mean = g.expect((u) => at(u).mean[i] ?? 0)
      const variance = g.expect((u) => (at(u).sd[i] ?? 0) ** 2 + ((at(u).mean[i] ?? 0) - mean) ** 2)
      return { mean, sd: Math.sqrt(variance) }
    },
  }
}

/** A Gaussian process on a line whose length scale, kernel variance and noise level are random: the noise goes through a custom function so that it is read like the others. */
const lineSpec = defineModel('gp-three-hyperparameters', (m) => {
  const x = m.data('x', { dims: ['i', 'input'], role: 'covariate' })
  const y = m.data('y', { dims: ['i'], role: 'response' })
  const ell = m.latent('ell', d.logNormal(0, 0.5))
  const variance = m.latent('variance', d.logNormal(0, 0.5))
  const sigma = m.latent('sigma', d.logNormal(NOISE_PRIOR.mean, NOISE_PRIOR.sd))
  const cov = m.custom(
    'gram',
    ([xs, scale, v]) =>
      gram(
        scaleKernel(rbfKernel({ lengthScale: (scale as { value: Value }).value }), (v as { value: Value }).value),
        (xs as { value: Value }).value,
        { jitter: 1e-6 },
      ),
    [x, ell, variance],
    { eventRank: 2 },
  )
  const noise = m.custom('noise', ([s]) => (s as { value: Value }).value, [sigma], {})
  const f = m.latent('f', d.multivariateNormal(new Array<number>(X1.length).fill(0), { covariance: cov }), {
    dims: ['i'],
  })
  m.observe('y', d.normal(f, noise), y)
})

describe('epGridMethod over several hyperparameters: a length scale, a variance and a noise level', () => {
  const bound = lineSpec.bind({ x: X1.map((v) => [v]), y: Y1 })
  const rows = X1.map((v) => [v])
  const ref = reference([{ mean: 0, sd: 0.5 }, { mean: 0, sd: 0.5 }, NOISE_PRIOR], 41, (u) =>
    gpPosteriorGivenHyperparameters(rows, Y1, {
      lengthScale: Math.exp(u[0] ?? 0),
      variance: Math.exp(u[1] ?? 0),
      noise: Math.exp(u[2] ?? 0),
    }),
  )

  it('a method over three hyperparameters opens on a walk, a coarser lattice and a larger budget; over one, as before', () => {
    const many = defaults(epGridMethod(bound).config) as Record<string, unknown>
    expect([many['order'], many['points'], many['warm'], many['maxNodes']]).toEqual(['walk', 15, true, 4000])
    const one = defaults(epGridKnobsFor(1)) as Record<string, unknown>
    expect([one['order'], one['points'], one['warm'], one['maxNodes']]).toEqual(['bisect', 31, false, 400])
  })

  it('the walk finds the evidence, the three hyperparameters and the function against the brute-force grid, on a fraction of the product', () => {
    const method = epGridMethod(bound)
    expect(method.check?.(analysisOf(bound)).applies).toBe(true)
    const run = session(method, { steps: 5000, mode: 'eager', config: { tol: 1e-10 } })
    const post = method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    expect(post.status).toBe('converged')
    expect(post.diagnostics?.['hyperparameters']).toEqual(['ell', 'variance', 'sigma'])
    // a product grid of 15 points per axis is 3375 nodes; the region with mass is a third of it or less
    expect(run.state.nodes.length).toBeLessThan(0.45 * 15 ** 3)
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - ref.logEvidence)).toBeLessThan(0.01)
    const rowOf = (site: string): { mean: number; sd: number } => {
      const r = post.summary().find((s) => s.site === site)
      return { mean: r?.mean ?? Number.NaN, sd: r?.sd ?? Number.NaN }
    }
    for (const [k, site] of ['ell', 'variance', 'sigma'].entries()) {
      const want = ref.hyper(k)
      const got = rowOf(site)
      expect(Math.abs(got.mean - want.mean) / want.sd, `${site} mean z`).toBeLessThan(0.05)
      expect(got.sd / want.sd, `${site} sd ratio`).toBeGreaterThan(0.97)
      expect(got.sd / want.sd, `${site} sd ratio`).toBeLessThan(1.03)
    }
    const [fMean, fSd] = [toFlat(post.mean('f')), toFlat(post.stddev('f'))]
    for (const i of [0, 4, 8]) {
      const want = ref.fn(i)
      const got = { mean: fMean[i] ?? Number.NaN, sd: fSd[i] ?? Number.NaN }
      expect(Math.abs(got.mean - want.mean) / want.sd, `f[${i}] mean z`).toBeLessThan(0.05)
      expect(got.sd / want.sd, `f[${i}] sd ratio`).toBeGreaterThan(0.97)
      expect(got.sd / want.sd, `f[${i}] sd ratio`).toBeLessThan(1.03)
    }
  })

  it("the atom `gp-hyperparameters` (the same three families, the suite's knobs) lands within 0.002 nats of the reference grid and passes its check", () => {
    const atom = epAtom('gp-hyperparameters')
    const problem = atom.problem(atom.defaults)
    const run = atomRun(problem, atom.ep ?? {}, { light: true })
    expect(run.status).toBe('ran')
    if (run.status !== 'ran') return
    expect(run.converged).toBe(true)
    expect(Math.abs((run.evidence ?? Number.NaN) - (problem.exactLogEvidence ?? Number.NaN))).toBeLessThan(0.002)
  })

  it('a budget that cuts the walk short leaves the posterior stopped, and says so', () => {
    const method = epGridMethod(bound)
    const run = session(method, { steps: 5000, mode: 'eager', config: { tol: 1e-10, maxNodes: 200 } })
    expect(run.state.nodes.length).toBe(200)
    const post = method.posterior?.(run.state, run.config)
    expect(post?.status).toBe('stopped')
    expect(post?.diagnostics?.['budgetReached']).toBe(true)
  })
})

describe('epGridMethod on an ARD kernel: a plated vector of length scales beside a variance', () => {
  const X2 = Array.from({ length: 10 }, (_, i) => [-2 + (4 * i) / 9, Math.sin(2.3 * i)])
  const Y2 = X2.map((r, i) => Math.sin(1.1 * (r[0] ?? 0)) + 0.1 * (r[1] ?? 0) + 0.15 * Math.cos(5 * i))
  const ardSpec = (noisy: boolean): ReturnType<typeof defineModel> =>
    defineModel('gp-ard', (m) => {
      const x = m.data('x', { dims: ['i', 'input'], role: 'covariate' })
      const y = m.data('y', { dims: ['i'], role: 'response' })
      // one length scale per input, independent, with the same prior
      const ell = m.latent('ell', d.logNormal(0, 0.5), { dims: ['input'] })
      const variance = m.latent('variance', d.logNormal(0, 0.5))
      const cov = m.custom(
        'ardGram',
        ([xs, scale, v]) =>
          gram(
            scaleKernel(rbfKernel({ lengthScale: (scale as { value: Value }).value }), (v as { value: Value }).value),
            (xs as { value: Value }).value,
            { jitter: 1e-6 },
          ),
        [x, ell, variance],
        { eventRank: 2 },
      )
      const f = m.latent('f', d.multivariateNormal(new Array<number>(X2.length).fill(0), { covariance: cov }), {
        dims: ['i'],
      })
      if (noisy) {
        // the noise sd is random too, and read through a custom function so that it sits on the grid
        const sigma = m.latent('sigma', d.logNormal(NOISE_PRIOR.mean, NOISE_PRIOR.sd))
        m.observe(
          'y',
          d.normal(
            f,
            m.custom('noise', ([s]) => (s as { value: Value }).value, [sigma], {}),
          ),
          y,
        )
      } else m.observe('y', d.normal(f, 0.3), y)
    })
  const bound = ardSpec(false).bind({ x: X2, y: Y2 })

  it('puts one axis on each component and matches the brute-force grid over the three', () => {
    const ref = reference(
      [
        { mean: 0, sd: 0.5 },
        { mean: 0, sd: 0.5 },
        { mean: 0, sd: 0.5 },
      ],
      41,
      (u) =>
        gpPosteriorGivenHyperparameters(X2, Y2, {
          lengthScale: [Math.exp(u[0] ?? 0), Math.exp(u[1] ?? 0)],
          variance: Math.exp(u[2] ?? 0),
          noise: 0.3,
        }),
    )
    const method = epGridMethod(bound)
    const run = session(method, { steps: 5000, mode: 'eager', config: { tol: 1e-10 } })
    const post = method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    expect(post.diagnostics?.['hyperparameters']).toEqual(['ell[0]', 'ell[1]', 'variance'])
    expect(post.status).toBe('converged')
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - ref.logEvidence)).toBeLessThan(0.01)
    for (const [k, site] of ['ell[0]', 'ell[1]', 'variance'].entries()) {
      const row = post.summary().find((s) => s.site === site)
      const want = ref.hyper(k)
      expect(Math.abs((row?.mean ?? Number.NaN) - want.mean) / want.sd, `${site} mean z`).toBeLessThan(0.05)
      // the second input's length scale is wide: the reference box ends where the prior's tail does, the grid extends past it
      expect((row?.sd ?? Number.NaN) / want.sd, `${site} sd ratio`).toBeGreaterThan(0.97)
      expect((row?.sd ?? Number.NaN) / want.sd, `${site} sd ratio`).toBeLessThan(1.05)
    }
    const f = ref.fn(3)
    const [mean, sd] = [toFlat(post.mean('f'))[3] ?? Number.NaN, toFlat(post.stddev('f'))[3] ?? Number.NaN]
    expect(Math.abs(mean - f.mean) / f.sd).toBeLessThan(0.05)
    expect(sd / f.sd).toBeGreaterThan(0.97)
    expect(sd / f.sd).toBeLessThan(1.03)
  })

  // The brute-force reference on 21 points per axis and EP over 11 points per axis (14641 nodes as a product): 20 s here, over 30 s on the CI runner.
  it('four axes (two length scales, the variance and the noise sd) on a coarser lattice: the evidence and the spreads against the brute-force grid', {
    timeout: 180_000,
  }, () => {
    const ref = reference([{ mean: 0, sd: 0.5 }, { mean: 0, sd: 0.5 }, { mean: 0, sd: 0.5 }, NOISE_PRIOR], 21, (u) =>
      gpPosteriorGivenHyperparameters(X2, Y2, {
        lengthScale: [Math.exp(u[0] ?? 0), Math.exp(u[1] ?? 0)],
        variance: Math.exp(u[2] ?? 0),
        noise: Math.exp(u[3] ?? 0),
      }),
    )
    const method = epGridMethod(ardSpec(true).bind({ x: X2, y: Y2 }))
    const run = session(method, { steps: 5000, mode: 'eager', config: { tol: 1e-10, points: 11 } })
    const post = method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    expect(post.diagnostics?.['hyperparameters']).toEqual(['ell[0]', 'ell[1]', 'variance', 'sigma'])
    expect(post.status).toBe('converged')
    // 11 points per axis is 14641 nodes as a product; the region with mass is a fifth of it or less
    expect(run.state.nodes.length).toBeLessThan(0.2 * 11 ** 4)
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - ref.logEvidence)).toBeLessThan(0.02)
    for (const [k, site] of ['ell[0]', 'ell[1]', 'variance', 'sigma'].entries()) {
      const row = post.summary().find((s) => s.site === site)
      const want = ref.hyper(k)
      expect(Math.abs((row?.mean ?? Number.NaN) - want.mean) / want.sd, `${site} mean z`).toBeLessThan(0.1)
      expect((row?.sd ?? Number.NaN) / want.sd, `${site} sd ratio`).toBeGreaterThan(0.93)
      expect((row?.sd ?? Number.NaN) / want.sd, `${site} sd ratio`).toBeLessThan(1.07)
    }
  })
})
