// Laplace-initialised guides: on a Gaussian model the Laplace fit is the posterior, so every guide family that contains it
// starts at the exact posterior, with no steps; on a model that is not Gaussian the start is far better than a random one; and a
// curvature that is not usable falls back to the mode with a scale of its own, and says why.
import { describe, expect, it } from 'vitest'
import { MODELS } from '../../testing/probprog.ts'
import { normal } from '../dists/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { add, item, mul, primalOf, square, toFlat, type Value } from '../tensor/index.ts'
import { stlElbo } from './elbo-extra.ts'
import { gaussianCondition, linearGaussianForm } from './engines/gaussian.ts'
import { svi } from './engines/svi.ts'
import {
  type AutoGuide,
  autoDelta,
  autoLowRankMultivariateNormal,
  autoMultivariateNormal,
  autoNormal,
} from './guides.ts'
import { laplaceInitialParams } from './guides-init.ts'
import { autoStructured } from './guides-structured.ts'
import { seed, traceOf } from './handlers.ts'
import { factor, type Model, observe, plate, sample } from './primitives.ts'

const explaining: Model = () => {
  const a = sample('a', normal(0, 1))
  const b = sample('b', normal(0, 1))
  observe('y', normal(add(a, b), 0.5), 1.0)
}
const Y = [0.9, -0.4, 1.6, 0.2]
const hierarchy: Model = () => {
  const mu = sample('mu', normal(0, 2))
  plate('j', 4, () => {
    const theta = sample('theta', normal(mu, 1))
    observe('y', normal(theta, 0.5), Y)
  })
}

function exactOf(model: Model): { mean: number[]; cov: number[][] } {
  const post = gaussianCondition(linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined)))
  const n = toFlat(post.mean).length
  const flat = toFlat(post.covariance)
  return {
    mean: [...toFlat(post.mean)],
    cov: Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => flat[i * n + j] ?? 0)),
  }
}

/** Draw from a guide's posterior at parameters and summarise the coordinates of `names`. */
function moments(
  guide: AutoGuide<void>,
  params: Record<string, Value>,
  names: readonly string[],
): { mean: number[]; cov: number[][] } {
  const n = 30000
  const draws = guide.posterior(params, undefined).sample(key(7), n)
  const cols = names.flatMap((name) => {
    const flat = toFlat(primalOf(draws[name] as Value))
    const size = flat.length / n
    return Array.from({ length: size }, (_, j) => Array.from({ length: n }, (_, i) => flat[i * size + j] ?? 0))
  })
  const mean = cols.map((c) => c.reduce((a, b) => a + b, 0) / n)
  return {
    mean,
    cov: cols.map((ci, i) =>
      cols.map((cj, j) => ci.reduce((a, x, k) => a + (x - (mean[i] ?? 0)) * ((cj[k] ?? 0) - (mean[j] ?? 0)), 0) / n),
    ),
  }
}

describe('on a Gaussian model the Laplace start is the exact posterior', () => {
  const cases: readonly [
    string,
    Model,
    readonly string[],
    () => AutoGuide<void>,
    Parameters<typeof laplaceInitialParams>[2],
  ][] = [
    [
      'autoMultivariateNormal',
      explaining,
      ['a', 'b'],
      () => autoMultivariateNormal(explaining),
      { guide: 'autoMultivariateNormal' },
    ],
    [
      'autoLowRankMultivariateNormal',
      explaining,
      ['a', 'b'],
      () => autoLowRankMultivariateNormal(explaining, { rank: 2 }),
      { guide: 'autoLowRankMultivariateNormal', rank: 2 },
    ],
    [
      'autoStructured',
      hierarchy,
      ['mu', 'theta'],
      () => autoStructured(hierarchy, { dependencies: 'parents' }),
      { guide: 'autoStructured', structured: { dependencies: 'parents' } },
    ],
  ]
  for (const [name, model, sites, make, options] of cases) {
    it(`${name}: mean and covariance, with no SVI step`, () => {
      const exact = exactOf(model)
      const init = laplaceInitialParams(model, undefined, options)
      expect(init.reason).toBeUndefined()
      const m = moments(make(), init.params as Record<string, Value>, sites)
      for (const [i, x] of exact.mean.entries()) expect(Math.abs((m.mean[i] ?? 0) - x), `mean ${i}`).toBeLessThan(0.03)
      for (const [i, row] of exact.cov.entries())
        for (const [j, x] of row.entries())
          expect(Math.abs((m.cov[i]?.[j] ?? 0) - x), `cov ${i},${j}`).toBeLessThan(0.03)
    })
  }

  it('autoNormal and autoDelta: the mode, and the marginal scales of the Laplace covariance', () => {
    const exact = exactOf(hierarchy)
    const n = laplaceInitialParams(hierarchy, undefined)
    expect(toFlat(n.params['mu_auto_loc'] as Value)[0]).toBeCloseTo(exact.mean[0] ?? 0, 6)
    expect(toFlat(n.params['theta_auto_scale'] as Value)[1]).toBeCloseTo(Math.sqrt(exact.cov[2]?.[2] ?? 0), 6)
    const d = laplaceInitialParams(hierarchy, undefined, { guide: 'autoDelta' })
    expect(toFlat(d.params['theta_auto_loc'] as Value)[2]).toBeCloseTo(exact.mean[3] ?? 0, 6)
    expect(Object.keys(autoDelta(hierarchy)).length).toBeGreaterThan(0)
  })
})

describe('on a model that is not Gaussian', () => {
  const model = MODELS['gammaPoisson'] as Model
  const decaying = adam({ learningRate: 0.02 })

  it('starts the fit far ahead of a random start', () => {
    const guide = autoNormal(model)
    const init = laplaceInitialParams(model, undefined)
    const run = (initParams: Record<string, Value> | undefined, steps: number) =>
      svi(model, undefined, {
        key: key(3),
        guide,
        numSteps: steps,
        optimizer: decaying,
        elbo: stlElbo({ numParticles: 4 }),
        ...(initParams === undefined ? {} : { initParams }),
      })
    const cold = run(undefined, 60)
    const warm = run(init.params as Record<string, Value>, 60)
    const mean = (r: { elbos: readonly number[] }) => r.elbos.slice(-20).reduce((a, b) => a + b, 0) / 20
    expect(mean(warm)).toBeGreaterThan(mean(cold))
    // After the same few steps the warm start is within a whisker of where long runs end.
    const long = run(undefined, 1500)
    expect(Math.abs(mean(warm) - mean(long))).toBeLessThan(0.15)
  })
})

describe('when the curvature is not usable', () => {
  it('keeps the mode, uses the fallback scale and says why', () => {
    const flat: Model = () => {
      const x = sample('x', normal(0, 1))
      factor('cancel', mul(square(x), 0.5))
    }
    const init = laplaceInitialParams(flat, undefined, { fallbackScale: 0.3 })
    expect(init.reason).toMatch(/positive definite/)
    expect(init.covariance).toBeUndefined()
    expect(item(init.params['x_auto_scale'] as Value)).toBeCloseTo(0.3, 12)
  })
})
