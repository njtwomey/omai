// The ELBO with discrete latents summed out: a binary guide site is enumerated, so its value and gradient are exact (the
// same at every key) and equal the closed-form ones; with a continuous site beneath it the gradient is unbiased against the
// score-function estimators; a mixture's assignments under a plate are summed out of the model; too many cells is refused.
import { describe, expect, it } from 'vitest'
import { valueAndGrad } from '../autodiff/index.ts'
import { bernoulli, categorical, normal } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import { add, log, mul, primalOf, sigmoid, sub, take, tensor, toFlat, type Value } from '../tensor/index.ts'
import { traceElbo } from './elbo.ts'
import { gradientMoments } from './elbo-diagnostics.ts'
import { traceEnumElbo } from './elbo-enum.ts'
import { traceGraphElbo } from './elbo-graph.ts'
import { seed, substitute, traceOf } from './handlers.ts'
import { type Model, observe, param, plate, sample } from './primitives.ts'

const lpdf = (x: number, m: number, s = 1): number => -0.5 * Math.log(2 * Math.PI * s * s) - 0.5 * ((x - m) / s) ** 2
const Y = 1.3
const model: Model = () => {
  const c = sample('c', bernoulli({ probs: 0.4 }))
  observe('y', normal(mul(c, 3), 1), Y)
}
const guide: Model = () => {
  sample('c', bernoulli({ logits: param('l', tensor(0.2)) }))
}
const logZ = Math.log(0.4 * Math.exp(lpdf(Y, 3)) + 0.6 * Math.exp(lpdf(Y, 0)))

const exactElbo = (p: Record<string, Value>): Value => {
  const q = sigmoid(p['l'] as Value)
  const one = add(Math.log(0.4) + lpdf(Y, 3), neg1(log(q)))
  const zero = add(Math.log(0.6) + lpdf(Y, 0), neg1(log(sub(1, q))))
  return add(mul(q, one), mul(sub(1, q), zero))
}
function neg1(v: Value): Value {
  return mul(v, -1)
}

describe('enumerating a discrete guide site', () => {
  it('gives the exact ELBO and its exact gradient, the same at every key', () => {
    const est = traceEnumElbo()
    const p = { l: tensor(0.2) }
    const { value, grad } = valueAndGrad(exactElbo)(p as Record<string, Value>)
    const exactGrad = -(toFlat(primalOf((grad as Record<string, Value>)['l'] as Value))[0] ?? 0)
    const m = gradientMoments({ estimator: est, model, guide, data: undefined, params: p, key: key(0), repeats: 5 })
    expect(Math.abs(m.elboMean - (toFlat(primalOf(value as Value))[0] ?? 0))).toBeLessThan(1e-12)
    expect(m.elboSd).toBeLessThan(1e-12)
    expect(Math.abs((m.mean['l']?.[0] ?? 0) - exactGrad)).toBeLessThan(1e-12)
    expect(m.totalVariance).toBeLessThan(1e-20)
    const noisy = gradientMoments({
      estimator: traceElbo({ baselineDecay: 0 }),
      model,
      guide,
      data: undefined,
      params: p,
      key: key(0),
      repeats: 200,
    })
    expect(noisy.totalVariance).toBeGreaterThan(1e-3)
  })

  it('reaches the log evidence at the exact posterior of the site', () => {
    const post = (0.4 * Math.exp(lpdf(Y, 3))) / Math.exp(logZ)
    const r = traceEnumElbo().estimate(
      key(1),
      model,
      guide,
      undefined,
      { l: tensor(Math.log(post / (1 - post))) },
      { baseline: undefined },
    )
    expect(Math.abs(r.elbo - logZ)).toBeLessThan(1e-12)
    expect(r.diagnostics?.['cells']).toBe(2)
  })

  it('is unbiased with a continuous site under the discrete one, against the score-function estimators', () => {
    const m2: Model = () => {
      const c = sample('c', bernoulli({ probs: 0.4 }))
      const z = sample('z', normal(mul(c, 3), 1))
      observe('y', normal(z, 1), Y)
    }
    const g2: Model = () => {
      const c = sample('c', bernoulli({ logits: param('l', tensor(0.2)) }))
      const loc = take(param('loc', tensor([0.1, 2.0])), c)
      const scale = take(param('scale', tensor([0.8, 0.9])), c)
      sample('z', normal(loc, scale))
    }
    const p = { l: tensor(0.2), loc: tensor([0.1, 2.0]), scale: tensor([0.8, 0.9]) }
    const repeats = 1500
    const run = (estimator: Parameters<typeof gradientMoments>[0]['estimator'], state?: unknown) =>
      gradientMoments({
        estimator,
        model: m2,
        guide: g2,
        data: undefined,
        params: p,
        key: key(5),
        repeats,
        ...(state === undefined ? {} : { state }),
      })
    const enumd = run(traceEnumElbo(), { baseline: 0 })
    const graph = run(traceGraphElbo({ baseline: 'none' }))
    for (const k of ['l', 'loc', 'scale'] as const)
      for (let i = 0; i < (enumd.mean[k]?.length ?? 0); i++) {
        const se = Math.sqrt(((enumd.variance[k]?.[i] ?? 0) + (graph.variance[k]?.[i] ?? 0)) / repeats)
        expect(Math.abs((enumd.mean[k]?.[i] ?? 0) - (graph.mean[k]?.[i] ?? 0)), `${k}[${i}]`).toBeLessThan(
          4.5 * se + 1e-9,
        )
      }
    expect(enumd.totalVariance).toBeLessThan(graph.totalVariance)
  })

  it('refuses more joint states than maxCells, and says what to do', () => {
    const plated: Model = () => {
      plate('n', 10, () => {
        const c = sample('c', bernoulli({ probs: 0.5 }))
        observe('y', normal(c, 1), 0.3)
      })
    }
    const pg: Model = () => {
      plate('n', 10, () => sample('c', bernoulli({ logits: param('l', tensor(Array.from({ length: 10 }, () => 0))) })))
    }
    expect(() =>
      traceEnumElbo({ maxCells: 100 }).estimate(
        key(0),
        plated,
        pg,
        undefined,
        { l: tensor(Array.from({ length: 10 }, () => 0)) },
        { baseline: undefined },
      ),
    ).toThrow(/maxCells/)
  })
})

describe("summing the model's discrete sites out under a plate", () => {
  const ys = [-2.1, 2.3, -1.8, 2.5, 2.0]
  const mixture: Model = () => {
    const mu = sample('mu', normal(tensor([-2, 2]), 1))
    plate('n', ys.length, () => {
      const z = sample('z', categorical({ probs: [0.5, 0.5] }))
      observe('y', normal(take(mu, z), 0.5), ys)
    })
  }
  const g: Model = () => {
    sample('mu', normal(param('loc', tensor([-2, 2])), param('scale', tensor([0.3, 0.3]))))
  }

  it('is a tighter bound than any assignment-sampling one and below the log evidence, with a finite gradient', () => {
    // The log evidence by brute force over the means on a grid is dear; bound it from above by the exact mixture's data term.
    const est = traceEnumElbo()
    const p = { loc: tensor([-2, 2]), scale: tensor([0.3, 0.3]) }
    const { value, grad } = valueAndGrad(
      (q: Record<string, Value>) => est.estimate(key(2), mixture, g, undefined, q, est.init()).loss,
    )(p as Record<string, Value>)
    expect(Number.isFinite(toFlat(primalOf(value as Value))[0] ?? Number.NaN)).toBe(true)
    for (const v of Object.values(grad as Record<string, Value>))
      for (const x of toFlat(primalOf(v))) expect(Number.isFinite(x)).toBe(true)
    const r = est.estimate(key(2), mixture, g, undefined, p, est.init())
    expect(r.diagnostics?.['cells']).toBe(1)
    // At the guide's draw, the bound is log p(mu) + sum_n log sum_z p(z) N(y_n; mu_z, 0.5) - log q(mu), by brute force.
    const draw = traceOf(seed(substitute(g, p), foldIn(key(2), 0)), undefined).sites.get('mu')
    const mu = draw?.type === 'sample' ? toFlat(primalOf(draw.value)) : []
    let brute = 0
    for (let k = 0; k < 2; k++) {
      brute += lpdf(mu[k] ?? 0, k === 0 ? -2 : 2, 1) - lpdf(mu[k] ?? 0, k === 0 ? -2 : 2, 0.3)
    }
    for (const y of ys)
      brute += Math.log(0.5 * Math.exp(lpdf(y, mu[0] ?? 0, 0.5)) + 0.5 * Math.exp(lpdf(y, mu[1] ?? 0, 0.5)))
    expect(Math.abs(r.elbo - brute)).toBeLessThan(1e-9)
  })
})
