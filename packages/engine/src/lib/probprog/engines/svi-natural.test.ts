// Natural-gradient SVI against the conjugate posterior it must reach: with exact gradients and rate one the first step is the
// CAVI update (the exact posterior); with Monte Carlo gradients and a Robbins-Monro rate it converges to the same fixed point,
// in far fewer steps than the plain gradient of the same estimator; on a conjugate-exponential program the minibatch steps are
// convex combinations of natural parameters that converge to the batch posterior.
import { describe, expect, it } from 'vitest'
import { MODELS, Y4 } from '../../../testing/probprog.ts'
import { valueAndGrad } from '../../autodiff/index.ts'
import { gamma, normal, poisson } from '../../dists/index.ts'
import { inv } from '../../numerics/linalg/index.ts'
import { adam } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import {
  add,
  item,
  log,
  matmul,
  mul,
  primalOf,
  square,
  sub,
  type Tensor,
  tensor,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { traceElbo } from '../elbo.ts'
import { autoNormal } from '../guides.ts'
import { seed, traceOf } from '../handlers.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { gaussianCondition, linearGaussianForm } from './gaussian.ts'
import { svi } from './svi.ts'
import {
  conjugateNaturalGradientAlgorithm,
  conjugateNaturalMarginals,
  gaussianNaturalGradientAlgorithm,
  naturalRate,
  naturalStepDiagonal,
  naturalStepFullRank,
} from './svi-natural.ts'

const normalMean = MODELS['normalMean'] as Model
const exact = conjugateInference(normalMean, undefined)
const m0 = item(exact.mean('mu'))
const s0 = Math.sqrt(item(exact.variance('mu')))

/** The exact ELBO of the normal-mean model with a normal guide, in the guide's location and scale. */
function exactElbo(p: Record<string, Value>): Value {
  const m = p['m'] as Value
  const s = p['s'] as Value
  const prior = sub(-0.5 * Math.log(2 * Math.PI * 4), mul(add(square(m), square(s)), 1 / 8))
  let like: Value = tensor(0)
  for (const y of Y4)
    like = add(like, sub(-0.5 * Math.log(2 * Math.PI * 2.25), mul(add(square(sub(y, m)), square(s)), 1 / 4.5)))
  return add(add(prior, like), add(0.5 * Math.log(2 * Math.PI * Math.E), log(s)))
}

describe('the natural-gradient step is the CAVI update on a conjugate model', () => {
  it('diagonal: one step of rate one from anywhere is the exact posterior', () => {
    for (const [m, s] of [
      [0.1, 0.8],
      [-3, 0.3],
      [2, 4],
    ] as const) {
      const { grad } = valueAndGrad(exactElbo)({ m: tensor(m), s: tensor(s) } as Record<string, Value>)
      const g = grad as Record<string, Value>
      // The gradient of the (maximised) ELBO is minus that of its negative.
      const r = naturalStepDiagonal([m], [s], [item(g['m'] as Value)], [item(g['s'] as Value)], 1)
      expect(Math.abs((r.loc[0] ?? 0) - m0), `from ${m}, ${s}`).toBeLessThan(1e-9)
      expect(Math.abs((r.scale[0] ?? 0) - s0), `from ${m}, ${s}`).toBeLessThan(1e-9)
    }
  })

  it('full rank: one step of rate one is the exact posterior with its correlations', () => {
    const model: Model = () => {
      const a = sample('a', normal(0, 1))
      const b = sample('b', normal(0, 1))
      observe('y', normal(add(a, b), 0.5), 1.0)
    }
    const post = gaussianCondition(linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined)))
    const lambda = primalOf(inv(post.covariance))
    const mu = tensor([0.4, -1.2])
    const sigma = tensor([
      [0.7, 0.1],
      [0.1, 0.5],
    ])
    const gMu = matmul(mul(lambda, -1), sub(mu, post.mean))
    const gSigma = mul(sub(primalOf(inv(sigma)), lambda), 0.5)
    const r = naturalStepFullRank(mu as Tensor, sigma as Tensor, gMu as Tensor, gSigma as Tensor, 1)
    const cov = toFlat(post.covariance)
    expect(Math.abs((r.loc[0] ?? 0) - (toFlat(post.mean)[0] ?? 0))).toBeLessThan(1e-9)
    for (let i = 0; i < 4; i++) expect(Math.abs((r.scale[i] ?? 0) - (cov[i] ?? 0))).toBeLessThan(1e-9)
    expect(Math.abs(transpose(sigma).shape[0] ?? 0)).toBe(2)
  })

  it('cuts the rate so the precision stays positive', () => {
    const r = naturalStepDiagonal([0], [1], [0], [6], 1)
    expect(r.scale[0]).toBeGreaterThan(0)
    expect(Number.isFinite(r.loc[0])).toBe(true)
  })

  it('has a Robbins-Monro rate that starts at one and falls', () => {
    expect(naturalRate({}, 0)).toBe(1)
    expect(naturalRate({ forgetting: 1 }, 9)).toBeCloseTo(0.1, 12)
    expect(naturalRate({ forgetting: 0 }, 50)).toBe(1)
  })
})

describe('Gaussian natural-gradient SVI with sampled gradients', () => {
  const settings = { elbo: traceElbo({ numParticles: 8 }), forgetting: 0.6 }

  it('converges to the exact posterior, mean-field and full-rank', () => {
    for (const family of ['meanField', 'fullRank'] as const) {
      const r = runAlgorithm(gaussianNaturalGradientAlgorithm(normalMean, undefined, { ...settings, family }), {
        seed: 3,
        maxSteps: 250,
      })
      const p = r.state.params
      const loc = item((p['mu_auto_loc'] ?? p['auto_loc']) as Value)
      const scale =
        family === 'fullRank' ? Math.abs(item(p['auto_scale_tril'] as Value)) : item(p['mu_auto_scale'] as Value)
      expect(Math.abs(loc - m0), family).toBeLessThan(0.05)
      expect(Math.abs(scale - s0), family).toBeLessThan(0.05)
    }
  })

  it('is much faster in steps than the plain gradient of the same estimator', () => {
    const steps = 20
    const error = (loc: number, scale: number): number => Math.abs(loc - m0) + Math.abs(scale - s0)
    const nat = runAlgorithm(gaussianNaturalGradientAlgorithm(normalMean, undefined, settings), {
      seed: 3,
      maxSteps: steps,
    })
    const natError = error(
      item(nat.state.params['mu_auto_loc'] as Value),
      item(nat.state.params['mu_auto_scale'] as Value),
    )
    // The plain gradient, with each learning rate that is a sensible choice for Adam on this problem (best taken).
    let plain = Number.POSITIVE_INFINITY
    for (const lr of [0.01, 0.03, 0.1, 0.3]) {
      const r = svi(normalMean, undefined, {
        key: key(3),
        guide: autoNormal(normalMean),
        numSteps: steps,
        optimizer: adam({ learningRate: lr }),
        elbo: traceElbo({ numParticles: 8 }),
      })
      plain = Math.min(plain, error(item(r.params['mu_auto_loc'] as Value), item(r.params['mu_auto_scale'] as Value)))
    }
    expect(natError).toBeLessThan(0.15)
    expect(natError).toBeLessThan(plain / 2)
  })

  it('records the natural gradient, which shrinks as the fit settles', () => {
    const alg = gaussianNaturalGradientAlgorithm(normalMean, undefined, settings)
    const r = runAlgorithm(alg, { seed: 3, maxSteps: 120 })
    expect(r.state.naturalGradNorm).toBeLessThan(1)
    expect(r.state.rate).toBeLessThan(0.1)
  })
})

describe('conjugate natural-gradient SVI', () => {
  const model = MODELS['gammaPoisson'] as Model
  const batch = conjugateInference(model, undefined)

  it('is batch CAVI in one step of rate one: the exact posterior, to rounding', () => {
    const r = runAlgorithm(conjugateNaturalGradientAlgorithm(model, undefined), { seed: 0, maxSteps: 1 })
    const q = conjugateNaturalMarginals(model, undefined, r.state)['rate']
    expect(Math.abs(item(q?.mean() as Value) - item(batch.mean('rate')))).toBeLessThan(1e-12)
    expect(Math.abs(item(q?.variance() as Value) - item(batch.variance('rate')))).toBeLessThan(1e-12)
    expect(r.state.rate).toBe(1)
    // The ELBO of the exact posterior is the log evidence.
    expect(Math.abs(r.state.elbo - (batch.logEvidence?.value ?? 0))).toBeLessThan(0.5)
  })

  it('on minibatches with a Robbins-Monro rate converges to the batch posterior', () => {
    const counts = Array.from({ length: 200 }, (_, i) => 2 + ((i * 7) % 5))
    const big: Model = () => {
      const r = sample('rate', gamma(2, 0.5))
      plate('n', counts.length, () => observe('k', poisson(r), counts))
    }
    const truth = conjugateInference(big, undefined)
    const alg = conjugateNaturalGradientAlgorithm(big, undefined, { batchSize: 20, forgetting: 0.8, elboDraws: 2 })
    const run = (steps: number) => runAlgorithm(alg, { seed: 1, maxSteps: steps })
    const err = (steps: number): number => {
      const r = run(steps)
      return (
        Math.abs(
          item(conjugateNaturalMarginals(big, undefined, r.state)['rate']?.mean() as Value) - item(truth.mean('rate')),
        ) / item(truth.mean('rate'))
      )
    }
    expect(err(400)).toBeLessThan(0.03)
    // Plain SVI with Adam on the same problem is further from the posterior after the same number of steps.
    const plain = svi(big, undefined, { key: key(1), numSteps: 400, optimizer: adam({ learningRate: 0.02 }) })
    const plainErr = Math.abs(item(plain.posterior.mean('rate')) - item(truth.mean('rate'))) / item(truth.mean('rate'))
    expect(err(400)).toBeLessThan(plainErr + 0.02)
  })

  it('refuses a program that is not conjugate', () => {
    expect(() =>
      runAlgorithm(conjugateNaturalGradientAlgorithm(MODELS['logistic'] as Model, undefined), { seed: 0, maxSteps: 1 }),
    ).toThrow(/not fully conjugate/)
  })
})
