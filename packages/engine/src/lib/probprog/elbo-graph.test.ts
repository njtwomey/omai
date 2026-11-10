// The Rao-Blackwellised score-function ELBO against a computable one: the discrete models below have an exact ELBO as a sum
// over states, so the exact gradient comes from differentiating it, and the mean of the estimator's gradient over many draws
// must equal it (within the Monte Carlo error) for every baseline, while the variance falls as the dropped costs and the
// baselines say it should.
import { describe, expect, it } from 'vitest'
import { valueAndGrad } from '../autodiff/index.ts'
import { bernoulli, normal } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import {
  add,
  item,
  log,
  mul,
  primalOf,
  sigmoid,
  sub,
  take,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { traceElbo } from './elbo.ts'
import { gradientMoments } from './elbo-diagnostics.ts'
import { dependencyGraph, type TraceGraphState, traceGraphElbo } from './elbo-graph.ts'
import { type Model, observe, param, plate, sample } from './primitives.ts'

const X1 = 1.2
const X2 = 0.4
// Two independent binary latents, each with its own observation: the cost of one is noise for the other.
const model: Model = () => {
  const a = sample('a', bernoulli({ probs: 0.3 }))
  const b = sample('b', bernoulli({ probs: 0.6 }))
  observe('x1', normal(mul(a, 2), 1), X1)
  observe('x2', normal(mul(b, 2), 1), X2)
}
const guide: Model = () => {
  sample('a', bernoulli({ logits: param('la', tensor(0.3)) }))
  sample('b', bernoulli({ logits: param('lb', tensor(-0.2)) }))
}
const params = { la: tensor(0.3), lb: tensor(-0.2) }

const normalLogPdf = (x: number, m: number): number => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2

/** sum over the state s of q(s) (log p(s) + log N(x; 2s, 1) - log q(s)), for q = P(s = 1). */
const binaryElbo = (q: Value, prior: number, x: number): Value =>
  sub(
    add(mul(q, Math.log(prior) + normalLogPdf(x, 2)), mul(sub(1, q), Math.log(1 - prior) + normalLogPdf(x, 0))),
    add(mul(q, log(q)), mul(sub(1, q), log(sub(1, q)))),
  )

/** The exact ELBO of the model and guide above as a differentiable function of the logits. */
const exactElbo = (p: Record<string, Value>): Value =>
  add(binaryElbo(sigmoid(p['la'] as Value), 0.3, X1), binaryElbo(sigmoid(p['lb'] as Value), 0.6, X2))

function exactGradient(): Record<string, number> {
  const { grad, value } = valueAndGrad(exactElbo)(params as Record<string, Value>)
  void value
  const g = grad as Record<string, Value>
  // The loss is minus the ELBO.
  return { la: -item(g['la'] as Value), lb: -item(g['lb'] as Value) }
}

describe('the dependency graph', () => {
  it('credits each site with the costs that depend on it', () => {
    const g = dependencyGraph(model, guide, undefined, params, key(0))
    expect([...g.guide].sort()).toEqual(['a', 'b'])
    const ofA = g.downstream['a']?.map((c) => c.join(':')).sort()
    const ofB = g.downstream['b']?.map((c) => c.join(':')).sort()
    expect(ofA).toEqual(['guide:a', 'model:a', 'model:x1'])
    expect(ofB).toEqual(['guide:b', 'model:b', 'model:x2'])
  })

  it('follows dependencies between guide sites, so a descendant of a draw is downstream of it', () => {
    const chain: Model = () => {
      const a = sample('a', bernoulli({ probs: 0.5 }))
      const b = sample('b', bernoulli({ probs: mul(a, 0.6) }))
      observe('y', normal(b, 1), 0.5)
    }
    const chainGuide: Model = () => {
      const a = sample('a', bernoulli({ logits: param('la', tensor(0)) }))
      sample('b', bernoulli({ logits: add(param('lb', tensor(0)), a) }))
    }
    const g = dependencyGraph(chain, chainGuide, undefined, { la: 0, lb: 0 }, key(1))
    expect(g.parents['b']).toEqual(['a'])
    expect(g.downstream['a']?.map((c) => c.join(':'))).toContain('model:y')
    expect(g.downstream['b']?.map((c) => c.join(':'))).not.toContain('guide:a')
  })
})

describe('the gradient is unbiased', () => {
  const exact = exactGradient()
  const draws = 1600
  const kinds = [
    ['none', traceGraphElbo({ baseline: 'none' })],
    ['decay', traceGraphElbo({ baseline: 'decay' })],
    ['learned', traceGraphElbo({ baseline: 'learned' })],
    ['leaveOneOut', traceGraphElbo({ baseline: 'leaveOneOut', numParticles: 2 })],
  ] as const
  for (const [name, estimator] of kinds) {
    it(`with the ${name} baseline, as against the exact ELBO's gradient`, () => {
      // A baseline fixed by hand: any constant that does not depend on the draw is a valid control variate.
      const state: TraceGraphState = { ...(estimator.init() as TraceGraphState), baselines: { a: -2.5, b: -1.5 } }
      const m = gradientMoments({
        estimator,
        model,
        guide,
        data: undefined,
        params,
        key: key(3),
        repeats: draws,
        state,
      })
      for (const k of ['la', 'lb'] as const) {
        const se = Math.sqrt((m.variance[k]?.[0] ?? 0) / draws)
        expect(Math.abs((m.mean[k]?.[0] ?? 0) - (exact[k] ?? 0)), `${name}: d ${k}`).toBeLessThan(4.5 * se + 1e-9)
      }
    })
  }
})

describe('the variance falls', () => {
  const measure = (estimator: Parameters<typeof gradientMoments>[0]['estimator'], state?: unknown) =>
    gradientMoments({
      estimator,
      model,
      guide,
      data: undefined,
      params,
      key: key(8),
      repeats: 800,
      ...(state === undefined ? {} : { state }),
    })

  it('when the costs a site cannot influence are dropped (against the whole-ELBO signal)', () => {
    const whole = measure(traceElbo({ baselineDecay: 0 }))
    const graph = measure(traceGraphElbo({ baseline: 'none' }))
    expect(graph.totalVariance).toBeLessThan(0.7 * whole.totalVariance)
  })

  it('with a baseline near the mean cost, and with leave-one-out over particles', () => {
    const none = measure(traceGraphElbo({ baseline: 'none' }))
    const decay = measure(traceGraphElbo({ baseline: 'decay' }), {
      ...(traceGraphElbo().init() as TraceGraphState),
      baselines: { a: -1.6, b: -1.0 },
    })
    expect(decay.totalVariance).toBeLessThan(0.8 * none.totalVariance)
    const loo = measure(traceGraphElbo({ baseline: 'leaveOneOut', numParticles: 2 }))
    const two = measure(traceGraphElbo({ baseline: 'none', numParticles: 2 }))
    expect(loo.totalVariance).toBeLessThan(two.totalVariance)
  })

  it('the learned baseline tracks the cost: after some steps it beats none', () => {
    const est = traceGraphElbo({ baseline: 'learned', forgetting: 0.98 })
    let state = est.init()
    for (let i = 0; i < 150; i++) state = est.estimate(foldIn(key(5), i), model, guide, undefined, params, state).state
    const learned = measure(est, state)
    const none = measure(traceGraphElbo({ baseline: 'none' }))
    expect(learned.totalVariance).toBeLessThan(none.totalVariance)
  })
})

describe('under plates', () => {
  const N = 3
  const ys = [0.9, -0.3, 1.4]
  const plated: Model = () => {
    plate('n', N, () => {
      const z = sample('z', bernoulli({ probs: 0.4 }))
      observe('y', normal(mul(z, 2), 1), ys)
    })
  }
  const platedGuide: Model = () => {
    const l = param('l', zeros([N]))
    plate('n', N, () => sample('z', bernoulli({ logits: l })))
  }
  const p0 = { l: tensor([0.2, -0.4, 0.1]) }
  const exact = (p: Record<string, Value>): Value => {
    let total: Value = tensor(0)
    for (let i = 0; i < N; i++) total = add(total, binaryElbo(sigmoid(take(p['l'] as Value, i)), 0.4, ys[i] ?? 0))
    return total
  }

  it('credits a local latent with its own element only, and is unbiased', () => {
    const g = dependencyGraph(plated, platedGuide, undefined, p0, key(0))
    expect(g.downstream['z']?.map((c) => c.join(':')).sort()).toEqual(['guide:z', 'model:y', 'model:z'])
    const { grad } = valueAndGrad(exact)(p0 as Record<string, Value>)
    const e = Array.from(toFlat(primalOf((grad as Record<string, Value>)['l'] as Value)), (x) => -x)
    const m = gradientMoments({
      estimator: traceGraphElbo({ baseline: 'none' }),
      model: plated,
      guide: platedGuide,
      data: undefined,
      params: p0,
      key: key(4),
      repeats: 1500,
    })
    for (let i = 0; i < N; i++) {
      const se = Math.sqrt((m.variance['l']?.[i] ?? 0) / 1500)
      expect(Math.abs((m.mean['l']?.[i] ?? 0) - (e[i] ?? 0))).toBeLessThan(4.5 * se + 1e-9)
    }
    // Against crediting each element with the whole ELBO, the per-element signal is far quieter.
    const whole = gradientMoments({
      estimator: traceElbo({ baselineDecay: 0 }),
      model: plated,
      guide: platedGuide,
      data: undefined,
      params: p0,
      key: key(4),
      repeats: 1500,
    })
    expect(m.totalVariance).toBeLessThan(0.7 * whole.totalVariance)
  })
})
