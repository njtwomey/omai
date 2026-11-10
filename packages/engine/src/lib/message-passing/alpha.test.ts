import { describe, expect, it } from 'vitest'
import { integrate } from '../../testing/message-passing.ts'
import { expectResumable } from '../../testing/mp-variants.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'

const density = (x: number, m: number, v: number): number =>
  Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)

/** A robust likelihood: an inlier N(y; x, 1) with probability 1 - w, clutter N(y; 0, 10) with probability w. */
const clutter = M.numericOperator({
  name: 'clutter',
  arity: 1,
  logDensity: ([x], p) =>
    Math.log(
      (1 - (p['w'] as number)) * density(p['y'] as number, x as number, 1) +
        (p['w'] as number) * density(p['y'] as number, 0, 10),
    ),
})
const operators = { ...M.defaultOperators(), clutter }

const ys = [2.1, 3.0, 4.4, 2.6, -2.5, 3.6, 1.2, 2.9]
const clutterGraph = (): M.CompiledGraph =>
  M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [
        M.gaussianPrior('x', 0, 100),
        ...ys.map((y) => ({ operator: 'clutter', args: ['x'], params: { y, w: 0.5 } })),
      ],
    },
    operators,
  )

const belief = (g: M.CompiledGraph, state: M.MpState, name = 'x'): { mean: number; variance: number } => {
  const m = M.marginalsOf(g, state)[name] as Record<string, number>
  return { mean: m['mean'] as number, variance: m['variance'] as number }
}

describe('alpha = 1 is EP', () => {
  const g = clutterGraph()
  it('gives the very same state as epAlgorithm', () => {
    const a = runAlgorithm(M.alphaEpAlgorithm(g, { alpha: 1, tol: 1e-10 }), { maxSteps: 300 })
    const b = runAlgorithm(M.epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 300 })
    expect(a.status).toBe('converged')
    expect(a.state.messages).toEqual(b.state.messages)
  })
})

describe('one alpha update minimises the alpha-divergence of the tilted distribution', () => {
  // x ~ N(0, 4) and a factor with two unequal bumps: the tilted distribution is bimodal, so the divergences differ.
  const bumps = M.numericOperator({
    name: 'bumps',
    arity: 1,
    logDensity: ([x]) => Math.log(0.7 * density(x as number, -1.5, 0.5) + 0.3 * density(x as number, 1.5, 0.5)),
  })
  const g = M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', 0, 4), { operator: 'bumps', args: ['x'] }],
    },
    { ...operators, bumps },
  )
  const tilted = (x: number): number => density(x, 0, 4) * (0.7 * density(x, -1.5, 0.5) + 0.3 * density(x, 1.5, 0.5))
  const mass = integrate(tilted, -15, 15, 200, 12)
  // J(m, v) = integral of p^alpha q^(1 - alpha): the alpha-divergence is (1 - J) / (alpha (1 - alpha)), so J is maximised.
  const overlap = (alpha: number, m: number, v: number): number =>
    integrate((x) => (tilted(x) / mass) ** alpha * density(x, m, v) ** (1 - alpha), -15, 15, 100, 12)
  const argmax = (alpha: number, start: [number, number]): [number, number] => {
    let [cm, cv] = start
    let hm = 0.5
    let hv = 0.5
    for (let round = 0; round < 24; round++) {
      let best: [number, number, number] = [cm, cv, Number.NEGATIVE_INFINITY]
      for (let i = -3; i <= 3; i++) {
        for (let j = -3; j <= 3; j++) {
          const m = cm + (hm * i) / 3
          const v = cv * Math.exp((hv * j) / 3)
          const value = overlap(alpha, m, v)
          if (value > best[2]) best = [m, v, value]
        }
      }
      ;[cm, cv] = [best[0], best[1]]
      hm *= 0.55
      hv *= 0.55
    }
    return [cm, cv]
  }
  for (const alpha of [0.75, 0.5, 0.25]) {
    it(`alpha ${alpha}: the fixed point is the maximiser of the overlap integral`, () => {
      const r = runAlgorithm(M.alphaEpAlgorithm(g, { alpha, tol: 1e-12 }), { maxSteps: 100 })
      expect(r.status).toBe('converged')
      const q = belief(g, r.state)
      const [m, v] = argmax(alpha, [q.mean, q.variance])
      expect(Math.abs(q.mean - m)).toBeLessThan(2e-3)
      expect(Math.abs(Math.log(q.variance / v))).toBeLessThan(5e-3)
      // and it beats the EP solution on the alpha-divergence it minimises
      const ep = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 100 })
      const e = belief(g, ep.state)
      expect(overlap(alpha, q.mean, q.variance)).toBeGreaterThan(overlap(alpha, e.mean, e.variance))
    })
  }
})

describe('the alpha sweep on a clutter problem interpolates between EP and the variational Gaussian', () => {
  const g = clutterGraph()
  const lp = (x: number): number =>
    density(x, 0, 100) * ys.reduce((a, y) => a * (0.5 * density(y, x, 1) + 0.5 * density(y, 0, 10)), 1)
  const Z = integrate(lp, -40, 40, 300, 12)
  const klpq = (b: { mean: number; variance: number }): number =>
    integrate(
      (x) => (lp(x) / Z) * (Math.log(lp(x) / Z + 1e-300) - Math.log(density(x, b.mean, b.variance) + 1e-300)),
      -40,
      40,
      300,
      12,
    )
  const klqp = (b: { mean: number; variance: number }): number =>
    integrate(
      (x) =>
        density(x, b.mean, b.variance) *
        (Math.log(density(x, b.mean, b.variance) + 1e-300) - Math.log(lp(x) / Z + 1e-300)),
      -40,
      40,
      300,
      12,
    )
  const alphas = [1, 0.5, 0.25, 0.1, 0.01]
  const fits = alphas.map((alpha) => {
    const r = runAlgorithm(M.alphaEpAlgorithm(g, { alpha, tol: 1e-10 }), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    return belief(g, r.state)
  })
  it('narrows as alpha falls (mass-covering at 1, mode-seeking toward 0), on this problem', () => {
    for (let i = 1; i < fits.length; i++)
      expect((fits[i] as { variance: number }).variance).toBeLessThan((fits[i - 1] as { variance: number }).variance)
  })
  it('KL(p || q) grows and KL(q || p) shrinks as alpha falls, on this problem', () => {
    const forward = fits.map(klpq)
    const reverse = fits.map(klqp)
    for (let i = 1; i < fits.length; i++) {
      expect(forward[i] as number).toBeGreaterThan(forward[i - 1] as number)
      expect(reverse[i] as number).toBeLessThan(reverse[i - 1] as number)
    }
  })
  it('alpha -> 0 reaches the Gaussian that minimises KL(q || p) of the whole posterior', () => {
    // a zooming grid search for the variational Gaussian
    let cm = 2.9
    let cv = 0.3
    let hm = 0.5
    let hv = 0.5
    for (let round = 0; round < 14; round++) {
      let best: [number, number, number] = [cm, cv, Number.POSITIVE_INFINITY]
      for (let i = -3; i <= 3; i++) {
        for (let j = -3; j <= 3; j++) {
          const m = cm + (hm * i) / 3
          const v = cv * Math.exp((hv * j) / 3)
          const k = klqp({ mean: m, variance: v })
          if (k < best[2]) best = [m, v, k]
        }
      }
      ;[cm, cv] = [best[0], best[1]]
      hm *= 0.5
      hv *= 0.5
    }
    const q = fits[fits.length - 1] as { mean: number; variance: number }
    expect(Math.abs(q.mean - cm)).toBeLessThan(2e-3)
    expect(Math.abs(Math.log(q.variance / cv))).toBeLessThan(5e-3)
  })
})

describe('alpha-divergence message passing: the contract', () => {
  const g = clutterGraph()
  it('refuses an alpha outside (0, 1], factors with powers, and graphs with nothing to project', () => {
    expect(() => M.alphaProjection(0)).toThrow(/alpha/)
    expect(() => M.alphaProjection(1.5)).toThrow(/alpha/)
    expect(() => M.alphaEpAlgorithm(M.withPowers(g, (f) => (f.operator.power === true ? 0.5 : 1)))).toThrow(/power/)
    const plain = M.compileGraph(
      { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [M.gaussianPrior('x', 0, 1)] },
      operators,
    )
    expect(() => M.alphaEpAlgorithm(plain, { alpha: 0.5 })).toThrow(/nothing to project/)
  })
  it('is a method with an alpha knob, a typed tape, and no evidence below alpha 1', () => {
    const method = M.alphaEpMethod(g)
    expect(method.name).toBe('alpha-ep')
    expect(Object.keys(method.config.spaces)).toContain('alpha')
    const rec = recorder()
    runAlgorithm(M.alphaEpAlgorithm(g, { alpha: 0.5 }), { maxSteps: 100, tracer: rec })
    const names = new Set(rec.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    expect(names.has('change')).toBe(true)
    expect(names.has('evidence')).toBe(false)
    const ep = recorder()
    runAlgorithm(M.alphaEpAlgorithm(g, { alpha: 1 }), { maxSteps: 100, tracer: ep })
    expect(ep.run().events.some((e) => e.type === 'metric' && e.name === 'evidence')).toBe(true)
  })
  it('serialises and resumes to the same state, and runs anytime in slices to the same marginals', () => {
    const alg = M.alphaEpAlgorithm(g, { alpha: 0.4, granularity: 'message', tol: 1e-10 })
    expectResumable(alg, 5)
    let now = 0
    const first = M.runAnytime(g, alg, { budgetMs: 15, snapshotEveryMs: 5, clock: () => (now += 1) })
    expect(first.status).toBe('budget')
    const second = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot.marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
})

describe('alpha-divergence message passing on probit regression', () => {
  it('is EP at alpha 1 and stays close to it at alpha one half', () => {
    const rows = [0.3, -1.1, 0.8, 1.6, -0.4]
    const labels = [1, 0, 1, 1, 0]
    const g = M.compileGraph(
      {
        variables: [
          { name: 'w', family: { kind: 'gaussian' } },
          ...labels.map((l, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: l })),
          ...rows.map((_, i): M.VariableSpec => ({ name: `s${i}`, family: { kind: 'gaussian' } })),
        ],
        factors: [
          M.gaussianPrior('w', 0, 4),
          ...rows.flatMap((a, i): M.FactorSpec[] => [
            { operator: 'times-constant', args: [`s${i}`, 'w'], params: { factor: a } },
            { operator: 'probit', args: [`y${i}`, `s${i}`] },
          ]),
        ],
      },
      operators,
    )
    const fit = (alpha: number): { mean: number; variance: number } =>
      belief(g, runAlgorithm(M.alphaEpAlgorithm(g, { alpha, tol: 1e-10 }), { maxSteps: 500 }).state, 'w')
    const ep = fit(1)
    const half = fit(0.5)
    expect(Math.abs(half.mean - ep.mean)).toBeLessThan(0.05 * Math.abs(ep.mean))
    expect(half.variance).toBeLessThan(ep.variance)
  })
})
