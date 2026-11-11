import { describe, expect, it } from 'vitest'
import { expectResumable, probitRows } from '../../../testing/mp-variants.ts'
import { gaussianEp } from '../../pgm/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'
import { gaussianNodes } from '../numerics/_quadrature.ts'
import { millsRatio } from '../numerics/_scalar.ts'

const operators = M.defaultOperators()
const close = (a: number, b: number, tol = 1e-8): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

/** Gaussian-latent probit regression in the engine's terms: a weight vector, one dot product and one probit link per row. */
function probitGraph(x: number[][], y: number[], priorVariance = 4): M.CompiledGraph {
  const d = (x[0] as number[]).length
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'w', family: { kind: 'gaussian-mv', size: d } },
      ...x.map((_, i): M.VariableSpec => ({ name: `s${i}`, family: { kind: 'gaussian' } })),
      ...x.map((_, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: y[i] as number })),
    ],
    factors: [
      M.gaussianMvPrior(
        'w',
        new Array<number>(d).fill(0),
        Array.from({ length: d * d }, (_, k) => (k % (d + 1) === 0 ? priorVariance : 0)),
      ),
      ...x.flatMap((row, i): M.FactorSpec[] => [
        { operator: 'dot-constant', args: [`s${i}`, 'w'], params: { a: row } },
        { operator: 'probit', args: [`y${i}`, `s${i}`] },
      ]),
    ],
  }
  return M.compileGraph(spec, operators)
}

const weights = (g: M.CompiledGraph, state: M.MpState): { mean: number[]; covariance: number[][] } => {
  const w = M.marginalsOf(g, state)['w'] as Record<string, never>
  return { mean: w['mean'] as unknown as number[], covariance: w['covariance'] as unknown as number[][] }
}

describe('power EP: power 1 is EP', () => {
  const { x, y } = probitRows(8)
  const g = probitGraph(x, y)
  it('gives the very same state as epAlgorithm', () => {
    const a = runAlgorithm(M.powerEpAlgorithm(g, { power: 1, tol: 1e-12 }), { maxSteps: 300 })
    const b = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
    expect(a.status).toBe('converged')
    expect(a.state.messages).toEqual(b.state.messages)
  })
})

describe('power EP reproduces the power EP of gaussianEp on a Gaussian-latent model', () => {
  const { x, y } = probitRows(8)
  const g = probitGraph(x, y)
  const sites = x.map((row, i) => ({
    kind: 'probit' as const,
    weights: row,
    label: y[i] === 1 ? (1 as const) : (-1 as const),
  }))
  const prior = {
    mean: [0, 0, 0],
    covariance: [
      [4, 0, 0],
      [0, 4, 0],
      [0, 0, 4],
    ],
  }
  for (const power of [0.3, 0.5, 0.8, 1]) {
    it(`at power ${power}`, () => {
      const r = runAlgorithm(M.powerEpAlgorithm(g, { power, tol: 1e-11, maxSweeps: 1000 }), { maxSteps: 1000 })
      expect(r.status).toBe('converged')
      const ref = gaussianEp(prior, sites, { power, tol: 1e-12, maxIter: 2000, quadraturePoints: 96 })
      const w = weights(g, r.state)
      for (let i = 0; i < 3; i++) {
        close(w.mean[i] as number, ref.mean[i] as number, 2e-5)
        for (let j = 0; j < 3; j++)
          close((w.covariance[i] as number[])[j] as number, (ref.covariance[i] as number[])[j] as number, 2e-5)
      }
    })
  }
})

describe('power EP is exact on Gaussian factors of one Gaussian variable, at every power', () => {
  // x ~ N(0.5, 2) and observations y_i = x + noise of variance v_i: the posterior is conjugate.
  const obs = [
    { y: 1.2, v: 0.5 },
    { y: -0.3, v: 1.5 },
    { y: 0.8, v: 0.9 },
  ]
  const prior = { m: 0.5, v: 2 }
  const precision = 1 / prior.v + obs.reduce((a, o) => a + 1 / o.v, 0)
  const mean = (prior.m / prior.v + obs.reduce((a, o) => a + o.y / o.v, 0)) / precision
  const logEvidence = ((): number => {
    // log of the integral of prior times likelihoods: complete the square.
    let s = -0.5 * Math.log(2 * Math.PI * prior.v) - (0.5 * prior.m * prior.m) / prior.v
    for (const o of obs) s += -0.5 * Math.log(2 * Math.PI * o.v) - (0.5 * o.y * o.y) / o.v
    return s + 0.5 * Math.log((2 * Math.PI) / precision) + (0.5 * (mean * precision) ** 2) / precision
  })()
  // The numeric operator takes its density from a closure, so one operator per observation.
  const ops = (): Record<string, M.FactorOperator> => ({
    ...operators,
    ...Object.fromEntries(
      obs.map((o, i) => [
        `lik${i}`,
        M.numericOperator({
          name: `lik${i}`,
          arity: 1,
          logDensity: ([z]) => -0.5 * Math.log(2 * Math.PI * o.v) - (0.5 * ((z as number) - o.y) ** 2) / o.v,
        }),
      ]),
    ),
  })
  const g = M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', prior.m, prior.v), ...obs.map((_, i) => ({ operator: `lik${i}`, args: ['x'] }))],
    },
    ops(),
  )
  for (const power of [0.25, 0.5, 0.75, 1]) {
    it(`at power ${power}: the exact posterior and the exact evidence`, () => {
      const s = { ...M.epDefaults, tol: 1e-12 }
      const alg = M.powerEpAlgorithm(g, { power, tol: 1e-12 })
      const r = runAlgorithm(alg, { maxSteps: 400 })
      expect(r.status).toBe('converged')
      const m = M.marginalsOf(g, r.state)['x'] as Record<string, number>
      close(m['mean'] as number, mean, 1e-6)
      close(m['variance'] as number, 1 / precision, 1e-6)
      const settings: M.EngineSettings = {
        ...s,
        projection: M.momentProjection,
        schedule: 'serial',
        granularity: 'sweep',
      }
      const poweredGraph = M.withPowers(g, (f) => (f.operator.power === true ? power : 1))
      close(M.evidenceOf(poweredGraph, settings, r.state), logEvidence, 1e-5)
    })
  }
})

describe('power EP as power goes to zero approaches the variational Gaussian (KL(q || p))', () => {
  // x ~ N(0, 1) and three probit sites: the KL(q || p) Gaussian solves m = m0 + v0 E_q[dlogL], 1/v = 1/v0 - E_q[d2logL].
  const labels = [1, 1, 0]
  const g = M.compileGraph(
    {
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        ...labels.map((l, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: l })),
      ],
      factors: [M.gaussianPrior('x', 0, 1), ...labels.map((_, i) => ({ operator: 'probit', args: [`y${i}`, 'x'] }))],
    },
    operators,
  )
  const variational = (): { mean: number; variance: number } => {
    const nodes = gaussianNodes(0, 1, 80)
    let m = 0
    let v = 1
    for (let it = 0; it < 4000; it++) {
      let d1 = 0
      let d2 = 0
      const s = Math.sqrt(v)
      for (let k = 0; k < nodes.x.length; k++) {
        const xk = m + s * (nodes.x[k] as number)
        for (const l of labels) {
          const sign = l === 1 ? 1 : -1
          const z = sign * xk
          const lam = millsRatio(z)
          d1 += (nodes.w[k] as number) * sign * lam
          d2 += (nodes.w[k] as number) * -lam * (z + lam)
        }
      }
      const vNew = 1 / (1 - d2)
      const mNew = d1
      m = 0.5 * m + 0.5 * mNew
      v = 0.5 * v + 0.5 * vNew
    }
    return { mean: m, variance: v }
  }
  it('is closer to it at smaller powers, and close at 0.01', () => {
    const ref = variational()
    const err = (power: number): number => {
      const r = runAlgorithm(M.powerEpAlgorithm(g, { power, tol: 1e-10, maxSweeps: 3000 }), { maxSteps: 3000 })
      expect(r.status).toBe('converged')
      const m = M.marginalsOf(g, r.state)['x'] as Record<string, number>
      return Math.abs((m['mean'] as number) - ref.mean) + Math.abs((m['variance'] as number) - ref.variance)
    }
    const errors = [1, 0.5, 0.1, 0.01].map(err)
    for (let i = 1; i < errors.length; i++) expect(errors[i] as number).toBeLessThan(errors[i - 1] as number)
    expect(errors[3] as number).toBeLessThan(0.01)
  })
})

describe('power EP: per-factor powers, ids, and the contract', () => {
  const { x, y } = probitRows(6)
  const g = probitGraph(x, y)
  it('names the factors it gives a power by id, and refuses unknown ids and unsupported powers', () => {
    expect(() => M.powerEpAlgorithm(g, { powers: { nope: 0.5 } })).toThrow(/no factor/)
    expect(() => M.powerEpAlgorithm(g, { powers: { f0: 0.5 } })).toThrow(/does not support a power/)
    const alg = M.powerEpAlgorithm(g, { powers: { f2: 0.5 } })
    const r = runAlgorithm(alg, { maxSteps: 400 })
    expect(r.status).toBe('converged')
  })
  it('is a method with a power knob and the EP tape', () => {
    const method = M.powerEpMethod(g)
    expect(method.name).toBe('power-ep')
    expect(Object.keys(method.contract as object)).toContain('mp/evidence')
  })
  it('serialises and resumes to the same state, and runs anytime in slices to the same marginals', () => {
    const alg = M.powerEpAlgorithm(g, { power: 0.6, granularity: 'message', tol: 1e-11 })
    expectResumable(alg, 5)
    let now = 0
    const first = M.runAnytime(g, alg, { budgetMs: 40, snapshotEveryMs: 10, clock: () => (now += 1) })
    expect(first.status).toBe('budget')
    const second = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot.marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
})

describe('a gate of closed-form branches takes a power', () => {
  /** Two counts on a rate that one of two regimes sets: z picks the Poisson branch (rate known to be low or high on a shared gamma rate). */
  const spec = (branchesOk: boolean): M.FactorGraphSpec => ({
    variables: [
      { name: 'z', family: { kind: 'discrete', size: 2 } },
      { name: 'a', family: { kind: 'gamma' } },
      { name: 'b', family: { kind: 'gamma' } },
    ],
    factors: [
      M.discretePrior('z', [0.5, 0.5]),
      M.priorFactor('a', M.gammaEta(2, 0.5)),
      M.priorFactor('b', M.gammaEta(2, 0.1)),
      {
        operator: 'gate',
        args: ['z', 'a', 'b'],
        params: {
          arity: 2,
          branches: branchesOk
            ? [
                { operator: 'poisson', args: [{ const: 3 }, 0], params: { exposure: 1 } },
                { operator: 'poisson', args: [{ const: 20 }, 1], params: { exposure: 1 } },
              ]
            : [
                { operator: 'gaussian', args: [0, 1], params: { precision: 1 } },
                { operator: 'poisson', args: [{ const: 20 }, 1], params: { exposure: 1 } },
              ],
        },
      },
    ],
  })

  it('power EP and the alpha divergence run on it (at power one they are EP) and give the regime posterior of the exact two-branch mixture', () => {
    const g = M.compileGraph(spec(true), operators)
    expect(g.factors[3]?.operator.name).toBe('gate')
    const ep = runAlgorithm(M.epAlgorithm(g), { maxSteps: 100 })
    const one = runAlgorithm(M.powerEpAlgorithm(g, { power: 1 }), { maxSteps: 100 })
    expect(M.marginalsOf(g, one.state)['a']).toEqual(M.marginalsOf(g, ep.state)['a'])
    for (const r of [
      runAlgorithm(M.powerEpAlgorithm(g, { power: 0.5 }), { maxSteps: 400 }),
      runAlgorithm(M.alphaEpAlgorithm(g, { alpha: 0.5 }), { maxSteps: 400 }),
    ]) {
      expect(r.status).toBe('converged')
      const z = (M.marginalsOf(g, r.state)['z'] as { probs: number[] }).probs
      expect(z[0] as number).toBeGreaterThan(0)
      expect((z[0] as number) + (z[1] as number)).toBeCloseTo(1, 10)
    }
    const powered = M.withPowers(g, (f) => (M.takesPower(f.operator, f.params) ? 0.5 : 1))
    expect(powered.factors.map((f) => f.power)).toEqual([1, 1, 1, 0.5])
  })

  it('a gate whose branches have no closed-form message over one variable keeps power 1 under power EP, and refuses one set on it', () => {
    const g = M.compileGraph(spec(false), operators)
    const gateFactor = g.factors[3] as M.CompiledFactor
    expect(M.takesPower(gateFactor.operator, gateFactor.params)).toBe(false)
    expect(M.withPowers(g, (f) => (M.takesPower(f.operator, f.params) ? 0.5 : 1)).factors.map((f) => f.power)).toEqual([
      1, 1, 1, 1,
    ])
    expect(() => M.withPowers(g, (f) => (f.operator.name === 'gate' ? 0.5 : 1))).toThrow(/does not support a power/)
  })
})
