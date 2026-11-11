import { describe, expect, it } from 'vitest'
import { integrate, integrateBox } from '../../../testing/message-passing.ts'
import { expectResumable } from '../../../testing/mp-variants.ts'
import { sigmoidOf } from '../../base/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'

const density = (x: number, m: number, v: number): number =>
  Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)
const close = (a: number, b: number, tol: number): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

/** The brute-force reference: log mass, mean and variance of N(m, v) f by integration on a wide interval. */
function reference(
  m: number,
  v: number,
  f: (x: number) => number,
): { logMass: number; mean: number; variance: number } {
  const lo = m - 14 * Math.sqrt(v)
  const hi = m + 14 * Math.sqrt(v)
  const mass = integrate((x) => density(x, m, v) * f(x), lo, hi, 80, 16)
  const mean = integrate((x) => x * density(x, m, v) * f(x), lo, hi, 80, 16) / mass
  const variance = integrate((x) => (x - mean) ** 2 * density(x, m, v) * f(x), lo, hi, 80, 16) / mass
  return { logMass: Math.log(mass), mean, variance }
}

const rules: [string, M.TiltedRule][] = [
  ['quadrature', M.quadratureRule],
  ['gauss-hermite(40)', M.gaussHermiteRule(40)],
  ['laplace', M.laplaceRule],
  ['unscented(2)', M.unscentedRule(2)],
  ['sample(16384)', M.sampleRule(16384, 7)],
]

describe('numeric rules on a factor with a Gaussian shape', () => {
  // exp(-(x - 1)^2 / 3) against N(0.3, 2): the tilted density is Gaussian, so the exact rules reproduce it.
  const logf = ([x]: readonly number[]): number => (-1 / 3) * ((x as number) - 1) ** 2
  const ref = reference(0.3, 2, (x) => Math.exp(logf([x])))
  for (const [name, rule, tol] of [
    ['quadrature', M.quadratureRule, 1e-9],
    ['gauss-hermite(40)', M.gaussHermiteRule(40), 1e-9],
    ['laplace', M.laplaceRule, 1e-6],
  ] as const) {
    it(`${name} is exact`, () => {
      const r = rule.moments({ bases: [{ mean: 0.3, variance: 2 }], to: 0, logf })
      close(r.logMass, ref.logMass, tol)
      close(r.mean, ref.mean, tol)
      close(r.variance, ref.variance, tol)
    })
  }
  it('unscented (three points) and sampling rules are close, not exact', () => {
    const u = M.unscentedRule(2).moments({ bases: [{ mean: 0.3, variance: 2 }], to: 0, logf })
    close(u.mean, ref.mean, 0.3)
    close(u.variance, ref.variance, 0.3)
    const s = M.sampleRule(16384, 7).moments({ bases: [{ mean: 0.3, variance: 2 }], to: 0, logf })
    close(s.mean, ref.mean, 0.03)
    close(s.variance, ref.variance, 0.05)
  })
})

describe('numeric rules on the logistic factor', () => {
  const m = 0.4
  const v = 1.7
  const logf = ([x]: readonly number[]): number => Math.log(sigmoidOf(x as number))
  const ref = reference(m, v, sigmoidOf)
  it('quadrature and Gauss-Hermite match the brute-force integral', () => {
    for (const rule of [M.quadratureRule, M.gaussHermiteRule(64)]) {
      const r = rule.moments({ bases: [{ mean: m, variance: v }], to: 0, logf })
      close(r.logMass, ref.logMass, 1e-8)
      close(r.mean, ref.mean, 1e-8)
      close(r.variance, ref.variance, 1e-8)
    }
  })
  it('Laplace finds the mode and the curvature of the density, and a close mass', () => {
    const r = M.laplaceRule.moments({ bases: [{ mean: m, variance: v }], to: 0, logf })
    // the mode solves sigmoid(-x) = (x - m) / v; the curvature is -sigmoid(x) sigmoid(-x) - 1 / v
    let x = m
    for (let i = 0; i < 60; i++) {
      const g = sigmoidOf(-x) - (x - m) / v
      const h = -sigmoidOf(x) * sigmoidOf(-x) - 1 / v
      x -= g / h
    }
    close(r.mean, x, 1e-6)
    close(r.variance, -1 / (-sigmoidOf(x) * sigmoidOf(-x) - 1 / v), 1e-5)
    expect(Math.abs(r.logMass - ref.logMass)).toBeLessThan(0.02)
    expect(Math.abs(r.mean - ref.mean)).toBeLessThan(0.1)
  })
  it('sampling is within its Monte Carlo error, and the unscented rule within its bias', () => {
    const s = M.sampleRule(32768, 3).moments({ bases: [{ mean: m, variance: v }], to: 0, logf })
    close(s.mean, ref.mean, 0.02)
    close(s.variance, ref.variance, 0.05)
    close(s.logMass, ref.logMass, 0.01)
    const u = M.unscentedRule(2).moments({ bases: [{ mean: m, variance: v }], to: 0, logf })
    close(u.mean, ref.mean, 0.15)
  })
})

describe('numeric rules in two arguments', () => {
  // f(x, y) = exp(-(x - y)^2 / 2) against N(0, 1) x N(1, 2): jointly Gaussian, so the marginal of x is closed form.
  const logf = ([x, y]: readonly number[]): number => -0.5 * ((x as number) - (y as number)) ** 2
  const bases = [
    { mean: 0, variance: 1 },
    { mean: 1, variance: 2 },
  ]
  const mass = integrateBox(
    (p) => density(p[0] as number, 0, 1) * density(p[1] as number, 1, 2) * Math.exp(logf(p)),
    [
      [-9, 9],
      [-14, 14],
    ],
  )
  const meanX =
    integrateBox(
      (p) => (p[0] as number) * density(p[0] as number, 0, 1) * density(p[1] as number, 1, 2) * Math.exp(logf(p)),
      [
        [-9, 9],
        [-14, 14],
      ],
    ) / mass
  it('quadrature, Gauss-Hermite and Laplace agree with the brute-force integral', () => {
    for (const [rule, tol] of [
      [M.quadratureRule, 1e-6],
      [M.gaussHermiteRule(24), 1e-8],
      [M.laplaceRule, 1e-5],
    ] as const) {
      const r = rule.moments({ bases, to: 0, logf })
      close(Math.exp(r.logMass), mass, tol)
      close(r.mean, meanX, tol)
    }
  })
  it('a point argument (an observed value) is held fixed', () => {
    const r = M.gaussHermiteRule(24).moments({
      bases: [
        { mean: 0, variance: 1 },
        { mean: 2, variance: 0 },
      ],
      to: 0,
      logf,
    })
    // x | y = 2: N(0, 1) exp(-(x - 2)^2 / 2) has mean 1 and variance 1 / 2
    close(r.mean, 1, 1e-9)
    close(r.variance, 0.5, 1e-9)
  })
  it('refuses a product rule that is too large, and no mass', () => {
    expect(() =>
      M.gaussHermiteRule(60).moments({
        bases: [bases[0] as never, bases[1] as never, { mean: 0, variance: 1 }],
        to: 0,
        logf: () => 0,
      }),
    ).toThrow(/cap/)
    expect(() =>
      M.gaussHermiteRule(8).moments({ bases: [{ mean: 0, variance: 1 }], to: 0, logf: () => Number.NEGATIVE_INFINITY }),
    ).toThrow(/no mass/)
  })
})

describe('numeric EP: the rule is a slot of the run', () => {
  const clutter = M.numericOperator({
    name: 'clutter',
    arity: 1,
    logDensity: ([x], p) =>
      Math.log(0.5 * density(p['y'] as number, x as number, 1) + 0.5 * density(p['y'] as number, 0, 10)),
  })
  const ys = [2.1, 3.0, 4.4, 2.6, -2.5, 3.6, 1.2, 2.9]
  const g = M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', 0, 100), ...ys.map((y) => ({ operator: 'clutter', args: ['x'], params: { y } }))],
    },
    { ...M.defaultOperators(), clutter },
  )
  const fit = (rule: M.TiltedRule): { mean: number; variance: number } => {
    const r = runAlgorithm(M.numericEpAlgorithm(g, { rule, tol: 1e-9 }), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    const m = M.marginalsOf(g, r.state)['x'] as Record<string, number>
    return { mean: m['mean'] as number, variance: m['variance'] as number }
  }
  const base = fit(M.quadratureRule)
  it('the default is the same as the operator without a rule: the EP of the engine', () => {
    const r = runAlgorithm(M.epAlgorithm(g, { tol: 1e-9 }), { maxSteps: 500 })
    const m = M.marginalsOf(g, r.state)['x'] as Record<string, number>
    expect(m['mean']).toBe(base.mean)
    expect(m['variance']).toBe(base.variance)
  })
  it('Gauss-Hermite and the unscented and sampling rules give EP posteriors near the quadrature one', () => {
    const gh = fit(M.gaussHermiteRule(96))
    close(gh.mean, base.mean, 1e-6)
    close(gh.variance, base.variance, 1e-4)
    const sam = fit(M.sampleRule(65536, 5))
    close(sam.mean, base.mean, 0.01)
    close(sam.variance, base.variance, 0.1)
  })
  it('Laplace propagation gives a posterior near it too, at the modes the factors have', () => {
    const lap = fit(M.laplaceRule)
    close(lap.mean, base.mean, 0.05)
    close(lap.variance, base.variance, 0.5)
  })
  it('each rule is a pure function of its input: a stopped and resumed run reaches the same state', () => {
    for (const rule of [M.gaussHermiteRule(32), M.laplaceRule, M.sampleRule(512, 2)])
      expectResumable(M.numericEpAlgorithm(g, { rule, granularity: 'message', tol: 1e-9 }), 4)
  })
  it('anytime snapshots resume to the same marginals', () => {
    const alg = M.numericEpAlgorithm(g, { rule: M.laplaceRule, granularity: 'message', tol: 1e-10 })
    let now = 0
    const first = M.runAnytime(g, alg, { budgetMs: 12, snapshotEveryMs: 4, clock: () => (now += 1) })
    const second = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot.marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
  it('a factor with no usable curvature is an improper update, counted and repaired, not a crash', () => {
    const blowup = M.numericOperator({ name: 'blowup', arity: 1, logDensity: ([x]) => (x as number) ** 2 })
    const h = M.compileGraph(
      {
        variables: [{ name: 'x', family: { kind: 'gaussian' } }],
        factors: [M.gaussianPrior('x', 0, 1), { operator: 'blowup', args: ['x'] }],
      },
      { ...M.defaultOperators(), blowup },
    )
    const r = runAlgorithm(M.numericEpAlgorithm(h, { rule: M.laplaceRule, maxSweeps: 4 }), { maxSteps: 10 })
    expect(r.state.improper.events).toBeGreaterThan(0)
  })
  it('the evidence of one factor is its mass: exact under quadrature, the Laplace approximation under Laplace', () => {
    const logistic = M.numericOperator({ name: 'sig', arity: 1, logDensity: ([x]) => Math.log(sigmoidOf(x as number)) })
    const one = M.compileGraph(
      {
        variables: [{ name: 'x', family: { kind: 'gaussian' } }],
        factors: [M.gaussianPrior('x', 0.4, 1.7), { operator: 'sig', args: ['x'] }],
      },
      { ...M.defaultOperators(), sig: logistic },
    )
    const ref = reference(0.4, 1.7, sigmoidOf).logMass
    for (const [rule, tol] of [
      [M.quadratureRule, 1e-8],
      [M.laplaceRule, 0.02],
    ] as const) {
      const settings: M.EngineSettings = { ...M.epDefaults, projection: M.numericProjection(rule), tol: 1e-12 }
      const r = runAlgorithm(M.numericEpAlgorithm(one, { rule, tol: 1e-12 }), { maxSteps: 100 })
      expect(Math.abs(M.evidenceOf(one, settings, r.state) - ref)).toBeLessThan(tol)
    }
  })
  it('the table lists the rules, and the method has a rule knob that builds each', () => {
    expect(Object.keys(M.numericRules)).toEqual(['quadrature', 'gauss-hermite', 'unscented', 'sample', 'laplace'])
    const method = M.numericEpMethod(g)
    expect(method.name).toBe('numeric-ep')
    expect(Object.keys(method.config.spaces)).toContain('rule')
    for (const kind of Object.keys(M.numericRules)) {
      const alg = method.make({ rule: { kind } } as never, { kind: 'key', words: [0, 0] })
      expect(alg.name).toBe('numeric-ep')
    }
  })
  it('names its rules', () => {
    expect(rules.map(([n]) => n).length).toBe(5)
    expect(M.numericProjection(M.gaussHermiteRule(8)).name).toBe('numeric(gauss-hermite(8))')
  })
})
