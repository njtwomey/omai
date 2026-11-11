// The variational messages of the operators added beyond the conjugate ones (numeric likelihoods, scales, shapes, softmax,
// deterministic functions, products of gammas, the select, the transition). A stochastic factor is checked against a brute-force
// reference of the bound: its mean-field fixed point must be a stationary maximum of the evidence lower bound, computed with a
// quadrature of its own (`bruteExpectation`: composite Gauss-Legendre, not the library's rule), and the engine's reported bound
// must equal that brute-force bound at the fixed point. A deterministic factor (which defines its output) is checked against the
// same bound of the model with the output substituted.
import { describe, expect, it } from 'vitest'
import { type BruteBelief, bruteExpectation } from '../../../testing/message-passing.ts'
import { lgammaOf } from '../_scalar.ts'
import * as M from '../index.ts'

const operators = M.defaultOperators()

type Kind = 'gaussian' | 'gamma' | 'beta' | 'discrete' | 'gaussian-mv'

interface Var {
  readonly name: string
  readonly kind: Kind
  /** The number of states of a discrete variable, the length of a vector. */
  readonly size?: number
  /** The natural parameters of the normalised prior on the variable. */
  readonly prior: readonly number[]
  readonly log?: boolean
  /** Where the log density has a kink in this variable's coordinate. */
  readonly kinks?: readonly number[]
  /** The panels of the brute-force rule, fewer where many variables make a product grid. */
  readonly panels?: number
}

interface Factor {
  readonly operator: string
  /** The arguments of the operator in port order: names of variables. */
  readonly args: readonly string[]
  readonly params?: M.Params
}

interface Setup extends Factor {
  readonly free: readonly Var[]
  /** Observed variables of the factor, by name and value. */
  readonly observed?: Readonly<Record<string, number>>
  /** Variables a deterministic factor defines (they are not part of the bound: the mean field of the model with them substituted). */
  readonly defined?: readonly { readonly name: string; readonly kind: Kind }[]
  /** The other factors of the graph (the readers of a defined variable). */
  readonly extra?: readonly Factor[]
  /** The log density of the factor at one coordinate per free variable (the order of `free`). */
  readonly logf: (x: readonly number[]) => number
}

const settings: M.EngineSettings = {
  projection: M.klqpProjection,
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0,
  improper: 'skip',
  tol: 1e-13,
  maxSweeps: 800,
}

/** Run variational message passing on the setup to its fixed point and return the beliefs and the engine's bound. */
function fit(s: Setup, damping = 0): { etas: number[][]; bound: number; sweeps: number } {
  const spec: M.FactorGraphSpec = {
    variables: [
      ...s.free.map((v) => ({ name: v.name, family: specOf(v) })),
      ...(s.defined ?? []).map((v) => ({ name: v.name, family: { kind: v.kind } as M.FamilySpec })),
      ...Object.entries(s.observed ?? {}).map(([name, value]) => ({
        name,
        family: { kind: 'gaussian' } as M.FamilySpec,
        observed: value,
      })),
    ],
    factors: [
      ...s.free.map((v) => ({ operator: 'prior', args: [v.name], params: { eta: [...v.prior] } })),
      { operator: s.operator, args: [...s.args], params: s.params ?? {} },
      ...(s.extra ?? []).map((f) => ({ operator: f.operator, args: [...f.args], params: f.params ?? {} })),
    ],
  }
  const g = M.compileGraph(spec, operators)
  const alg = M.vmpAlgorithm(g, { tol: 1e-13, maxSweeps: 800, damping })
  let state = alg.init(undefined as never)
  let used = 0
  for (let k = 0; k < 800; k++) {
    state = alg.step(state, undefined as never)
    used = k + 1
    if (state.change < 1e-12 && k > 3) break
  }
  const etas = s.free.map((v) => [
    ...M.beliefOf(
      g,
      state.messages,
      g.variables.findIndex((x) => x.name === v.name),
    ),
  ])
  return { etas, bound: M.evidenceOf(g, settings, state), sweeps: used }
}

const specOf = (v: Var): M.FamilySpec =>
  (v.size === undefined ? { kind: v.kind } : { kind: v.kind, size: v.size }) as M.FamilySpec
const familyOf = (v: Var): M.Family => M.familyOf(specOf(v))

/** The brute-force evidence lower bound at beliefs `etas`: the priors' expected logs, the factor's by the brute rule, the entropies. */
function bruteBound(s: Setup, etas: readonly (readonly number[])[]): number {
  let total = 0
  for (const [i, v] of s.free.entries()) {
    const f = familyOf(v)
    const eta = etas[i] as readonly number[]
    total += f.pair(v.prior as number[], f.moments(eta as number[])) - f.logPartition(v.prior as number[])
    total += f.entropy(eta as number[])
  }
  const beliefs: BruteBelief[] = s.free.map((v, i) => ({
    kind: v.kind === 'gaussian-mv' ? 'gaussian' : v.kind,
    eta: etas[i] as readonly number[],
    ...(v.log === true ? { log: true } : {}),
    ...(v.kinks === undefined ? {} : { kinks: v.kinks }),
    ...(v.panels === undefined ? {} : { panels: v.panels }),
  }))
  return total + bruteExpectation(beliefs, s.logf)
}

/** The bound of the setup as a function of the natural parameters, perturbed in one coordinate. */
function perturbed(etas: readonly (readonly number[])[], at: [number, number], by: number): number[][] {
  return etas.map((e, i) => e.map((v, j) => (i === at[0] && j === at[1] ? v + by : v)))
}

/**
 * Check that the fixed point of variational message passing is a stationary maximum of the brute-force bound: the gradient in
 * every natural parameter is zero (a central difference), and the bound does not rise under perturbations of the beliefs. Also that the
 * engine's bound is the brute-force one there.
 */
function expectStationary(s: Setup, options: { damping?: number; gradient?: number; bound?: number } = {}): void {
  const { etas, bound } = fit(s, options.damping ?? 0)
  const here = bruteBound(s, etas)
  expect(bound).toBeCloseTo(here, Math.round(-Math.log10(options.bound ?? 1e-5)))
  const h = 1e-3
  for (const [i, e] of etas.entries())
    for (let j = 0; j < e.length; j++) {
      const up = bruteBound(s, perturbed(etas, [i, j], h))
      const down = bruteBound(s, perturbed(etas, [i, j], -h))
      expect(Math.abs(up - down) / (2 * h)).toBeLessThan(options.gradient ?? 2e-4)
    }
  // A local maximum: small steps along each coordinate lower the bound (the stationary point is not a saddle).
  for (const [i, e] of etas.entries())
    for (let j = 0; j < e.length; j++)
      for (const by of [0.04, -0.04]) {
        const moved = perturbed(etas, [i, j], by)
        const proper = moved.every((m, k) => familyOf(s.free[k] as Var).isProper(m))
        if (proper) expect(bruteBound(s, moved)).toBeLessThan(here + 1e-9)
      }
}

const g = (mean: number, variance: number): number[] => [...M.gaussianEta(mean, variance)]
const lgam = (shape: number, rate: number): number[] => [...M.gammaEta(shape, rate)]

describe('the quadrature of the non-conjugate message', () => {
  it('a quadratic log density gives the conjugate message exactly, in any family', () => {
    // log f = -(x - 2)^2 / (2 * 0.5) is a Gaussian likelihood of precision 2 and mean 2: the message is [4, 2].
    const x = M.messageArg(M.gaussian, M.gaussianEta(0.3, 1.7))
    const ports = [x].map((a) => M.quadraturePort(a))
    expect(M.numericMessage(ports, 0, (v) => -(((v[0] as number) - 2) ** 2))).toEqual([
      expect.closeTo(4, 10),
      expect.closeTo(2, 10),
    ])
    // log f = 3 log x - 2 x on a gamma: the message is the gamma's own [3, -2].
    const t = M.messageArg(M.gamma, M.gammaEta(2, 1.5))
    const m = M.numericMessage([M.quadraturePort(t)], 0, (v) => 3 * Math.log(v[0] as number) - 2 * (v[0] as number))
    expect(m[0]).toBeCloseTo(3, 8)
    expect(m[1]).toBeCloseTo(-2, 8)
    // log f = 2 log p + 5 log(1 - p) on a beta: the message is [2, 5].
    const b = M.messageArg(M.beta, M.betaEta(3, 2))
    const mb = M.numericMessage(
      [M.quadraturePort(b)],
      0,
      (v) => 2 * Math.log(v[0] as number) + 5 * Math.log(1 - (v[0] as number)),
    )
    expect(mb[0]).toBeCloseTo(2, 8)
    expect(mb[1]).toBeCloseTo(5, 8)
  })

  it('a port with no density is refused, not sent a message', () => {
    expect(() => M.quadraturePort(M.messageArg(M.gaussian, [0, -1]))).toThrow(M.ImproperMessageError)
    expect(() => M.quadraturePort(M.messageArg(M.gamma, [0, 1]))).toThrow(M.ImproperMessageError)
  })
})

describe('numeric likelihoods of a Gaussian predictor: the fixed point maximises the brute-force bound', () => {
  const x: Var = { name: 'x', kind: 'gaussian', prior: g(0.4, 2) }
  it('a Poisson count with a log link', () => {
    const y = 4
    expectStationary({
      free: [x],
      operator: 'likelihood-site',
      args: ['x'],
      params: { likelihood: 'poisson-log', y, exposure: 1 },
      logf: ([u]) => y * (u as number) - Math.exp(u as number) - lgammaOf(y + 1),
    })
  })
  it('a Student-t outcome (heavy tails: not log-concave, so damped)', () => {
    const y = 1.5
    const df = 4
    const scale = 0.8
    expectStationary(
      {
        free: [x],
        operator: 'student-t-site',
        args: ['x'],
        params: { y, df, scale },
        logf: ([u]) =>
          lgammaOf((df + 1) / 2) -
          lgammaOf(df / 2) -
          0.5 * Math.log(df * Math.PI) -
          Math.log(scale) -
          ((df + 1) / 2) * Math.log1p(((y - (u as number)) / scale) ** 2 / df),
      },
      { damping: 0.3 },
    )
  })
  it('a Laplace outcome (a kink at the outcome: a looser quadrature)', () => {
    const y = 0.9
    const scale = 0.7
    expectStationary(
      {
        free: [{ ...x, kinks: [y] }],
        operator: 'laplace-site',
        args: ['x'],
        params: { y, scale },
        logf: ([u]) => -Math.log(2 * scale) - Math.abs(y - (u as number)) / scale,
      },
      { gradient: 2e-3, bound: 1e-4 },
    )
  })
  it('a Weibull survival time with a log scale', () => {
    const y = 2.2
    const shape = 1.6
    expectStationary({
      free: [x],
      operator: 'likelihood-site',
      args: ['x'],
      params: { likelihood: 'weibull-log', y, shape, censored: 0 },
      logf: ([u]) =>
        Math.log(shape) -
        (u as number) +
        (shape - 1) * (Math.log(y) - (u as number)) -
        Math.exp(shape * (Math.log(y) - (u as number))),
    })
  })
})

describe('the Gaussian with a random log scale', () => {
  it('gaussian-scale: x ~ N(m, exp(2 s)) with a random mean and a random log scale', () => {
    const xObs = 1.3
    expectStationary({
      free: [
        { name: 'm', kind: 'gaussian', prior: g(0, 2) },
        { name: 's', kind: 'gaussian', prior: g(0, 1) },
      ],
      observed: { x: xObs },
      operator: 'gaussian-scale',
      args: ['x', 'm', 's'],
      logf: ([m, s]) =>
        -(s as number) - 0.5 * Math.log(2 * Math.PI) - 0.5 * (xObs - (m as number)) ** 2 * Math.exp(-2 * (s as number)),
    })
  })
  it('gaussian-log-scale: x ~ N(loc, exp(2 s))', () => {
    const xObs = -0.8
    expectStationary({
      free: [{ name: 's', kind: 'gaussian', prior: g(0.2, 1.5) }],
      observed: { x: xObs },
      operator: 'gaussian-log-scale',
      args: ['x', 's'],
      params: { loc: 0.1 },
      logf: ([s]) =>
        -(s as number) - 0.5 * Math.log(2 * Math.PI) - 0.5 * (xObs - 0.1) ** 2 * Math.exp(-2 * (s as number)),
    })
  })
})

describe('two random scalars and the shapes of a beta', () => {
  it('likelihood-pair: Laplace noise with a Gaussian location and a gamma scale (a gamma port read through its log)', () => {
    const y = 0.7
    expectStationary(
      {
        free: [
          { name: 'u', kind: 'gaussian', prior: g(0.2, 1), kinks: [y] },
          { name: 'b', kind: 'gamma', prior: lgam(2, 2), log: true },
        ],
        operator: 'likelihood-pair',
        args: ['u', 'b'],
        params: { likelihood: 'laplace-scale', y },
        logf: ([u, t]) => -Math.LN2 - (t as number) - Math.abs(y - (u as number)) * Math.exp(-(t as number)),
      },
      { gradient: 2e-3, bound: 1e-4 },
    )
  })
  it('negative-binomial-dispersion: two Gaussian ports', () => {
    const y = 3
    const logFactorial = lgammaOf(y + 1)
    expectStationary(
      {
        free: [
          { name: 'u', kind: 'gaussian', prior: g(1, 0.6) },
          { name: 's', kind: 'gaussian', prior: g(-1, 0.6) },
        ],
        operator: 'negative-binomial-dispersion',
        args: ['u', 's'],
        params: { y, exposure: 1 },
        logf: ([u, s]) => {
          const r = Math.exp(-(s as number))
          const mu = Math.exp(u as number)
          return lgammaOf(y + r) - lgammaOf(r) - logFactorial + r * Math.log(r / (r + mu)) + y * Math.log(mu / (r + mu))
        },
      },
      { gradient: 1e-3, bound: 1e-4 },
    )
  })
  it('dirichlet-concentration: a beta with a gamma concentration (non-conjugate in the gamma)', () => {
    expectStationary({
      free: [
        { name: 'p', kind: 'beta', prior: [...M.betaEta(3, 2)] },
        { name: 'a', kind: 'gamma', prior: lgam(2, 1) },
      ],
      operator: 'dirichlet-concentration',
      args: ['p', 'a'],
      params: { weights: [1, 1] },
      // E over q(p) of log Beta(p; a, a): the Beta normaliser and (a - 1) (E log p + E log (1 - p)).
      logf: ([p, a]) =>
        lgammaOf(2 * (a as number)) -
        2 * lgammaOf(a as number) +
        ((a as number) - 1) * (Math.log(p as number) + Math.log(1 - (p as number))),
    })
  })
})

describe('the variational messages are conjugate where the factor is', () => {
  it('mixture-weight: the switch summed out, a beta weight', () => {
    expectStationary({
      free: [{ name: 'p', kind: 'beta', prior: [...M.betaEta(2, 3)] }],
      operator: 'mixture-weight',
      args: ['p'],
      params: { l1: 3, l0: 0.5 },
      logf: ([p]) => Math.log(3 * (p as number) + 0.5 * (1 - (p as number))),
    })
  })
})

describe('the softmax over class scores', () => {
  it('multiclass-site: three Gaussian scores, the observed class is the second', () => {
    expectStationary({
      free: [
        { name: 'z0', kind: 'gaussian', prior: g(0.3, 1.2) },
        { name: 'z1', kind: 'gaussian', prior: g(-0.2, 0.8) },
        { name: 'z2', kind: 'gaussian', prior: g(0.1, 1.5) },
      ],
      operator: 'multiclass-site',
      args: ['z0', 'z1', 'z2'],
      params: { y: 1, classes: 3 },
      logf: (z) => {
        const top = Math.max(...z)
        return (z[1] as number) - top - Math.log(z.reduce((a, v) => a + Math.exp(v - top), 0))
      },
    })
  })
})

describe('a beta with two random shapes', () => {
  it('beta-two-shapes: gamma shapes, non-conjugate in both', () => {
    expectStationary({
      free: [
        { name: 'p', kind: 'beta', prior: [...M.betaEta(2.5, 1.5)], panels: 8 },
        { name: 'a', kind: 'gamma', prior: lgam(3, 1.5), panels: 10 },
        { name: 'b', kind: 'gamma', prior: lgam(2, 1), panels: 10 },
      ],
      operator: 'beta-two-shapes',
      args: ['p', 'a', 'b'],
      logf: ([p, a, b]) =>
        lgammaOf((a as number) + (b as number)) -
        lgammaOf(a as number) -
        lgammaOf(b as number) +
        ((a as number) - 1) * Math.log(p as number) +
        ((b as number) - 1) * Math.log(1 - (p as number)),
    })
  })
})

describe('deterministic factors define their output: the fixed point is the mean field of the substituted model', () => {
  const tau = 3
  const reading = (): Factor => ({ operator: 'gaussian', args: ['o', 'y'], params: { precision: tau } })
  const logReading = (o: number, y: number): number => 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (o - y) ** 2

  it('gamma-product: a precision that is a local times a global one, read by a Gaussian', () => {
    const o = 0.9
    expectStationary({
      free: [
        { name: 'u', kind: 'gamma', prior: lgam(3, 2) },
        { name: 't', kind: 'gamma', prior: lgam(4, 3) },
      ],
      defined: [{ name: 'p', kind: 'gamma' }],
      observed: { o, m: 0.2 },
      operator: 'gamma-product',
      args: ['p', 'u', 't'],
      extra: [{ operator: 'gaussian-gamma-precision', args: ['o', 'm', 'p'], params: {} }],
      logf: ([u, t]) =>
        0.5 * Math.log((u as number) * (t as number)) -
        0.5 * Math.log(2 * Math.PI) -
        0.5 * (u as number) * (t as number) * (o - 0.2) ** 2,
    })
  })

  it('product: y = a b read by a Gaussian', () => {
    const o = 1.1
    expectStationary({
      free: [
        { name: 'a', kind: 'gaussian', prior: g(0.8, 0.5) },
        { name: 'b', kind: 'gaussian', prior: g(1.2, 0.6) },
      ],
      defined: [{ name: 'y', kind: 'gaussian' }],
      observed: { o },
      operator: 'product',
      args: ['y', 'a', 'b'],
      extra: [reading()],
      logf: ([a, b]) => logReading(o, (a as number) * (b as number)),
    })
  })

  for (const [name, f] of [
    ['exp', Math.exp],
    ['tanh', Math.tanh],
    ['softplus', (x: number) => Math.log1p(Math.exp(x))],
    ['square', (x: number) => x * x],
  ] as const) {
    it(`apply ${name}: y = ${name}(x) read by a Gaussian`, () => {
      const o = 0.7
      expectStationary({
        free: [{ name: 'x', kind: 'gaussian', prior: g(0.9, 0.5) }],
        defined: [{ name: 'y', kind: 'gaussian' }],
        observed: { o },
        operator: 'apply',
        args: ['y', 'x'],
        params: { fn: name },
        extra: [reading()],
        logf: ([x]) => logReading(o, f(x as number)),
      })
    })
  }

  for (const [name, f] of [
    ['max', Math.max],
    ['min', Math.min],
  ] as const) {
    it(`${name}: y = ${name}(a, b) read by a Gaussian (the kink on the diagonal: a looser quadrature)`, () => {
      const o = 0.5
      expectStationary(
        {
          free: [
            { name: 'a', kind: 'gaussian', prior: g(0.4, 0.7) },
            { name: 'b', kind: 'gaussian', prior: g(0.1, 0.9) },
          ],
          defined: [{ name: 'y', kind: 'gaussian' }],
          observed: { o },
          operator: name,
          args: ['y', 'a', 'b'],
          extra: [reading()],
          logf: ([a, b]) => logReading(o, f(a as number, b as number)),
        },
        { gradient: 3e-3, bound: 1e-3 },
      )
    })
  }

  it('select: y = c_z read by a Gaussian, with the switch a free discrete variable', () => {
    const o = 0.6
    expectStationary({
      free: [
        { name: 'c0', kind: 'gaussian', prior: g(-0.5, 1) },
        { name: 'c1', kind: 'gaussian', prior: g(1, 0.8) },
        { name: 'z', kind: 'discrete', size: 2, prior: [Math.log(0.4), Math.log(0.6)] },
      ],
      defined: [{ name: 'y', kind: 'gaussian' }],
      observed: { o },
      operator: 'select',
      args: ['y', 'z', 'c0', 'c1'],
      params: { components: 2 },
      extra: [reading()],
      logf: ([c0, c1, z]) => logReading(o, z === 0 ? (c0 as number) : (c1 as number)),
    })
  })
})

describe('the Gaussian step of a linear system', () => {
  it('gaussian-transition: x ~ N(f w + c, q) with vectors of one entry, the mean field of two Gaussians', () => {
    const f = 1.4
    const c = 0.3
    const q = 0.5
    expectStationary({
      free: [
        { name: 'x', kind: 'gaussian-mv', size: 1, prior: g(0.2, 1.5) },
        { name: 'w', kind: 'gaussian-mv', size: 1, prior: g(-0.4, 1.2) },
      ],
      operator: 'gaussian-transition',
      args: ['x', 'w'],
      params: { matrix: [f], offset: [c], covariance: [q] },
      logf: ([x, w]) => -0.5 * Math.log(2 * Math.PI * q) - ((x as number) - f * (w as number) - c) ** 2 / (2 * q),
    })
  })

  it('matmul-constant: the messages of a one-row matrix are those of dot-constant', () => {
    const w = M.gaussianMv(2)
    const y = M.gaussianMv(1)
    const belief = M.messageArg(w, M.gaussianMvEta([0.4, -0.7], [1, 0.2, 0.2, 2]))
    const cavity = M.messageArg(y, M.gaussianMvEta([1.3], [1 / 2.5]))
    const toW = M.matmulConstant.vmp?.({
      mode: 'vmp',
      to: 1,
      args: [cavity, belief],
      params: { matrix: [1.5, -2] },
      power: 1,
    })
    const dot = M.dotConstant.vmp?.({
      mode: 'vmp',
      to: 1,
      args: [M.messageArg(M.gaussian, M.gaussianEta(1.3, 1 / 2.5)), belief],
      params: { a: [1.5, -2] },
      power: 1,
    })
    expect(toW?.message).toEqual(dot?.message.map((v) => expect.closeTo(v, 10)))
    const fromW = M.matmulConstant.vmp?.({
      mode: 'vmp',
      to: 0,
      args: [cavity, belief],
      params: { matrix: [1.5, -2] },
      power: 1,
    })
    const mean = 1.5 * 0.4 + -2 * -0.7
    expect(M.gaussianMvMoments(fromW?.message as number[], 1).mean[0]).toBeCloseTo(mean, 10)
  })
})

describe('a mixture density as one factor, and a regression row', () => {
  it('mixture-site: a spike and a slab (not log-concave, damped)', () => {
    const params = { weights: [0.5, 0.5], means: [0, 0], sds: [0.3, 3] }
    expectStationary(
      {
        free: [{ name: 'x', kind: 'gaussian', prior: g(0.8, 4) }],
        operator: 'mixture-site',
        args: ['x'],
        params,
        logf: ([x]) => {
          const d = (sd: number): number => Math.exp(-0.5 * ((x as number) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
          return Math.log(0.5 * d(0.3) + 0.5 * d(3))
        },
      },
      { damping: 0.5, gradient: 1e-3 },
    )
  })

  it('a row on a weight vector is the factor on the predictor it defines (probit, logistic, Student-t, Poisson)', () => {
    const a = [0.8, -1.1]
    const priorW = M.gaussianMvEta([0.2, -0.1], [1.5, 0.3, 0.3, 1])
    const cases: { row: string; site: string; params: M.Params; outcome: boolean }[] = [
      { row: 'probit-row', site: 'probit', params: { noise: 1 }, outcome: true },
      { row: 'logistic-row', site: 'logistic', params: {}, outcome: true },
      { row: 'student-t-row', site: 'student-t-site', params: { y: 0.4, df: 3, scale: 0.7 }, outcome: false },
      {
        row: 'likelihood-row',
        site: 'likelihood-site',
        params: { likelihood: 'poisson-log', y: 3, exposure: 1 },
        outcome: false,
      },
    ]
    for (const c of cases) {
      const outcome = c.outcome ? [{ name: 'b', family: { kind: 'bernoulli' } as M.FamilySpec, observed: 1 }] : []
      const variables = (extra: M.VariableSpec[]): M.VariableSpec[] => [
        { name: 'w', family: { kind: 'gaussian-mv', size: 2 } },
        ...extra,
        ...outcome,
      ]
      const rowGraph = M.compileGraph(
        {
          variables: variables([]),
          factors: [
            { operator: 'prior', args: ['w'], params: { eta: [...priorW] } },
            { operator: c.row, args: [...(c.outcome ? ['b'] : []), 'w'], params: { ...c.params, a } },
          ],
        },
        operators,
      )
      const expandedGraph = M.compileGraph(
        {
          variables: variables([{ name: 'u', family: { kind: 'gaussian' } }]),
          factors: [
            { operator: 'prior', args: ['w'], params: { eta: [...priorW] } },
            { operator: 'dot-constant', args: ['u', 'w'], params: { a } },
            { operator: c.site, args: [...(c.outcome ? ['b'] : []), 'u'], params: c.params },
          ],
        },
        operators,
      )
      const run = (graph: M.CompiledGraph): number[] => {
        const alg = M.vmpAlgorithm(graph, { tol: 1e-13, maxSweeps: 500, damping: 0.2 })
        let state = alg.init(undefined as never)
        for (let k = 0; k < 500; k++) state = alg.step(state, undefined as never)
        return [
          ...M.beliefOf(
            graph,
            state.messages,
            graph.variables.findIndex((v) => v.name === 'w'),
          ),
        ]
      }
      const row = run(rowGraph)
      const expanded = run(expandedGraph)
      for (const [i, v] of row.entries()) expect(v, `${c.row} entry ${i}`).toBeCloseTo(expanded[i] as number, 6)
    }
  })
})

describe('an operator with no variational message says why, and the engine reports it', () => {
  it('a hard step, an interval test and a spin carry a reason for the capability check', () => {
    for (const name of ['greater-than', 'is-between', 'spin', 'spin-row']) {
      const op = operators[name] as M.FactorOperator
      expect(op.vmp, name).toBeUndefined()
      expect(op.noRule?.vmp, name).toMatch(/infinite|point masses/)
    }
  })

  it('a row of a registry whose expected log is infinite is refused with its reason, the others of the registry are not', () => {
    const pair = operators['likelihood-pair'] as M.FactorOperator
    expect(pair.unsupported?.('vmp', { likelihood: 'weibull-shape', y: 1 })).toMatch(/double exponential/)
    expect(pair.unsupported?.('vmp', { likelihood: 'laplace-scale', y: 1 })).toBeUndefined()
    expect(pair.unsupported?.('ep', { likelihood: 'weibull-shape', y: 1 })).toBeUndefined()
  })
})
