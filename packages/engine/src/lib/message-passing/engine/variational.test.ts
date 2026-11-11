import { describe, expect, it } from 'vitest'
import { integrate, integrateBox } from '../../../testing/message-passing.ts'
import * as M from '../index.ts'
import { gaussianExpectation } from '../numerics/_quadrature.ts'
import { logPhi } from '../numerics/_scalar.ts'

const operators = M.defaultOperators()
const gaussian = M.gaussian
const msg = (eta: M.Eta): M.Arg => M.messageArg(gaussian, eta)
const ctx = (to: number, args: M.Arg[], params: M.Params = {}): M.OperatorContext => ({
  mode: 'vmp',
  to,
  args,
  params,
  power: 1,
})

const settings: M.EngineSettings = {
  projection: M.klqpProjection,
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0,
  improper: 'skip',
  tol: 1e-12,
  maxSweeps: 500,
}

/** Sweeps of VMP, with the bound after each. */
function sweeps(g: M.CompiledGraph, n: number): { state: M.MpState; bounds: number[] } {
  const alg = M.vmpAlgorithm(g)
  let state = alg.init(undefined as never)
  const bounds: number[] = []
  for (let k = 0; k < n; k++) {
    state = alg.step(state, undefined as never)
    bounds.push(M.evidenceOf(g, settings, state))
  }
  return { state, bounds }
}

const stat = (g: M.CompiledGraph, state: M.MpState, name: string): { mean: number; variance: number } => {
  const m = M.marginalsOf(g, state)[name] as { mean: number; variance: number }
  return { mean: m.mean, variance: m.variance }
}

describe('the affine operators have a variational message of the mean field', () => {
  // The message to input j is exp E[log cavity(d + sum c_i a_i)] over the other inputs, as a function of a_j: checked by
  // quadrature of the definition, up to the constant the message is defined to.
  const cases: { op: M.FactorOperator; params: M.Params; coeffs: number[]; offset: number }[] = [
    { op: M.linear, params: { coeffs: [2, -1, 0.5], offset: 1.5 }, coeffs: [2, -1, 0.5], offset: 1.5 },
    { op: M.plus, params: {}, coeffs: [1, 1], offset: 0 },
    { op: M.difference, params: {}, coeffs: [1, -1], offset: 0 },
    { op: M.timesConstant, params: { factor: -3 }, coeffs: [-3], offset: 0 },
    { op: M.sum, params: { n: 3 }, coeffs: [1, 1, 1], offset: 0 },
  ]
  for (const { op, params, coeffs, offset } of cases) {
    it(`${op.name}: to an input it is the expected log of the cavity, to the output the pushforward`, () => {
      const beliefs = coeffs.map((_, i) => M.gaussianEta(0.4 * (i + 1) - 1, 0.5 + 0.3 * i))
      const cavity: M.Eta = [0.7, 2.2]
      const args = [msg(cavity), ...beliefs.map(msg)]
      const logCavity = (s: number): number => (cavity[0] as number) * s - 0.5 * (cavity[1] as number) * s * s
      for (const j of coeffs.keys()) {
        const m = op.vmp?.(ctx(j + 1, args, params)).message as M.Eta
        // E over the other inputs of log cavity(offset + c_j a + sum others), at a few values of a_j.
        const expected = (a: number): number => {
          const others = coeffs.map((_, i) => i).filter((i) => i !== j)
          const walk = (k: number, acc: number): number => {
            const i = others[k]
            if (i === undefined) return logCavity(acc)
            const mean = M.gaussian.moments(beliefs[i] as M.Eta)[0] as number
            const variance = (M.gaussian.moments(beliefs[i] as M.Eta)[1] as number) - mean * mean
            return gaussianExpectation(mean, variance, (x) => walk(k + 1, acc + (coeffs[i] as number) * x), 24)
          }
          return walk(0, offset + (coeffs[j] as number) * a)
        }
        const residual = (a: number): number => expected(a) - ((m[0] as number) * a - 0.5 * (m[1] as number) * a * a)
        const base = residual(0)
        for (const a of [-2, -0.5, 0.9, 3]) expect(residual(a)).toBeCloseTo(base, 8)
      }
      // The output's message is the pushforward of the beliefs.
      let mean = offset
      let variance = 0
      for (const [i, k] of coeffs.entries()) {
        const mo = M.gaussian.moments(beliefs[i] as M.Eta)
        mean += k * (mo[0] as number)
        variance += k * k * ((mo[1] as number) - (mo[0] as number) ** 2)
      }
      const toY = op.vmp?.(ctx(0, args, params)).message as M.Eta
      expect((toY[0] as number) / (toY[1] as number)).toBeCloseTo(mean, 10)
      expect(1 / (toY[1] as number)).toBeCloseTo(variance, 10)
      expect(op.expectedLog?.(ctx(0, args, params))).toBe(0)
      expect(op.defines).toBe(0)
    })
  }

  it('dot-constant: the message to w is the rank-one cavity of s, to s the pushforward of the belief of w', () => {
    const a = [1, -2]
    const w = M.gaussianMv(2)
    const belief = M.gaussianMvEta([0.5, 1], [2, 0.3, 0.3, 1])
    const cavity: M.Eta = [0.4, 1.7]
    const args = [msg(cavity), M.messageArg(w, belief)]
    const toW = M.dotConstant.vmp?.(ctx(1, args, { a })).message as M.Eta
    expect(toW).toEqual([0.4 * 1, 0.4 * -2, 1.7, -3.4, -3.4, 6.8])
    // The same by the definition: E log cavity(a . w) has no other random input, so the message is the function itself.
    const toS = M.dotConstant.vmp?.(ctx(0, args, { a })).message as M.Eta
    expect((toS[0] as number) / (toS[1] as number)).toBeCloseTo(0.5 - 2, 12)
    // Var(a . w) = a' Sigma a
    const cov = M.gaussianMvMoments(belief, 2).covariance
    const v = 1 * (cov[0] as number) + 2 * (1 * -2) * (cov[1] as number) + 4 * (cov[3] as number)
    expect(1 / (toS[1] as number)).toBeCloseTo(v, 12)
  })
})

describe('the probit link under the Albert-Chib augmentation', () => {
  const sigmas = [1, 0.6]
  for (const sigma of sigmas) {
    for (const y of [1, 0]) {
      it(`noise ${sigma}, outcome ${y}: the message and the bound equal the integrals of the augmented model`, () => {
        const m = 0.45
        const v = 1.3
        const belief = M.gaussianEta(m, v)
        const args = [M.valueArg(M.bernoulli, y), msg(belief)]
        const message = M.probit.vmp?.(ctx(1, args, { noise: sigma })).message as M.Eta
        const sign = y === 1 ? 1 : -1
        // q(z) is N(m, sigma^2) cut to the side of the outcome; the message is the expected log N(z; x, sigma^2) over it.
        const lo = sign === 1 ? 0 : -40
        const hi = sign === 1 ? 40 : 0
        const dens = (z: number): number => Math.exp(-0.5 * ((z - m) / sigma) ** 2)
        const mass = integrate(dens, lo, hi, 24, 16)
        const ez = integrate((z) => z * dens(z), lo, hi, 24, 16) / mass
        expect(message[0]).toBeCloseTo(ez / sigma ** 2, 9)
        expect(message[1]).toBeCloseTo(1 / sigma ** 2, 12)
        // The bound: log of the integral over the side of exp(E_q(x) log N(z; x, sigma^2)), by two nested quadratures.
        const inner = (z: number): number =>
          gaussianExpectation(m, v, (x) => -0.5 * Math.log(2 * Math.PI * sigma ** 2) - 0.5 * ((z - x) / sigma) ** 2, 48)
        const bound = Math.log(integrate((z) => Math.exp(inner(z)), lo, hi, 24, 16))
        expect(M.probit.expectedLog?.(ctx(1, args, { noise: sigma }))).toBeCloseTo(bound, 8)
      })
    }
  }
  it('has no rule for a latent outcome, and greater-than is refused with its reason', () => {
    const latent = [M.messageArg(M.bernoulli, M.bernoulliEta(0.3)), msg(M.gaussianEta(0, 1))]
    expect(() => M.probit.vmp?.(ctx(1, latent))).toThrow(/observed outcome/)
    expect(M.greaterThan.vmp).toBeUndefined()
    expect(M.greaterThan.noRule?.vmp).toMatch(/infinite/)
  })
})

describe('variational message passing through deterministic factors', () => {
  /** a ~ N(1, 2), b ~ N(-1, 1), s = a + b, 2 ~ N(s, 1/4). */
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'a', family: { kind: 'gaussian' } },
      { name: 'b', family: { kind: 'gaussian' } },
      { name: 's', family: { kind: 'gaussian' } },
      { name: 'obs', family: { kind: 'gaussian' }, observed: 2 },
    ],
    factors: [
      M.gaussianPrior('a', 1, 2),
      M.gaussianPrior('b', -1, 1),
      { operator: 'plus', args: ['s', 'a', 'b'] },
      { operator: 'gaussian', args: ['obs', 's'], params: { precision: 4 } },
    ],
  }
  const g = M.compileGraph(spec, operators)

  it('reaches the closed-form coordinate updates of the mean field over (a, b)', () => {
    // The mean field of the model with s substituted: q(a) q(b), each updated against the mean of the other.
    const pa = 1 / 2
    const pb = 1
    const tau = 4
    let ma = 0
    let mb = 0
    for (let k = 0; k < 500; k++) {
      ma = (pa * 1 + tau * (2 - mb)) / (pa + tau)
      mb = (pb * -1 + tau * (2 - ma)) / (pb + tau)
    }
    const { state } = sweeps(g, 200)
    const a = stat(g, state, 'a')
    const b = stat(g, state, 'b')
    expect(a.mean).toBeCloseTo(ma, 9)
    expect(b.mean).toBeCloseTo(mb, 9)
    expect(a.variance).toBeCloseTo(1 / (pa + tau), 12)
    expect(b.variance).toBeCloseTo(1 / (pb + tau), 12)
    // The defined variable's belief is the pushforward of q(a) q(b) and nothing the observation said to it.
    const s = stat(g, state, 's')
    expect(s.mean).toBeCloseTo(ma + mb, 9)
    expect(s.variance).toBeCloseTo(a.variance + b.variance, 12)
  })

  it('has the exact posterior means (the mean field of a Gaussian is exact in the mean) and a smaller variance', () => {
    const { state } = sweeps(g, 400)
    const exact = M.epAlgorithm(g, { tol: 1e-12 })
    const run = (() => {
      let st = exact.init(undefined as never)
      for (let k = 0; k < 20; k++) st = exact.step(st, undefined as never)
      return st
    })()
    for (const name of ['a', 'b']) {
      const v = stat(g, state, name)
      const e = stat(g, run, name)
      expect(v.mean).toBeCloseTo(e.mean, 7)
      expect(v.variance).toBeLessThan(e.variance)
    }
  })

  it('raises the bound at every sweep, to the bound of the substituted model', () => {
    const { bounds, state } = sweeps(g, 30)
    for (let k = 1; k < bounds.length; k++)
      expect(bounds[k] as number).toBeGreaterThanOrEqual((bounds[k - 1] as number) - 1e-10)
    expect(bounds[bounds.length - 1] as number).toBeGreaterThan(bounds[0] as number)
    // Evaluated directly: E log p(a) + E log p(b) + E log N(2; a + b, 1/4) + H(q(a)) + H(q(b)).
    const a = stat(g, state, 'a')
    const b = stat(g, state, 'b')
    const expectedLogNormal = (mean: number, mv: number, m: number, v: number): number =>
      -0.5 * Math.log(2 * Math.PI * v) - (0.5 * (mv + (mean - m) ** 2)) / v
    const direct =
      expectedLogNormal(a.mean, a.variance, 1, 2) +
      expectedLogNormal(b.mean, b.variance, -1, 1) +
      expectedLogNormal(2, 0, a.mean + b.mean, 1 / 4) -
      (0.5 * (a.variance + b.variance)) / (1 / 4) +
      0.5 * Math.log(2 * Math.PI * Math.E * a.variance) +
      0.5 * Math.log(2 * Math.PI * Math.E * b.variance)
    expect(bounds[bounds.length - 1]).toBeCloseTo(direct, 8)
  })

  it('does not change expectation propagation, which still treats the sum as an exact factor', () => {
    const ep = M.epAlgorithm(g, { tol: 1e-12 })
    let st = ep.init(undefined as never)
    for (let k = 0; k < 10; k++) st = ep.step(st, undefined as never)
    // The exact posterior of (a, b) given 2 = a + b + noise: precision matrix [[1/2 + 4, 4], [4, 1 + 4]].
    const det = 4.5 * 5 - 16
    expect(stat(g, st, 'a').variance).toBeCloseTo(5 / det, 9)
    expect(stat(g, st, 's').mean).toBeCloseTo(stat(g, st, 'a').mean + stat(g, st, 'b').mean, 9)
  })

  it('a chain of definitions is the mean field of the substituted combination', () => {
    const chain: M.FactorGraphSpec = {
      variables: [
        { name: 'a', family: { kind: 'gaussian' } },
        { name: 'b', family: { kind: 'gaussian' } },
        { name: 'c', family: { kind: 'gaussian' } },
        { name: 's', family: { kind: 'gaussian' } },
        { name: 't', family: { kind: 'gaussian' } },
        { name: 'obs', family: { kind: 'gaussian' }, observed: 1.5 },
      ],
      factors: [
        M.gaussianPrior('a', 0.5, 2),
        M.gaussianPrior('b', -1, 1),
        M.gaussianPrior('c', 1, 3),
        { operator: 'plus', args: ['s', 'a', 'b'] },
        { operator: 'linear', args: ['t', 's', 'c'], params: { coeffs: [2, -1], offset: 0.5 } },
        { operator: 'gaussian', args: ['obs', 't'], params: { precision: 3 } },
      ],
    }
    const direct: M.FactorGraphSpec = {
      variables: [
        { name: 'a', family: { kind: 'gaussian' } },
        { name: 'b', family: { kind: 'gaussian' } },
        { name: 'c', family: { kind: 'gaussian' } },
        { name: 't', family: { kind: 'gaussian' } },
        { name: 'obs', family: { kind: 'gaussian' }, observed: 1.5 },
      ],
      factors: [
        M.gaussianPrior('a', 0.5, 2),
        M.gaussianPrior('b', -1, 1),
        M.gaussianPrior('c', 1, 3),
        { operator: 'linear', args: ['t', 'a', 'b', 'c'], params: { coeffs: [2, 2, -1], offset: 0.5 } },
        { operator: 'gaussian', args: ['obs', 't'], params: { precision: 3 } },
      ],
    }
    const g1 = M.compileGraph(chain, operators)
    const g2 = M.compileGraph(direct, operators)
    const r1 = sweeps(g1, 1500)
    const r2 = sweeps(g2, 1500)
    for (const name of ['a', 'b', 'c', 't']) {
      expect(stat(g1, r1.state, name).mean).toBeCloseTo(stat(g2, r2.state, name).mean, 9)
      expect(stat(g1, r1.state, name).variance).toBeCloseTo(stat(g2, r2.state, name).variance, 11)
    }
    expect(r1.bounds[r1.bounds.length - 1]).toBeCloseTo(r2.bounds[r2.bounds.length - 1] as number, 8)
    for (let k = 1; k < r1.bounds.length; k++)
      expect(r1.bounds[k] as number).toBeGreaterThanOrEqual((r1.bounds[k - 1] as number) - 1e-10)
  })
})

describe('Bayesian linear regression on scalar weights', () => {
  const xs = [-1.2, -0.4, 0.3, 0.9, 1.6, 2.1]
  const ys = [-0.9, -0.2, 0.6, 0.8, 2.1, 2.4]
  const tau = 5
  const prior = { a: { m: 0, v: 4 }, b: { m: 0, v: 4 } }
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'a', family: { kind: 'gaussian' } },
      { name: 'b', family: { kind: 'gaussian' } },
      ...xs.flatMap((_, i) => [
        { name: `mu${i}`, family: { kind: 'gaussian' as const } },
        { name: `y${i}`, family: { kind: 'gaussian' as const }, observed: ys[i] as number },
      ]),
    ],
    factors: [
      M.gaussianPrior('a', prior.a.m, prior.a.v),
      M.gaussianPrior('b', prior.b.m, prior.b.v),
      ...xs.flatMap((x, i) => [
        { operator: 'linear', args: [`mu${i}`, 'a', 'b'], params: { coeffs: [1, x], offset: 0 } },
        { operator: 'gaussian', args: [`y${i}`, `mu${i}`], params: { precision: tau } },
      ]),
    ],
  }
  const g = M.compileGraph(spec, operators)

  /** The closed-form coordinate ascent of the mean field q(a) q(b) of y_i ~ N(a + b x_i, 1/tau). */
  function closedForm(): { ma: number; mb: number; va: number; vb: number } {
    const n = xs.length
    const sx = xs.reduce((s, x) => s + x, 0)
    const sxx = xs.reduce((s, x) => s + x * x, 0)
    const va = 1 / (1 / prior.a.v + tau * n)
    const vb = 1 / (1 / prior.b.v + tau * sxx)
    let ma = 0
    let mb = 0
    for (let k = 0; k < 1000; k++) {
      ma = va * (prior.a.m / prior.a.v + tau * (ys.reduce((s, y) => s + y, 0) - mb * sx))
      mb = vb * (prior.b.m / prior.b.v + tau * (xs.reduce((s, x, i) => s + x * (ys[i] as number), 0) - ma * sx))
    }
    return { ma, mb, va, vb }
  }

  it('the fixed point is the mean field of the coordinate updates, the bound rises, the means are exact', () => {
    const { state, bounds } = sweeps(g, 120)
    const want = closedForm()
    const a = stat(g, state, 'a')
    const b = stat(g, state, 'b')
    expect(a.mean).toBeCloseTo(want.ma, 8)
    expect(b.mean).toBeCloseTo(want.mb, 8)
    expect(a.variance).toBeCloseTo(want.va, 12)
    expect(b.variance).toBeCloseTo(want.vb, 12)
    for (let k = 1; k < bounds.length; k++)
      expect(bounds[k] as number).toBeGreaterThanOrEqual((bounds[k - 1] as number) - 1e-9)
    // The exact posterior (a Gaussian with this precision) has the same mean.
    const n = xs.length
    const sx = xs.reduce((s, x) => s + x, 0)
    const sxx = xs.reduce((s, x) => s + x * x, 0)
    const P = [
      [1 / prior.a.v + tau * n, tau * sx],
      [tau * sx, 1 / prior.b.v + tau * sxx],
    ] as const
    const h = [tau * ys.reduce((s, y) => s + y, 0), tau * xs.reduce((s, x, i) => s + x * (ys[i] as number), 0)]
    const det = P[0][0] * P[1][1] - P[0][1] * P[1][0]
    const exactA = (P[1][1] * (h[0] as number) - P[0][1] * (h[1] as number)) / det
    const exactB = (P[0][0] * (h[1] as number) - P[1][0] * (h[0] as number)) / det
    expect(a.mean).toBeCloseTo(exactA, 6)
    expect(b.mean).toBeCloseTo(exactB, 6)
    expect(a.variance).toBeLessThan(P[1][1] / det)
  })

  it('the per-row means are the pushforward and the schedule order does not change the fixed point', () => {
    const { state } = sweeps(g, 120)
    const a = stat(g, state, 'a')
    const b = stat(g, state, 'b')
    for (const [i, x] of xs.entries()) {
      const mu = stat(g, state, `mu${i}`)
      expect(mu.mean).toBeCloseTo(a.mean + x * b.mean, 8)
      expect(mu.variance).toBeCloseTo(a.variance + x * x * b.variance, 10)
    }
    const flooding = M.vmpAlgorithm(g, { schedule: 'flooding' })
    let st = flooding.init(undefined as never)
    for (let k = 0; k < 400; k++) st = flooding.step(st, undefined as never)
    expect(stat(g, st, 'a').mean).toBeCloseTo(a.mean, 5)
  })
})

describe('probit regression by variational message passing', () => {
  it('raises the bound and is the mean field of the augmented model', () => {
    const xs = [-1.5, -0.8, -0.2, 0.1, 0.7, 1.3, 1.9, -1.1]
    const ys = [0, 0, 1, 0, 1, 1, 1, 0]
    const prior = 4
    const spec: M.FactorGraphSpec = {
      variables: [
        { name: 'a', family: { kind: 'gaussian' } },
        { name: 'b', family: { kind: 'gaussian' } },
        ...xs.flatMap((_, i) => [
          { name: `z${i}`, family: { kind: 'gaussian' as const } },
          { name: `y${i}`, family: { kind: 'bernoulli' as const }, observed: ys[i] as number },
        ]),
      ],
      factors: [
        M.gaussianPrior('a', 0, prior),
        M.gaussianPrior('b', 0, prior),
        ...xs.flatMap((x, i) => [
          { operator: 'linear', args: [`z${i}`, 'a', 'b'], params: { coeffs: [1, x], offset: 0 } },
          { operator: 'probit', args: [`y${i}`, `z${i}`] },
        ]),
      ],
    }
    const g = M.compileGraph(spec, operators)
    const { state, bounds } = sweeps(g, 80)
    for (let k = 1; k < bounds.length; k++)
      expect(bounds[k] as number).toBeGreaterThanOrEqual((bounds[k - 1] as number) - 1e-9)
    // The same mean field written out by hand: q(z_i) the outcome's side of N(E a + x_i E b, 1), q(a) and q(b) Gaussian
    // with the closed-form updates (Albert and Chib 1993; the variational form of Girolami and Rogers 2006).
    const phiDensity = (z: number): number => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
    const va = 1 / (1 / prior + xs.length)
    const vb = 1 / (1 / prior + xs.reduce((t, x) => t + x * x, 0))
    let ma = 0
    let mb = 0
    for (let k = 0; k < 500; k++) {
      const ez = xs.map((x, i) => {
        const m = ma + x * mb
        const sign = ys[i] === 1 ? 1 : -1
        return m + (sign * phiDensity(m)) / Math.exp(logPhi(sign * m))
      })
      ma = va * ez.reduce((t, z, i) => t + (z - (xs[i] as number) * mb), 0)
      mb = vb * ez.reduce((t, z, i) => t + (xs[i] as number) * (z - ma), 0)
    }
    expect(stat(g, state, 'a').mean).toBeCloseTo(ma, 6)
    expect(stat(g, state, 'b').mean).toBeCloseTo(mb, 6)
    expect(stat(g, state, 'a').variance).toBeCloseTo(va, 12)
    // The exact posterior by a product grid over (a, b): the mean field shrinks the slope towards zero, as it does for probit.
    const logPost = ([a, b]: readonly number[]): number => {
      let t = (-0.5 * ((a as number) ** 2 + (b as number) ** 2)) / prior
      for (const [i, x] of xs.entries()) {
        const z = (a as number) + x * (b as number)
        t += logPhi(ys[i] === 1 ? z : -z)
      }
      return t
    }
    const box: [number, number][] = [
      [-6, 6],
      [-6, 6],
    ]
    const Z = integrateBox((p) => Math.exp(logPost(p)), box, 12, 14)
    const exactB = integrateBox((p) => (p[1] as number) * Math.exp(logPost(p)), box, 12, 14) / Z
    expect(stat(g, state, 'b').mean).toBeGreaterThan(0)
    expect(stat(g, state, 'b').mean).toBeLessThan(exactB)
    expect(stat(g, state, 'b').mean).toBeGreaterThan(0.6 * exactB)
    // The bound is below the log evidence (the grid's mass is the evidence without the prior's constant).
    expect(bounds[bounds.length - 1] as number).toBeLessThan(Math.log(Z) - Math.log(2 * Math.PI * prior))
  })
})

describe('a copy and the evidence of a state with nothing in it', () => {
  it('equal defines a variable as the copy of another: no double counting under the mean field', () => {
    const copy: M.FactorGraphSpec = {
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'c', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'gaussian' }, observed: 3 },
      ],
      factors: [
        M.gaussianPrior('x', 0, 4),
        { operator: 'equal', args: ['c', 'x'] },
        { operator: 'gaussian', args: ['y', 'c'], params: { precision: 1 } },
      ],
    }
    const g = M.compileGraph(copy, operators)
    const { state } = sweeps(g, 50)
    // Posterior of x given y = 3: precision 1/4 + 1, mean 3 / (1/4 + 1).
    expect(stat(g, state, 'x').mean).toBeCloseTo(3 / 1.25, 10)
    expect(stat(g, state, 'c').mean).toBeCloseTo(3 / 1.25, 10)
    expect(stat(g, state, 'c').variance).toBeCloseTo(1 / 1.25, 10)
  })
})
