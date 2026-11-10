import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  type BrutePort,
  bruteTilted,
  integrate,
  integrateBox,
  integratePositive,
} from '../../testing/message-passing.ts'
import { sigmoidOf } from '../base/index.ts'
import * as M from './index.ts'

const runs = { numRuns: 12 }
const mean = fc.double({ min: -2, max: 2, noNaN: true })
const variance = fc.double({ min: 0.3, max: 3, noNaN: true })
const gaussianArg = fc.tuple(mean, variance).map(([m, v]) => M.gaussianEta(m, v))

const msg = (fam: M.Family, eta: M.Eta): M.Arg => M.messageArg(fam, eta)
const ctx = (to: number, args: M.Arg[], params: M.Params = {}, mode: M.Mode = 'ep'): M.OperatorContext => ({
  mode,
  to,
  args,
  params,
  power: 1,
})

/** The moments of the cavity times the message: what the projection of the tilted distribution must match. */
function implied(fam: M.Family, cavity: M.Eta, message: M.Eta): number[] {
  return [...fam.moments(fam.product(cavity, message))]
}

const near = (got: readonly number[], want: readonly number[], tol = 1e-6): void => {
  expect(got.length).toBe(want.length)
  for (const [i, v] of got.entries())
    expect(Math.abs(v - (want[i] as number)), `entry ${i}: ${v} vs ${want[i]}`).toBeLessThan(
      tol * (1 + Math.abs(want[i] as number)),
    )
}

const boxOf = (mw: ArrayLike<number>, cw: ArrayLike<number>): [number, number][] =>
  [0, 3].map(
    (k, i) => [(mw[i] ?? 0) - 8 * Math.sqrt(cw[k] ?? 1), (mw[i] ?? 0) + 8 * Math.sqrt(cw[k] ?? 1)] as [number, number],
  )

const gp = (eta: M.Eta): BrutePort => ({ kind: 'gaussian', eta })
const gaussian = M.gaussian

describe('Gaussian operators equal the integral of the exact message', () => {
  it('gaussian (fixed precision): EP to x and to the mean, evidence', () => {
    fc.assert(
      fc.property(gaussianArg, gaussianArg, fc.double({ min: 0.5, max: 4, noNaN: true }), (ex, em, tau) => {
        const params = { precision: tau }
        const logf = ([x, m]: readonly number[]): number =>
          0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * ((x ?? 0) - (m ?? 0)) ** 2
        for (const to of [0, 1]) {
          const args = [msg(gaussian, ex), msg(gaussian, em)]
          const m = M.gaussianOperator.ep?.(ctx(to, args, params)).message as M.Eta
          const brute = bruteTilted([gp(ex), gp(em)], logf, to)
          near(implied(gaussian, to === 0 ? ex : em, m), brute.moments)
        }
        const brute = bruteTilted([gp(ex), gp(em)], logf, 0)
        expect(M.gaussianOperator.logEvidence?.(ctx(0, [msg(gaussian, ex), msg(gaussian, em)], params))).toBeCloseTo(
          brute.logZ,
          6,
        )
      }),
      runs,
    )
  })
  it('gaussian with an observed argument is the likelihood of the mean', () => {
    const em = M.gaussianEta(0.3, 1.2)
    const m = M.gaussianOperator.ep?.(ctx(1, [M.valueArg(gaussian, 1.5), msg(gaussian, em)], { precision: 2 }))
      .message as M.Eta
    near(m, M.gaussianEta(1.5, 0.5))
  })
  it('linear, plus, difference, times-constant and sum: EP to every port', () => {
    fc.assert(
      fc.property(gaussianArg, gaussianArg, gaussianArg, (ey, ea, eb) => {
        const cases: { op: M.FactorOperator; params: M.Params; coeffs: number[]; offset: number; etas: M.Eta[] }[] = [
          { op: M.plus, params: {}, coeffs: [1, 1], offset: 0, etas: [ey, ea, eb] },
          { op: M.difference, params: {}, coeffs: [1, -1], offset: 0, etas: [ey, ea, eb] },
          {
            op: M.linear,
            params: { coeffs: [2, -0.5], offset: 1.5 },
            coeffs: [2, -0.5],
            offset: 1.5,
            etas: [ey, ea, eb],
          },
          { op: M.timesConstant, params: { factor: -3 }, coeffs: [-3], offset: 0, etas: [ey, ea] },
          { op: M.sum, params: { n: 2 }, coeffs: [1, 1], offset: 0, etas: [ey, ea, eb] },
        ]
        for (const { op, params, coeffs, offset, etas } of cases) {
          const args = etas.map((e) => msg(gaussian, e))
          for (let to = 0; to < etas.length; to++) {
            const out = to === 0 ? 0 : -1
            // The output is the free port when it is the target; otherwise one input is derived from the output.
            const solveFor = to === 0 ? -1 : to
            const ports: BrutePort[] = etas.map((e, i) => {
              if (i === 0 && solveFor < 0)
                return {
                  kind: 'gaussian',
                  eta: e,
                  derive: (x: readonly number[]) => offset + coeffs.reduce((s, k, j) => s + k * (x[j + 1] ?? 0), 0),
                }
              if (i === solveFor) {
                return {
                  kind: 'gaussian',
                  eta: e,
                  derive: (x: readonly number[]) =>
                    ((x[0] ?? 0) -
                      offset -
                      coeffs.reduce((s, k, j) => (j + 1 === i ? s : s + k * (x[j + 1] ?? 0)), 0)) /
                    (coeffs[i - 1] ?? 1),
                }
              }
              return { kind: 'gaussian', eta: e }
            })
            // The derived input has a Jacobian 1/|c| in the delta; it is a constant that moment matching ignores.
            const brute = bruteTilted(ports, () => 0, to === 0 ? 0 : to, { panels: 14, order: 14 })
            const message = op.ep?.(ctx(to, args, params)).message as M.Eta
            near(implied(gaussian, etas[to] as M.Eta, message), brute.moments, 3e-4)
            expect(out).toBeLessThanOrEqual(0)
          }
        }
      }),
      { numRuns: 4 },
    )
  })
  it('product: EP to the output and to each factor', () => {
    fc.assert(
      fc.property(gaussianArg, gaussianArg, gaussianArg, (ey, ea, eb) => {
        const args = [msg(gaussian, ey), msg(gaussian, ea), msg(gaussian, eb)]
        const free: BrutePort[] = [
          { kind: 'gaussian', eta: ey, derive: (x) => (x[1] ?? 0) * (x[2] ?? 0) },
          gp(ea),
          gp(eb),
        ]
        near(
          implied(gaussian, ey, M.product.ep?.(ctx(0, args)).message as M.Eta),
          bruteTilted(free, () => 0, 0, { panels: 14, order: 14 }).moments,
          3e-4,
        )
        // To a: b is integrated out; y = a b is the derived port of a free (a, b) pair.
        near(
          implied(gaussian, ea, M.product.ep?.(ctx(1, args)).message as M.Eta),
          bruteTilted(free, () => 0, 1, { panels: 14, order: 14 }).moments,
          1e-4,
        )
        near(
          implied(gaussian, eb, M.product.ep?.(ctx(2, args)).message as M.Eta),
          bruteTilted(free, () => 0, 2, { panels: 14, order: 14 }).moments,
          1e-4,
        )
      }),
      { numRuns: 4 },
    )
  })
  it('product with nothing known about y is the matched pushforward', () => {
    const ea = M.gaussianEta(2, 1)
    const eb = M.gaussianEta(3, 4)
    const m = M.product.ep?.(ctx(0, [msg(gaussian, gaussian.flat), msg(gaussian, ea), msg(gaussian, eb)]))
      .message as M.Eta
    const s = gaussian.summary(m)
    expect(s['mean']).toBeCloseTo(6, 12)
    expect(s['variance']).toBeCloseTo(1 * 4 + 1 * 9 + 4 * 4, 12)
  })
})

describe('vector Gaussian operators', () => {
  const w2 = M.gaussianMv(2)
  const cavityW = M.gaussianMvEta([0.4, -0.3], [1.5, 0.4, 0.4, 0.8])
  it('dot-constant: EP to s and to w equal the integrals over w', () => {
    const a = [1, -2]
    const es = M.gaussianEta(0.5, 1.4)
    const { mean: mw, covariance: cw } = M.gaussianMvMoments(cavityW, 2)
    const box = boxOf(mw, cw)
    const logw = (w: readonly number[]): number =>
      (cavityW[0] ?? 0) * (w[0] ?? 0) +
      (cavityW[1] ?? 0) * (w[1] ?? 0) -
      0.5 *
        ((cavityW[2] ?? 0) * (w[0] ?? 0) ** 2 +
          2 * (cavityW[3] ?? 0) * (w[0] ?? 0) * (w[1] ?? 0) +
          (cavityW[5] ?? 0) * (w[1] ?? 0) ** 2)
    const s = (w: readonly number[]): number => (a[0] ?? 0) * (w[0] ?? 0) + (a[1] ?? 0) * (w[1] ?? 0)
    // To s: the tilted distribution of s = a . w with the cavity of s as weight.
    const weight = (w: readonly number[]): number =>
      Math.exp(logw(w) + (es[0] ?? 0) * s(w) - 0.5 * (es[1] ?? 0) * s(w) ** 2)
    const z = integrateBox(weight, box, 10, 12)
    const m1 = integrateBox((w) => weight(w) * s(w), box, 10, 12) / z
    const m2 = integrateBox((w) => weight(w) * s(w) ** 2, box, 10, 12) / z
    const toS = M.dotConstant.ep?.(ctx(0, [msg(gaussian, es), msg(w2, cavityW)], { a })).message as M.Eta
    near(implied(gaussian, es, toS), [m1, m2], 1e-6)
    // To w: the message is [h_s a, J_s a a'].
    const toW = M.dotConstant.ep?.(ctx(1, [msg(gaussian, es), msg(w2, cavityW)], { a })).message as M.Eta
    near(toW, [
      (es[0] ?? 0) * 1,
      (es[0] ?? 0) * -2,
      (es[1] ?? 0) * 1,
      (es[1] ?? 0) * -2,
      (es[1] ?? 0) * -2,
      (es[1] ?? 0) * 4,
    ])
    expect(M.dotConstant.logEvidence?.(ctx(0, [msg(gaussian, es), msg(w2, cavityW)], { a }))).toBeCloseTo(
      Math.log(z),
      5,
    )
  })
  it('matmul-constant: the message to y is the pushforward, the message to w is natural', () => {
    const A = [1, 2, -1, 0.5]
    const y2 = M.gaussianMv(2)
    const ey = M.gaussianMvEta([0.2, 0.1], [2, 0.3, 0.3, 1.5])
    const toY = M.matmulConstant.ep?.(ctx(0, [msg(y2, y2.flat), msg(w2, cavityW)], { matrix: A })).message as M.Eta
    const { mean, covariance } = M.gaussianMvMoments(cavityW, 2)
    expect(y2.summary(toY)['mean']).toEqual(
      [
        (A[0] ?? 0) * (mean[0] ?? 0) + (A[1] ?? 0) * (mean[1] ?? 0),
        (A[2] ?? 0) * (mean[0] ?? 0) + (A[3] ?? 0) * (mean[1] ?? 0),
      ].map((v) => expect.closeTo(v, 10)),
    )
    const toW = M.matmulConstant.ep?.(ctx(1, [msg(y2, ey), msg(w2, cavityW)], { matrix: A })).message as M.Eta
    // A' J A of the message.
    const j = (r: number, c: number): number => ey[2 + r * 2 + c] ?? 0
    let a00 = 0
    for (let r = 0; r < 2; r++) for (let q = 0; q < 2; q++) a00 += (A[r * 2] ?? 0) * j(r, q) * (A[q * 2] ?? 0)
    expect(toW[2]).toBeCloseTo(a00, 12)
    expect(covariance.length).toBe(4)
  })
  it('linear-regression-row: EP to w with an observed and a Gaussian y, and to y', () => {
    const a = [0.7, -1.1]
    const tau = 2.5
    const { mean: mw, covariance: cw } = M.gaussianMvMoments(cavityW, 2)
    const box = boxOf(mw, cw)
    const logw = (w: readonly number[]): number =>
      (cavityW[0] ?? 0) * (w[0] ?? 0) +
      (cavityW[1] ?? 0) * (w[1] ?? 0) -
      0.5 *
        ((cavityW[2] ?? 0) * (w[0] ?? 0) ** 2 +
          2 * (cavityW[3] ?? 0) * (w[0] ?? 0) * (w[1] ?? 0) +
          (cavityW[5] ?? 0) * (w[1] ?? 0) ** 2)
    const yObs = 0.9
    const like = (w: readonly number[]): number =>
      Math.exp(logw(w) - 0.5 * tau * (yObs - (a[0] ?? 0) * (w[0] ?? 0) - (a[1] ?? 0) * (w[1] ?? 0)) ** 2)
    const z = integrateBox(like, box, 10, 12)
    const m = [0, 1].map((k) => integrateBox((w) => like(w) * (w[k] ?? 0), box, 10, 12) / z)
    const toW = M.linearRegressionRow.ep?.(
      ctx(1, [M.valueArg(gaussian, yObs), msg(w2, cavityW)], { a, precision: tau }),
    ).message as M.Eta
    near(w2.summary(w2.product(cavityW, toW))['mean'] as number[], m, 1e-6)
    // With a Gaussian y of finite variance the effective noise grows by that variance.
    const ey = M.gaussianEta(yObs, 0.5)
    const toWy = M.linearRegressionRow.ep?.(ctx(1, [msg(gaussian, ey), msg(w2, cavityW)], { a, precision: tau }))
      .message as M.Eta
    const effective = 1 / (0.5 + 1 / tau)
    near(toWy.slice(2), [
      effective * (a[0] as number) * (a[0] as number),
      effective * (a[0] as number) * (a[1] as number),
      effective * (a[1] as number) * (a[0] as number),
      effective * (a[1] as number) * (a[1] as number),
    ])
    // To y: N(a . m, a' S a + 1 / tau).
    const toY = M.linearRegressionRow.ep?.(ctx(0, [M.valueArg(gaussian, 0), msg(w2, cavityW)], { a, precision: tau }))
      .message as M.Eta
    const mean0 = (a[0] as number) * (mw[0] as number) + (a[1] as number) * (mw[1] as number)
    expect(gaussian.summary(toY)['mean']).toBeCloseTo(mean0, 10)
    expect(
      M.linearRegressionRow.logEvidence?.(
        ctx(1, [M.valueArg(gaussian, yObs), msg(w2, cavityW)], { a, precision: tau }),
      ),
    ).toBeCloseTo(Math.log(z) + 0.5 * Math.log(tau / (2 * Math.PI)), 5)
  })
  it('gaussian-joint: the Schur complement of the potential, and the evidence', () => {
    fc.assert(
      fc.property(gaussianArg, gaussianArg, (e0, e1) => {
        const J = [2, -0.8, -0.8, 1.5]
        const h = [0.3, -0.2]
        const logf = (x: readonly number[]): number =>
          (h[0] ?? 0) * (x[0] ?? 0) +
          (h[1] ?? 0) * (x[1] ?? 0) -
          0.5 *
            ((J[0] ?? 0) * (x[0] ?? 0) ** 2 +
              2 * (J[1] ?? 0) * (x[0] ?? 0) * (x[1] ?? 0) +
              (J[3] ?? 0) * (x[1] ?? 0) ** 2)
        const args = [msg(gaussian, e0), msg(gaussian, e1)]
        for (const to of [0, 1]) {
          const message = M.gaussianJoint.ep?.(ctx(to, args, { J, h })).message as M.Eta
          const brute = bruteTilted([gp(e0), gp(e1)], logf, to)
          near(implied(gaussian, to === 0 ? e0 : e1, message), brute.moments, 1e-6)
        }
        expect(M.gaussianJoint.logEvidence?.(ctx(0, args, { J, h }))).toBeCloseTo(
          bruteTilted([gp(e0), gp(e1)], logf, 0).logZ,
          6,
        )
      }),
      runs,
    )
  })
})

describe('truncation operators', () => {
  const bern = M.bernoulli
  it('probit: observed outcomes, an uncertain outcome, the message to b, and the evidence', () => {
    fc.assert(
      fc.property(
        gaussianArg,
        fc.double({ min: 0.05, max: 0.95, noNaN: true }),
        fc.double({ min: 0.5, max: 2, noNaN: true }),
        (ex, pi, noise) => {
          const params = { noise }
          const norm = (z: number): number =>
            Math.exp(M.truncatedNormalMoments(0, 1, Number.NEGATIVE_INFINITY, z).logMass)
          const logf = ([b, x]: readonly number[]): number =>
            Math.log((b ?? 0) > 0.5 ? norm((x ?? 0) / noise) : norm(-(x ?? 0) / noise))
          for (const label of [1, 0]) {
            const m = M.probit.ep?.(ctx(1, [M.valueArg(bern, label), msg(gaussian, ex)], params)).message as M.Eta
            const brute = bruteTilted([{ kind: 'bernoulli', value: label }, gp(ex)], logf, 1)
            near(implied(gaussian, ex, m), brute.moments, 1e-6)
            expect(M.probit.logEvidence?.(ctx(1, [M.valueArg(bern, label), msg(gaussian, ex)], params))).toBeCloseTo(
              bruteTilted([{ kind: 'bernoulli', value: label }, gp(ex)], logf, 1).logZ,
              6,
            )
          }
          const eb = M.bernoulliEta(pi)
          const m = M.probit.ep?.(ctx(1, [msg(bern, eb), msg(gaussian, ex)], params)).message as M.Eta
          near(implied(gaussian, ex, m), bruteTilted([{ kind: 'bernoulli', eta: eb }, gp(ex)], logf, 1).moments, 1e-6)
          const toB = M.probit.ep?.(ctx(0, [msg(bern, eb), msg(gaussian, ex)], params)).message as M.Eta
          near(
            bern.moments(bern.product(eb, toB)),
            bruteTilted([{ kind: 'bernoulli', eta: eb }, gp(ex)], logf, 0).moments,
            1e-6,
          )
          expect(M.probit.logEvidence?.(ctx(1, [msg(bern, eb), msg(gaussian, ex)], params))).toBeCloseTo(
            bruteTilted([{ kind: 'bernoulli', eta: eb }, gp(ex)], logf, 1).logZ,
            6,
          )
        },
      ),
      runs,
    )
  })
  it('greater-than and is-between: observed outcomes truncate, an uncertain outcome mixes the regions', () => {
    fc.assert(
      fc.property(gaussianArg, fc.double({ min: 0.05, max: 0.95, noNaN: true }), (ex, pi) => {
        for (const [op, params, inside] of [
          [M.greaterThan, { threshold: 0.4 }, (x: number) => x > 0.4],
          [M.isBetween, { lower: -0.7, upper: 0.9 }, (x: number) => x > -0.7 && x < 0.9],
        ] as const) {
          const logf = ([b, x]: readonly number[]): number =>
            (b ?? 0) > 0.5 === inside(x ?? 0) ? 0 : Number.NEGATIVE_INFINITY
          // The indicator is discontinuous: integrate on panels that end at the breakpoints.
          const mid = (b: BrutePort): { moments: number[]; logZ: number } => {
            const breaks = 'upper' in params ? [-14, -0.7, 0.9, 14] : [-14, 0.4, 14]
            let total = { z: 0, m1: 0, m2: 0 }
            for (let k = 0; k + 1 < breaks.length; k++) {
              const lo = breaks[k] as number
              const hi = breaks[k + 1] as number
              const r = bruteTilted(
                [b, { kind: 'gaussian', eta: ex, range: [lo, hi] }],
                ([bb, x]) => ((bb ?? 0) > 0.5 === inside(x ?? 0) ? 0 : Number.NEGATIVE_INFINITY),
                1,
              )
              const z = Math.exp(r.logZ)
              if (!(z > 0)) continue
              total = { z: total.z + z, m1: total.m1 + z * (r.moments[0] ?? 0), m2: total.m2 + z * (r.moments[1] ?? 0) }
            }
            return { logZ: Math.log(total.z), moments: [total.m1 / total.z, total.m2 / total.z] }
          }
          void logf
          for (const label of [1, 0]) {
            const m = op.ep?.(ctx(1, [M.valueArg(bern, label), msg(gaussian, ex)], params)).message as M.Eta
            const brute = mid({ kind: 'bernoulli', value: label })
            near(implied(gaussian, ex, m), brute.moments, 1e-6)
            expect(op.logEvidence?.(ctx(1, [M.valueArg(bern, label), msg(gaussian, ex)], params))).toBeCloseTo(
              brute.logZ,
              6,
            )
          }
          const eb = M.bernoulliEta(pi)
          const m = op.ep?.(ctx(1, [msg(bern, eb), msg(gaussian, ex)], params)).message as M.Eta
          const brute = mid({ kind: 'bernoulli', eta: eb })
          near(implied(gaussian, ex, m), brute.moments, 1e-6)
          expect(op.logEvidence?.(ctx(1, [msg(bern, eb), msg(gaussian, ex)], params))).toBeCloseTo(brute.logZ, 6)
          // To b: the mass inside against outside.
          const toB = op.ep?.(ctx(0, [msg(bern, eb), msg(gaussian, ex)], params)).message as M.Eta
          const [lo, hi] = 'upper' in params ? [-0.7, 0.9] : [0.4, 14]
          const inMass = integrate((x) => Math.exp((ex[0] ?? 0) * x - 0.5 * (ex[1] ?? 0) * x * x), lo, hi, 40, 12)
          const total = integrate((x) => Math.exp((ex[0] ?? 0) * x - 0.5 * (ex[1] ?? 0) * x * x), -14, 14, 40, 12)
          expect(toB[0]).toBeCloseTo(Math.log(inMass) - Math.log(total - inMass), 5)
        }
      }),
      { numRuns: 6 },
    )
  })
  it('truncatedNormalMoments equals integration, in the upper tail too', () => {
    for (const [m, v, lo, hi] of [
      [0.3, 1.2, -1, 2],
      [0, 1, 0, Infinity],
      [1, 0.5, 3, 5],
      [-0.5, 2, -Infinity, -4],
    ] as const) {
      const t = M.truncatedNormalMoments(m, v, lo, hi)
      const a = Math.max(lo, m - 12 * Math.sqrt(v))
      const b = Math.min(hi, m + 12 * Math.sqrt(v))
      const dens = (x: number): number => Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)
      const z = integrate(dens, a, b, 30, 12)
      const mu = integrate((x) => x * dens(x), a, b, 30, 12) / z
      const va = integrate((x) => (x - mu) ** 2 * dens(x), a, b, 30, 12) / z
      expect(Math.exp(t.logMass)).toBeCloseTo(z, 8)
      expect(t.mean).toBeCloseTo(mu, 7)
      expect(t.variance).toBeCloseTo(va, 7)
    }
  })
})

describe('logistic', () => {
  it('EP is the quadrature of the tilted distribution; the variational messages are the Jaakkola bound and E[x]', () => {
    const ex = M.gaussianEta(0.4, 1.3)
    const logf = ([b, x]: readonly number[]): number => Math.log(sigmoidOf((b ?? 0) > 0.5 ? (x ?? 0) : -(x ?? 0)))
    for (const label of [1, 0]) {
      const m = M.logistic.ep?.(ctx(1, [M.valueArg(M.bernoulli, label), msg(gaussian, ex)])).message as M.Eta
      const brute = bruteTilted([{ kind: 'bernoulli', value: label }, gp(ex)], logf, 1)
      near(implied(gaussian, ex, m), brute.moments, 1e-7)
      expect(M.logistic.logEvidence?.(ctx(1, [M.valueArg(M.bernoulli, label), msg(gaussian, ex)]))).toBeCloseTo(
        brute.logZ,
        7,
      )
    }
    // Variational: the message to b is E[x]; the message to x is a quadratic below the log logistic, tangent at xi.
    const belief = M.gaussianEta(0.4, 1.3)
    const toB = M.logistic.vmp?.(ctx(0, [msg(M.bernoulli, [0]), msg(gaussian, belief)], {}, 'vmp')).message as M.Eta
    expect(toB[0]).toBeCloseTo(0.4, 12)
    const toX = M.logistic.vmp?.(ctx(1, [M.valueArg(M.bernoulli, 1), msg(gaussian, belief)], {}, 'vmp'))
      .message as M.Eta
    const xi = Math.sqrt(1.3 + 0.4 ** 2)
    const [h, j] = toX as [number, number]
    expect(h).toBe(0.5)
    const lambda = j / 2
    const c = Math.log(sigmoidOf(xi)) - xi / 2 + lambda * xi * xi
    for (let x = -6; x <= 6; x += 0.25)
      expect(c + h * x - 0.5 * j * x * x).toBeLessThanOrEqual(Math.log(sigmoidOf(x)) + 1e-12)
    expect(c + h * xi - 0.5 * j * xi * xi).toBeCloseTo(Math.log(sigmoidOf(xi)), 12)
  })
})

describe('conjugate operators', () => {
  it('bernoulli with a beta: variational messages are the expected logs, EP matches the log moments of the mixture', () => {
    const cavity = M.betaEta(2.5, 4)
    const beta = M.beta
    const bern = M.bernoulli
    for (const pi of [1, 0, 0.3]) {
      const b = pi === 0.3 ? msg(bern, M.bernoulliEta(0.3)) : M.valueArg(bern, pi)
      const m = M.bernoulliBeta.ep?.(ctx(1, [b, msg(beta, cavity)])).message as M.Eta
      const factor = (p: number): number => pi * p + (1 - pi) * (1 - p)
      const dens = (p: number): number => Math.exp(1.5 * Math.log(p) + 3 * Math.log(1 - p)) * factor(p)
      const z = integrate(dens, 0, 1, 40, 12)
      const e1 = integrate((p) => Math.log(p) * dens(p), 0, 1, 60, 12) / z
      const e2 = integrate((p) => Math.log(1 - p) * dens(p), 0, 1, 60, 12) / z
      near(beta.moments(beta.product(cavity, m)), [e1, e2], 1e-5)
    }
    const toB = M.bernoulliBeta.ep?.(ctx(0, [msg(bern, bern.flat), msg(beta, cavity)])).message as M.Eta
    expect(bern.moments(toB)[0]).toBeCloseTo(2.5 / (2.5 + 4), 12)
    const vp = M.bernoulliBeta.vmp?.(ctx(1, [msg(bern, M.bernoulliEta(0.2)), msg(beta, cavity)], {}, 'vmp'))
      .message as M.Eta
    near(vp, [0.2, 0.8], 1e-12)
    const vb = M.bernoulliBeta.vmp?.(ctx(0, [msg(bern, bern.flat), msg(beta, cavity)], {}, 'vmp')).message as M.Eta
    const mo = beta.moments(cavity)
    expect(vb[0]).toBeCloseTo((mo[0] as number) - (mo[1] as number), 12)
  })
  it('discrete with a Dirichlet: EP matches the log moments of the mixture, variational messages are exact', () => {
    const k = 3
    const cavity = [1.5, 0.8, 2.2]
    const dir = M.dirichlet(k)
    const q = [0.2, 0.5, 0.3]
    const dz = M.discrete(k)
    const m = M.discreteDirichlet.ep?.(ctx(1, [msg(dz, M.discreteEta(q)), msg(dir, cavity)])).message as M.Eta
    const alpha = cavity.map((v) => v + 1)
    const dens = (p1: number, p2: number): number => {
      const p3 = 1 - p1 - p2
      return (
        p1 ** (cavity[0] as number) *
        p2 ** (cavity[1] as number) *
        p3 ** (cavity[2] as number) *
        ((q[0] as number) * p1 + (q[1] as number) * p2 + (q[2] as number) * p3)
      )
    }
    const tri = (f: (a: number, b: number) => number): number =>
      integrate((a) => integrate((b) => f(a, b), 0, 1 - a, 12, 12), 0, 1, 24, 12)
    const z = tri(dens)
    const want = [0, 1, 2].map((j) => tri((a, b) => Math.log([a, b, 1 - a - b][j] as number) * dens(a, b)) / z)
    near(dir.moments(dir.product(cavity, m)), want, 1e-5)
    void alpha
    const toZ = M.discreteDirichlet.ep?.(ctx(0, [msg(dz, dz.flat), msg(dir, cavity)])).message as M.Eta
    near(M.discreteProbs(toZ), [2.5 / 7.5, 1.8 / 7.5, 3.2 / 7.5], 1e-12)
    const vz = M.discreteDirichlet.vmp?.(ctx(0, [msg(dz, dz.flat), msg(dir, cavity)], {}, 'vmp')).message as M.Eta
    near(vz, [...dir.moments(cavity)], 1e-12)
    const vp = M.discreteDirichlet.vmp?.(ctx(1, [msg(dz, M.discreteEta(q)), msg(dir, cavity)], {}, 'vmp'))
      .message as M.Eta
    near(vp, q, 1e-12)
  })
  it('poisson and binomial: the message is the likelihood, equal to the integral', () => {
    const cavity = M.gammaEta(2.2, 1.4)
    const m = M.poisson.ep?.(ctx(1, [M.valueArg(gaussian, 3), msg(M.gamma, cavity)], { exposure: 1.7 }))
      .message as M.Eta
    const dens = (l: number): number => Math.exp(1.2 * Math.log(l) - 1.4 * l) * l ** 3 * Math.exp(-1.7 * l)
    const z = integratePositive(dens, 40)
    near(
      M.gamma.moments(M.gamma.product(cavity, m)),
      [integratePositive((l) => Math.log(l) * dens(l), 40) / z, integratePositive((l) => l * dens(l), 40) / z],
      1e-6,
    )
    const bc = M.betaEta(2, 3)
    const mb = M.binomial.ep?.(ctx(1, [M.valueArg(gaussian, 4), msg(M.beta, bc)], { n: 9 })).message as M.Eta
    expect(mb).toEqual([4, 5])
    expect(
      M.poisson.logEvidence?.(ctx(1, [M.valueArg(gaussian, 3), msg(M.gamma, cavity)], { exposure: 1.7 })),
    ).toBeCloseTo(Math.log((1.7 ** 3 / 6) * z), 6)
  })
})

describe('discrete operators', () => {
  const d2 = M.discrete(2)
  const d3 = M.discrete(3)
  it('table: BP messages and evidence equal the sums; variational messages are the expected log table', () => {
    const values = [0.2, 0.5, 0.3, 0.9, 0.1, 0.4]
    const params = { states: [2, 3], values }
    const q0 = [0.3, 0.7]
    const q1 = [0.2, 0.5, 0.3]
    const args = [msg(d2, M.discreteEta(q0)), msg(d3, M.discreteEta(q1))]
    const toX0 = M.table.bp?.(ctx(0, args, params, 'bp')).message as M.Eta
    const want0 = [0, 1].map((a) =>
      [0, 1, 2].reduce((s, b) => s + (values[a * 3 + b] as number) * (q1[b] as number), 0),
    )
    near(
      M.discreteProbs(toX0),
      want0.map((v) => v / ((want0[0] as number) + (want0[1] as number))),
      1e-12,
    )
    const toX1 = M.table.bp?.(ctx(1, args, params, 'bp')).message as M.Eta
    const want1 = [0, 1, 2].map((b) =>
      [0, 1].reduce((s, a) => s + (values[a * 3 + b] as number) * (q0[a] as number), 0),
    )
    near(
      M.discreteProbs(toX1),
      want1.map((v) => v / want1.reduce((a, c) => a + c, 0)),
      1e-12,
    )
    let z = 0
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 3; b++) z += (values[a * 3 + b] as number) * (q0[a] as number) * (q1[b] as number)
    expect(M.table.logEvidence?.(ctx(0, args, params, 'bp'))).toBeCloseTo(Math.log(z), 12)
    // Observed state.
    const toObs = M.table.bp?.(ctx(1, [M.valueArg(d2, 1), msg(d3, d3.flat)], params, 'bp')).message as M.Eta
    near(M.discreteProbs(toObs), [0.9 / 1.4, 0.1 / 1.4, 0.4 / 1.4], 1e-12)
    // Variational.
    const vq = M.table.vmp?.(ctx(0, args, params, 'vmp')).message as M.Eta
    const wantV = [0, 1].map((a) =>
      [0, 1, 2].reduce((s, b) => s + Math.log(values[a * 3 + b] as number) * (q1[b] as number), 0),
    )
    near(M.discreteProbs(vq), M.discreteProbs(wantV), 1e-12)
    expect(M.table.expectedLog?.(ctx(0, args, params, 'vmp'))).toBeCloseTo(
      [0, 1].reduce((s, a) => s + (q0[a] as number) * (wantV[a] as number), 0),
      12,
    )
  })
  it('logic tables behave as the gates they name', () => {
    expect(M.logicTable('or', 2).values).toEqual([1, 0, 0, 0, 0, 1, 1, 1])
    expect(M.logicTable('not', 1).values).toEqual([0, 1, 1, 0])
    expect(M.logicTable('xor', 2).values.length).toBe(8)
    // P(a and b) from independent Bernoulli(0.3), Bernoulli(0.6): the message to the output.
    const params = M.logicTable('and', 2)
    const args = [
      msg(M.bernoulli, M.bernoulli.flat),
      msg(M.bernoulli, M.bernoulliEta(0.3)),
      msg(M.bernoulli, M.bernoulliEta(0.6)),
    ]
    const out = M.table.bp?.(ctx(0, args, params, 'bp')).message as M.Eta
    expect(M.bernoulli.moments(out)[0]).toBeCloseTo(0.18, 12)
  })
  it('select (the gate): the messages are the moment-matched mixtures', () => {
    fc.assert(
      fc.property(
        gaussianArg,
        gaussianArg,
        gaussianArg,
        fc.double({ min: 0.1, max: 0.9, noNaN: true }),
        (ey, e0, e1, p) => {
          const dz = M.discrete(2)
          const ez = M.discreteEta([p, 1 - p])
          const args = [msg(gaussian, ey), msg(dz, ez), msg(gaussian, e0), msg(gaussian, e1)]
          const params = { components: 2 }
          const normal = (eta: M.Eta, x: number): number =>
            Math.exp((eta[0] ?? 0) * x - 0.5 * (eta[1] ?? 0) * x * x - gaussian.logPartition(eta))
          const range = (eta: M.Eta): [number, number] => {
            const s = gaussian.summary(eta)
            return [
              (s['mean'] as number) - 10 * Math.sqrt(s['variance'] as number),
              (s['mean'] as number) + 10 * Math.sqrt(s['variance'] as number),
            ]
          }
          const wide: [number, number] = [-14, 14]
          void range
          // To y: tilted(y) = cav_y(y) (p q0(y) + (1 - p) q1(y)).
          const tiltedY = (y: number): number =>
            Math.exp((ey[0] ?? 0) * y - 0.5 * (ey[1] ?? 0) * y * y) * (p * normal(e0, y) + (1 - p) * normal(e1, y))
          const zY = integrate(tiltedY, ...wide, 40, 12)
          const mY = [
            integrate((y) => y * tiltedY(y), ...wide, 40, 12) / zY,
            integrate((y) => y * y * tiltedY(y), ...wide, 40, 12) / zY,
          ]
          near(implied(gaussian, ey, M.select.ep?.(ctx(0, args, params)).message as M.Eta), mY, 1e-6)
          // To z: the overlap of the cavity of y with each component.
          const over = [e0, e1].map((e) =>
            integrate((y) => Math.exp((ey[0] ?? 0) * y - 0.5 * (ey[1] ?? 0) * y * y) * normal(e, y), ...wide, 40, 12),
          )
          near(
            M.discreteProbs(M.select.ep?.(ctx(1, args, params)).message as M.Eta),
            over.map((v) => v / ((over[0] as number) + (over[1] as number))),
            1e-6,
          )
          // To c_0: tilted(c) = cav_c(c) (p q_y(c) + (1 - p) O_1).
          const qy = (y: number): number => normal(ey, y)
          const o1 = integrate((y) => qy(y) * normal(e1, y), ...wide, 40, 12)
          const tiltedC = (c: number): number =>
            Math.exp((e0[0] ?? 0) * c - 0.5 * (e0[1] ?? 0) * c * c) * (p * qy(c) + (1 - p) * o1)
          const zC = integrate(tiltedC, ...wide, 40, 12)
          const mC = [
            integrate((y) => y * tiltedC(y), ...wide, 40, 12) / zC,
            integrate((y) => y * y * tiltedC(y), ...wide, 40, 12) / zC,
          ]
          near(implied(gaussian, e0, M.select.ep?.(ctx(2, args, params)).message as M.Eta), mC, 1e-6)
          // Evidence: the mass of the factor under the messages.
          const zz = integrate(
            (y) =>
              Math.exp((ey[0] ?? 0) * y - 0.5 * (ey[1] ?? 0) * y * y) *
              (p * Math.exp((e0[0] ?? 0) * y - 0.5 * (e0[1] ?? 0) * y * y) * Math.exp(gaussian.logPartition(e1)) +
                (1 - p) *
                  Math.exp((e1[0] ?? 0) * y - 0.5 * (e1[1] ?? 0) * y * y) *
                  Math.exp(gaussian.logPartition(e0))),
            ...wide,
            40,
            12,
          )
          expect(M.select.logEvidence?.(ctx(0, args, params))).toBeCloseTo(Math.log(zz), 6)
        },
      ),
      { numRuns: 5 },
    )
  })
})

describe('variational operators equal the expected log factor', () => {
  /** The expected log factor as a function of the target's statistics, up to a constant, against the message's pairing. */
  it('gaussian with a gamma precision: messages to x, mean and precision', () => {
    const bx = M.gaussianEta(0.5, 2)
    const bm = M.gaussianEta(-0.3, 1.5)
    const bl = M.gammaEta(3, 2)
    const args = [msg(gaussian, bx), msg(gaussian, bm), msg(M.gamma, bl)]
    // E over (mu, lambda) of log N(x; mu, 1 / lambda) as a function of x: differences against h x - J x^2 / 2.
    const lambdaMean = 1.5
    const muMean = -0.3
    const toX = M.gaussianGammaPrecision.vmp?.(ctx(0, args, {}, 'vmp')).message as M.Eta
    const g = (x: number): number => -0.5 * lambdaMean * ((x - muMean) ** 2 + 1.5)
    const pair = (e: M.Eta, x: number): number => (e[0] as number) * x - 0.5 * (e[1] as number) * x * x
    expect(g(1.2) - g(0.1)).toBeCloseTo(pair(toX, 1.2) - pair(toX, 0.1), 12)
    const toL = M.gaussianGammaPrecision.vmp?.(ctx(2, args, {}, 'vmp')).message as M.Eta
    // E_{x, mu} log N(x; mu, 1/l) as a function of l: 1/2 log l - l/2 E (x - mu)^2.
    const e2 = 2 + 1.5 + (0.5 + 0.3) ** 2
    const gl = (l: number): number => 0.5 * Math.log(l) - 0.5 * l * e2
    const pl = (l: number): number => (toL[0] as number) * Math.log(l) + (toL[1] as number) * l
    expect(gl(2.5) - gl(0.7)).toBeCloseTo(pl(2.5) - pl(0.7), 12)
    const lp = M.gaussianGammaPrecision.expectedLog?.(ctx(0, args, {}, 'vmp')) as number
    // E log f = 1/2 (E log l - log 2 pi) - 1/2 E l E (x - mu)^2.
    const mo = M.gamma.moments(bl)
    expect(lp).toBeCloseTo(0.5 * ((mo[0] as number) - Math.log(2 * Math.PI)) - 0.5 * lambdaMean * e2, 12)
  })
  it('gaussian-normal-gamma and the mixture: messages to the normal-gamma and to the assignment', () => {
    const ng = M.normalGammaEta(0.4, 2, 3, 1.5)
    const x = 1.3
    const toTheta = M.gaussianNormalGamma.vmp?.(ctx(1, [M.valueArg(gaussian, x), msg(M.normalGamma, ng)], {}, 'vmp'))
      .message as M.Eta
    // E_x log N(x | mu, 1/l) for the observed x is 1/2 log l - l (x - mu)^2 / 2: a function of the statistics (l mu, l mu^2, l, log l).
    const stat = (mu: number, l: number): number[] => [l * mu, l * mu * mu, l, Math.log(l)]
    const dot = (a: readonly number[], b: readonly number[]): number =>
      a.reduce((s, v, i) => s + v * (b[i] as number), 0)
    const gx = (mu: number, l: number): number => 0.5 * Math.log(l) - 0.5 * l * (x - mu) ** 2
    const d1 = gx(0.7, 2.1) - gx(-0.4, 0.9)
    expect(d1).toBeCloseTo(dot(toTheta, stat(0.7, 2.1)) - dot(toTheta, stat(-0.4, 0.9)), 12)
    // The mixture: responsibility times that message, and E log N under each component to z.
    const dz = M.discrete(2)
    const t0 = M.normalGammaEta(0, 1, 2, 1)
    const t1 = M.normalGammaEta(5, 2, 4, 3)
    const args = [
      M.valueArg(gaussian, x),
      msg(dz, M.discreteEta([0.25, 0.75])),
      msg(M.normalGamma, t0),
      msg(M.normalGamma, t1),
    ]
    const toT1 = M.gaussianMixture.vmp?.(ctx(3, args, { components: 2 }, 'vmp')).message as M.Eta
    near(
      toT1,
      toTheta.map(
        (v, i) => 0.75 * (i === 0 || i === 3 ? (i === 0 ? x : 0.5) : i === 1 ? -0.5 : -0.5 * x * x) * (v === 0 ? 1 : 1),
      ),
      1e-12,
    )
    const toZ = M.gaussianMixture.vmp?.(ctx(1, args, { components: 2 }, 'vmp')).message as M.Eta
    const expectLog = (t: M.Eta): number => {
      const mo = M.normalGamma.moments(t)
      return (
        0.5 * ((mo[3] as number) - Math.log(2 * Math.PI)) -
        0.5 * (x * x * (mo[2] as number) - 2 * x * (mo[0] as number) + (mo[1] as number))
      )
    }
    near(toZ, [expectLog(t0), expectLog(t1)], 1e-12)
  })
  it('gaussian: the expected log density and the messages', () => {
    const bx = M.gaussianEta(0.5, 2)
    const bm = M.gaussianEta(-0.3, 1.5)
    const toX = M.gaussianOperator.vmp?.(ctx(0, [msg(gaussian, bx), msg(gaussian, bm)], { precision: 3 }, 'vmp'))
      .message as M.Eta
    expect(toX[0]).toBeCloseTo(-0.9, 12)
    expect(toX[1]).toBe(3)
    const lp = M.gaussianOperator.expectedLog?.(
      ctx(0, [msg(gaussian, bx), msg(gaussian, bm)], { precision: 3 }, 'vmp'),
    ) as number
    const e = 2 + 1.5 + (0.5 + 0.3) ** 2
    expect(lp).toBeCloseTo(0.5 * Math.log(3 / (2 * Math.PI)) - 1.5 * e, 12)
  })
})

describe('prior, equal and the numeric operator', () => {
  it('the prior is its own message and its mass is the ratio of partitions', () => {
    const eta = M.gaussianEta(1, 0.5)
    const cav = M.gaussianEta(0.2, 2)
    expect(M.prior.ep?.(ctx(0, [msg(gaussian, cav)], { eta })).message).toEqual(eta)
    const brute = bruteTilted(
      [gp(cav)],
      ([x]) => (-0.5 * ((x ?? 0) - 1) ** 2) / 0.5 - 0.5 * Math.log(2 * Math.PI * 0.5),
      0,
    )
    expect(M.prior.logEvidence?.(ctx(0, [msg(gaussian, cav)], { eta }))).toBeCloseTo(brute.logZ, 7)
    expect(M.prior.check?.({ eta }, [gaussian])).toBeUndefined()
    expect(M.prior.check?.({ eta: [1] }, [gaussian])).toBeDefined()
  })
  it('equal passes the other side along', () => {
    const e = M.gammaEta(2, 3)
    expect(M.equal.ep?.(ctx(0, [msg(M.gamma, M.gamma.flat), msg(M.gamma, e)])).message).toEqual(e)
  })
  it('numericOperator: the tilted moments of the clutter factor equal the integral', () => {
    const w = 0.25
    const y = 1.4
    const logf = ([x]: readonly number[]): number =>
      Math.log(
        ((1 - w) * Math.exp(-0.5 * (y - (x ?? 0)) ** 2)) / Math.sqrt(2 * Math.PI) +
          (w * Math.exp((-0.5 * (y * y)) / 10)) / Math.sqrt(20 * Math.PI),
      )
    const op = M.numericOperator({ name: 'clutter', arity: 1, logDensity: logf })
    const ex = M.gaussianEta(0.3, 4)
    const t = op.tilted?.(ctx(0, [msg(gaussian, ex)])) as M.TiltedMoments
    const brute = bruteTilted([gp(ex)], logf, 0)
    near(t.moments, brute.moments, 2e-5)
    expect(t.logZ).toBeCloseTo(brute.logZ, 5)
    const two = M.numericOperator({ name: 'pair', arity: 2, logDensity: ([a, b]) => -0.5 * ((a ?? 0) - (b ?? 0)) ** 2 })
    const e2 = [msg(gaussian, M.gaussianEta(0, 2)), msg(gaussian, M.gaussianEta(1, 3))]
    const tp = two.tilted?.(ctx(1, e2)) as M.TiltedMoments
    near(
      tp.moments,
      bruteTilted([gp(M.gaussianEta(0, 2)), gp(M.gaussianEta(1, 3))], ([a, b]) => -0.5 * ((a ?? 0) - (b ?? 0)) ** 2, 1)
        .moments,
      1e-6,
    )
  })
  it('the registry lists the operators by name and checks its ports', () => {
    const ops = M.defaultOperators()
    for (const name of ['prior', 'gaussian', 'probit', 'plus', 'table', 'select']) expect(ops[name]?.name).toBe(name)
    expect(Object.keys(ops).length).toBeGreaterThanOrEqual(26)
  })
})
