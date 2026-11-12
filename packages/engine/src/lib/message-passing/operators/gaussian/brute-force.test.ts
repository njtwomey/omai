// The Gaussian and vector Gaussian operators against the integral of the exact message (brute-force quadrature).
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { type BrutePort, bruteTilted, integrateBox } from '../../../../testing/message-passing.ts'
import {
  boxOf,
  ctx,
  gaussian,
  gaussianArg,
  gp,
  implied,
  mean,
  msg,
  near,
  runs,
  scale,
} from '../../../../testing/operator-checks.ts'
import * as M from '../../index.ts'

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
  it('gaussian-log-scale: EP to the log scale matches the brute-force tilted moments and evidence', () => {
    fc.assert(
      fc.property(gaussianArg, fc.double({ min: -3, max: 3, noNaN: true }), mean, (es, y, loc) => {
        const logf = ([, s]: readonly number[]): number =>
          -0.5 * Math.log(2 * Math.PI) - (s ?? 0) - 0.5 * (y - loc) ** 2 * Math.exp(-2 * (s ?? 0))
        const args = [M.valueArg(gaussian, y), msg(gaussian, es)]
        const m = M.gaussianLogScale.ep?.(ctx(1, args, { loc })).message as M.Eta
        const ports: BrutePort[] = [{ kind: 'gaussian', value: y }, gp(es)]
        const brute = bruteTilted(ports, logf, 1)
        near(implied(gaussian, es, m), brute.moments, 1e-4)
        expect(M.gaussianLogScale.logEvidence?.(ctx(1, args, { loc }))).toBeCloseTo(brute.logZ, 4)
      }),
      // The counterexample of seed 1395539355 (a precise outcome, a sharp peak in the log scale): the reference adapts to it now.
      { ...runs, examples: [[[-0.3959524544892656, 0.3395338044369565], -2.999999999999082, 0]] },
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
      { numRuns: 4 * scale },
    )
  })
  it('product: EP to the output and to each factor', () => {
    // The product has a ridge along a b = y that the 14-panel grid resolves only to about 2e-4 when y is precise and a, b are wide.
    const brute = { panels: 30, order: 14 }
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
          bruteTilted(free, () => 0, 0, brute).moments,
          3e-4,
        )
        // To a: b is integrated out; y = a b is the derived port of a free (a, b) pair.
        near(
          implied(gaussian, ea, M.product.ep?.(ctx(1, args)).message as M.Eta),
          bruteTilted(free, () => 0, 1, brute).moments,
          1e-4,
        )
        near(
          implied(gaussian, eb, M.product.ep?.(ctx(2, args)).message as M.Eta),
          bruteTilted(free, () => 0, 2, brute).moments,
          1e-4,
        )
      }),
      // The counterexample that 14 panels could not resolve (a precise y against wide a and b: the brute-force error was 2.2e-4).
      {
        numRuns: 4 * scale,
        examples: [
          [
            [0, 3.3333333333333335],
            [-0.11952662893229665, 0.38647342995170597],
            [0, 0.3333333333340008],
          ],
        ],
      },
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
