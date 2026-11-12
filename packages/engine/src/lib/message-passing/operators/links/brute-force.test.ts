// The truncation operators and the logistic link against the integral of the exact message (brute-force quadrature).
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { type BrutePort, bruteTilted, integrate } from '../../../../testing/message-passing.ts'
import { ctx, gaussian, gaussianArg, gp, implied, msg, near, runs, scale } from '../../../../testing/operator-checks.ts'
import { sigmoidOf } from '../../../base/index.ts'
import { truncatedNormal } from '../../../dists/index.ts'
import { item } from '../../../tensor/index.ts'
import * as M from '../../index.ts'

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
      { numRuns: 6 * scale },
    )
  })
  it('truncatedNormalMoments agrees with the truncated normal distribution of dists', () => {
    for (const [m, v, lo, hi] of [
      [0.3, 1.2, -1, 2],
      [1, 0.5, 0, 4],
    ] as const) {
      const t = M.truncatedNormalMoments(m, v, lo, hi)
      const d = truncatedNormal(m, Math.sqrt(v), lo, hi)
      expect(t.mean).toBeCloseTo(item(d.mean()), 10)
      expect(t.variance).toBeCloseTo(item(d.variance()), 10)
    }
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
