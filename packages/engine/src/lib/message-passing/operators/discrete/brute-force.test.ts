// The conjugate and discrete operators against the integral of the exact message (brute-force quadrature).
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { integrate, integratePositive } from '../../../../testing/message-passing.ts'
import { ctx, gaussian, gaussianArg, implied, msg, near, scale } from '../../../../testing/operator-checks.ts'
import * as M from '../../index.ts'

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
      { numRuns: 5 * scale },
    )
  })
})
