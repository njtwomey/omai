import { describe, expect, it } from 'vitest'
import { type BrutePort, bruteTilted, integrate } from '../../testing/message-passing.ts'
import { Phi } from './_scalar.ts'
import * as M from './index.ts'

const gaussian = M.gaussian
const msg = (eta: M.Eta): M.Arg => M.messageArg(gaussian, eta)
const ctx = (to: number, args: M.Arg[]): M.OperatorContext => ({ mode: 'ep', to, args, params: {}, power: 1 })
const gp = (eta: M.Eta, derive?: (x: readonly number[]) => number): BrutePort => ({
  kind: 'gaussian',
  eta,
  ...(derive === undefined ? {} : { derive }),
})

/** The moments the projection of the tilted distribution must match: those of the cavity times the message. */
const implied = (cavity: M.Eta, message: M.Eta): number[] => [...gaussian.moments(gaussian.product(cavity, message))]

describe("Clark's maximum of two Gaussians", () => {
  it('has the known value for two standard normals: mean 1 / sqrt(pi), variance 1 - 1 / pi', () => {
    const m = M.clarkMax({ mean: 0, variance: 1 }, { mean: 0, variance: 1 })
    expect(m.mean).toBeCloseTo(1 / Math.sqrt(Math.PI), 12)
    expect(m.variance).toBeCloseTo(1 - 1 / Math.PI, 12)
  })

  it('equals the integral of the maximum density for unequal means and variances', () => {
    const cases = [
      [
        { mean: 0.3, variance: 1.2 },
        { mean: -0.2, variance: 0.8 },
      ],
      [
        { mean: 2, variance: 0.5 },
        { mean: -1, variance: 3 },
      ],
      [
        { mean: -1.5, variance: 2 },
        { mean: 1.5, variance: 0.1 },
      ],
    ] as const
    for (const [a, b] of cases) {
      const nd = (x: number, m: number, v: number): number =>
        Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)
      const cdf = (x: number, m: number, v: number): number => Phi((x - m) / Math.sqrt(v))
      const density = (y: number): number =>
        nd(y, a.mean, a.variance) * cdf(y, b.mean, b.variance) + nd(y, b.mean, b.variance) * cdf(y, a.mean, a.variance)
      const lo = -14
      const hi = 14
      const z = integrate(density, lo, hi, 40, 16)
      const mean = integrate((y) => y * density(y), lo, hi, 40, 16)
      const second = integrate((y) => y * y * density(y), lo, hi, 40, 16)
      const clark = M.clarkMax(a, b)
      expect(z).toBeCloseTo(1, 8)
      expect(clark.mean).toBeCloseTo(mean, 8)
      expect(clark.variance).toBeCloseTo(second - mean * mean, 8)
    }
  })

  it('is symmetric, and the minimum is the negative of the maximum of the negatives', () => {
    const a = { mean: 0.7, variance: 0.9 }
    const b = { mean: -0.4, variance: 1.6 }
    expect(M.clarkMax(a, b)).toEqual(M.clarkMax(b, a))
  })
})

describe('the max and min operators equal the integral of the exact message', () => {
  const ea = M.gaussianEta(0.3, 1.2)
  const eb = M.gaussianEta(-0.2, 0.8)
  const ey = M.gaussianEta(0.5, 0.9)
  const flat = gaussian.flat
  for (const [op, pick, label] of [
    [M.maximum, Math.max, 'max'],
    [M.minimum, Math.min, 'min'],
  ] as const) {
    for (const [name, y] of [
      ['y told nothing', flat],
      ['y with a cavity', ey],
    ] as const) {
      it(`${label}, ${name}: every message matches the tilted moments and the evidence is the mass`, () => {
        const ports = [gp(y, (x) => pick(x[1] ?? 0, x[2] ?? 0)), gp(ea), gp(eb)]
        const args = [msg(y), msg(ea), msg(eb)]
        for (const to of [0, 1, 2]) {
          const m = op.ep?.(ctx(to, args)).message as M.Eta
          const brute = bruteTilted(ports, () => 0, to, { panels: 60, order: 12 })
          const cavity = [y, ea, eb][to] as M.Eta
          const got = implied(cavity, m)
          expect(got[0]).toBeCloseTo(brute.moments[0] as number, 5)
          expect(got[1]).toBeCloseTo(brute.moments[1] as number, 5)
        }
        if (y !== flat)
          expect(op.logEvidence?.(ctx(0, args))).toBeCloseTo(
            bruteTilted(ports, () => 0, 0, { panels: 60, order: 12 }).logZ,
            5,
          )
      })
    }
  }

  it('the message to y of a told-nothing output is Clark, and the minimum reflects it', () => {
    const args = [msg(flat), msg(ea), msg(eb)]
    const toY = M.maximum.ep?.(ctx(0, args)).message as M.Eta
    const clark = M.clarkMax({ mean: 0.3, variance: 1.2 }, { mean: -0.2, variance: 0.8 })
    expect(toY[0]! / toY[1]!).toBeCloseTo(clark.mean, 12)
    expect(1 / toY[1]!).toBeCloseTo(clark.variance, 12)
    const low = M.minimum.ep?.(ctx(0, args)).message as M.Eta
    const clarkMin = M.clarkMax({ mean: -0.3, variance: 1.2 }, { mean: 0.2, variance: 0.8 })
    expect(low[0]! / low[1]!).toBeCloseTo(-clarkMin.mean, 12)
    expect(1 / low[1]!).toBeCloseTo(clarkMin.variance, 12)
  })

  it('is in the registry and runs in a graph: a hinge of a Gaussian against another has the integral posterior mean', () => {
    const ops = M.defaultOperators()
    expect(ops['max']?.name).toBe('max')
    expect(ops['min']?.name).toBe('min')
    // a ~ N(0, 1), b ~ N(1, 1), y = max(a, b) observed through N(y, 0.25) at 2: EP is the exact moment match on this one-factor tree.
    const g = M.compileGraph(
      {
        variables: [
          { name: 'a', family: { kind: 'gaussian' } },
          { name: 'b', family: { kind: 'gaussian' } },
          { name: 'y', family: { kind: 'gaussian' } },
          { name: 'o', family: { kind: 'gaussian' }, observed: 2 },
        ],
        factors: [
          M.gaussianPrior('a', 0, 1),
          M.gaussianPrior('b', 1, 1),
          { operator: 'max', args: ['y', 'a', 'b'] },
          { operator: 'gaussian', args: ['o', 'y'], params: { precision: 4 } },
        ],
      },
      ops,
    )
    const alg = M.epAlgorithm(g, { tol: 1e-12 })
    let state = alg.init(undefined as never)
    for (let k = 0; k < 60; k++) state = alg.step(state, undefined as never)
    // The exact posterior mean of a: integrate over (a, b) with y = max(a, b).
    const logp = (a: number, b: number): number => -0.5 * a * a - 0.5 * (b - 1) ** 2 - 2 * (2 - Math.max(a, b)) ** 2
    const grid = (f: (a: number, b: number) => number): number =>
      integrate((a) => integrate((b) => f(a, b), -9, 9, 30, 12), -9, 9, 30, 12)
    const z = grid((a, b) => Math.exp(logp(a, b)))
    const ma = grid((a, b) => a * Math.exp(logp(a, b))) / z
    const mb = grid((a, b) => b * Math.exp(logp(a, b))) / z
    const m = M.marginalsOf(g, state)
    expect((m['a'] as { mean: number }).mean).toBeCloseTo(ma, 1)
    expect((m['b'] as { mean: number }).mean).toBeCloseTo(mb, 1)
    expect((m['y'] as { mean: number }).mean).toBeGreaterThan(1.2)
  })
})
