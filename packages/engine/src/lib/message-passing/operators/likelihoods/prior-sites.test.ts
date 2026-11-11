// The exact numeric prior site against brute force: the moments of the cavity times the density from the independent scan-and-trapezoid
// of `densityMoments`, the evidence against the same integral, and the first message (a flat cavity) against the one-off projection
// the compiler used to make (`projectedPrior`).
import { describe, expect, it } from 'vitest'
import { densityMoments, projectedPrior } from '../../core/priors.ts'
import { FLAT_CAVITY_PRECISION, operatorMessage } from '../../engine/engine.ts'
import { momentProjection } from '../../engine/projections.ts'
import * as M from '../../index.ts'

const logPhi = (z: number): number => Math.log(0.5 * (1 + erf(z / Math.SQRT2)))
function erf(x: number): number {
  // Abramowitz-Stegun 7.1.26, enough for a normaliser checked to 1e-6.
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}

interface Case {
  readonly name: string
  readonly params: M.Params
  /** The log density of the variable, normalised. */
  readonly logf: (x: number) => number
  readonly lo: number
  readonly hi: number
  readonly start: number
}

const cases: readonly Case[] = [
  {
    name: 'half-normal on the log',
    params: { y: 0, base: 'half-normal', log: 1, scale: 2 },
    logf: (l) => Math.log(2) - 0.5 * Math.log(2 * Math.PI * 4) - Math.exp(2 * l) / 8 + l,
    lo: Number.NEGATIVE_INFINITY,
    hi: Number.POSITIVE_INFINITY,
    start: Math.log(2),
  },
  {
    name: 'half-Cauchy on the log',
    params: { y: 0, base: 'half-cauchy', log: 1, scale: 0.5 },
    logf: (l) => Math.log(2 / (Math.PI * 0.5)) - Math.log1p((Math.exp(l) / 0.5) ** 2) + l,
    lo: Number.NEGATIVE_INFINITY,
    hi: Number.POSITIVE_INFINITY,
    start: Math.log(0.5),
  },
  {
    name: 'uniform on the value',
    params: { y: 0, base: 'uniform', log: 0, low: -1, high: 3 },
    logf: () => -Math.log(4),
    lo: -1,
    hi: 3,
    start: 1,
  },
  {
    name: 'uniform on the log (zero lower bound)',
    params: { y: 0, base: 'uniform', log: 1, low: 0, high: 5 },
    logf: (l) => -Math.log(5) + l,
    lo: Number.NEGATIVE_INFINITY,
    hi: Math.log(5),
    start: Math.log(5) - 1,
  },
  {
    name: 'truncated normal on the value',
    params: { y: 0, base: 'truncated-normal', log: 0, loc: 0.5, scale: 1.5, low: -1, high: 4 },
    logf: (x) =>
      -Math.log(1.5) -
      0.5 * Math.log(2 * Math.PI) -
      0.5 * ((x - 0.5) / 1.5) ** 2 -
      Math.log(Math.exp(logPhi((4 - 0.5) / 1.5)) - Math.exp(logPhi((-1 - 0.5) / 1.5))),
    lo: -1,
    hi: 4,
    start: 0.5,
  },
  {
    name: 'truncated normal on the log',
    params: { y: 0, base: 'truncated-normal', log: 1, loc: 3, scale: 0.5, low: 0, high: 10 },
    logf: (l) =>
      -Math.log(0.5) -
      0.5 * Math.log(2 * Math.PI) -
      0.5 * ((Math.exp(l) - 3) / 0.5) ** 2 -
      Math.log(Math.exp(logPhi((10 - 3) / 0.5)) - Math.exp(logPhi((0 - 3) / 0.5))) +
      l,
    lo: Number.NEGATIVE_INFINITY,
    hi: Math.log(10),
    start: Math.log(3),
  },
]

const factor = { operator: M.priorSite, params: {}, power: 1, ports: [] } as never
const ep = (params: M.Params, eta: M.Eta): M.Eta => {
  const p = operatorMessage({ ...(factor as object), params } as never, momentProjection, 0, [
    M.messageArg(M.gaussian, eta),
  ])
  if (p.kind !== 'message') throw new Error(`no message: ${p.kind}`)
  return p.eta
}

describe('prior-site against brute force', () => {
  for (const c of cases) {
    describe(c.name, () => {
      for (const [label, cavity] of [
        ['a narrow cavity', M.gaussianEta(c.start, 0.05)],
        ['a cavity of the width of the prior', M.gaussianEta(c.start + 0.5, 2)],
        ['a wide cavity far from the mass', M.gaussianEta(c.start - 3, 50)],
      ] as const) {
        it(`moments and evidence with ${label}`, () => {
          const { mean: cm, variance: cv } = M.gaussian.summary(cavity) as { mean: number; variance: number }
          const logTilted = (x: Float64Array): Float64Array =>
            x.map((v) => (v >= c.lo && v <= c.hi ? c.logf(v) - (0.5 * (v - cm) ** 2) / cv : Number.NEGATIVE_INFINITY))
          const brute = densityMoments(logTilted, c.lo, c.hi, Math.min(Math.max(cm, c.lo), c.hi))
          const message = ep(c.params, cavity)
          const posterior = M.gaussian.summary(M.gaussian.product(cavity, message)) as {
            mean: number
            variance: number
          }
          // The brute force (a trapezoid on a window found by a scan) is itself good to about a quarter of a percent on an exponential tail.
          expect(Math.abs(posterior.mean - brute.mean)).toBeLessThan(5e-3 * Math.sqrt(brute.variance) + 1e-6)
          expect(Math.abs(posterior.variance / brute.variance - 1)).toBeLessThan(1e-2)
        })
      }
      it('a flat cavity gives the projection of the density alone (what the compiler used to project once)', () => {
        const got = M.gaussian.summary(ep(c.params, M.gaussian.flat)) as { mean: number; variance: number }
        const old = projectedPrior('x', (x) => x.map(c.logf), { support: [c.lo, c.hi], start: c.start }).params?.[
          'eta'
        ] as number[]
        const want = M.gaussian.summary(old) as { mean: number; variance: number }
        // Within the wide cavity the engine shows (precision 1e-6): its curvature moves the mean by about variance * 1e-6 * mean.
        expect(Math.abs(got.mean - want.mean)).toBeLessThan(1e-3 * Math.sqrt(want.variance) + 1e-3)
        expect(Math.abs(got.variance / want.variance - 1)).toBeLessThan(2e-3)
        expect(FLAT_CAVITY_PRECISION).toBe(1e-6)
      })
    })
  }

  it('a flat cavity on the log of a uniform on (0, 5) is the exact log of a uniform: mean log 5 - 1, variance 1', () => {
    const got = M.gaussian.summary(ep((cases[3] as Case).params, M.gaussian.flat)) as { mean: number; variance: number }
    expect(got.mean).toBeCloseTo(Math.log(5) - 1, 3)
    expect(got.variance).toBeCloseTo(1, 3)
  })

  it('the evidence is the log mass of the cavity times the density', () => {
    const cavity = M.gaussianEta(0.3, 0.8)
    const c = cases[0] as Case
    const { mean: cm, variance: cv } = M.gaussian.summary(cavity) as { mean: number; variance: number }
    // integral of N(l; cm, cv) f(l) dl by a fine trapezoid
    let z = 0
    const h = 1e-3
    for (let l = -30; l < 12; l += h)
      z += Math.exp(c.logf(l) - 0.5 * Math.log(2 * Math.PI * cv) - (l - cm) ** 2 / (2 * cv)) * h
    const got = M.priorSite.logEvidence?.({
      mode: 'ep',
      to: 0,
      args: [M.messageArg(M.gaussian, cavity)],
      params: c.params,
      power: 1,
    })
    // logEvidence is the log partition of the unnormalised cavity message plus the log of the integral of the normalised cavity against f
    expect((got ?? Number.NaN) - M.gaussian.logPartition(cavity)).toBeCloseTo(Math.log(z), 3)
  })

  it('the mean-field message is the projection of the density alone, whatever the belief (the message the one-off projection gave)', () => {
    for (const c of cases) {
      const flat = ep(c.params, M.gaussian.flat)
      for (const belief of [M.gaussian.flat, M.gaussianEta(c.start, 0.1), M.gaussianEta(c.start + 1, 5)]) {
        const m = M.priorSite.vmp?.({
          mode: 'vmp',
          to: 0,
          args: [M.messageArg(M.gaussian, belief)],
          params: c.params,
          power: 1,
        })
        expect(m?.message[0]).toBeCloseTo(flat[0] as number, 6)
        expect(m?.message[1]).toBeCloseTo(flat[1] as number, 6)
      }
    }
  })

  it('refuses parameters it cannot use, with the reason', () => {
    expect(M.priorSite.check?.({ y: 0, base: 'gamma', log: 0 }, [M.gaussian])).toMatch(/base must be one of/)
    expect(M.priorSite.check?.({ y: 0, base: 'uniform', log: 0, low: 2, high: 1 }, [M.gaussian])).toMatch(
      /low must be below high/,
    )
    expect(M.priorSite.check?.({ y: 0, base: 'uniform', log: 1, low: -1, high: 1 }, [M.gaussian])).toMatch(
      /positive support/,
    )
    expect(M.priorSite.check?.({ y: 0, base: 'half-normal', log: 1, scale: -1 }, [M.gaussian])).toMatch(
      /scale must be positive/,
    )
  })
})
