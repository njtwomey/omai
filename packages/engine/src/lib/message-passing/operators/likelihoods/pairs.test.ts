import { describe, expect, it } from 'vitest'
import { massOf, messageArg, valueArg } from '../../core/args.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg, Eta, FactorOperator, Family } from '../../core/types.ts'
import { digammaOf, lgammaOf } from '../../numerics/_scalar.ts'
import { defaultOperators } from '../registry.ts'
import { pairLikelihoodOperator, pairLikelihoods } from './pairs.ts'

/*
 * Each case integrates the density against the two cavities on a fine midpoint grid in the coordinates, with no code of the
 * operator in it, and compares the tilted moments of both ports and the log evidence.
 */

const op: FactorOperator = pairLikelihoodOperator()

interface Cavity {
  readonly family: Family
  readonly eta: Eta
  /** The unnormalised log density of the cavity in the coordinate the likelihood reads (the value, or the log of a gamma value). */
  readonly logDensity: (x: number) => number
  readonly lo: number
  readonly hi: number
}

const gaussianCavity = (m: number, v: number, span = 10): Cavity => ({
  family: gaussian,
  eta: gaussianEta(m, v),
  logDensity: (x) => -0.5 * Math.log(2 * Math.PI * v) - (0.5 * (x - m) ** 2) / v,
  lo: m - span * Math.sqrt(v),
  hi: m + span * Math.sqrt(v),
})

const gammaCavity = (shape: number, rate: number): Cavity => ({
  family: gamma,
  eta: gammaEta(shape, rate),
  // The normalised gamma density in t = log a: shape t - rate e^t - log Gamma(shape) + shape log rate.
  logDensity: (t) => shape * t - rate * Math.exp(t) - lgammaOf(shape) + shape * Math.log(rate),
  lo: -14,
  hi: 6,
})

/** The mass and the moments of the coordinates under the integrand by a midpoint grid of n by n cells. */
function brute(
  cavities: readonly [Cavity, Cavity],
  logf: (a: number, b: number) => number,
  n = 900,
): { logMass: number; mean: number[]; second: number[]; meanExp: number[] } {
  const [c0, c1] = cavities
  const h0 = (c0.hi - c0.lo) / n
  const h1 = (c1.hi - c1.lo) / n
  let mass = 0
  const m = [0, 0]
  const m2 = [0, 0]
  const me = [0, 0]
  for (let i = 0; i < n; i++) {
    const a = c0.lo + (i + 0.5) * h0
    for (let j = 0; j < n; j++) {
      const b = c1.lo + (j + 0.5) * h1
      const w = Math.exp(c0.logDensity(a) + c1.logDensity(b) + logf(a, b)) * h0 * h1
      mass += w
      m[0] = (m[0] as number) + w * a
      m[1] = (m[1] as number) + w * b
      m2[0] = (m2[0] as number) + w * a * a
      m2[1] = (m2[1] as number) + w * b * b
      me[0] = (me[0] as number) + w * Math.exp(a)
      me[1] = (me[1] as number) + w * Math.exp(b)
    }
  }
  return {
    logMass: Math.log(mass),
    mean: m.map((v) => v / mass),
    second: m2.map((v) => v / mass),
    meanExp: me.map((v) => v / mass),
  }
}

/** The target's tilted moments in the family's statistics, from the brute force. */
function expected(cav: Cavity, k: number, b: ReturnType<typeof brute>): number[] {
  return cav.family.kind === 'gamma'
    ? [b.mean[k] as number, b.meanExp[k] as number]
    : [b.mean[k] as number, b.second[k] as number]
}

function check(
  likelihood: string,
  y: number,
  extra: Record<string, number>,
  cavities: readonly [Cavity, Cavity],
  tolerance = { moments: 2e-4, evidence: 2e-4 },
  n = 900,
): void {
  const params = { likelihood, y, ...extra }
  const density = pairLikelihoods()[likelihood]?.density(y, params) as (a: number, b: number) => number
  const b = brute(cavities, density, n)
  const args: Arg[] = cavities.map((c) => messageArg(c.family, c.eta))
  for (const k of [0, 1] as const) {
    const t = op.tilted?.({ mode: 'ep', to: k, args, params, power: 1 })
    const want = expected(cavities[k], k, b)
    for (const [i, w] of want.entries()) {
      const got = t?.moments[i] as number
      expect(Math.abs(got - w), `${likelihood} port ${k} moment ${i}: ${got} against ${w}`).toBeLessThan(
        tolerance.moments * (1 + Math.abs(w)),
      )
    }
    const wantZ = b.logMass + massOf(args[0] as Arg) + massOf(args[1] as Arg)
    expect(Math.abs((t?.logZ as number) - wantZ), `${likelihood} evidence ${t?.logZ} against ${wantZ}`).toBeLessThan(
      tolerance.evidence,
    )
  }
  const z = op.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
  expect(Math.abs(z - (b.logMass + massOf(args[0] as Arg) + massOf(args[1] as Arg)))).toBeLessThan(tolerance.evidence)
}

describe('likelihood-pair: tilted moments and log evidence against a brute-force integral', () => {
  it('laplace-scale, narrow cavities', () => {
    check('laplace-scale', 1.3, {}, [gaussianCavity(0.5, 0.6), gaussianCavity(-0.4, 0.4)])
  })
  it('laplace-scale, a location cavity much wider than the likelihood', () => {
    check(
      'laplace-scale',
      0.2,
      {},
      [gaussianCavity(0, 9, 4.7), gaussianCavity(-0.5, 0.5, 8)],
      { moments: 5e-4, evidence: 5e-4 },
      2800,
    )
  })
  it('laplace-scale with a gamma scale', () => {
    check('laplace-scale', -0.7, {}, [gaussianCavity(0, 2), gammaCavity(2.5, 3)])
  })
  it('asymmetric-laplace-scale, a log-normal and a gamma scale', () => {
    check('asymmetric-laplace-scale', 0.9, { tau: 0.8 }, [gaussianCavity(0.2, 0.7), gaussianCavity(-0.5, 0.4)])
    check('asymmetric-laplace-scale', -0.6, { tau: 0.25 }, [gaussianCavity(0, 1.5), gammaCavity(2.5, 3)])
  })
  it('student-t-df, a gaussian log df', () => {
    check('student-t-df', 2.4, { scale: 0.8 }, [gaussianCavity(0.3, 1), gaussianCavity(1.2, 0.5)])
  })
  it('student-t-scale, a gaussian log scale and a gamma scale', () => {
    check('student-t-scale', 2.4, { df: 3 }, [gaussianCavity(0.3, 1), gaussianCavity(-0.2, 0.4)], {
      moments: 3e-3,
      evidence: 3e-3,
    })
    check('student-t-scale', -1.1, { df: 5.5 }, [gaussianCavity(0, 1.5), gammaCavity(6, 5)])
  })
  it('student-t-df, a gamma df', () => {
    check('student-t-df', -1.1, { scale: 1 }, [gaussianCavity(0, 1.5), gammaCavity(3, 0.4)])
  })
  it('weibull-shape, uncensored and right-censored', () => {
    check('weibull-shape', 1.6, { censored: 0 }, [gaussianCavity(0.2, 0.5), gammaCavity(4, 2.5)])
    check('weibull-shape', 2.2, { censored: 1 }, [gaussianCavity(0.5, 0.5), gaussianCavity(0.3, 0.3)])
  })
  it('beta-shapes, two log-normal shapes and two gamma shapes', () => {
    check('beta-shapes', 0.3, {}, [gaussianCavity(0.4, 0.5), gaussianCavity(0.8, 0.4)])
    check('beta-shapes', 0.8, {}, [gammaCavity(2, 1), gammaCavity(3, 1.5)])
  })
  it('the column form of a density is the density', () => {
    const cases: [string, number, Record<string, number>][] = [
      ['student-t-df', 1.3, { scale: 0.7 }],
      ['student-t-scale', 1.3, { df: 4 }],
      ['weibull-shape', 1.6, { censored: 0 }],
      ['weibull-shape', 2.2, { censored: 1 }],
      ['beta-shapes', 0.35, {}],
    ]
    for (const [name, y, extra] of cases) {
      const row = pairLikelihoods()[name]
      const flat = row?.density(y, { y, ...extra }) as (a: number, b: number) => number
      const column = row?.column?.(y, { y, ...extra }) as (b: number) => (a: number) => number
      for (const [a, b] of [
        [0.3, 0.9],
        [-1.2, 1.7],
        [0.8, -0.4],
      ] as const)
        expect(column(b)(a), `${name} at ${a}, ${b}`).toBeCloseTo(flat(a, b), 10)
    }
  })
  it('an observed second coordinate is a point', () => {
    const params = { likelihood: 'laplace-scale', y: 1 }
    const c = gaussianCavity(0, 1)
    const t = op.tilted?.({
      mode: 'ep',
      to: 0,
      args: [messageArg(c.family, c.eta), valueArg(gaussian, 0.2)],
      params,
      power: 1,
    })
    // A single Laplace site with scale exp(0.2): the log mass is the integral of exp(-|y - u| / b) / (2 b) against N(0, 1).
    const b = Math.exp(0.2)
    let mass = 0
    for (let i = 0; i < 20000; i++) {
      const u = -10 + (i + 0.5) * 0.001
      mass += Math.exp(-0.5 * u * u - Math.abs(1 - u) / b) * 0.001
    }
    mass /= Math.sqrt(2 * Math.PI) * 2 * b
    expect(t?.logZ).toBeCloseTo(Math.log(mass) + massOf(messageArg(c.family, c.eta)), 4)
  })
  it('is in the default registry, carries a log evidence and refuses a wrong family', () => {
    const registry = defaultOperators()
    expect(registry['likelihood-pair']?.logEvidence).toBeDefined()
    expect(digammaOf(1)).toBeLessThan(0)
    expect(op.check?.({ likelihood: 'student-t-df', y: 1, scale: 1 }, [gamma, gamma])).toMatch(/not a gamma variable/)
    expect(op.check?.({ likelihood: 'student-t-df', y: 1, scale: 1 }, [gaussian, gamma])).toBeUndefined()
    expect(op.check?.({ likelihood: 'nope', y: 1 }, [])).toMatch(/unknown likelihood/)
    expect(op.check?.({ likelihood: 'beta-shapes', y: 1.5 }, [])).toMatch(/\(0, 1\)/)
  })
})
