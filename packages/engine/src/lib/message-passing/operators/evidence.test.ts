import { describe, expect, it } from 'vitest'
import { massOf, messageArg, valueArg } from '../core/args.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../core/families.ts'
import { discrete, discreteEta, gaussianMv, gaussianMvEta } from '../core/families-vector.ts'
import type { Arg, FactorOperator, Params } from '../core/types.ts'
import { lgammaOf } from '../numerics/_scalar.ts'
import { gaussianGammaPrecision } from './gaussian/gaussian.ts'
import { defaultOperators } from './registry.ts'
import { gammaProduct, gammaRate } from './scales/gamma-scale.ts'
import { gaussianMvPrecisionScale } from './scales/scale.ts'

/*
 * The log evidence of a factor is log of the integral of the factor times the unnormalised messages of its ports, exp(eta . T).
 * Each test integrates that on a grid, with no code of the operator in it, and compares.
 */

const ops = defaultOperators()
const evidence = (op: FactorOperator, args: readonly Arg[], params: Params, to = 0): number =>
  op.logEvidence?.({ mode: 'ep', to, args: [...args], params, power: 1 }) as number

/** The log of a sum of exponentials. */
function logSum(terms: readonly number[]): number {
  let top = Number.NEGATIVE_INFINITY
  for (const l of terms) if (l > top) top = l
  return top + Math.log(terms.reduce((a, l) => a + Math.exp(l - top), 0))
}

/** The integral of exp(f) over a rectangle by the midpoint rule, in log: f takes the coordinates. */
function logIntegral(
  axes: readonly (readonly [number, number, number])[],
  f: (x: readonly number[]) => number,
): number {
  const terms: number[] = []
  const walk = (k: number, x: number[], logCell: number): void => {
    if (k === axes.length) {
      terms.push(f(x) + logCell)
      return
    }
    const [lo, hi, n] = axes[k] as readonly [number, number, number]
    const h = (hi - lo) / n
    for (let i = 0; i < n; i++) walk(k + 1, [...x, lo + (i + 0.5) * h], logCell + Math.log(h))
  }
  walk(0, [], 0)
  return logSum(terms)
}

const logNormalDensity = (x: number, m: number, v: number): number =>
  -0.5 * Math.log(2 * Math.PI * v) - (0.5 * (x - m) ** 2) / v

type Vec2 = readonly [number, number]
/** A 2 x 2 matrix, row-major. */
type Mat2 = readonly [number, number, number, number]

const invert2 = (v: Mat2): Mat2 => {
  const det = v[0] * v[3] - v[1] * v[2]
  return [v[3] / det, -v[1] / det, -v[2] / det, v[0] / det]
}

/** log N(x; m, V) of a pair, from the precision matrix of V. */
function logNormal2(x: Vec2, m: Vec2, p: Mat2): number {
  const dx = x[0] - m[0]
  const dy = x[1] - m[1]
  const det = p[0] * p[3] - p[1] * p[2]
  return -Math.log(2 * Math.PI) + 0.5 * Math.log(det) - 0.5 * (p[0] * dx * dx + 2 * p[1] * dx * dy + p[3] * dy * dy)
}

/** log of the unnormalised gamma cavity exp(eta . (log x, x)) at x = exp(t), in the coordinate t = log x (with its Jacobian). */
const logGammaInLog = (t: number, shape: number, rate: number): number => shape * t - rate * Math.exp(t)

describe('gaussian-gamma-precision: the log evidence', () => {
  it.each([
    [1, 0],
    [2.5, 0],
    [0.4, 0.9],
  ])('a latent x and a known mean, scale %d offset %d: the two-dimensional integral', (scale, offset) => {
    const [m, v, shape, rate, u] = [0.7, 1.3, 3, 1.5, 0.2]
    const args = [
      messageArg(gaussian, gaussianEta(m, v)),
      valueArg(gaussian, u),
      messageArg(gamma, gammaEta(shape, rate)),
    ]
    const brute = logIntegral(
      [
        [-9, 9, 360],
        [-14, 8, 360],
      ],
      ([x, t]) => {
        const p = scale * Math.exp(t as number) + offset
        return (
          logNormalDensity(x as number, m, v) +
          logNormalDensity(x as number, u, 1 / p) +
          logGammaInLog(t as number, shape, rate)
        )
      },
    )
    const want = massOf(args[0] as Arg) + brute
    expect(evidence(gaussianGammaPrecision, args, { scale, offset })).toBeCloseTo(want, 6)
  })

  it('both values latent: the three-dimensional integral', () => {
    const [mx, vx, mu, vu, shape, rate] = [0.5, 0.8, -0.3, 0.6, 2.2, 1.1]
    const args = [
      messageArg(gaussian, gaussianEta(mx, vx)),
      messageArg(gaussian, gaussianEta(mu, vu)),
      messageArg(gamma, gammaEta(shape, rate)),
    ]
    const brute = logIntegral(
      [
        [-7, 7, 100],
        [-7, 7, 100],
        [-12, 7, 160],
      ],
      ([x, u, t]) =>
        logNormalDensity(x as number, mx, vx) +
        logNormalDensity(u as number, mu, vu) +
        logNormalDensity(x as number, u as number, 1 / Math.exp(t as number)) +
        logGammaInLog(t as number, shape, rate),
    )
    const want = massOf(args[0] as Arg) + massOf(args[1] as Arg) + brute
    expect(evidence(gaussianGammaPrecision, args, {})).toBeCloseTo(want, 4)
  })

  it('an observed value and an observed mean: the gamma integral of the likelihood', () => {
    const args = [valueArg(gaussian, 1.4), valueArg(gaussian, 0.1), messageArg(gamma, gammaEta(2, 0.8))]
    const brute = logIntegral(
      [[-14, 8, 800]],
      ([t]) => logNormalDensity(1.4, 0.1, 1 / Math.exp(t as number)) + logGammaInLog(t as number, 2, 0.8),
    )
    expect(evidence(gaussianGammaPrecision, args, {})).toBeCloseTo(brute, 7)
  })

  it('a flat cavity of the value has no evidence yet', () => {
    const args = [messageArg(gaussian, gaussian.flat), valueArg(gaussian, 0), messageArg(gamma, gammaEta(2, 1))]
    expect(Number.isNaN(evidence(gaussianGammaPrecision, args, {}))).toBe(true)
  })
})

describe('the gamma hierarchies: the log evidence', () => {
  it('gamma-rate: the integral of the factor Gamma(x; a, c r) against both cavities', () => {
    const [a, c] = [0.7, 1.6]
    const [sx, bx, sr, br] = [2.5, 1.2, 3.1, 2.4]
    const args = [messageArg(gamma, gammaEta(sx, bx)), messageArg(gamma, gammaEta(sr, br))]
    const brute = logIntegral(
      [
        [-9, 5, 420],
        [-9, 5, 420],
      ],
      ([tx, tr]) => {
        const x = Math.exp(tx as number)
        const r = Math.exp(tr as number)
        const logFactor = a * Math.log(c * r) + (a - 1) * Math.log(x) - c * r * x - lgammaOf(a)
        return logGammaInLog(tx as number, sx, bx) + logGammaInLog(tr as number, sr, br) + logFactor
      },
    )
    expect(evidence(gammaRate, args, { shape: a, scale: c })).toBeCloseTo(brute, 5)
  })

  it('gamma-product: p = u tau, the integral over u and tau of the three cavities', () => {
    const [sp, bp, su, bu, st, bt] = [3, 2, 2.4, 1.1, 4, 3]
    const args = [
      messageArg(gamma, gammaEta(sp, bp)),
      messageArg(gamma, gammaEta(su, bu)),
      messageArg(gamma, gammaEta(st, bt)),
    ]
    // The point mass p = u tau integrates p out: the cavity of p at u tau, exp((sp - 1) log p - bp p).
    const brute = logIntegral(
      [
        [-9, 6, 420],
        [-9, 6, 420],
      ],
      ([tu, tt]) => {
        const u = Math.exp(tu as number)
        const tau = Math.exp(tt as number)
        const logP = (tu as number) + (tt as number)
        return (
          logGammaInLog(tu as number, su, bu) + logGammaInLog(tt as number, st, bt) + (sp - 1) * logP - bp * u * tau
        )
      },
    )
    expect(evidence(gammaProduct, args, {})).toBeCloseTo(brute, 5)
  })
})

describe('the vector and mixture sites: the log evidence', () => {
  it('gaussian-mv-precision-scale: the integral over the vector and the precision multiplier', () => {
    const v2 = gaussianMv(2)
    const m: Vec2 = [0.8, -0.4]
    const cov: Mat2 = [1.2, 0.5, 0.5, 0.9]
    const [shape, rate] = [2, 1.5]
    const scale: Vec2 = [1, 0.2]
    const offset: Vec2 = [0, 0.5]
    const prec = invert2(cov)
    const args = [messageArg(v2, gaussianMvEta(m, cov)), messageArg(gamma, gammaEta(shape, rate))]
    const brute = logIntegral(
      [
        [-5, 5, 90],
        [-5, 5, 90],
        [-9, 6, 120],
      ],
      ([x, y, t]) => {
        const lam = Math.exp(t as number)
        const p0 = scale[0] * lam + offset[0]
        const p1 = scale[1] * lam + offset[1]
        const logFactor =
          -Math.log(2 * Math.PI) + 0.5 * Math.log(p0 * p1) - 0.5 * (p0 * (x as number) ** 2 + p1 * (y as number) ** 2)
        return logNormal2([x as number, y as number], m, prec) + logFactor + logGammaInLog(t as number, shape, rate)
      },
    )
    const want = massOf(args[0] as Arg) + brute
    expect(evidence(gaussianMvPrecisionScale, args, { scale, offset })).toBeCloseTo(want, 3)
  })

  it('mixture-site: the integral of the Gaussian cavity against the mixture', () => {
    const [m, v] = [0.9, 1.4]
    const params = { weights: [0.3, 0.7], means: [0, 0.5], sds: [0.2, 3] }
    const args = [messageArg(gaussian, gaussianEta(m, v))]
    const brute = logIntegral([[-14, 14, 4000]], ([x]) => {
      const f =
        0.3 * Math.exp(logNormalDensity(x as number, 0, 0.04)) + 0.7 * Math.exp(logNormalDensity(x as number, 0.5, 9))
      return logNormalDensity(x as number, m, v) + Math.log(f)
    })
    expect(evidence(ops['mixture-site'] as FactorOperator, args, params)).toBeCloseTo(massOf(args[0] as Arg) + brute, 6)
  })

  it.each([
    ['laplace-site', { y: 0.4, scale: 0.8 }, (x: number) => -Math.log(1.6) - Math.abs(0.4 - x) / 0.8],
    [
      'student-t-site',
      { y: -0.3, scale: 0.7, df: 3 },
      (x: number) => {
        const [df, s, z] = [3, 0.7, (-0.3 - x) / 0.7]
        return (
          lgammaOf((df + 1) / 2) -
          lgammaOf(df / 2) -
          0.5 * Math.log(df * Math.PI) -
          Math.log(s) -
          ((df + 1) / 2) * Math.log(1 + (z * z) / df)
        )
      },
    ],
  ])('%s: the integral of the cavity against the density', (name, params, logDensity) => {
    const [m, v] = [0.2, 1.8]
    const args = [messageArg(gaussian, gaussianEta(m, v))]
    const brute = logIntegral([[-16, 16, 6400]], ([x]) => logNormalDensity(x as number, m, v) + logDensity(x as number))
    expect(evidence(ops[name] as FactorOperator, args, params)).toBeCloseTo(massOf(args[0] as Arg) + brute, 5)
  })

  it('student-t-row: a row on two weights is the site on the predictor', () => {
    const v2 = gaussianMv(2)
    const m: Vec2 = [0.5, -0.2]
    const cov: Mat2 = [1.1, 0.4, 0.4, 0.7]
    const a: Vec2 = [1, 2]
    const params = { a, y: 0.3, scale: 0.9, df: 4 }
    const prec = invert2(cov)
    const args = [messageArg(v2, gaussianMvEta(m, cov))]
    const brute = logIntegral(
      [
        [-6, 6, 220],
        [-6, 6, 220],
      ],
      ([x, y]) => {
        const z = (0.3 - (a[0] * (x as number) + a[1] * (y as number))) / 0.9
        const logT =
          lgammaOf(2.5) - lgammaOf(2) - 0.5 * Math.log(4 * Math.PI) - Math.log(0.9) - 2.5 * Math.log(1 + (z * z) / 4)
        return logNormal2([x as number, y as number], m, prec) + logT
      },
    )
    expect(evidence(ops['student-t-row'] as FactorOperator, args, params)).toBeCloseTo(
      massOf(args[0] as Arg) + brute,
      3,
    )
  })
})

describe('the numeric and the gate operators: the log evidence', () => {
  it('negative-binomial-dispersion: the integral over the predictor and the log dispersion', () => {
    const [mu, vu, ms, vs] = [1, 0.5, -1, 0.5]
    const params = { y: 3, exposure: 1.5 }
    const args = [messageArg(gaussian, gaussianEta(mu, vu)), messageArg(gaussian, gaussianEta(ms, vs))]
    const brute = logIntegral(
      [
        [-6, 8, 280],
        [-8, 6, 280],
      ],
      ([u, s]) => {
        const r = Math.exp(-(s as number))
        const logMu = Math.log(1.5) + (u as number)
        const logDen = Math.log(r + Math.exp(logMu))
        const logPmf = lgammaOf(3 + r) - lgammaOf(r) - Math.log(6) + r * Math.log(r) - (r + 3) * logDen + 3 * logMu
        return logNormalDensity(u as number, mu, vu) + logNormalDensity(s as number, ms, vs) + logPmf
      },
    )
    const want = massOf(args[0] as Arg) + massOf(args[1] as Arg) + brute
    expect(evidence(ops['negative-binomial-dispersion'] as FactorOperator, args, params)).toBeCloseTo(want, 4)
  })

  it('the gate: the mass of z and the two cavities times the mixture of the branches', () => {
    const y = 0.7
    const pz = [0.3, 0.7]
    const [ma, va, mb, vb] = [0.2, 0.9, 1.5, 1.4]
    const params = {
      arity: 2,
      branches: [
        { operator: 'gaussian', args: [{ const: y }, 0], params: { precision: 4 } },
        { operator: 'gaussian', args: [{ const: y }, 1], params: { precision: 1 } },
      ],
    }
    const args = [
      messageArg(discrete(2), discreteEta(pz)),
      messageArg(gaussian, gaussianEta(ma, va)),
      messageArg(gaussian, gaussianEta(mb, vb)),
    ]
    // Branch 0 reads a only (precision 4), branch 1 reads b only (precision 1): the integrals separate, but are done on the grid.
    const per = [0, 1].map((k) =>
      logIntegral(
        [
          [-7, 7, 700],
          [-7, 7, 700],
        ],
        ([a, b]) =>
          logNormalDensity(a as number, ma, va) +
          logNormalDensity(b as number, mb, vb) +
          (k === 0 ? logNormalDensity(y, a as number, 0.25) : logNormalDensity(y, b as number, 1)),
      ),
    )
    const want =
      massOf(args[0] as Arg) +
      massOf(args[1] as Arg) +
      massOf(args[2] as Arg) +
      logSum(pz.map((p, k) => Math.log(p) + (per[k] as number)))
    expect(evidence(ops['gate'] as FactorOperator, args, params)).toBeCloseTo(want, 5)
  })
})

describe('matmul-constant: the log evidence', () => {
  it('y = A w with Gaussian cavities of both: the two-dimensional integral over w of the cavity of y at A w', () => {
    const v2 = gaussianMv(2)
    const matrix: Mat2 = [1, 0.5, -0.3, 2]
    const my: Vec2 = [0.4, -0.6]
    const vy: Mat2 = [0.9, 0.2, 0.2, 0.7]
    const mw: Vec2 = [0.5, 0.1]
    const vw: Mat2 = [1.1, -0.3, -0.3, 0.8]
    const [py, pw] = [invert2(vy), invert2(vw)]
    const args = [messageArg(v2, gaussianMvEta(my, vy)), messageArg(v2, gaussianMvEta(mw, vw))]
    // The point mass puts y at A w; the integral over y is its cavity at A w, so the integral is over w only.
    const brute = logIntegral(
      [
        [-6, 6, 260],
        [-6, 6, 260],
      ],
      ([a, b]) => {
        const w: Vec2 = [a as number, b as number]
        const image: Vec2 = [matrix[0] * w[0] + matrix[1] * w[1], matrix[2] * w[0] + matrix[3] * w[1]]
        return logNormal2(image, my, py) + logNormal2(w, mw, pw)
      },
    )
    const want = massOf(args[0] as Arg) + massOf(args[1] as Arg) + brute
    expect(evidence(ops['matmul-constant'] as FactorOperator, args, { matrix })).toBeCloseTo(want, 4)
  })
})

describe('the registry: every operator with an expectation propagation rule has a log evidence', () => {
  it('lists the ones that have none, and expects none', () => {
    const missing = Object.values(ops)
      .filter((op) => op.support.ep !== undefined && op.logEvidence === undefined)
      .map((op) => op.name)
    expect(missing).toEqual([])
  })
})
