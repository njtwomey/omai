import { describe, expect, it } from 'vitest'
import { ndtr } from '../../numerics/special/_kernels.ts'
import { gaussian, gaussianEta } from '../families.ts'
import type { Arg } from '../types.ts'
import { massOf, messageArg, valueArg } from './args.ts'
import { multiclassSite } from './multiclass.ts'
import { defaultOperators } from './registry.ts'

/*
 * The tilted means and variances of the class scores and the log evidence against a brute-force integral over a fine grid of the
 * scores, with no code of the operator in it.
 */

const logSoftmax = (z: readonly number[], y: number): number => {
  const top = Math.max(...z)
  return (z[y] as number) - top - Math.log(z.reduce((a, v) => a + Math.exp(v - top), 0))
}

/** The multinomial probit probability by a fine one-dimensional integral over the noise of the chosen class. */
const logProbit = (z: readonly number[], y: number): number => {
  const n = 160
  let total = 0
  for (let i = 0; i < n; i++) {
    const e = -8 + ((i + 0.5) * 16) / n
    let p = Math.exp(-0.5 * e * e - 0.5 * Math.log(2 * Math.PI))
    for (let j = 0; j < z.length; j++) if (j !== y) p *= ndtr((z[y] as number) - (z[j] as number) + e)
    total += (p * 16) / n
  }
  return Math.log(total)
}

interface Cavity {
  readonly m: number
  readonly v: number
}

/** The brute force over a grid of K = 3 scores (or fewer free ones: a null cavity is an observed value). */
function brute(
  cavities: readonly Cavity[],
  logf: (z: readonly number[]) => number,
  n = 70,
  span = 6,
): { logMass: number; mean: number[]; variance: number[] } {
  const axes = cavities.map((c) => {
    const sd = Math.sqrt(c.v)
    return Array.from({ length: n }, (_, i) => c.m + sd * (-span + ((i + 0.5) * 2 * span) / n))
  })
  const steps = cavities.map((c) => (Math.sqrt(c.v) * 2 * span) / n)
  const k = cavities.length
  let mass = 0
  const m1 = new Array<number>(k).fill(0)
  const m2 = new Array<number>(k).fill(0)
  const z = new Array<number>(k).fill(0)
  const walk = (d: number, logw: number): void => {
    if (d === k) {
      const w = Math.exp(logw + logf(z))
      mass += w
      for (let i = 0; i < k; i++) {
        m1[i] = (m1[i] as number) + w * (z[i] as number)
        m2[i] = (m2[i] as number) + w * (z[i] as number) ** 2
      }
      return
    }
    const c = cavities[d] as Cavity
    for (let i = 0; i < n; i++) {
      const x = (axes[d] as number[])[i] as number
      z[d] = x
      walk(
        d + 1,
        logw - (0.5 * (x - c.m) ** 2) / c.v - 0.5 * Math.log(2 * Math.PI * c.v) + Math.log(steps[d] as number),
      )
    }
  }
  walk(0, 0)
  const mean = m1.map((v) => v / mass)
  return { logMass: Math.log(mass), mean, variance: m2.map((v, i) => v / mass - (mean[i] as number) ** 2) }
}

function check(link: 'logit' | 'probit', y: number, cavities: readonly Cavity[], tolerance = 2e-3): void {
  const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
  const params = { y, classes: cavities.length, link }
  const logf = link === 'logit' ? (z: readonly number[]) => logSoftmax(z, y) : (z: readonly number[]) => logProbit(z, y)
  const b = brute(cavities, logf, cavities.length === 3 ? (link === 'probit' ? 26 : 60) : 200)
  for (let k = 0; k < cavities.length; k++) {
    const got = multiclassSite.ep?.({ mode: 'ep', to: k, args, params, power: 1 }).message as readonly number[]
    const c = cavities[k] as Cavity
    // The site is the tilted Gaussian over the cavity in natural parameters.
    const wantPrecision = 1 / (b.variance[k] as number) - 1 / c.v
    const wantH = (b.mean[k] as number) / (b.variance[k] as number) - c.m / c.v
    expect(Math.abs((got[1] as number) - wantPrecision), `${link} class ${k} precision`).toBeLessThan(
      tolerance * (1 + Math.abs(wantPrecision)),
    )
    expect(Math.abs((got[0] as number) - wantH), `${link} class ${k} mean term`).toBeLessThan(
      tolerance * (1 + Math.abs(wantH)),
    )
  }
  const want = b.logMass + args.reduce((s, a) => s + massOf(a), 0)
  const got = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
  expect(Math.abs(got - want), `${link} evidence ${got} against ${want}`).toBeLessThan(tolerance)
}

describe('multiclass-site: tilted moments and log evidence against a brute-force integral', () => {
  it('softmax, three classes, moderate cavities', () => {
    check('logit', 1, [
      { m: 0.3, v: 1 },
      { m: -0.2, v: 0.8 },
      { m: 0.1, v: 1.5 },
    ])
  })
  it('softmax, three classes, wide cavities (the first sweeps of a run)', () => {
    check(
      'logit',
      2,
      [
        { m: 0, v: 9 },
        { m: 0, v: 9 },
        { m: 0, v: 9 },
      ],
      8e-3,
    )
  })
  it('softmax, two classes is the logistic link', () => {
    check('logit', 0, [
      { m: 0.4, v: 1.2 },
      { m: -0.3, v: 0.6 },
    ])
  })
  it('multinomial probit, three classes', () => {
    check(
      'probit',
      0,
      [
        { m: 0.2, v: 1 },
        { m: -0.1, v: 0.7 },
        { m: 0.4, v: 1.1 },
      ],
      1.5e-2,
    )
  }, 30_000)
  it('multinomial probit with two classes is the binary probit of the difference over sqrt 2', () => {
    // P(y = 0 | z) = Phi((z0 - z1) / sqrt 2): the tilted mean of z0 - z1 has the closed form of the probit site on d with sd sqrt 2.
    check(
      'probit',
      0,
      [
        { m: 0.5, v: 1 },
        { m: -0.2, v: 0.8 },
      ],
      4e-3,
    )
  }, 30_000)
  it('an observed score is a constant coordinate', () => {
    const args = [
      messageArg(gaussian, gaussianEta(0.2, 1)),
      valueArg(gaussian, 0.5),
      messageArg(gaussian, gaussianEta(-0.3, 0.9)),
    ]
    const params = { y: 1, classes: 3 }
    const b = brute(
      [
        { m: 0.2, v: 1 },
        { m: -0.3, v: 0.9 },
      ],
      ([a, c]) => logSoftmax([a as number, 0.5, c as number], 1),
      200,
    )
    const want = b.logMass + massOf(args[0] as Arg) + massOf(args[2] as Arg)
    expect(multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 })).toBeCloseTo(want, 3)
  })
  it('is in the registry with its evidence, and refuses a bad class', () => {
    expect(defaultOperators()['multiclass-site']?.logEvidence).toBeDefined()
    expect(multiclassSite.check?.({ y: 3, classes: 3 }, [])).toMatch(/class must be/)
    expect(multiclassSite.check?.({ y: 1, classes: 3, link: 'cloglog' }, [])).toMatch(/link/)
  })
})
