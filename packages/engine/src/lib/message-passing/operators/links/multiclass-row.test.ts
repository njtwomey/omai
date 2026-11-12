import { describe, expect, it } from 'vitest'
import { bruteVectorSite } from '../../../../testing/vector-site.ts'
import { ndtr } from '../../../numerics/special/_kernels.ts'
import { massOf, messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import type { Arg, Params } from '../../core/types.ts'
import { defaultOperators } from '../registry.ts'
import { multiclassSite } from './multiclass.ts'
import { multiclassRow } from './multiclass-row.ts'

/*
 * The message of the multiclass row (and its evidence) against a brute-force integral over the Gaussian vector itself: a fine grid of
 * the standardised coordinates of w, the likelihood of the scores A w + b at each node, the tilted mean and covariance of w, and the
 * site as the quotient of the tilted natural parameters by the cavity's. Nothing of the operator's own factorisation is used.
 */

const logSoftmax = (z: readonly number[], y: number): number => {
  const top = Math.max(...z)
  return (z[y] as number) - top - Math.log(z.reduce((a, v) => a + Math.exp(v - top), 0))
}

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

interface Case {
  readonly link: 'logit' | 'probit'
  readonly y: number
  readonly mean: readonly number[]
  readonly covariance: readonly number[]
  readonly matrix: readonly number[]
  readonly offset?: readonly number[]
  readonly power?: number
  readonly grid?: number
  readonly tolerance?: number
}

function check(c: Case): void {
  const d = c.mean.length
  const classes = c.matrix.length / d
  const offset = c.offset ?? new Array<number>(classes).fill(0)
  const power = c.power ?? 1
  const w = gaussianMv(d)
  const args: Arg[] = [messageArg(w, gaussianMvEta(c.mean, c.covariance))]
  const params: Params = {
    y: c.y,
    classes,
    link: c.link,
    matrix: [...c.matrix],
    ...(c.offset === undefined ? {} : { offset: [...c.offset] }),
  }
  const log =
    c.link === 'logit' ? (z: readonly number[]) => logSoftmax(z, c.y) : (z: readonly number[]) => logProbit(z, c.y)
  const brute = bruteVectorSite(
    { mean: c.mean, covariance: c.covariance, eta: args[0]?.eta as number[] },
    c.matrix,
    offset,
    log,
    power,
    c.grid ?? (d === 3 ? 44 : d === 2 ? 150 : 600),
  )
  const want = brute.message
  const got = multiclassRow.ep?.({ mode: 'ep', to: 0, args, params, power }).message as readonly number[]
  const tolerance = c.tolerance ?? 3e-3
  for (let i = 0; i < want.length; i++)
    expect(
      Math.abs((got[i] as number) - (want[i] as number)),
      `message entry ${i}: ${got[i]} against ${want[i]}`,
    ).toBeLessThan(tolerance * (1 + Math.abs(want[i] as number)))
  const evidence = multiclassRow.logEvidence?.({ mode: 'ep', to: 0, args, params, power }) as number
  const wantEvidence = brute.logMass + massOf(args[0] as Arg)
  expect(Math.abs(evidence - wantEvidence), `evidence ${evidence} against ${wantEvidence}`).toBeLessThan(tolerance)
}

describe('multiclass-row: the message to the vector and the evidence against a brute-force integral', () => {
  const v3 = [1, 0.4, -0.2, 0.4, 0.8, 0.3, -0.2, 0.3, 1.2]
  it('softmax of three class slopes with a correlated cavity: scores x w_k', () => {
    check({ link: 'logit', y: 1, mean: [0.3, -0.2, 0.1], covariance: v3, matrix: [0.9, 0, 0, 0, 0.9, 0, 0, 0, 0.9] })
  })
  it('softmax, three classes, wide cavity (the first sweeps)', () => {
    check({
      link: 'logit',
      y: 2,
      mean: [0, 0, 0],
      covariance: [9, 1, 0, 1, 9, 0.5, 0, 0.5, 9],
      matrix: [-1.4, 0, 0, 0, -1.4, 0, 0, 0, -1.4],
      grid: 70,
      tolerance: 6e-3,
    })
  })
  it('softmax of eight alternatives with two shared coefficients: a rank-two integral', () => {
    const matrix = [-1.2, 0.3, 0.5, 1.1, -0.4, -0.9, 1.3, 0.2, -0.7, 0.8, 0.1, -1.4, 0.9, 1.0, -0.3, 0.6]
    check({ link: 'logit', y: 3, mean: [-0.5, 0.4], covariance: [0.8, -0.3, -0.3, 0.6], matrix })
  })
  it('multinomial probit of three scores with an offset', () => {
    check({
      link: 'probit',
      y: 0,
      mean: [0.2, -0.1, 0.4],
      covariance: v3,
      matrix: [0.8, 0, 0, 0, 0.8, 0, 0, 0, 0.8],
      offset: [0.1, -0.2, 0.3],
      grid: 36,
      tolerance: 6e-3,
    })
  })
  it('a score the vector does not move (a zero row of the matrix) lowers the rank: a known class', () => {
    check({
      link: 'logit',
      y: 0,
      mean: [0.3, -0.2, 0.1],
      covariance: v3,
      matrix: [0, 0, 0, 0, 0.9, 0, 0, 0, 0.9],
      offset: [0.2, 0, 0],
    })
  })
  it('the power divides the message: fractional EP', () => {
    check({
      link: 'logit',
      y: 1,
      mean: [0.3, -0.2, 0.1],
      covariance: v3,
      matrix: [0.9, 0, 0, 0, 0.9, 0, 0, 0, 0.9],
      power: 0.5,
    })
  })
  it('agrees with the scalar site on the tilted marginals of independent scores, and adds the correlation between them', () => {
    // Independent cavities and one score per coordinate: the tilted mean and variance of each score are the same under both sites.
    const diag = [1, 0, 0, 0, 0.8, 0, 0, 0, 1.2]
    const mean = [0.3, -0.2, 0.1]
    const args: Arg[] = [messageArg(gaussianMv(3), gaussianMvEta(mean, diag))]
    const eye = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    const joint = multiclassRow.ep?.({ mode: 'ep', to: 0, args, params: { y: 1, classes: 3, matrix: eye }, power: 1 })
      .message as number[]
    const cavity = args[0]?.eta as number[]
    const tiltedEta = cavity.map((x, i) => x + (joint[i] as number))
    const { mean: mt, covariance: ct } = gaussianMvMoments(tiltedEta, 3)
    const scalars = [0, 1, 2].map((k) => messageArg(gaussian, gaussianEta(mean[k] as number, diag[k * 4] as number)))
    for (let k = 0; k < 3; k++) {
      const s = multiclassSite.ep?.({ mode: 'ep', to: k, args: scalars, params: { y: 1, classes: 3 }, power: 1 })
        .message as number[]
      const cav = scalars[k]?.eta as number[]
      const precision = (cav[1] as number) + (s[1] as number)
      expect(Math.abs((mt[k] as number) - ((cav[0] as number) + (s[0] as number)) / precision)).toBeLessThan(1e-5)
      expect(Math.abs((ct[k * 4] as number) - 1 / precision)).toBeLessThan(1e-5)
    }
    // The joint site has the off-diagonal terms the scalar site cannot: the labels tie the three scores together.
    expect(Math.abs(ct[1] as number)).toBeGreaterThan(1e-3)
    expect(Math.abs(joint[3 + 1] as number)).toBeGreaterThan(1e-3)
  })
  it('is in the registry, takes a vector port and refuses what the product grid cannot reach', () => {
    const op = defaultOperators()['multiclass-row']
    expect(op).toBe(multiclassRow)
    const reason = multiclassRow.check?.({ y: 0, classes: 8, matrix: new Array<number>(8 * 8).fill(0) }, [
      gaussianMv(8),
    ])
    expect(reason).toMatch(/product grid/)
    expect(multiclassRow.check?.({ y: 0, classes: 3, matrix: [1, 0] }, [gaussianMv(3)])).toMatch(/matrix/)
    expect(multiclassRow.noRule?.vmp).toBeDefined()
  })
})
