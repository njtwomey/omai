import { describe, expect, it } from 'vitest'
import { ndtr } from '../../../numerics/special/_kernels.ts'
import { massOf, messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import type { Arg, Params } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
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

/** The tilted natural parameters of w (and the log mass) by a grid over the coordinates of w's cavity. */
function brute(
  m: readonly number[],
  v: readonly number[],
  matrix: readonly number[],
  offset: readonly number[],
  logf: (z: readonly number[]) => number,
  n: number,
): { logMass: number; mean: number[]; covariance: number[] } {
  const d = m.length
  const k = offset.length
  const l = cholesky(v, d)
  const span = 6
  const h = (2 * span) / n
  const u = new Array<number>(d).fill(0)
  let mass = 0
  const m1 = new Array<number>(d).fill(0)
  const m2 = new Array<number>(d * d).fill(0)
  const w = new Array<number>(d).fill(0)
  const z = new Array<number>(k).fill(0)
  const walk = (i: number, logw: number): void => {
    if (i === d) {
      for (let a = 0; a < d; a++) {
        let s = m[a] as number
        for (let b = 0; b <= a; b++) s += (l[a * d + b] as number) * (u[b] as number)
        w[a] = s
      }
      for (let a = 0; a < k; a++) {
        let s = offset[a] as number
        for (let b = 0; b < d; b++) s += (matrix[a * d + b] as number) * (w[b] as number)
        z[a] = s
      }
      const p = Math.exp(logw + logf(z))
      mass += p
      for (let a = 0; a < d; a++) {
        m1[a] = (m1[a] as number) + p * (w[a] as number)
        for (let b = 0; b < d; b++) m2[a * d + b] = (m2[a * d + b] as number) + p * (w[a] as number) * (w[b] as number)
      }
      return
    }
    for (let j = 0; j < n; j++) {
      u[i] = -span + (j + 0.5) * h
      walk(i + 1, logw - 0.5 * (u[i] as number) ** 2 - 0.5 * Math.log(2 * Math.PI) + Math.log(h))
    }
  }
  walk(0, 0)
  const mean = m1.map((x) => x / mass)
  const covariance = m2.map((x, i) => x / mass - (mean[Math.floor(i / d)] as number) * (mean[i % d] as number))
  return { logMass: Math.log(mass), mean, covariance }
}

/** Lower Cholesky factor, row-major. */
function cholesky(a: readonly number[], n: number): number[] {
  const l = new Array<number>(n * n).fill(0)
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j] as number
      for (let k = 0; k < j; k++) s -= (l[i * n + k] as number) * (l[j * n + k] as number)
      l[i * n + j] = i === j ? Math.sqrt(s) : s / (l[j * n + j] as number)
    }
  return l
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
  const b = brute(
    c.mean,
    c.covariance,
    c.matrix,
    offset,
    (z) => power * log(z),
    c.grid ?? (d === 3 ? 44 : d === 2 ? 150 : 600),
  )
  const cavity = args[0]?.eta as number[]
  // The tilted natural parameters of w, and the site as their quotient by the cavity, over the power.
  const precision = spd(b.covariance, d)?.inverse() as Float64Array
  const hTilt: number[] = []
  for (let i = 0; i < d; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += (precision[i * d + j] as number) * (b.mean[j] as number)
    hTilt.push(s)
  }
  const want = [...hTilt, ...precision].map((x, i) => (x - (cavity[i] as number)) / power)
  const got = multiclassRow.ep?.({ mode: 'ep', to: 0, args, params, power }).message as readonly number[]
  const tolerance = c.tolerance ?? 3e-3
  for (let i = 0; i < want.length; i++)
    expect(
      Math.abs((got[i] as number) - (want[i] as number)),
      `message entry ${i}: ${got[i]} against ${want[i]}`,
    ).toBeLessThan(tolerance * (1 + Math.abs(want[i] as number)))
  const evidence = multiclassRow.logEvidence?.({ mode: 'ep', to: 0, args, params, power }) as number
  const wantEvidence = b.logMass + massOf(args[0] as Arg)
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
