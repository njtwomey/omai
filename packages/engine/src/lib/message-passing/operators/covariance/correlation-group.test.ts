import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { beta, betaEta, gamma, gammaEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, OperatorContext } from '../../core/types.ts'
import { choleskyInPlace, choleskyLogDet, choleskySolveInPlace } from '../../numerics/_bartlett.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { mvnSdCorr } from './correlation.ts'
import { mvnSdCorrGroup } from './correlation-group.ts'

const G = gaussianMv(2)
const cav = { a1: 3, b1: 2.5, a2: 2.5, b2: 2, p: 2, q: 2 }
const meanCavity = { m: [0.1, 0.2], v: [0.6, 0.1, 0.1, 0.5] }
const vectorCavities = [
  { m: [1.1, 0.6], v: [0.5, 0.2, 0.2, 0.4] },
  { m: [-0.4, -0.5], v: [0.4, 0, 0, 0.7] },
  { m: [0.9, 1.3], v: [0.6, 0.15, 0.15, 0.5] },
]

const gaussianArg = (c: { m: readonly number[]; v: readonly number[] }): Arg =>
  messageArg(G, gaussianMvEta([...c.m], [...c.v]))

const ctx = (to: number, args: readonly Arg[], groups: number, params: object = {}): OperatorContext => ({
  mode: 'ep',
  to,
  args,
  params: { groups, ...params },
  power: 1,
})

/** The group's arguments: `groups` vectors, the mean, the two precisions and the correlation's beta. */
const argsOf = (groups: number): Arg[] => [
  ...vectorCavities.slice(0, groups).map(gaussianArg),
  gaussianArg(meanCavity),
  messageArg(gamma, gammaEta(cav.a1, cav.b1)),
  messageArg(gamma, gammaEta(cav.a2, cav.b2)),
  messageArg(beta, betaEta(cav.p, cav.q)),
]

/** The inverse of a symmetric 2 by 2 matrix given as four entries. */
const inv2 = (v: readonly number[]): number[] => {
  const det = (v[0] as number) * (v[3] as number) - (v[1] as number) * (v[2] as number)
  return [(v[3] as number) / det, -(v[1] as number) / det, -(v[2] as number) / det, (v[0] as number) / det]
}

/**
 * The brute force: for each node of a midpoint grid over `(t1, t2, u)` the integral over the vectors and the mean is a dense Gaussian integral of
 * the joint precision matrix of all `2 (J + 1)` coordinates (no algebra of the factor's closed form): its log value, and the mean and covariance
 * of every Gaussian given the node. The node weights then give the moments of the parameters and the mixtures of the Gaussians.
 */
function brute(groups: number, N: number) {
  const dim = 2 * (groups + 1)
  const box = [
    [0.02, 9],
    [0.02, 9],
    [0.002, 0.998],
  ] as const
  const steps = box.map((b) => (b[1] - b[0]) / N)
  const logGamma = (x: number, a: number, b: number): number =>
    a * Math.log(b) - lgammaOf(a) + (a - 1) * Math.log(x) - b * x
  const logBeta = (u: number): number =>
    lgammaOf(cav.p + cav.q) -
    lgammaOf(cav.p) -
    lgammaOf(cav.q) +
    (cav.p - 1) * Math.log(u) +
    (cav.q - 1) * Math.log(1 - u)
  // the cavities of the Gaussians as natural parameters: h and W
  const sides = [...vectorCavities.slice(0, groups), meanCavity].map((c) => {
    const w = inv2(c.v)
    const h = [
      (w[0] as number) * (c.m[0] as number) + (w[1] as number) * (c.m[1] as number),
      (w[2] as number) * (c.m[0] as number) + (w[3] as number) * (c.m[1] as number),
    ]
    return { w, h }
  })
  const nodes: { w: number; t1: number; t2: number; u: number; mean: number[]; cov: number[][] }[] = []
  const P = new Float64Array(dim * dim)
  const work = new Float64Array(dim * dim)
  const sol = new Float64Array(dim)
  let top = Number.NEGATIVE_INFINITY
  const logs: number[] = []
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      for (let k = 0; k < N; k++) {
        const t1 = box[0][0] + (i + 0.5) * (steps[0] as number)
        const t2 = box[1][0] + (j + 0.5) * (steps[1] as number)
        const u = box[2][0] + (k + 0.5) * (steps[2] as number)
        const rho = 2 * u - 1
        const [s1, s2] = [1 / Math.sqrt(t1), 1 / Math.sqrt(t2)]
        const sigma = [s1 * s1, rho * s1 * s2, rho * s1 * s2, s2 * s2]
        const lam = inv2(sigma)
        const logDetLam = Math.log(1 / ((sigma[0] as number) * (sigma[3] as number) - (sigma[1] as number) ** 2))
        P.fill(0)
        const add = (r: number, c: number, m: readonly number[], sign: number): void => {
          for (let a = 0; a < 2; a++)
            for (let b = 0; b < 2; b++)
              P[(2 * r + a) * dim + 2 * c + b] =
                (P[(2 * r + a) * dim + 2 * c + b] as number) + sign * (m[a * 2 + b] as number)
        }
        for (let g = 0; g < groups; g++) {
          add(g, g, (sides[g] as { w: number[] }).w, 1)
          add(g, g, lam, 1)
          add(groups, groups, lam, 1)
          add(g, groups, lam, -1)
          add(groups, g, lam, -1)
        }
        add(groups, groups, (sides[groups] as { w: number[] }).w, 1)
        const h = new Float64Array(dim)
        for (let g = 0; g <= groups; g++) {
          h[2 * g] = (sides[g] as { h: number[] }).h[0] as number
          h[2 * g + 1] = (sides[g] as { h: number[] }).h[1] as number
        }
        work.set(P)
        if (!choleskyInPlace(work, dim)) continue
        sol.set(h)
        choleskySolveInPlace(work, dim, sol)
        let quad = 0
        for (let a = 0; a < dim; a++) quad += (h[a] as number) * (sol[a] as number)
        const logI =
          groups * (0.5 * logDetLam - Math.log(2 * Math.PI)) +
          0.5 * (dim * Math.log(2 * Math.PI) - choleskyLogDet(work, dim) + quad)
        const lw = logGamma(t1, cav.a1, cav.b1) + logGamma(t2, cav.a2, cav.b2) + logBeta(u) + logI
        const cov: number[][] = []
        for (let a = 0; a < dim; a++) {
          const e = new Float64Array(dim)
          e[a] = 1
          choleskySolveInPlace(work, dim, e)
          cov.push(Array.from(e))
        }
        nodes.push({ w: lw, t1, t2, u, mean: Array.from(sol), cov })
        logs.push(lw)
        if (lw > top) top = lw
      }
  const cell = (steps[0] as number) * (steps[1] as number) * (steps[2] as number)
  let total = 0
  for (const n of nodes) {
    n.w = Math.exp(n.w - top)
    total += n.w
  }
  const acc = { lt1: 0, t1: 0, lt2: 0, t2: 0, u: 0, u2: 0 }
  const mean = new Float64Array(dim)
  const second = new Float64Array(dim * dim)
  for (const n of nodes) {
    const w = n.w / total
    acc.lt1 += w * Math.log(n.t1)
    acc.t1 += w * n.t1
    acc.lt2 += w * Math.log(n.t2)
    acc.t2 += w * n.t2
    acc.u += w * n.u
    acc.u2 += w * n.u * n.u
    for (let a = 0; a < dim; a++) {
      mean[a] = (mean[a] as number) + w * (n.mean[a] as number)
      for (let b = 0; b < dim; b++)
        second[a * dim + b] =
          (second[a * dim + b] as number) +
          w * ((n.cov[a] as number[])[b] as number) +
          w * (n.mean[a] as number) * (n.mean[b] as number)
    }
  }
  const covOf = (g: number): number[] =>
    [0, 1, 2, 3].map(
      (e) =>
        (second[(2 * g + Math.floor(e / 2)) * dim + 2 * g + (e % 2)] as number) -
        (mean[2 * g + Math.floor(e / 2)] as number) * (mean[2 * g + (e % 2)] as number),
    )
  return {
    logMass: top + Math.log(total * cell),
    ...acc,
    uVar: acc.u2 - acc.u ** 2,
    means: Array.from({ length: groups + 1 }, (_, g) => [mean[2 * g] as number, mean[2 * g + 1] as number]),
    covs: Array.from({ length: groups + 1 }, (_, g) => covOf(g)),
  }
}

/** The belief a port ends with: the cavity times the message. */
const belief = (cavity: Eta, message: Eta): { mean: number[]; cov: number[] } => {
  const s = G.summary(G.product(cavity, message)) as unknown as { mean: number[]; covariance: number[][] }
  return { mean: s.mean, cov: s.covariance.flat() }
}

describe('mvn-sd-corr-group', () => {
  it('one vector is mvn-sd-corr: the same messages to the parameters, the vector and the mean, and the same evidence', () => {
    const a = argsOf(1)
    const single = [a[0], a[1], a[2], a[3], a[4]] as Arg[]
    for (const to of [2, 3, 4, 0, 1]) {
      const g = mvnSdCorrGroup.ep?.(ctx(to, a, 1)).message as number[]
      const s = mvnSdCorr.ep?.({ ...ctx(to, single, 1), params: {} }).message as number[]
      // the group's ports are [x0, mean, t1, t2, rho]: the same order as the single factor's
      for (let i = 0; i < g.length; i++) expect(g[i]).toBeCloseTo(s[i] as number, 9)
    }
    const evGroup = mvnSdCorrGroup.logEvidence?.(ctx(2, a, 1)) as number
    const evSingle = mvnSdCorr.logEvidence?.({ ...ctx(2, single, 1), params: {} }) as number
    expect(evGroup).toBeCloseTo(evSingle, 9)
  })

  for (const groups of [2, 3]) {
    it(`${groups} vectors and a random mean: the messages match a brute-force grid with the dense joint Gaussian at every node`, () => {
      const a = argsOf(groups)
      const b = brute(groups, groups === 2 ? 56 : 40)
      // the gammas match (E log t, E t) and the beta the mean and variance of u
      const t1 = gamma.moments(
        gamma.product(gammaEta(cav.a1, cav.b1), mvnSdCorrGroup.ep?.(ctx(groups + 1, a, groups)).message as number[]),
      )
      const t2 = gamma.moments(
        gamma.product(gammaEta(cav.a2, cav.b2), mvnSdCorrGroup.ep?.(ctx(groups + 2, a, groups)).message as number[]),
      )
      // (the brute-force grid is a coarse midpoint rule over the precisions themselves: 1% on their means)
      expect(t1[0]).toBeCloseTo(b.lt1, 2)
      expect(Math.abs((t1[1] as number) / b.t1 - 1)).toBeLessThan(0.01)
      expect(t2[0]).toBeCloseTo(b.lt2, 2)
      expect(Math.abs((t2[1] as number) / b.t2 - 1)).toBeLessThan(0.01)
      const toRho = mvnSdCorrGroup.ep?.(ctx(groups + 3, a, groups)).message as number[]
      const pp = (betaEta(cav.p, cav.q)[0] as number) + (toRho[0] as number) + 1
      const qq = (betaEta(cav.p, cav.q)[1] as number) + (toRho[1] as number) + 1
      expect(pp / (pp + qq)).toBeCloseTo(b.u, 2)
      expect(Math.sqrt((pp * qq) / ((pp + qq) ** 2 * (pp + qq + 1)))).toBeCloseTo(Math.sqrt(b.uVar), 2)
      // every vector and the mean match the mixture of the conditionals over the grid
      for (let g = 0; g <= groups; g++) {
        const cavityEta = (a[g] as Arg).eta as Eta
        const msg = mvnSdCorrGroup.ep?.(ctx(g, a, groups)).message as number[]
        const s = belief(cavityEta, msg)
        expect(s.mean[0]).toBeCloseTo((b.means[g] as number[])[0] as number, 2)
        expect(s.mean[1]).toBeCloseTo((b.means[g] as number[])[1] as number, 2)
        for (let e = 0; e < 4; e++) expect(s.cov[e]).toBeCloseTo((b.covs[g] as number[])[e] as number, 2)
      }
      const massOf =
        gamma.logPartition(gammaEta(cav.a1, cav.b1)) +
        gamma.logPartition(gammaEta(cav.a2, cav.b2)) +
        beta.logPartition(betaEta(cav.p, cav.q))
      expect(mvnSdCorrGroup.logEvidence?.(ctx(0, a, groups)) as number).toBeCloseTo(b.logMass + massOf, 2)
    })
  }

  it('observed vectors and known precisions are single nodes of their axes; a flat mean cavity gives a proper message', () => {
    const a: Arg[] = [
      valueArg(G, [1.1, 0.6]),
      valueArg(G, [-0.4, -0.5]),
      messageArg(G, G.flat as Eta),
      valueArg(gamma, 1.5),
      valueArg(gamma, 2),
      messageArg(beta, betaEta(2, 2)),
    ]
    const toMean = mvnSdCorrGroup.ep?.(ctx(2, a, 2)).message as number[]
    expect(G.isProper(toMean)).toBe(true)
    // the data pull the mean to their average
    expect((G.summary(toMean) as unknown as { mean: number[] }).mean[0]).toBeCloseTo(0.35, 1)
    const toRho = mvnSdCorrGroup.ep?.(ctx(5, a, 2)).message as number[]
    expect(toRho.length).toBe(2)
  })

  it('declares its ports from the number of vectors, checks them, and has no variational rule', () => {
    const ports =
      typeof mvnSdCorrGroup.ports === 'function' ? mvnSdCorrGroup.ports({ groups: 3 }) : mvnSdCorrGroup.ports
    expect(ports.map((p) => p.name)).toEqual(['x0', 'x1', 'x2', 'mean', 'precision 1', 'precision 2', 'correlation'])
    expect(mvnSdCorrGroup.check?.({ groups: 2 }, [G, G, G, gamma, gamma, beta])).toBeUndefined()
    expect(mvnSdCorrGroup.check?.({ groups: 2 }, [G, G, gamma, gamma, beta])).toMatch(/one port each/)
    expect(mvnSdCorrGroup.check?.({ groups: 1 }, [gaussianMv(3), G, gamma, gamma, beta])).toMatch(/two entries/)
    expect(mvnSdCorrGroup.noRule?.vmp).toMatch(/closed form/)
  })
})
