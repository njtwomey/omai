// dirichlet-concentration: the tilted moments of the concentration and of E log p against a brute-force double integral.
import { describe, expect, it } from 'vitest'
import { lgammaOf } from '../_scalar.ts'
import { beta, gamma, gammaEta, gaussian, gaussianEta } from '../families.ts'
import { messageArg } from './args.ts'
import { dirichletConcentration } from './concentration.ts'

/** The reference: the joint density of (p, alpha) on a fine grid, p ~ Beta(w1 a + o1, w2 a + o2) times the cavities. */
function brute(
  pEta: readonly number[],
  logCavityAlpha: (a: number) => number,
  w: readonly number[],
  o: readonly number[],
): { z: number; ea: number; elogA: number; elogP: readonly number[] } {
  const NA = 1500
  const NP = 3000
  const aLo = 1e-4
  const aHi = 60
  let z = 0
  let ea = 0
  let elogA = 0
  let l0 = 0
  let l1 = 0
  for (let i = 0; i < NA; i++) {
    const a = aLo + ((i + 0.5) * (aHi - aLo)) / NA
    const k0 = (w[0] as number) * a + (o[0] as number)
    const k1 = (w[1] as number) * a + (o[1] as number)
    const norm = lgammaOf(k0 + k1) - lgammaOf(k0) - lgammaOf(k1)
    // The p integral by the midpoint rule (the integrand is a smooth Beta kernel for these parameters).
    let sp = 0
    let s0 = 0
    let s1 = 0
    for (let j = 0; j < NP; j++) {
      const p = (j + 0.5) / NP
      const lp = Math.log(p)
      const lq = Math.log(1 - p)
      const dens = Math.exp(norm + (k0 - 1 + (pEta[0] as number)) * lp + (k1 - 1 + (pEta[1] as number)) * lq)
      sp += dens
      s0 += dens * lp
      s1 += dens * lq
    }
    sp /= NP
    s0 /= NP
    s1 /= NP
    const wa = Math.exp(logCavityAlpha(a)) * ((aHi - aLo) / NA)
    z += wa * sp
    ea += wa * sp * a
    elogA += wa * sp * Math.log(a)
    l0 += wa * s0
    l1 += wa * s1
  }
  return { z, ea: ea / z, elogA: elogA / z, elogP: [l0 / z, l1 / z] }
}

describe('dirichlet-concentration', () => {
  const pEta = [2.5, 1.5]
  it('with a gamma concentration: the tilted moments of alpha and of log p, and the evidence, match the double integral', () => {
    const [s, r] = [2.5, 1.2]
    const ref = brute(pEta, (a) => (s - 1) * Math.log(a) - r * a + Math.log(a) * 0, [1, 1], [0, 0])
    const ctx = (to: number): Parameters<NonNullable<typeof dirichletConcentration.ep>>[0] => ({
      mode: 'ep',
      to,
      args: [messageArg(beta, pEta as never), messageArg(gamma, gammaEta(s, r))],
      params: { weights: [1, 1] },
      power: 1,
    })
    const toAlpha = dirichletConcentration.ep?.(ctx(1)).message as number[]
    const alphaBelief = gammaEta(s, r).map((v, i) => v + (toAlpha[i] as number))
    const mo = gamma.moments(alphaBelief)
    expect(mo[1]).toBeCloseTo(ref.ea, 3)
    expect(mo[0]).toBeCloseTo(ref.elogA, 3)
    const toP = dirichletConcentration.ep?.(ctx(0)).message as number[]
    const pBelief = pEta.map((v, i) => v + (toP[i] as number))
    const m = beta.moments(pBelief)
    expect(m[0]).toBeCloseTo(ref.elogP[0] as number, 3)
    expect(m[1]).toBeCloseTo(ref.elogP[1] as number, 3)
    const logZ = dirichletConcentration.logEvidence?.(ctx(1)) as number
    // The brute force integrated the cavity density of alpha without its Jacobian-free normalisation: compare in the same measure.
    expect(logZ).toBeCloseTo(Math.log(ref.z * 1), 3)
  })

  it('with a log-normal concentration and an offset (Beta(alpha, 2))', () => {
    const [m0, v0] = [0.4, 0.6]
    const ref = brute(
      pEta,
      (a) => (-0.5 * (Math.log(a) - m0) ** 2) / v0 - Math.log(a) + (m0 * m0) / (2 * v0),
      [1, 0],
      [0, 2],
    )
    const ctx = (to: number): Parameters<NonNullable<typeof dirichletConcentration.ep>>[0] => ({
      mode: 'ep',
      to,
      args: [messageArg(beta, pEta as never), messageArg(gaussian, gaussianEta(m0, v0))],
      params: { weights: [1, 0], offsets: [0, 2], log: true },
      power: 1,
    })
    // The brute force integrates over alpha, the operator over t = log alpha: the density in t carries the Jacobian alpha, which the cavity above (a density in t) lacks: compare the moments of t.
    const toT = dirichletConcentration.ep?.(ctx(1)).message as number[]
    const belief = gaussianEta(m0, v0).map((v, i) => v + (toT[i] as number))
    const mean = (belief[0] as number) / (belief[1] as number)
    expect(mean).toBeCloseTo(ref.elogA, 3)
    const toP = dirichletConcentration.ep?.(ctx(0)).message as number[]
    const m = beta.moments(pEta.map((v, i) => v + (toP[i] as number)))
    expect(m[0]).toBeCloseTo(ref.elogP[0] as number, 3)
    expect(m[1]).toBeCloseTo(ref.elogP[1] as number, 3)
  })
})
