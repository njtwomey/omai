// dirichlet-concentration: the tilted moments of the concentration and of E log p against a brute-force double integral.
import { describe, expect, it } from 'vitest'
import { massOf, messageArg } from '../../core/args.ts'
import { beta, betaEta, gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg } from '../../core/types.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { betaTwoShapes, dirichletConcentration } from './concentration.ts'

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

describe('beta-two-shapes: the tilted moments of the shapes and E log p against a brute-force triple integral', () => {
  // The cavity of p is Beta(eta + 1); a is a Gaussian on its log (a log-normal shape), b a gamma shape.
  const pEta = [2.5, 1.2]
  const [ma, va, shapeB, rateB] = [0.4, 0.5, 2.5, 1.5]
  const args = [
    messageArg(beta, betaEta((pEta[0] as number) + 1, (pEta[1] as number) + 1)),
    messageArg(gaussian, gaussianEta(ma, va)),
    messageArg(gamma, gammaEta(shapeB, rateB)),
  ]
  const NA = 500
  const NB = 500
  const NP = 1500
  const [tLo, tHi, uLo, uHi] = [ma - 9 * Math.sqrt(va), ma + 9 * Math.sqrt(va), -12, 5]
  /** The reference at the power `alpha` of the factor: the cavities times `Beta(p; a, b)^alpha`, integrated on three grids. */
  const brute = (alpha: number) => {
    let z = 0
    let ea = 0
    let eb = 0
    let ea2 = 0
    let elogP = 0
    let elogQ = 0
    for (let i = 0; i < NA; i++) {
      const ta = tLo + ((i + 0.5) * (tHi - tLo)) / NA
      const a = Math.exp(ta)
      const wa = Math.exp((-0.5 * (ta - ma) ** 2) / va) / Math.sqrt(2 * Math.PI * va)
      for (let j = 0; j < NB; j++) {
        const tb = uLo + ((j + 0.5) * (uHi - uLo)) / NB
        const b = Math.exp(tb)
        const wb = Math.exp(shapeB * tb - rateB * b - lgammaOf(shapeB) + shapeB * Math.log(rateB))
        const norm = alpha * (lgammaOf(a + b) - lgammaOf(a) - lgammaOf(b))
        let sp = 0
        let l0 = 0
        let l1 = 0
        for (let k = 0; k < NP; k++) {
          const p = (k + 0.5) / NP
          const dens = Math.exp(
            norm +
              (alpha * (a - 1) + (pEta[0] as number)) * Math.log(p) +
              (alpha * (b - 1) + (pEta[1] as number)) * Math.log(1 - p),
          )
          sp += dens
          l0 += dens * Math.log(p)
          l1 += dens * Math.log(1 - p)
        }
        const w = (wa * wb * sp) / NP
        z += w
        ea += w * ta
        ea2 += w * ta * ta
        eb += w * tb
        elogP += (wa * wb * l0) / NP
        elogQ += (wa * wb * l1) / NP
      }
    }
    return { z, ea, ea2, eb, elogP, elogQ }
  }
  const run = (to: number, power: number): readonly number[] =>
    betaTwoShapes.ep?.({ mode: 'ep', to, args, params: {}, power }).message as readonly number[]
  // Power 1 is plain EP; 0.5 is fractional EP: the factor enters the tilt to the power and its site is divided by it.
  for (const power of [1, 0.5]) {
    const ref = brute(power)
    describe(`power ${power}`, () => {
      it('the message to a is the Gaussian site that matches the tilted mean and variance of log a, over the power', () => {
        const mean = ref.ea / ref.z
        const variance = ref.ea2 / ref.z - mean * mean
        const want = [(mean / variance - ma / va) / power, (1 / variance - 1 / va) / power]
        const got = run(1, power)
        expect(got[0]).toBeCloseTo(want[0] as number, 2)
        expect(got[1]).toBeCloseTo(want[1] as number, 2)
      })
      it('the message to p matches E log p and E log (1 - p): the Beta with those statistics, divided by the cavity and the power', () => {
        const projected = beta.fromMoments([ref.elogP / ref.z, ref.elogQ / ref.z])
        const got = run(0, power)
        expect(got[0]).toBeCloseTo(((projected[0] as number) - (pEta[0] as number)) / power, 2)
        expect(got[1]).toBeCloseTo(((projected[1] as number) - (pEta[1] as number)) / power, 2)
      })
      it('the log evidence is the log of the integral of the powered factor plus the masses of the shapes', () => {
        const want =
          Math.log(ref.z * (((tHi - tLo) / NA) * ((uHi - uLo) / NB))) + massOf(args[1] as Arg) + massOf(args[2] as Arg)
        expect(betaTwoShapes.logEvidence?.({ mode: 'ep', to: 0, args, params: {}, power })).toBeCloseTo(want, 3)
      })
    })
  }
})
