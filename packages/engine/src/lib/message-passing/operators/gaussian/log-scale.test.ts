import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianScale } from './log-scale.ts'

/** The moments of m and s under N(m; mm, vm) N(s; ms, vs) N(x; m, exp(2 s)) on a 2-D grid, and the log mass. */
function brute(x: number, mm: number, vm: number, ms: number, vs: number) {
  const n = 1200
  const sdm = Math.sqrt(vm)
  const sds = Math.sqrt(vs)
  const m0 = mm - 9 * sdm
  const s0 = ms - 9 * sds
  const hm = (18 * sdm) / n
  const hs = (18 * sds) / n
  let z = 0
  let em = 0
  let es = 0
  let em2 = 0
  let es2 = 0
  for (let j = 0; j <= n; j++) {
    const s = s0 + j * hs
    const wsd = Math.exp(-0.5 * ((s - ms) / sds) ** 2)
    const sig2 = Math.exp(2 * s)
    for (let i = 0; i <= n; i++) {
      const m = m0 + i * hm
      const w = wsd * Math.exp(-0.5 * ((m - mm) / sdm) ** 2 - (0.5 * (x - m) ** 2) / sig2 - 0.5 * Math.log(sig2))
      z += w
      em += w * m
      em2 += w * m * m
      es += w * s
      es2 += w * s * s
    }
  }
  const mean = (a: number): number => a / z
  return {
    m: { mean: mean(em), variance: mean(em2) - mean(em) ** 2 },
    s: { mean: mean(es), variance: mean(es2) - mean(es) ** 2 },
    logMass: Math.log((z * hm * hs) / (2 * Math.PI * sdm * sds * Math.sqrt(2 * Math.PI))),
  }
}

const implied = (site: readonly number[], m: number, v: number): { mean: number; variance: number } => {
  const j = 1 / v + (site[1] ?? 0)
  return { mean: (m / v + (site[0] ?? 0)) / j, variance: 1 / j }
}

describe('x ~ N(m, exp(2 s)) with a random mean and log scale', () => {
  const cases = [
    { x: 2, mm: 0, vm: 1, ms: 0, vs: 1 },
    { x: -1.3, mm: 0.5, vm: 0.2, ms: -1, vs: 2 },
    { x: 0.1, mm: 0, vm: 9, ms: 0.5, vs: 0.5 },
    { x: 5, mm: 4.5, vm: 0.05, ms: 1, vs: 4 },
  ]
  for (const c of cases)
    it(`x = ${c.x}, mean N(${c.mm}, ${c.vm}), log scale N(${c.ms}, ${c.vs}): the messages match a two-dimensional grid`, () => {
      const ref = brute(c.x, c.mm, c.vm, c.ms, c.vs)
      const args = [
        valueArg(gaussian, c.x),
        messageArg(gaussian, gaussianEta(c.mm, c.vm)),
        messageArg(gaussian, gaussianEta(c.ms, c.vs)),
      ]
      const toS = implied(
        gaussianScale.ep?.({ mode: 'ep', to: 2, args, params: {}, power: 1 })?.message ?? [],
        c.ms,
        c.vs,
      )
      expect(Math.abs(toS.mean - ref.s.mean) / Math.sqrt(ref.s.variance)).toBeLessThan(2e-3)
      expect(toS.variance / ref.s.variance).toBeCloseTo(1, 2)
      const toM = implied(
        gaussianScale.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 })?.message ?? [],
        c.mm,
        c.vm,
      )
      expect(Math.abs(toM.mean - ref.m.mean) / Math.sqrt(ref.m.variance)).toBeLessThan(2e-3)
      expect(toM.variance / ref.m.variance).toBeCloseTo(1, 2)
      const ev = gaussianScale.logEvidence?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }) as number
      expect(ev).toBeCloseTo(
        gaussian.logPartition(gaussianEta(c.mm, c.vm)) + gaussian.logPartition(gaussianEta(c.ms, c.vs)) + ref.logMass,
        3,
      )
    })

  it('a known mean is the same factor with a point cavity', () => {
    const args = [valueArg(gaussian, 2), valueArg(gaussian, 0.3), messageArg(gaussian, gaussianEta(0, 1))]
    const site = gaussianScale.ep?.({ mode: 'ep', to: 2, args, params: {}, power: 1 })?.message ?? []
    expect(site.length).toBe(2)
    expect(site[1]).toBeGreaterThan(-1)
  })
})
