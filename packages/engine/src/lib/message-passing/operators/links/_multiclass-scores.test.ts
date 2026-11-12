import { describe, expect, it } from 'vitest'
import { logNdtr } from '../../../numerics/special/_kernels.ts'
import { massOf, messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg } from '../../core/types.ts'
import { logProbit } from './_multiclass-scores.ts'
import { multiclassSite } from './multiclass.ts'

/*
 * The multinomial probit's log probability far in the tail, against references with none of the code of the rule in them: the closed
 * form for two classes (`P(y = 0) = Phi((z_0 - z_1) / sqrt 2)`), and a dense one-dimensional integral over the noise of the chosen
 * class, taken in log space on a wide fine grid with `logNdtr` (accurate to the far tail), for more.
 */

/** log P(y | z) by the trapezoid rule on [-30, 80] with a step of 0.002, the log weights shifted by their maximum. */
function denseLogProbit(z: readonly number[], y: number): number {
  const n = 55000
  const h = 110 / n
  const logw: number[] = []
  for (let i = 0; i <= n; i++) {
    const e = -30 + i * h
    let s = -0.5 * e * e - 0.5 * Math.log(2 * Math.PI)
    for (let j = 0; j < z.length; j++) if (j !== y) s += logNdtr((z[y] as number) - (z[j] as number) + e)
    logw.push(s)
  }
  const top = logw.reduce((a, v) => Math.max(a, v), Number.NEGATIVE_INFINITY)
  return top + Math.log(logw.reduce((a, v) => a + Math.exp(v - top), 0) * h)
}

describe('logProbit in the far tail', () => {
  it('two classes: the closed form Phi((z_0 - z_1) / sqrt 2), from 0 to 60 below', () => {
    for (const gap of [0, 2, 5, 8, 9, 12, 20, 40, 60]) {
      const want = logNdtr(-gap / Math.SQRT2)
      const got = logProbit([0, gap], 0)
      expect(Math.abs(got - want), `gap ${gap}: ${got} against ${want}`).toBeLessThan(1e-9 * (1 + Math.abs(want)))
    }
  })
  it('three and seven classes against a dense log-space integral, from moderate to extreme gaps', () => {
    for (const gap of [2, 5, 9, 12, 20, 40]) {
      for (const z of [
        [0, gap, gap + 0.7],
        [0, gap, gap - 1, gap + 0.5, 0.5 * gap, gap + 2, gap - 0.3],
        [0, gap, gap, gap, gap, gap, gap],
      ]) {
        const want = denseLogProbit(z, 0)
        const got = logProbit(z, 0)
        expect(Math.abs(got - want), `gap ${gap}, ${z.length} classes: ${got} against ${want}`).toBeLessThan(
          1e-7 * (1 + Math.abs(want)),
        )
      }
    }
  }, 120_000)
  it('an ordinary evaluation keeps the fixed-node rule it always had: the same bits (a pinned value), within 1e-6 of the dense integral', () => {
    const z = [0.2, -0.1, 0.4]
    expect(logProbit(z, 0)).toBe(-1.0821879150714095)
    expect(Math.abs(logProbit(z, 0) - denseLogProbit(z, 0))).toBeLessThan(1e-6)
  })
})

describe('multiclass-site, the multinomial probit of a class far below the others', () => {
  it('the tilted moments of two scores whose cavities put the chosen class 20 below the other: finite, against a dense grid', () => {
    const cavities = [
      { m: -10, v: 1 },
      { m: 10, v: 1 },
    ]
    const args: Arg[] = cavities.map((c) => messageArg(gaussian, gaussianEta(c.m, c.v)))
    const params = { y: 0, classes: 2, link: 'probit' }
    // The dense grid over the two scores with log weights shifted by their maximum (P = Phi((z_0 - z_1) / sqrt 2) in closed form).
    const n = 600
    const axis = (m: number): number[] => Array.from({ length: n + 1 }, (_, i) => m - 12 + (24 * i) / n)
    const [a0, a1] = [axis(-10), axis(10)]
    const lw: number[] = []
    const pts: [number, number][] = []
    for (const x of a0 as number[])
      for (const y of a1 as number[]) {
        lw.push(-0.5 * (x + 10) ** 2 - 0.5 * (y - 10) ** 2 + logNdtr((x - y) / Math.SQRT2))
        pts.push([x, y])
      }
    const top = lw.reduce((a, v) => Math.max(a, v), Number.NEGATIVE_INFINITY)
    const w = lw.map((v) => Math.exp(v - top))
    const total = w.reduce((a, v) => a + v, 0)
    const mean = [0, 1].map(
      (k) => w.reduce((a, v, i) => a + v * ((pts[i] as [number, number])[k] as number), 0) / total,
    )
    const variance = [0, 1].map(
      (k) =>
        w.reduce((a, v, i) => a + v * (((pts[i] as [number, number])[k] as number) - (mean[k] as number)) ** 2, 0) /
        total,
    )
    for (const k of [0, 1]) {
      const c = cavities[k] as { m: number; v: number }
      const site = multiclassSite.ep?.({ mode: 'ep', to: k, args, params, power: 1 }).message as readonly number[]
      const precision = 1 / c.v + (site[1] as number)
      const gotMean = (c.m / c.v + (site[0] as number)) / precision
      expect(Math.abs(gotMean - (mean[k] as number)), `mean of score ${k}`).toBeLessThan(5e-3)
      expect(Math.abs(1 / precision / (variance[k] as number) - 1), `variance of score ${k}`).toBeLessThan(5e-3)
    }
    const evidence = multiclassSite.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
    const logMass = top + Math.log(total * ((24 / n) ** 2 / (2 * Math.PI)))
    expect(Math.abs(evidence - (logMass + args.reduce((s, a) => s + massOf(a), 0)))).toBeLessThan(5e-3)
  })
})
