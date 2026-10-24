import { describe, expect, it } from 'vitest'
import { coverage, expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { binomial, foldIn, key, multinomial } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const table = (k: ReturnType<typeof key>, n: number, probs: readonly number[]): number[][] => {
  const c = toFlat(multinomial(k, n, probs, 1))
  return [c.slice(0, 2), c.slice(2, 4)]
}

describe('contingency tables', () => {
  it('tests of independence are valid under the null', () => {
    const probs = [0.12, 0.28, 0.18, 0.42]
    expectUniformPValues(
      nullPValues(300, 1, (k) => T.chi2Contingency(table(k, 80, probs), { correction: false }).pValue),
    )
    expectUniformPValues(nullPValues(300, 2, (k) => T.gTest(table(k, 80, probs)).pValue))
    expectUniformPValues(
      nullPValues(300, 3, (k) => T.fisherExact(table(k, 30, probs)).pValue),
      { discrete: true },
    )
  })

  it("Fisher's r x c test agrees with the 2 x 2 test in rejecting strong association", () => {
    const strong = [
      [12, 1, 2],
      [1, 10, 2],
    ]
    expect(T.fisherExact(strong, { seed: 1, nResamples: 999 }).pValue).toBeLessThan(0.01)
    expect(T.chi2Contingency(strong).pValue).toBeLessThan(0.01)
  })

  it('odds-ratio and relative-risk intervals cover the truth', () => {
    const p1 = 0.4
    const p0 = 0.2
    const trueOr = p1 / (1 - p1) / (p0 / (1 - p0))
    const ors: [number, number][] = []
    const rrs: [number, number][] = []
    for (let i = 0; i < 200; i++) {
      const k = foldIn(key(5), i)
      const a = toFlat(binomial(foldIn(k, 0), 60, p1, []))[0] ?? 0
      const c = toFlat(binomial(foldIn(k, 1), 60, p0, []))[0] ?? 0
      ors.push(
        (T.oddsRatio(
          [
            [a, 60 - a],
            [c, 60 - c],
          ],
          { kind: 'sample' },
        ).interval ?? [0, 0]) as [number, number],
      )
      rrs.push((T.relativeRisk(a, 60, c, 60).interval ?? [0, 0]) as [number, number])
    }
    expect(coverage(ors, trueOr)).toBeGreaterThan(0.9)
    expect(coverage(rrs, p1 / p0)).toBeGreaterThan(0.9)
  })

  it('McNemar and Cochran agree on two conditions', () => {
    const x = [
      [1, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [1, 0],
      [1, 0],
      [0, 0],
      [1, 0],
    ]
    const b = x.filter(([u, v]) => u === 1 && v === 0).length
    const c = x.filter(([u, v]) => u === 0 && v === 1).length
    const q = T.cochranQ(x).statistic
    const m = T.mcnemar(
      [
        [1, b],
        [c, 1],
      ],
      { exact: false, correction: false },
    ).statistic
    expect(q).toBeCloseTo(m, 12)
  })
})
