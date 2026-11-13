import { describe, expect, it } from 'vitest'
import { run } from './compute.ts'
import { CHECKPOINTS, FIRST_SEED, ROWS, series, TRUTH } from './data.ts'
import { logistic, oddsText, readBeliefs, readZeros } from './reading.ts'

describe('the too-many-zeros data', () => {
  it('has the story’s counts: structural zeros are zeros, and the same seed gives the same counts', () => {
    const a = series(FIRST_SEED)
    expect(a.y).toHaveLength(ROWS)
    expect(a.x).toHaveLength(ROWS)
    expect(a.y.every((v) => Number.isInteger(v) && v >= 0)).toBe(true)
    expect(a.structural.every((s, i) => s === 0 || a.y[i] === 0)).toBe(true)
    expect(a.structural.some((s) => s === 1)).toBe(true)
    expect(series(FIRST_SEED)).toEqual(a)
    expect(series(FIRST_SEED + 1)).not.toEqual(a)
    expect(CHECKPOINTS.at(-1)).toBe(ROWS)
    expect(TRUTH.inflation).toBeGreaterThan(0)
  })
})

describe('the too-many-zeros computation', () => {
  it('returns plain data of the right shapes on a small run, and the result survives a structured clone', () => {
    const said: string[] = []
    const r = run(FIRST_SEED, (s) => said.push(s), { rows: 40, checkpoints: [10, 40] })
    expect(r.y).toHaveLength(40)
    expect(r.chance).toHaveLength(40)
    expect(r.gate).toHaveLength(40)
    expect(r.prior).toHaveLength(40)
    expect(r.beliefs.map((b) => b.rows)).toEqual([10, 40])
    expect(r.beliefs.every((b) => b.density.length === r.grid.length)).toBe(true)
    expect(r.beliefs.every((b) => b.lower < b.mean && b.mean < b.upper)).toBe(true)
    // the belief about the chance narrows as counts arrive
    expect(r.beliefs[1]?.sd).toBeLessThan(r.beliefs[0]?.sd as number)
    expect(r.settled).toBe(true)
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
  }, 120_000)

  it('gives a count above zero no chance of being structural, and each zero the chance its two messages make', () => {
    const r = run(FIRST_SEED, undefined, { rows: 40, checkpoints: [40] })
    r.y.forEach((y, i) => {
      if (y > 0) expect(r.chance[i]).toBe(0)
      else expect(r.chance[i]).toBeCloseTo(logistic((r.prior[i] as number) + (r.gate[i] as number)), 2)
    })
    // a higher rate makes an ordinary zero less likely, so the gate's message favours structural more at larger x
    const z = readZeros(r).zeros
    expect(z.length).toBeGreaterThan(5)
    expect(z.at(-1)?.gate).toBeGreaterThan(z[0]?.gate as number)
    expect(z.at(-1)?.chance).toBeGreaterThan(z[0]?.chance as number)
  }, 120_000)

  it('agrees with the library’s exact posterior, and the switch lifts the intercept the zeros pull down', () => {
    const r = run(FIRST_SEED, undefined, { rows: 60, checkpoints: [60] })
    const b = readBeliefs(r)
    expect(b.meanGap).toBeLessThan(0.25)
    expect(r.exact.chanceGap).toBeLessThan(0.05)
    expect(b.sdRatio[0]).toBeGreaterThan(0.7)
    expect(b.sdRatio[1]).toBeLessThan(1.3)
    expect(r.withoutSwitch.b0.mean).toBeLessThan(r.withSwitch.b0.mean)
  }, 120_000)
})

describe('the too-many-zeros reading', () => {
  it('writes odds in words', () => {
    expect(oddsText(Math.log(48))).toBe('48 to 1')
    expect(oddsText(Math.log(3.2))).toBe('3.2 to 1')
    expect(oddsText(Math.log(0.5))).toBe('1 to 2.0')
  })
})
