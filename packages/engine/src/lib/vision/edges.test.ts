import { describe, expect, it } from 'vitest'
import { canny, cannyStages, houghLine } from '../signal/image/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { CANNY_STAGES, cannyAlgorithm, houghLineAlgorithm, houghLinePeaks } from './index.ts'

// A bright square and a bright disc on a dark ground.
const scene = Array.from({ length: 32 }, (_, r) =>
  Array.from({ length: 32 }, (_, c) => {
    const square = r >= 5 && r < 15 && c >= 5 && c < 15
    const disc = (r - 22) ** 2 + (c - 22) ** 2 <= 25
    return square || disc ? 1 : 0
  }),
)

describe('cannyAlgorithm', () => {
  it('shows each stage and ends at the one-shot detector', () => {
    const options = { sigma: 1.5, lowThreshold: 0.1, highThreshold: 0.3 }
    const alg = cannyAlgorithm(scene, options)
    const stages = cannyStages(scene, options)
    let s = alg.init({} as never)
    const seen: string[] = [s.stage]
    const ctx = { t: 0, key: {} as never, log: () => undefined, wants: () => false, emit: () => undefined } as never
    while (alg.done?.(s, s.step) === undefined) {
      s = alg.step(s, ctx)
      if (seen[seen.length - 1] !== s.stage) {
        seen.push(s.stage)
        const expected = {
          smoothed: stages.smoothed,
          gradient: stages.magnitude,
          suppressed: stages.suppressed,
          thresholded: stages.thresholded,
        }[s.stage as 'smoothed']
        if (expected !== undefined) expect(toFlat(s.image)).toEqual(toFlat(expected))
      }
    }
    expect(seen).toEqual([...CANNY_STAGES])
    expect(toFlat(s.image)).toEqual(toFlat(canny(scene, options)))
    expect(toFlat(s.image)).toEqual(toFlat(stages.edges))
    expect(toFlat(s.image).some((v) => v === 1)).toBe(true)
  })
  it('hysteresis only grows, from the strong pixels', () => {
    const alg = cannyAlgorithm(scene, { sigma: 1, lowThreshold: 0.05, highThreshold: 0.4 })
    let s = alg.init({} as never)
    const ctx = { t: 0, key: {} as never, log: () => undefined, wants: () => false, emit: () => undefined } as never
    const counts: number[] = []
    while (alg.done?.(s, s.step) === undefined) {
      s = alg.step(s, ctx)
      if (s.stage === 'hysteresis') counts.push(s.edgePixels)
    }
    for (let i = 1; i < counts.length; i++) expect(counts[i] as number).toBeGreaterThanOrEqual(counts[i - 1] as number)
    expect(counts.length).toBeGreaterThanOrEqual(2)
    const strong = toFlat(cannyStages(scene, { sigma: 1, lowThreshold: 0.05, highThreshold: 0.4 }).thresholded).filter(
      (v) => v === 2,
    ).length
    expect(counts[0]).toBe(strong)
  })
  it('serialises its state', () => {
    const alg = cannyAlgorithm(scene)
    const r = runAlgorithm(alg, { maxSteps: 3 })
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(back?.stage).toBe(r.state.stage)
    expect(toFlat(back?.image as never)).toEqual(toFlat(r.state.image))
  })
})

// Three planted lines in a 40 x 40 image: x = 10 (rho 10, theta 0), y = 25 (rho -25, theta -pi/2), x = y (rho 0, theta -pi/4).
const lines = Array.from({ length: 40 }, (_, y) =>
  Array.from({ length: 40 }, (_, x) => (x === 10 || y === 25 || x === y ? 1 : 0)),
)

describe('houghLineAlgorithm', () => {
  it('fills one pixel per step and ends at the one-shot accumulator', () => {
    const { algorithm, geometry } = houghLineAlgorithm(lines)
    const r = runAlgorithm(algorithm, { maxSteps: 500 })
    expect(r.status).toBe('completed')
    expect(r.state.voted).toBe(
      toFlat(fromStorage(Float64Array.from(lines.flat()), [40, 40], 'float64')).filter((v) => v !== 0).length,
    )
    const one = houghLine(lines)
    expect(Array.from(r.state.accumulator)).toEqual(toFlat(one.accumulator))
    expect(geometry.shape).toEqual(one.accumulator.shape)
    const partial = runAlgorithm(algorithm, { maxSteps: 10 })
    expect(partial.state.voted).toBe(10)
    expect(Math.max(...partial.state.accumulator)).toBeLessThanOrEqual(10)
  })
  it('the peaks recover the planted lines', () => {
    const { algorithm, geometry } = houghLineAlgorithm(lines, { pixelsPerStep: 50 })
    const acc = runAlgorithm(algorithm, { maxSteps: 100 }).state.accumulator
    const peaks = houghLinePeaks(fromStorage(acc, [...geometry.shape], 'float64'), geometry, { numPeaks: 3 })
    expect(peaks).toHaveLength(3)
    const near = (rho: number, theta: number): boolean =>
      peaks.some((p) => Math.abs(p.rho - rho) <= 1 && Math.abs(p.theta - theta) <= 0.03)
    expect(near(10, 0)).toBe(true)
    expect(near(-25, -Math.PI / 2)).toBe(true)
    expect(near(0, -Math.PI / 4)).toBe(true)
    for (const p of peaks) expect(p.votes).toBeGreaterThanOrEqual(38)
  })
  it('batched voting gives the same accumulator', () => {
    const a = runAlgorithm(houghLineAlgorithm(lines, { pixelsPerStep: 7 }).algorithm, { maxSteps: 200 }).state
      .accumulator
    const b = runAlgorithm(houghLineAlgorithm(lines).algorithm, { maxSteps: 500 }).state.accumulator
    expect(Array.from(a)).toEqual(Array.from(b))
  })
})
