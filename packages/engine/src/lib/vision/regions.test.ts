import { describe, expect, it } from 'vitest'
import { image } from '../signal/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { meanShiftSegmentation, meanShiftSegmentationAlgorithm } from './index.ts'

const noCtx = { t: 0, key: {} as never, log: () => undefined, wants: () => false, emit: () => undefined } as never

// Two basins (columns 0-6 and 8-14) separated by a ridge in column 7, on a 9 x 15 image.
const basins = Array.from({ length: 9 }, (_, r) =>
  Array.from({ length: 15 }, (_, c) => (c === 7 ? 10 : Math.abs(c - (c < 7 ? 3 : 11)) + 0.1 * Math.abs(r - 4))),
)
const seeds = Array.from({ length: 9 }, (_, r) =>
  Array.from({ length: 15 }, (_, c) => (r === 4 && c === 3 ? 1 : r === 4 && c === 11 ? 2 : 0)),
)

describe('watershedAlgorithm', () => {
  it('floods two planted basins to two regions, ending at the one-shot result', () => {
    const alg = image.watershedAlgorithm(basins, seeds, { levels: 6 })
    const r = runAlgorithm(alg, { maxSteps: 50 })
    expect(r.status).toBe('completed')
    const one = toFlat(image.watershed(basins, seeds))
    expect(Array.from(r.state.labels)).toEqual(one)
    expect(new Set(r.state.labels)).toEqual(new Set([1, 2]))
    // The left basin is the left side of the ridge.
    expect(r.state.labels[4 * 15 + 1]).toBe(1)
    expect(r.state.labels[4 * 15 + 13]).toBe(2)
  })
  it('labels more pixels at each level, low ground first', () => {
    const alg = image.watershedAlgorithm(basins, seeds, { levels: 8 })
    let s = alg.init({} as never)
    const counts: number[] = []
    const levels: number[] = []
    while (alg.done?.(s, s.step) === undefined) {
      s = alg.step(s, noCtx)
      counts.push(Array.from(s.labels).filter((v) => v !== 0).length)
      levels.push(s.level)
    }
    for (let i = 1; i < counts.length; i++) expect(counts[i] as number).toBeGreaterThanOrEqual(counts[i - 1] as number)
    for (let i = 1; i < levels.length; i++) expect(levels[i] as number).toBeGreaterThan(levels[i - 1] as number)
    // The ridge (value 10) is flooded last.
    expect(counts[0] as number).toBeLessThan(counts[counts.length - 1] as number)
    const ridge = 4 * 15 + 7
    const before = alg.step(alg.init({} as never), noCtx)
    expect(before.labels[ridge]).toBe(0)
  })
  it('matches the one-shot result for any number of levels, and serialises the queue', () => {
    for (const levels of [1, 3, 40]) {
      const r = runAlgorithm(image.watershedAlgorithm(basins, seeds, { levels }), { maxSteps: 100 })
      expect(Array.from(r.state.labels)).toEqual(toFlat(image.watershed(basins, seeds)))
    }
    const alg = image.watershedAlgorithm(basins, seeds, { levels: 6 })
    const mid = runAlgorithm(alg, { maxSteps: 2 })
    const back = alg.deserialise?.(alg.serialise?.(mid.state) as never)
    const finish = runAlgorithm(alg, { maxSteps: 50, start: { state: back, step: 2 } })
    expect(Array.from(finish.state.labels)).toEqual(toFlat(image.watershed(basins, seeds)))
  })
})

// Two flat regions of grey 0.2 and 0.8 with a deterministic ripple of at most 0.02.
const ripple = (r: number, c: number): number => 0.02 * Math.sin(7 * r + 3 * c)
const planted = Array.from({ length: 8 }, (_, r) =>
  Array.from({ length: 12 }, (_, c) => (c < 6 ? 0.2 : 0.8) + ripple(r, c)),
)

describe('meanShiftSegmentationAlgorithm', () => {
  it('the modes are the planted region colours, and it ends at the one-shot labels', () => {
    const options = { spatialRadius: 3, colourRadius: 0.3, maxIter: 20 }
    const alg = meanShiftSegmentationAlgorithm(planted, options)
    const r = runAlgorithm(alg, { maxSteps: 30 })
    expect(r.state.regions).toBe(2)
    const one = meanShiftSegmentation(planted, options)
    expect(Array.from(r.state.labels)).toEqual(toFlat(one.labels))
    // Every pixel's mode colour is within the ripple of its region's colour.
    const d = 3
    for (let p = 0; p < 8 * 12; p++) {
      const c = p % 12
      expect(Math.abs((r.state.modes[p * d + 2] as number) - (c < 6 ? 0.2 : 0.8))).toBeLessThan(0.03)
    }
    // Modes within a region agree with each other to the window's resolution.
    const left = Array.from({ length: 8 }, (_, i) => r.state.modes[i * 12 * d + 2] as number)
    expect(Math.max(...left) - Math.min(...left)).toBeLessThan(0.02)
  })
  it('windows settle: the moving count falls to zero', () => {
    const alg = meanShiftSegmentationAlgorithm(planted, { spatialRadius: 3, colourRadius: 0.3, maxIter: 40 })
    let s = alg.init({} as never)
    const moving: number[] = []
    while (alg.done?.(s, s.iteration) === undefined) {
      s = alg.step(s, noCtx)
      moving.push(s.moving)
    }
    expect(moving[0] as number).toBeGreaterThan(0)
    expect(moving[moving.length - 1]).toBe(0)
    expect(alg.done?.(s, s.iteration)).toBe('converged')
  })
})
