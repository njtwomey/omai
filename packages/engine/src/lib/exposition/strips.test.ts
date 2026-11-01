import { describe, expect, it } from 'vitest'
import { runLengthStrip } from './strips.ts'

describe('runLengthStrip', () => {
  const rows = [
    [0, 1],
    [0, 0.1, 0.9],
    [0, 0.8, 0.1, 0.1],
    [0, 0.05, 0.9, 0.025, 0.025],
  ]

  it('lays the posterior out as tracks over steps, with the MAP path and the true changepoints', () => {
    const s = runLengthStrip({ runLength: rows, changepoints: [2] })
    expect(s.tokens).toEqual(['0', '1', '2', '3'])
    expect(s.tracks.map((t) => t.name)).toEqual(['run length 1', 'run length 2', 'run length 3', 'run length 4'])
    expect(s.tracks[0]?.values).toEqual([1, 0.1, 0.8, 0.05])
    expect(s.path).toEqual([0, 1, 0, 1])
    expect(s.marks).toEqual([2])
    expect(s.range).toEqual([0, 1])
  })

  it('caps the run lengths shown and draws longer MAP run lengths on the last track', () => {
    const s = runLengthStrip({ runLength: rows, maxRunLength: 2, mapRunLength: [1, 2, 3, 4] })
    expect(s.tracks).toHaveLength(2)
    expect(s.path).toEqual([0, 1, 1, 1])
  })

  it('checks its shapes', () => {
    expect(() => runLengthStrip({ runLength: rows, mapRunLength: [1] })).toThrow(/mapRunLength/)
    expect(() => runLengthStrip({ runLength: rows, changepoints: [9] })).toThrow(/marks/)
  })
})
