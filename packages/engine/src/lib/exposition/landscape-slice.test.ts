import { describe, expect, it } from 'vitest'
import { landscapeSlice, sampleLandscape, sliceCoordinates } from './landscape-slice.ts'
import { isExposition } from './record.ts'

describe('the landscape slice record', () => {
  const x = [-1, 0, 1]
  it('is a record, sampled from a loss over a line or a plane', () => {
    const line = sampleLandscape((a) => a * a, x)
    expect(line.dims).toBe(1)
    expect(line.curve).toEqual([1, 0, 1])
    const plane = sampleLandscape((a, b) => a + 10 * b, x, [0, 1])
    expect(plane.grid).toEqual([
      [-1, 0, 1],
      [9, 10, 11],
    ])
    expect(isExposition(line)).toBe(true)
  })

  it('checks the grid, the axes, the paths and the guides against the dimensions', () => {
    expect(() => landscapeSlice({ dims: 1, axes: ['a', 'b'], x, curve: [1, 2, 3] })).toThrow(/2 axes/)
    expect(() => landscapeSlice({ dims: 1, axes: ['a'], x, curve: [1, 2] })).toThrow(/curve needs 3/)
    expect(() => landscapeSlice({ dims: 1, axes: ['a'], x: [0, 0, 1], curve: [1, 2, 3] })).toThrow(/not increasing/)
    expect(() => landscapeSlice({ dims: 2, axes: ['a', 'b'], x, y: [0, 1], grid: [[1, 2, 3]] })).toThrow(/2 rows/)
    expect(() => landscapeSlice({ dims: 2, axes: ['a', 'b'], x, curve: [1, 2, 3] })).toThrow(/needs y and a grid/)
    expect(() => landscapeSlice({ dims: 1, axes: ['a'], x, curve: [1, 2, 3], paths: [{ coords: [[0, 1]] }] })).toThrow(
      /paths\[0\].coords\[0\]/,
    )
    expect(() =>
      landscapeSlice({ dims: 1, axes: ['a'], x, curve: [1, 2, 3], paths: [{ coords: [[0]], loss: [1, 2] }] }),
    ).toThrow(/loss has 2/)
    expect(() => landscapeSlice({ dims: 1, axes: ['a'], x, curve: [1, 2, 3], at: [0, 0] })).toThrow(/at has 2/)
  })

  it('projects iterates onto the directions of a slice', () => {
    expect(
      sliceCoordinates(
        [
          [1, 1],
          [3, 2],
        ],
        [1, 0],
        [
          [2, 0],
          [0, 1],
        ],
      ),
    ).toEqual([
      [0, 1],
      [1, 2],
    ])
    // The origin itself is the origin of the slice, and a vector of the wrong length is refused.
    expect(sliceCoordinates([[5, 5, 5]], [5, 5, 5], [[1, 0, 0]])).toEqual([[0]])
    expect(() => sliceCoordinates([[1]], [1, 2], [[1, 0]])).toThrow(/iterate 0/)
  })
})
