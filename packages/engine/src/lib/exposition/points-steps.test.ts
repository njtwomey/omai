import { describe, expect, it } from 'vitest'
import { pointsAt, pointsSteps } from './points-steps.ts'

describe('points with a value channel', () => {
  const r = pointsSteps({
    dims: 2,
    valueName: 'speed',
    steps: [
      { step: 0, points: [[0, 0]], values: [1] },
      {
        step: 4,
        points: [
          [1, 1],
          [2, 0],
        ],
        values: [2, 5],
      },
    ],
  })

  it('checks the values against the points', () => {
    expect(() => pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]], values: [1, 2] }] })).toThrow(
      /values has 2/,
    )
  })

  it('collects the values and the step each point arrived at', () => {
    const at = pointsAt(r, 4)
    expect(at.values).toEqual([1, 2, 5])
    expect(at.born).toEqual([0, 4, 4])
    expect(pointsAt(r, 1).values).toEqual([1])
  })

  it('has no values unless every step does, and shows only the latest set when it does not accumulate', () => {
    const mixed = pointsSteps({
      dims: 2,
      steps: [
        { step: 0, points: [[0, 0]] },
        { step: 1, points: [[1, 1]], values: [3] },
      ],
    })
    expect(pointsAt(mixed, 1).values).toBeUndefined()
    const moving = pointsSteps({ ...r, accumulate: false })
    expect(pointsAt(moving, 4).born).toEqual([4, 4])
  })
})
