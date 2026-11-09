import { describe, expect, it } from 'vitest'
import { edgeRecovery, graphEdges, precisionGraph } from './evaluation.ts'

describe('graphs of precision matrices', () => {
  it('the graph is where the entries are non-zero, off the diagonal, symmetric', () => {
    const g = precisionGraph([
      [2, -0.5, 0, 1e-12],
      [-0.5, 2, 0.3, 0],
      [0, 0.3, 1, 0],
      [1e-12, 0, 0, 1],
    ])
    expect(graphEdges(g)).toEqual([
      [0, 1],
      [1, 2],
    ])
    expect(
      precisionGraph(
        [
          [1, 0.001],
          [0.001, 1],
        ],
        0.01,
      )[0]?.[1],
    ).toBe(false)
  })

  it('counts edges once and reports precision, recall, F1 and the Hamming distance', () => {
    const truth = [
      [false, true, false],
      [true, false, true],
      [false, true, false],
    ]
    expect(edgeRecovery(truth, truth)).toMatchObject({ precision: 1, recall: 1, f1: 1, hamming: 0 })
    const found = [
      [false, true, true],
      [true, false, false],
      [true, false, false],
    ]
    expect(edgeRecovery(found, truth)).toMatchObject({
      truePositives: 1,
      falsePositives: 1,
      falseNegatives: 1,
      precision: 0.5,
      recall: 0.5,
      hamming: 2,
    })
    const empty = [
      [false, false],
      [false, false],
    ]
    expect(edgeRecovery(empty, empty)).toMatchObject({ precision: 1, recall: 1, f1: 1 })
    expect(
      edgeRecovery(empty, [
        [false, true],
        [true, false],
      ]),
    ).toMatchObject({ recall: 0, f1: 0 })
    expect(() => edgeRecovery(empty, truth)).toThrow(/differ/)
  })
})
