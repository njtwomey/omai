import { describe, expect, it } from 'vitest'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  bistochasticNormalize,
  bistochasticNormalizeAlgorithm,
  logNormalize,
  makeNonnegative,
  scaleNormalize,
} from './bicluster-normalise.ts'

const x = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 10],
  [2, 9, 1],
]
const sums = (m: number[][]): { rows: number[]; columns: number[] } => ({
  rows: m.map((r) => r.reduce((a, b) => a + b, 0)),
  columns: (m[0] ?? []).map((_, j) => m.reduce((a, r) => a + (r[j] ?? 0), 0)),
})

describe('bicluster normalisations', () => {
  it('makeNonnegative shifts by the minimum only when it is below the floor', () => {
    expect(
      makeNonnegative([
        [-2, 1],
        [3, 4],
      ]),
    ).toEqual([
      [0, 3],
      [5, 6],
    ])
    expect(makeNonnegative(x)).toEqual(x)
    expect(makeNonnegative([[0.5, 2]], 1)).toEqual([[1, 2.5]])
    expect(() => makeNonnegative([])).toThrow(/empty/)
    expect(() => makeNonnegative([[1, 2], [3]])).toThrow(/differ/)
    expect(() => makeNonnegative([[1, Number.NaN]])).toThrow(/non-finite/)
  })

  it('scaling gives the factors that produce it, and rejects a zero row', () => {
    const { matrix, rowScale, columnScale } = scaleNormalize(x)
    for (const [i, r] of matrix.entries())
      for (const [j, v] of r.entries())
        expect(v).toBeCloseTo((rowScale[i] ?? 0) * (x[i]?.[j] ?? 0) * (columnScale[j] ?? 0), 14)
    expect(() =>
      scaleNormalize([
        [0, 0],
        [1, 2],
      ]),
    ).toThrow(/sums to zero/)
  })

  it('the bistochastic iteration equalises row sums and column sums, and is a traced algorithm', () => {
    const b = bistochasticNormalize(x, { tol: 1e-12 })
    const { rows, columns } = sums(b)
    for (const r of rows) expect(r).toBeCloseTo(rows[0] ?? 0, 6)
    for (const c of columns) expect(c).toBeCloseTo(columns[0] ?? 0, 6)
    const run = runAlgorithm(bistochasticNormalizeAlgorithm(x), { maxSteps: 200 })
    expect(run.status).toBe('converged')
    expect(stateFromJson(stateToJson(run.state))).toEqual(run.state)
  })

  it('log normalisation double-centres the log matrix', () => {
    const l = logNormalize(x)
    const { rows, columns } = sums(l)
    for (const v of [...rows, ...columns]) expect(v).toBeCloseTo(0, 12)
    expect(
      logNormalize([
        [2, 2],
        [2, 2],
      ]),
    ).toEqual([
      [0, 0],
      [0, 0],
    ])
  })
})
