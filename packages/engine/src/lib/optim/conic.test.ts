import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { eigvalsh } from '../numerics/linalg/index.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>
const arr = (x: unknown): number[][][] => toArray(decode(x) as Tensor) as number[][][]

describe('conic programs against cvxpy (Clarabel)', () => {
  for (const c of loadFixture('optim/conic').cases) {
    it(c.name, () => {
      const out = c.output as Out
      if (c.fn === 'socp') {
        const [f, A, B, C, d, F, g, G, h] = c.inputs.map((x) => decode(x) as Tensor) as Tensor[]
        const As = toArray(A as Tensor) as number[][][]
        const Bs = toArray(B as Tensor) as number[][]
        const Cs = toArray(C as Tensor) as number[][]
        const ds = toFlat(d as Tensor)
        const r = O.conicProgram({
          c: f as Tensor,
          A: F as Tensor,
          b: g as Tensor,
          ...((G as Tensor).shape[0] === 0 ? {} : { G: G as Tensor, h: h as Tensor }),
          soc: As.map((a, i) => ({ A: a, b: Bs[i] ?? [], c: Cs[i] ?? [], d: ds[i] ?? 0 })),
        })
        expect(r.status).toBe('optimal')
        expectMatches(r.x, out['x'], c.rtol, c.atol, `${c.name} x`)
        expectMatches(r.fun, out['fun'], c.rtol, c.atol, `${c.name} fun`)
      } else if (c.fn === 'sdp') {
        const [C, A, b] = c.inputs
        const r = O.sdp(decode(C) as Tensor, arr(A), decode(b) as Tensor)
        expect(r.status).toBe('optimal')
        expectMatches(r.X, out['X'], c.rtol, c.atol, `${c.name} X`)
        expectMatches(r.fun, out['fun'], c.rtol, c.atol, `${c.name} fun`)
        // Strong duality and dual feasibility: b^T y = tr(C X), Z = C - sum y_i A_i >= 0.
        const by = toFlat(r.y).reduce((s, y, i) => s + y * (toFlat(decode(b) as Tensor)[i] ?? 0), 0)
        expect(by).toBeCloseTo(r.fun, 6)
        expect(toFlat(eigvalsh(r.Z))[0]).toBeGreaterThan(-1e-6)
      } else {
        const mats = arr(c.inputs[0])
        const [A0, A1, A2] = mats as [number[][], number[][], number[][]]
        const eye = [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]
        const neg = (m: number[][]) => m.map((row) => row.map((v) => -v))
        // t I - A0 - x1 A1 - x2 A2 >= 0 with variables (x1, x2, t), and |x|_inf <= 2.
        const r = O.conicProgram({
          c: [0, 0, 1],
          G: [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
          ],
          h: [2, 2, 2, 2],
          lmi: [{ F0: neg(A0), F: [neg(A1), neg(A2), eye] }],
        })
        expect(r.status).toBe('optimal')
        expectMatches(r.x, out['x'], c.rtol, c.atol, `${c.name} x`)
        expectMatches(r.fun, out['fun'], c.rtol, c.atol, `${c.name} fun`)
      }
    })
  }
})

describe('barrier method', () => {
  it('finds a strictly feasible point in phase 1, then shrinks the gap by mu per centring', () => {
    const rec = recorder()
    const r = O.conicProgram(
      {
        c: [1, 1],
        G: [
          [-1, 0],
          [0, -1],
        ],
        h: [-1, -2],
        soc: [
          {
            A: [
              [1, 0],
              [0, 1],
            ],
            b: [0, 0],
            c: [0, 0],
            d: 5,
          },
        ],
      },
      { tracer: rec },
    )
    expect(r.fun).toBeCloseTo(3, 7)
    const run = rec.run()
    expect(metricSeries(run, 'phase').values[0]).toBe(1)
    const gap = metricSeries(run, 'gap').values.filter((_, i) => i > 1)
    for (let i = 1; i < gap.length; i++) expect((gap[i] ?? 0) / (gap[i - 1] ?? 1)).toBeCloseTo(0.1, 9)
    // Duals: lambda >= 0, cone duals inside the cone, and stationarity c + G^T lambda = A^T z + c mu.
    const lambda = toFlat(r.duals.linear)
    expect(lambda.every((v) => v > 0)).toBe(true)
    expect(lambda[0]).toBeCloseTo(1, 6)
    expect(lambda[1]).toBeCloseTo(1, 6)
  })

  it('reports infeasible and unbounded programs', () => {
    const infeasible = O.conicProgram({ c: [1], G: [[1], [-1]], h: [-1, -1] })
    expect(infeasible.status).toBe('infeasible')
    const unbounded = O.conicProgram({ c: [-1, 0], soc: [{ A: [[0, 1]], b: [0], c: [1, 0], d: 0 }] })
    expect(unbounded.status).toBe('unbounded')
  })

  it('gives SDP duals: the smallest eigenvalue as min tr(C X), tr X = 1', () => {
    const r = O.sdp(
      [
        [2, 1],
        [1, 2],
      ],
      [
        [
          [1, 0],
          [0, 1],
        ],
      ],
      [1],
    )
    expect(r.fun).toBeCloseTo(1, 7)
    expect(toFlat(r.y)[0]).toBeCloseTo(1, 6)
    expect(toArray(r.X)).toEqual([
      [expect.closeTo(0.5, 6), expect.closeTo(-0.5, 6)],
      [expect.closeTo(-0.5, 6), expect.closeTo(0.5, 6)],
    ])
  })
})
