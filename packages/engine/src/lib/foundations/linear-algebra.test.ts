import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { tensor } from '../tensor/index.ts'
import * as F from './index.ts'

type M = number[][]
const mm = (a: M, b: M): M =>
  a.map((r) => (b[0] ?? []).map((_, j) => r.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0)))
const closeM = (a: M, b: M, digits = 10): void => {
  for (const [i, r] of a.entries())
    for (const [j, v] of r.entries()) expect(v).toBeCloseTo(b[i]?.[j] ?? Number.NaN, digits)
}
const intMatrix = (m: number, n: number): fc.Arbitrary<M> =>
  fc.array(fc.array(fc.integer({ min: -5, max: 5 }), { minLength: n, maxLength: n }), { minLength: m, maxLength: m })

describe('elimination', () => {
  it('every recorded step is the stated row operation applied to the previous matrix', () => {
    fc.assert(
      fc.property(intMatrix(3, 4), (a) => {
        const r = F.rref(a)
        let current = a.map((row) => [...row])
        for (const step of r.steps) {
          const op = step.operation
          const next = current.map((row) => [...row])
          if (op.kind === 'swap') {
            const [i, j] = op.rows
            next[i] = [...(current[j] ?? [])]
            next[j] = [...(current[i] ?? [])]
          } else if (op.kind === 'scale') next[op.row] = (current[op.row] ?? []).map((x) => x * op.factor)
          else
            next[op.target] = (current[op.target] ?? []).map((x, k) => x + op.factor * (current[op.source]?.[k] ?? 0))
          closeM(step.matrix, next, 9)
          current = step.matrix
        }
        closeM(r.matrix, current, 12)
      }),
      { numRuns: 40 },
    )
  })

  it('rank-nullity, and the null space is annihilated', () => {
    fc.assert(
      fc.property(intMatrix(3, 5), (a) => {
        const s = F.fundamentalSubspaces(a)
        expect(s.rank + s.nullity).toBe(5)
        expect(s.leftNullSpace.length).toBe(3 - s.rank)
        for (const v of s.nullSpace)
          for (const row of a) expect(row.reduce((t, x, j) => t + x * (v[j] ?? 0), 0)).toBeCloseTo(0, 10)
        expect(F.basisFromVectors(s.columnSpace.length > 0 ? s.columnSpace : [[0, 0, 0]]).indices.length).toBe(s.rank)
      }),
      { numRuns: 40 },
    )
  })

  it('first-nonzero pivoting gives the same reduced form', () => {
    const a = [
      [0, 2, 4],
      [1, 1, 1],
      [2, 4, 6],
    ]
    closeM(F.rref(a, { pivoting: 'first' }).matrix, F.rref(a).matrix, 12)
    expect(F.rowEchelon(a, { pivoting: 'first' }).steps[0]?.description).toBe('R1 <-> R2')
  })

  it('solves consistent systems in general form and detects inconsistent ones', () => {
    const s = F.solveByElimination(
      [
        [1, 2, -1],
        [2, 4, -2],
      ],
      [3, 6],
    )
    expect(s.consistent).toBe(true)
    expect(s.freeColumns).toEqual([1, 2])
    const bad = F.solveByElimination(
      [
        [1, 1],
        [1, 1],
      ],
      [1, 2],
    )
    expect(bad.consistent).toBe(false)
    expect(bad.particular).toBeNull()
    expect(() => F.solveByElimination([[1, 1]], [1, 2])).toThrow()
  })
})

describe('LU and determinants', () => {
  it('P L U rebuilds A and the three determinants agree', () => {
    fc.assert(
      fc.property(intMatrix(4, 4), (a) => {
        const { p, l, u } = F.luSteps(a)
        closeM(mm(p, mm(l, u)), a, 10)
        const byCof = F.detByCofactors(a).value
        expect(F.detByElimination(a).value).toBeCloseTo(byCof, 8)
        const poly = F.characteristicPolynomial(a).coefficients
        // det(t I - A) at t = 0 is (-1)^n det A.
        expect(poly[4] ?? 0).toBeCloseTo(byCof, 7)
      }),
      { numRuns: 40 },
    )
    expect(() => F.detByCofactors(Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => 1)))).toThrow()
    expect(
      F.detByElimination([
        [1, 2],
        [2, 4],
      ]).value,
    ).toBe(0)
    expect(F.detByCofactors([[7]]).value).toBe(7)
  })
})

describe('eigenvalues, diagonalisation and quadratic forms', () => {
  it('diagonalises symmetric matrices and rebuilds them', () => {
    const a = [
      [2, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ]
    const d = F.diagonalise(a)
    expect(d.diagonalisable).toBe(true)
    const pm = d.p ?? []
    const dm = d.d.map((v, i) => d.d.map((_, j) => (i === j ? v : 0)))
    closeM(mm(pm, mm(dm, d.pInverse ?? [])), a, 8)
  })

  it('reports multiplicities, defective matrices and complex eigenvalues', () => {
    const shear = F.smallEigen([
      [2, 1],
      [0, 2],
    ])
    expect(shear).toHaveLength(1)
    expect(shear[0]?.algebraicMultiplicity).toBe(2)
    expect(shear[0]?.eigenvectors).toHaveLength(1)
    const rotation = F.smallEigen([
      [0, -1],
      [1, 0],
    ])
    expect(rotation.map((p) => p.im)).toEqual([-1, 1])
    expect(rotation[0]?.eigenvectors).toEqual([])
    expect(
      F.diagonalise([
        [0, -1],
        [1, 0],
      ]).diagonalisable,
    ).toBe(false)
    const identity = F.smallEigen([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])
    expect(identity[0]?.eigenvectors).toHaveLength(3)
  })

  it('classifies quadratic forms by inertia', () => {
    expect(
      F.classifyQuadraticForm([
        [2, 0],
        [0, 3],
      ]).definiteness,
    ).toBe('positive definite')
    expect(
      F.classifyQuadraticForm([
        [-2, 0],
        [0, -3],
      ]).definiteness,
    ).toBe('negative definite')
    expect(
      F.classifyQuadraticForm([
        [-1, 1],
        [1, -1],
      ]).definiteness,
    ).toBe('negative semidefinite')
    expect(
      F.classifyQuadraticForm([
        [0, 0],
        [0, 0],
      ]).definiteness,
    ).toBe('positive semidefinite')
    const axes = F.principalAxes([
      [2, 1],
      [1, 2],
    ])
    expect(axes.eigenvalues[0]).toBeCloseTo(1, 14)
    expect(
      Number.isNaN(
        F.principalAxes([
          [-1, 0],
          [0, 1],
        ]).semiAxes[0],
      ),
    ).toBe(true)
  })
})

describe('pictures of linear maps', () => {
  it('the unit cell maps to the parallelogram of the columns with area |det|', () => {
    const g = F.gridTransformation([
      [2, 1],
      [1, 3],
    ])
    const [o, a, b, c] = g.unitCell.after
    expect(o).toEqual([0, 0])
    expect(c).toEqual([1, 3])
    const area = Math.abs((a?.[0] ?? 0) * (c?.[1] ?? 0) - (a?.[1] ?? 0) * (c?.[0] ?? 0))
    expect(area).toBeCloseTo(Math.abs(g.determinant), 12)
    expect(b).toEqual([3, 4])
    expect(g.before).toHaveLength(10)
    const g3 = F.gridTransformation(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      { lines: 2 },
    )
    expect(g3.before).toHaveLength(12)
    expect(() => F.gridTransformation([[1]])).toThrow()
  })

  it('draws scenes in 2-D and 3-D', () => {
    const s = F.linearMapScene([
      [1, 1],
      [0, 1],
    ])
    expect(s.title).toContain('det = 1')
    expect(s.children.filter((n) => n.kind === 'line' && n.arrow === true)).toHaveLength(2)
    expect(() =>
      F.linearMapScene([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    ).toThrow()
    expect(
      F.linearMapScene3d([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
      ]).title,
    ).toContain('det = 2')
    expect(() =>
      F.linearMapScene3d([
        [1, 0],
        [0, 1],
      ]),
    ).toThrow()
  })

  it('the SVD stages turn the circle into the image ellipse', () => {
    const a = [
      [2, 1],
      [0.5, 1.5],
    ]
    const g = F.svdGeometry(a, 16)
    const [circle, , , image] = g.stages
    for (const [i, p] of (circle ?? []).entries()) {
      const want = a.map((r) => r.reduce((s, v, j) => s + v * (p[j] ?? 0), 0))
      for (const [k, v] of want.entries()) expect(image?.[i]?.[k]).toBeCloseTo(v, 12)
    }
    expect(Math.hypot(...(g.axes[0] ?? []))).toBeCloseTo(g.singularValues[0] ?? 0, 12)
    expect(F.svdGeometry(a, 8).stages[0]).toHaveLength(8)
    expect(
      F.svdGeometry(
        [
          [1, 0, 0],
          [0, 2, 0],
          [0, 0, 3],
        ],
        8,
      ).stages[0],
    ).toHaveLength(24)
  })

  it('least squares residual is orthogonal to the columns', () => {
    const r = F.leastSquaresProjection(
      [
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
      ],
      [1, 2, 2, 5],
    )
    for (const v of r.orthogonality) expect(v).toBeCloseTo(0, 12)
    expect(r.normalMatrix).toEqual([
      [4, 6],
      [6, 14],
    ])
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    expectConformant(F.projectionMatrix, [
      tensor([
        [1, 0.5],
        [0.2, 2],
        [1, 1],
      ]),
    ])
    expectConformant(F.quadraticForm, [
      tensor([
        [2, 0.5],
        [0.3, 1],
      ]),
      tensor([
        [1, -2],
        [0.5, 0.7],
      ]),
    ])
  })
})
