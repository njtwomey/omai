import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

function problemOf(inputs: readonly unknown[]): O.LinearProgram {
  const [c, aUb, bUb, aEq, bEq, bounds] = inputs.map((x) => decode(x) as Tensor) as Tensor[]
  const b = toArray(bounds as Tensor) as number[][]
  return {
    c: c as Tensor,
    ...((aUb as Tensor).shape[0] === 0 ? {} : { aUb: aUb as Tensor, bUb: bUb as Tensor }),
    ...((aEq as Tensor).shape[0] === 0 ? {} : { aEq: aEq as Tensor, bEq: bEq as Tensor }),
    bounds: b.map(([lo, hi]) => [Number.isFinite(lo) ? (lo ?? 0) : null, Number.isFinite(hi) ? (hi ?? 0) : null]),
  }
}

const DEGENERATE = new Set(['mixed-bounds'])

const METHODS = { simplex: O.linprogSimplex, interiorPoint: O.linprogInteriorPoint } as const

describe('linprog against scipy (HiGHS)', () => {
  for (const c of loadFixture('optim/linear_programming').cases) {
    for (const [name, method] of Object.entries(METHODS)) {
      it(`${c.name} by ${name}`, () => {
        const out = c.output as Out
        const problem = problemOf(c.inputs)
        const r = O.linprog(problem, { method })
        expect(r.status).toBe(out['status'])
        if (out['status'] !== 'optimal') return
        const tol = name === 'simplex' ? c.atol : 1e-6
        expectMatches(r.x, out['x'], tol, tol, `${c.name} x`)
        expectMatches(r.fun, out['fun'], tol, tol, `${c.name} fun`)
        // Duals of a degenerate vertex are not unique: the interior point finds another optimal dual. Check that it is
        // one: stationarity c = A_ub^T ineqlin + A_eq^T eqlin + lower + upper, and strong duality.
        if (name === 'interiorPoint' && DEGENERATE.has(c.name)) {
          const [cv, aUb, bUb, aEq, bEq, bounds] = c.inputs.map((x) => toArray(decode(x) as Tensor)) as number[][][]
          const m = r.marginals
          const [yu, ye, lo, up] = [m.ineqlin, m.eqlin, m.lower, m.upper].map((t) => toFlat(t))
          const grad = (cv as unknown as number[]).map(
            (_, j) =>
              (aUb ?? []).reduce((s, row, i) => s + (row[j] ?? 0) * (yu?.[i] ?? 0), 0) +
              (aEq ?? []).reduce((s, row, i) => s + (row[j] ?? 0) * (ye?.[i] ?? 0), 0) +
              (lo?.[j] ?? 0) +
              (up?.[j] ?? 0),
          )
          grad.forEach((g, j) => {
            expect(g).toBeCloseTo((cv as unknown as number[])[j] ?? 0, 6)
          })
          let dual = 0
          for (const [i, v] of ((bUb as unknown as number[]) ?? []).entries()) dual += v * (yu?.[i] ?? 0)
          for (const [i, v] of ((bEq as unknown as number[]) ?? []).entries()) dual += v * (ye?.[i] ?? 0)
          for (const [j, [l, u]] of (bounds ?? []).entries()) {
            if (Number.isFinite(l)) dual += (l ?? 0) * (lo?.[j] ?? 0)
            if (Number.isFinite(u)) dual += (u ?? 0) * (up?.[j] ?? 0)
          }
          expect(dual).toBeCloseTo(r.fun, 6)
          return
        }
        expectMatches(r.marginals.lower, out['lower'], tol, tol, `${c.name} lower`)
        expectMatches(r.marginals.upper, out['upper'], tol, tol, `${c.name} upper`)
        expectMatches(r.marginals.ineqlin, out['ineqlin'], tol, tol, `${c.name} ineqlin`)
        if (c.name === 'redundant') {
          // Duals of dependent rows are not unique; their combination A_eq^T y is.
          const y = toFlat(r.marginals.eqlin)
          const want = toFlat(decode(out['eqlin']) as Tensor)
          const combo = (v: number[]) => [v[0] ?? 0, v[1] ?? 0].reduce((s, q, i) => s + q * (i + 1), 0)
          expect(combo(y)).toBeCloseTo(combo(want), 6)
          expect(y[2] ?? 0).toBeCloseTo(want[2] ?? 0, 6)
        } else expectMatches(r.marginals.eqlin, out['eqlin'], tol, tol, `${c.name} eqlin`)
      })
    }
  }
})

describe('simplex', () => {
  const textbook: O.LinearProgram = {
    c: [-1, -2],
    aUb: [
      [1, 1],
      [1, 3],
    ],
    bUb: [4, 6],
  }

  it('reports slack, residuals and the vertex path', () => {
    const r = O.linprog(textbook)
    expect(toFlat(r.slack)).toEqual([0, 0])
    expect(r.path.map((p) => p.phase)).toEqual([2, 2, 2])
    expect(r.path.at(-1)?.objective).toBe(-5)
    expect(r.basis).toHaveLength(2)
  })

  it('agrees under both pivot rules and is traced pivot by pivot', () => {
    const dantzig = O.linprog(textbook, { pivotRule: 'dantzig' })
    expect(toFlat(dantzig.x).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([3, 1])
    expect(dantzig.path.map((p) => p.x.map((v) => Math.round(v * 1e12) / 1e12))).toEqual([
      [0, 0],
      [0, 2],
      [3, 1],
    ])
    const rec = recorder()
    O.linprog(textbook, { tracer: rec, policy: tracePolicy({ messages: true }) })
    const run = rec.run()
    expect(messages(run, 'pivot')).toHaveLength(2)
    expect(metricSeries(run, 'objective').values.at(-1)).toBe(-5)
  })

  it('runs phase 1 when the origin is infeasible', () => {
    const r = O.linprog({
      c: [3, 2],
      aUb: [
        [-1, -1],
        [-2, -1],
      ],
      bUb: [-3, -4],
    })
    expect(r.path[0]?.phase).toBe(1)
    expect(toFlat(r.x)).toEqual([1, 2])
  })

  it('gives sensitivity ranges that predict when the basis changes', () => {
    const r = O.linprog(textbook)
    const s = r.sensitivity as O.Sensitivity
    // c_1 in [-2, -2/3], c_2 in [-3, -1], b_1 in [2, 6], b_2 in [4, 12].
    expect(s.cost.map(([lo, hi]) => [lo, hi].map((v) => Math.round(v * 1e9) / 1e9))).toEqual([
      [-2, -0.666666667],
      [-3, -1],
    ])
    expect(s.bUb).toEqual([
      [2, 6],
      [4, 12],
    ])
    // Inside the cost range the solution stays put; outside it moves.
    expect(toFlat(O.linprog({ ...textbook, c: [-1.9, -2] }).x)).toEqual([3, 1])
    expect(toFlat(O.linprog({ ...textbook, c: [-2.1, -2] }).x)).toEqual([4, 0])
    // Inside the rhs range the dual prices stay the same.
    expect(toFlat(O.linprog({ ...textbook, bUb: [5.5, 6] }).marginals.ineqlin)).toEqual(
      toFlat(O.linprog(textbook).marginals.ineqlin),
    )
  })

  it('handles a degenerate program without cycling (Beale)', () => {
    // Beale's classic cycling example under Dantzig's rule with textbook ties; Bland's rule terminates.
    const r = O.linprog({
      c: [-0.75, 150, -0.02, 6],
      aUb: [
        [0.25, -60, -0.04, 9],
        [0.5, -90, -0.02, 3],
        [0, 0, 1, 0],
      ],
      bUb: [0, 0, 1],
    })
    expect(r.status).toBe('optimal')
    expect(r.fun).toBeCloseTo(-0.05, 12)
  })

  it('stops at the iteration limit', () => {
    const r = O.linprog(textbook, { maxIterations: 1 })
    expect(r.status).toBe('iteration limit')
    expect(r.success).toBe(false)
  })

  it('converts to standard form and back', () => {
    const s = O.standardForm({
      c: [1, 1, 1],
      bounds: [
        [-1, 2],
        [null, 3],
        [null, null],
      ],
    })
    expect(s.columns.map((c) => c.kind)).toEqual(['shift', 'reflect', 'plus', 'minus', 'slack'])
    expect(Array.from(O.fromStandard(s, [1, 1, 2, 5, 0]))).toEqual([0, 2, -3])
    expect(s.offset).toBe(-1 + 3)
  })
})

describe('interior point', () => {
  it('records residuals that fall to the tolerance', () => {
    const form = O.standardForm({
      c: [-1, -2],
      aUb: [
        [1, 1],
        [1, 3],
      ],
      bUb: [4, 6],
    })
    const rec = recorder()
    const r = runAlgorithm(O.lpInteriorPointAlgorithm(form), { maxSteps: 100, tracer: rec })
    expect(r.status).toBe('converged')
    const gap = metricSeries(rec.run(), 'gap').values
    expect(gap.at(-1)).toBeLessThan(1e-8)
    expect(gap[0]).toBeGreaterThan(1e-3)
  })

  it('detects an inconsistent system of equalities before iterating', () => {
    const r = O.linprogInteriorPoint({
      c: [1, 1],
      aEq: [
        [1, 1],
        [2, 2],
      ],
      bEq: [1, 3],
    })
    expect([r.status, r.iterations]).toEqual(['infeasible', 0])
  })
})

// The welfare-maximising correlated equilibrium of a bimatrix game as an LP in the 16 joint probabilities: the
// programs on which the simplex used to report an infeasible point as optimal (payoff differences of 1e-6 beside 1).
function correlatedEquilibriumLp(a: readonly (readonly number[])[], b: readonly (readonly number[])[]) {
  const n = a.length
  const m = (a[0] ?? []).length
  const idx = (i: number, j: number): number => i * m + j
  const aUb: number[][] = []
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      if (k === i) continue
      const row = new Array<number>(n * m).fill(0)
      for (let j = 0; j < m; j++) row[idx(i, j)] = (a[k]?.[j] ?? 0) - (a[i]?.[j] ?? 0)
      aUb.push(row)
    }
  for (let j = 0; j < m; j++)
    for (let k = 0; k < m; k++) {
      if (k === j) continue
      const row = new Array<number>(n * m).fill(0)
      for (let i = 0; i < n; i++) row[idx(i, j)] = (b[i]?.[k] ?? 0) - (b[i]?.[j] ?? 0)
      aUb.push(row)
    }
  const c = Array.from(
    { length: n * m },
    (_, t) => -((a[Math.floor(t / m)]?.[t % m] ?? 0) + (b[Math.floor(t / m)]?.[t % m] ?? 0)),
  )
  return { c, aUb, bUb: new Array<number>(aUb.length).fill(0), aEq: [new Array<number>(n * m).fill(1)], bEq: [1] }
}

describe('simplex robustness on near-tied programs', () => {
  it('solves the correlated-equilibrium program that used to come back infeasible-but-optimal', () => {
    const A = [
      [-0.5, -0.06416795641680001, -0.49998299999830004, -0.4999959999996],
      [-0.3002129800213, -0.4999939999994, -0.4999819999982, -0.4999859999986],
      [-0.4999909999991, -0.49999499999950003, 0.39738708973870007, -0.4999929999993],
      [-0.49999199999919997, -0.4999819999982, -0.4999929999993, 0.34787108478710005],
    ]
    const B = [
      [-0.49999199999919997, -0.4999909999991, -0.49999499999950003, -0.4999929999993],
      [-0.4999819999982, -0.4999889999989, -0.3126139812614, -0.49999699999970004],
      [0.49999009999900007, -0.25201697520169997, 0.49999809999980005, -0.1439759643976],
      [0.49999309999930003, -0.49999499999950003, -0.1422289642229, 0.49998309999829993],
    ]
    const lp = correlatedEquilibriumLp(A, B)
    // The simplex alone either finds a feasible optimum or reports numerical difficulties; linprog always succeeds.
    const raw = O.linprogSimplex(lp)
    if (raw.status === 'optimal') {
      expect(Math.abs(toFlat(raw.con)[0] ?? 1)).toBeLessThan(1e-9)
      for (const v of toFlat(raw.slack)) expect(v).toBeGreaterThan(-1e-9)
    } else expect(raw.status).toBe('numerical difficulties')
    const r = O.linprog(lp)
    expect(r.status).toBe('optimal')
    expect(Math.abs(toFlat(r.con)[0] ?? 1)).toBeLessThan(1e-7)
    for (const v of toFlat(r.slack)) expect(v).toBeGreaterThan(-1e-7)
    expect(r.fun).toBeCloseTo(O.linprogInteriorPoint(lp).fun, 6)
  })

  it('does not pivot on a tiny entry of a degenerate program (a coarse equilibrium program on nearly tied payoffs)', () => {
    // fast-check found this program (seed 2075411073): every right-hand side is 0, so every ratio ties and the simplex
    // took pivots of 1e-6 beside entries of 1 and then 9e-9 of rounding noise, ended on a singular basis with
    // 'numerical difficulties' for both rules, and the interior point called the bounded program unbounded.
    const lp = {
      c: [
        0.9710649971065, 0.9999939999994, 0.5832659583266, 0.10772791077280003, 1, 0.8674859867486, -0.9999531999953,
        0.512329951233, 3.9000004000144806e-6,
      ],
      aUb: [
        [0, 0, 0, -0.8025397616566224, 0, -1.0298279364100302e-6, -1, -0.5022161897190217, 7.208795554412876e-6],
        [1, 0, 1.2832111075521467e-6, 0, 0, 0, -0.2460441809584304, -0.6257835607825021, 1.0265688859847313e-5],
        [
          1, 0.5022161897190217, -7.208795554412876e-6, 0.19746023834337761, 0.5022161897190217, -8.238623490822907e-6,
          0, 0, 0,
        ],
        [
          0, -4.000064001041373e-6, -0.41673166770668335, 0, 0.08404534472551563, -0.04846277540440647, 0, 1,
          -1.2000192003068606e-5,
        ],
        [
          4.000016000081356e-6, 0, -0.41672266689066756, -0.08404433617734472, 0, -0.13250653002612012,
          -0.9999879999519997, 0, -1,
        ],
        [
          0.41672666690666765, 0.41672266689066756, 0, 0.0484621938487754, 0.13250653002612012, 0,
          1.2000048000188558e-5, 1, 0,
        ],
      ],
      bUb: [0, 0, 0, 0, 0, 0],
      aEq: [[1, 1, 1, 1, 1, 1, 1, 1, 1]],
      bEq: [1],
      bounds: Array.from({ length: 9 }, () => [0, null] as const),
    }
    for (const pivotRule of ['bland', 'dantzig'] as const) {
      const r = O.linprogSimplex(lp, { pivotRule })
      expect(r.status).toBe('optimal')
      expect(r.fun).toBeCloseTo(0.8674859867486, 9)
      expect(Math.abs(toFlat(r.con)[0] ?? 1)).toBeLessThan(1e-9)
      for (const v of toFlat(r.slack)) expect(v).toBeGreaterThan(-1e-9)
    }
  })

  it('cross-checks false verdicts on near-tied programs (simplex said infeasible, interior point unbounded)', () => {
    const base = [1, -1, 0, 1, 3, 2, 0, 0, 0, -1, -1, -3, 0, -1, 0, 0, 1, 0]
    const jitter = [1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 2, 5, -4, 1, 0, 0, 0]
    const pay = (k: number): number => (base[k] ?? 0) / 2 + (jitter[k] ?? 0) * 1e-6
    const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => pay(i * 3 + j)))
    const B = [0, 1, 2].map((i) => [0, 1, 2].map((j) => pay(9 + i * 3 + j)))
    const lp = correlatedEquilibriumLp(A, B)
    expect(['optimal', 'numerical difficulties']).toContain(O.linprogSimplex(lp).status)
    for (const r of [O.linprog(lp)]) {
      expect(r.status).toBe('optimal')
      expect(Math.abs(toFlat(r.con)[0] ?? 1)).toBeLessThan(1e-7)
      for (const v of toFlat(r.slack)) expect(v).toBeGreaterThan(-1e-7)
    }
  })

  it('reports a feasible optimum on random near-tied games', () => {
    // Objective values are not compared with other solvers: on near ties a 1e-12 violation (within any solver's
    // tolerance) can move the optimum a long way, as with HiGHS against the interior point here. What the simplex
    // must never do is call an infeasible point optimal, or a feasible program infeasible.
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -3, max: 3 }), { minLength: 18, maxLength: 18 }),
        fc.array(fc.integer({ min: -5, max: 5 }), { minLength: 18, maxLength: 18 }),
        (base, jitter) => {
          // Payoffs on a coarse grid plus 1e-6-sized jitter: many near ties.
          const pay = (k: number): number => (base[k] ?? 0) / 2 + (jitter[k] ?? 0) * 1e-6
          const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => pay(i * 3 + j)))
          const B = [0, 1, 2].map((i) => [0, 1, 2].map((j) => pay(9 + i * 3 + j)))
          const lp = correlatedEquilibriumLp(A, B)
          // The simplex alone never calls an infeasible point optimal (it may give up with numerical difficulties)...
          const raw = O.linprogSimplex(lp)
          if (raw.status === 'optimal') {
            expect(Math.abs(toFlat(raw.con)[0] ?? 1)).toBeLessThan(1e-8)
            for (const v of toFlat(raw.slack)) expect(v).toBeGreaterThan(-1e-8)
          } else expect(['numerical difficulties', 'infeasible', 'unbounded']).toContain(raw.status)
          // ...and linprog, falling back to the interior point, always finds the feasible optimum (to the
          // interior point's accuracy on these deliberately ill-conditioned programs).
          const r = O.linprog(lp)
          expect(r.status).toBe('optimal')
          expect(Math.abs(toFlat(r.con)[0] ?? 1)).toBeLessThan(1e-6)
          for (const v of toFlat(r.slack)) expect(v).toBeGreaterThan(-1e-6)
        },
      ),
      { numRuns: 60 },
    )
  })
})
