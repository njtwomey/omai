import { describe, expect, it } from 'vitest'
import type { RunStatus } from '../trace/index.ts'
import {
  admmLasso,
  boxQp,
  brent,
  conicProgram,
  linearSvmDualCd,
  linprog,
  linprogSimplex,
  milp,
  nnls,
  orthogonalMatchingPursuit,
  quadprog,
  svcDual,
} from './index.ts'

// The failure convention across the optimisation routines (see `trace/events.ts`): every result carries `converged` and
// the run in the one vocabulary; the mathematical-programming results keep their finer `status` (`optimal`, `infeasible`,
// ...) and add `runStatus`, which is what `runStatusOf` makes of it. Invalid data (a NaN, an infinity where a number is
// needed) is an error, not an answer.
interface Outcome {
  readonly converged: boolean
  readonly runStatus?: RunStatus
  readonly status: string
}

const lp = {
  c: [-1, -1],
  aUb: [
    [1, 2],
    [3, 1],
  ],
  bUb: [4, 6],
}
const lp3 = {
  c: [-3, -2, -4],
  aUb: [
    [1, 1, 2],
    [2, 0, 3],
    [2, 1, 3],
  ],
  bUb: [4, 5, 7],
}
const qp = {
  P: [
    [2, 0],
    [0, 2],
  ],
  q: [-2, -5],
  G: [
    [1, 2],
    [-1, 2],
    [-1, -1],
  ],
  h: [6, 2, -1],
}
const soc = {
  c: [0, 1],
  soc: [
    {
      A: [
        [1, 0],
        [1, 0],
      ],
      b: [-3, 1],
      c: [0, 1],
      d: 0,
    },
  ],
}
const knapsack = {
  c: [-5, -4],
  aUb: [
    [6, 4],
    [1, 2],
  ],
  bUb: [24, 6],
  integrality: [1, 1],
}

const programs: readonly (readonly [string, () => Outcome, string, RunStatus])[] = [
  ['linprog, a solution', () => linprog(lp), 'optimal', 'converged'],
  ['linprog, no point', () => linprog({ c: [1], aUb: [[1]], bUb: [-1] }), 'infeasible', 'failed'],
  ['linprog, no minimum', () => linprog({ c: [-1], bounds: [[0, null]] }), 'unbounded', 'failed'],
  ['linprogSimplex, one pivot', () => linprogSimplex(lp3, { maxIterations: 1 }), 'iteration limit', 'completed'],
  ['quadprog, a solution', () => quadprog(qp), 'optimal', 'converged'],
  ['quadprog, no point', () => quadprog({ P: [[1]], q: [0], G: [[1], [-1]], h: [-1, -1] }), 'infeasible', 'failed'],
  ['conicProgram, a solution', () => conicProgram(soc), 'optimal', 'converged'],
  ['conicProgram, one step', () => conicProgram(soc, { maxIterations: 1 }), 'iteration limit', 'completed'],
  ['milp, a solution', () => milp(knapsack), 'optimal', 'converged'],
  ['milp, one node', () => milp(knapsack, { maxNodes: 1 }), 'node limit', 'completed'],
]

const spd = [
  [4, 1, 0],
  [1, 3, 1],
  [0, 1, 2],
]
const rowsOf = (x: number[][]): number[][] => x
const solvers: readonly (readonly [string, () => Outcome, () => Outcome])[] = [
  [
    'brent',
    () => ({ ...brent((x) => (x - 1) ** 2 + 1) }),
    () => ({ ...brent((x) => (x - 1) ** 2 + 1, { maxIterations: 2, tol: 1e-14 }) }),
  ],
  [
    'nnls',
    () =>
      nnls(
        rowsOf([
          [1, 0],
          [0, 1],
          [1, 1],
        ]),
        [1, 2, 3],
      ),
    () =>
      nnls(
        rowsOf([
          [1, 2, 0],
          [3, 1, 1],
          [0, 1, 4],
          [2, 2, 2],
        ]),
        [5, -1, 3, 4],
        { maxIterations: 1 },
      ),
  ],
  [
    'boxQp',
    () =>
      boxQp(
        [
          [2, 0],
          [0, 2],
        ],
        [-4, 2],
        [0, 0],
        [1, 1],
      ),
    () => boxQp(spd, [-9, -8, -7], [0, 0, 0], [1, 1, 1], { maxIterations: 0 }),
  ],
  [
    'svcDual',
    () =>
      svcDual(
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ],
        [1, 1, -1, -1],
        1,
      ),
    () =>
      svcDual(
        [
          [1, 0.9, 0.1, 0],
          [0.9, 1, 0, 0.1],
          [0.1, 0, 1, 0.9],
          [0, 0.1, 0.9, 1],
        ],
        [1, 1, -1, -1],
        1,
        { maxIterations: 1, tol: 1e-12 },
      ),
  ],
  [
    'linearSvmDualCd',
    () =>
      linearSvmDualCd(
        [
          [2, 0],
          [0, 2],
          [-2, 0],
          [0, -2],
        ],
        [1, 1, -1, -1],
        1,
      ),
    () =>
      linearSvmDualCd(
        [
          [2, 1],
          [1, 2],
          [-2, 1],
          [1, -2],
          [0.5, 0.5],
        ],
        [1, 1, -1, -1, 1],
        10,
        { maxSweeps: 1, tol: 1e-14 },
      ),
  ],
]

describe('the optimisation routines share one failure convention', () => {
  it('a programming result keeps its finer status and adds converged and runStatus, aligned with runStatusOf', () => {
    for (const [name, run, status, runStatus] of programs) {
      const r = run()
      expect(r.status, `${name}: status`).toBe(status)
      expect(r.runStatus, `${name}: runStatus`).toBe(runStatus)
      expect(r.converged, `${name}: converged`).toBe(runStatus === 'converged')
    }
  })

  it('the solvers with a plain status report converged, or completed when the budget ran out', () => {
    for (const [name, finishes, runsOut] of solvers) {
      expect(finishes(), name).toMatchObject({ converged: true, status: 'converged' })
      expect(runsOut(), name).toMatchObject({ converged: false, status: 'completed' })
    }
  })

  it('orthogonal matching pursuit is converged when nothing is left to explain, stopped when a repeated atom blocks it', () => {
    const twin = orthogonalMatchingPursuit(
      [
        [1, 1],
        [0, 0],
      ],
      [1, 0],
      { nNonzeroCoefs: 2 },
    )
    expect([twin.converged, twin.status]).toEqual([true, 'converged'])
    const blocked = orthogonalMatchingPursuit(
      [
        [1, 1],
        [0, 0],
      ],
      [1, 2],
      { nNonzeroCoefs: 2 },
    )
    expect([blocked.converged, blocked.status, blocked.support]).toEqual([false, 'stopped', [0]])
  })

  it('invalid data throws value instead of an answer: NaN, an infinity, crossed bounds, a negative weight', () => {
    const calls: readonly (readonly [string, () => unknown])[] = [
      ['linprog with a NaN cost', () => linprog({ ...lp, c: [Number.NaN, 1] })],
      ['linprog with an infinite right-hand side', () => linprog({ ...lp, bUb: [Number.POSITIVE_INFINITY, 6] })],
      [
        'linprog with a NaN in the matrix',
        () =>
          linprog({
            ...lp,
            aUb: [
              [1, Number.NaN],
              [3, 1],
            ],
          }),
      ],
      [
        'linprog with a NaN bound',
        () =>
          linprog({
            ...lp,
            bounds: [
              [Number.NaN, null],
              [0, null],
            ],
          }),
      ],
      [
        'quadprog with a NaN in P',
        () =>
          quadprog({
            ...qp,
            P: [
              [1, 0],
              [0, Number.NaN],
            ],
          }),
      ],
      ['conicProgram with a NaN right-hand side', () => conicProgram({ ...soc, h: [Number.NaN], G: [[1, 0]] })],
      [
        'a dual with a NaN kernel',
        () =>
          svcDual(
            [
              [1, Number.NaN],
              [Number.NaN, 1],
            ],
            [1, -1],
            1,
          ),
      ],
      [
        'nnls with a NaN in A',
        () =>
          nnls(
            [
              [1, 0],
              [0, Number.NaN],
            ],
            [1, 1],
          ),
      ],
      ['boxQp with crossed bounds', () => boxQp([[1]], [0], [1], [0])],
      [
        'admmLasso with a negative lambda',
        () =>
          admmLasso(
            [
              [1, 0],
              [0, 1],
            ],
            [1, 1],
            -1,
          ),
      ],
    ]
    for (const [name, call] of calls) expect(call, name).toThrow(expect.objectContaining({ code: 'value' }))
  })
})
