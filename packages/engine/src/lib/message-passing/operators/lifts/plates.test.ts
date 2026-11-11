import { describe, expect, it } from 'vitest'
import { probitRows } from '../../../../testing/mp-variants.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as M from '../../index.ts'
import { logPhi } from '../../numerics/_scalar.ts'

const operators = M.defaultOperators()
const close = (a: number, b: number, tol: number): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

const prior = (d: number, variance: number): M.FactorSpec =>
  M.gaussianMvPrior(
    'w',
    new Array<number>(d).fill(0),
    Array.from({ length: d * d }, (_, k) => (k % (d + 1) === 0 ? variance : 0)),
  )

/** The expanded graph: one factor per row on the weights (the row operators). */
function expanded(rows: number[][], labels: number[], op: string, variance = 4): M.CompiledGraph {
  const d = (rows[0] as number[]).length
  return M.compileGraph(
    {
      variables: [
        { name: 'w', family: { kind: 'gaussian-mv', size: d } },
        ...labels.map((l, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: l })),
      ],
      factors: [
        prior(d, variance),
        ...rows.map((a, i): M.FactorSpec => ({ operator: op, args: [`y${i}`, 'w'], params: { a } })),
      ],
    },
    operators,
  )
}

/** The plated graph: one factor for every row. */
function plated(rows: number[][], labels: number[], op: string, variance = 4): M.CompiledGraph {
  const d = (rows[0] as number[]).length
  return M.compileGraph(
    {
      variables: [{ name: 'w', family: { kind: 'gaussian-mv', size: d } }],
      factors: [prior(d, variance), { operator: op, args: ['w'], params: { X: rows.flat(), y: labels } }],
    },
    operators,
  )
}

const settings = (projection: M.Projection): M.EngineSettings => ({
  ...M.epDefaults,
  projection,
  tol: 1e-12,
  maxSweeps: 500,
})

const posterior = (g: M.CompiledGraph, state: M.MpState): { mean: number[]; covariance: number[][] } => {
  const w = M.marginalsOf(g, state)['w'] as Record<string, never>
  return { mean: w['mean'] as unknown as number[], covariance: w['covariance'] as unknown as number[][] }
}

describe('a plate of probit or logistic rows is the expanded graph in one factor', () => {
  for (const [plate, row] of [
    ['probit-plate', 'probit-row'],
    ['logistic-plate', 'logistic-row'],
  ] as const) {
    it(`${plate}: the posterior and the evidence of EP equal the expanded graph's`, () => {
      const { x, y } = probitRows(30)
      const ge = expanded(x, y, row)
      const gp = plated(x, y, plate)
      const se = settings(M.momentProjection)
      const re = runAlgorithm(M.messagePassingAlgorithm(ge, se), { maxSteps: 500 })
      const rp = runAlgorithm(M.messagePassingAlgorithm(gp, se), { maxSteps: 500 })
      expect(re.status).toBe('converged')
      expect(rp.status).toBe('converged')
      const a = posterior(ge, re.state)
      const b = posterior(gp, rp.state)
      for (let i = 0; i < 3; i++) {
        close(b.mean[i] as number, a.mean[i] as number, 1e-7)
        for (let j = 0; j < 3; j++)
          close((b.covariance[i] as number[])[j] as number, (a.covariance[i] as number[])[j] as number, 1e-7)
      }
      close(M.evidenceOf(gp, se, rp.state), M.evidenceOf(ge, se, re.state), 1e-6)
      // The plate is a graph of one variable and two factors, whatever the number of rows.
      expect([gp.variables.length, gp.factors.length]).toEqual([1, 2])
    })
  }

  it('is a graph that does not grow with the rows: 10 rows and 400 give the same shape', () => {
    for (const n of [10, 400]) {
      const { x, y } = probitRows(n)
      const g = plated(x, y, 'probit-plate')
      expect([g.variables.length, g.factors.length, g.edges.length]).toEqual([1, 2, 2])
    }
  })

  it('refuses a design matrix that does not match the outcomes and the weights', () => {
    expect(() => plated([[1, 2]], [1, 0], 'probit-plate')).toThrow(/entries/)
    expect(() => M.glmPlate('none', (({ ep: _e, ...r }) => r)(M.probit), 'x')).toThrow(/no EP rule/)
  })
})

describe('the plate under variational message passing', () => {
  it('probit: the fixed point is the Albert-Chib mean field with a full-covariance block over the weights', () => {
    const { x, y } = probitRows(25)
    const g = plated(x, y, 'probit-plate')
    const alg = M.vmpAlgorithm(g, { tol: 1e-13 })
    const s = settings(M.klqpProjection)
    let state = alg.init(undefined as never)
    let last = Number.NEGATIVE_INFINITY
    for (let k = 0; k < 60; k++) {
      state = alg.step(state, undefined as never)
      const bound = M.evidenceOf(g, s, state)
      expect(bound).toBeGreaterThanOrEqual(last - 1e-9)
      last = bound
    }
    // By hand: q(z_i) the outcome's side of N(a_i . mu, 1); q(w) Gaussian with precision I/4 + X'X and mean Sigma X' E[z].
    let mu = [0, 0, 0]
    const d = 3
    const xtx = new Array<number>(d * d).fill(0)
    for (const row of x)
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++)
          xtx[i * d + j] = (xtx[i * d + j] as number) + (row[i] as number) * (row[j] as number)
    const precision = xtx.map((v, k) => v + (k % (d + 1) === 0 ? 1 / 4 : 0))
    const solve = (b: number[]): number[] => {
      // Gaussian elimination on a 3 x 3 system.
      const m = Array.from({ length: d }, (_, i) => [...precision.slice(i * d, (i + 1) * d), b[i] as number])
      for (let c = 0; c < d; c++) {
        const p = m.findIndex(
          (r, i) => i >= c && Math.abs(r[c] as number) === Math.max(...m.slice(c).map((q) => Math.abs(q[c] as number))),
        )
        ;[m[c], m[p]] = [m[p] as number[], m[c] as number[]]
        for (let r = 0; r < d; r++) {
          if (r === c) continue
          const f = (m[r]?.[c] as number) / (m[c]?.[c] as number)
          for (let k = c; k <= d; k++) (m[r] as number[])[k] = (m[r]?.[k] as number) - f * (m[c]?.[k] as number)
        }
      }
      return m.map((r, i) => (r[d] as number) / (r[i] as number))
    }
    for (let k = 0; k < 300; k++) {
      const ez = x.map((row, i) => {
        const m =
          (row[0] as number) * (mu[0] as number) +
          (row[1] as number) * (mu[1] as number) +
          (row[2] as number) * (mu[2] as number)
        const sign = y[i] === 1 ? 1 : -1
        return m + (sign * Math.exp(-0.5 * m * m)) / Math.sqrt(2 * Math.PI) / Math.exp(logPhi(sign * m))
      })
      mu = solve(
        Array.from({ length: d }, (_, j) => x.reduce((t, row, i) => t + (row[j] as number) * (ez[i] as number), 0)),
      )
    }
    const p = posterior(g, state)
    for (let i = 0; i < d; i++) close(p.mean[i] as number, mu[i] as number, 1e-6)
  })

  it('gaussian: the exact posterior of Bayesian linear regression, and the bound is the evidence of EP', () => {
    const xs = probitRows(40).x
    const ys = xs.map(
      (r) => 0.4 + 0.9 * (r[1] as number) - 0.5 * (r[2] as number) + 0.1 * Math.sin(7 * (r[1] as number)),
    )
    const d = 3
    const tau = 6
    const xtx = new Array<number>(d * d).fill(0)
    const xty = new Array<number>(d).fill(0)
    let yty = 0
    for (const [i, row] of xs.entries()) {
      yty += (ys[i] as number) ** 2
      for (let a = 0; a < d; a++) {
        xty[a] = (xty[a] as number) + (row[a] as number) * (ys[i] as number)
        for (let b = 0; b < d; b++)
          xtx[a * d + b] = (xtx[a * d + b] as number) + (row[a] as number) * (row[b] as number)
      }
    }
    const plate = M.compileGraph(
      {
        variables: [{ name: 'w', family: { kind: 'gaussian-mv', size: d } }],
        factors: [
          prior(d, 4),
          { operator: 'linear-regression-plate', args: ['w'], params: { xtx, xty, yty, n: 40, precision: tau } },
        ],
      },
      operators,
    )
    const rows = M.compileGraph(
      {
        variables: [
          { name: 'w', family: { kind: 'gaussian-mv', size: d } },
          ...ys.map((v, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'gaussian' }, observed: v })),
        ],
        factors: [
          prior(d, 4),
          ...xs.map(
            (a, i): M.FactorSpec => ({
              operator: 'linear-regression-row',
              args: [`y${i}`, 'w'],
              params: { a, precision: tau },
            }),
          ),
        ],
      },
      operators,
    )
    for (const projection of [M.momentProjection, M.klqpProjection]) {
      const s = settings(projection)
      const a = runAlgorithm(M.messagePassingAlgorithm(plate, s), { maxSteps: 50 })
      const b = runAlgorithm(M.messagePassingAlgorithm(rows, s), { maxSteps: 50 })
      const pa = posterior(plate, a.state)
      const pb = posterior(rows, b.state)
      for (let i = 0; i < d; i++) close(pa.mean[i] as number, pb.mean[i] as number, 1e-9)
      close(M.evidenceOf(plate, s, a.state), M.evidenceOf(rows, s, b.state), 1e-9)
    }
    // The posterior of a Gaussian model is Gaussian with this precision: the mean solves the normal equations.
    const p = posterior(
      plate,
      runAlgorithm(M.messagePassingAlgorithm(plate, settings(M.momentProjection)), { maxSteps: 20 }).state,
    )
    for (let i = 0; i < d; i++) {
      let lhs = (p.mean[i] as number) / 4
      for (let j = 0; j < d; j++) lhs += tau * (xtx[i * d + j] as number) * (p.mean[j] as number)
      close(lhs, tau * (xty[i] as number), 1e-9)
    }
  })
})
