import { describe, expect, it } from 'vitest'
import { integrateBox } from '../../../../testing/message-passing.ts'
import { probitRows } from '../../../../testing/mp-variants.ts'
import { sigmoidOf } from '../../../base/index.ts'
import { gaussianEp } from '../../../pgm/index.ts'
import { eye, mul, toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as M from '../../index.ts'
import { zeroArray } from '../../numerics/_arrays.ts'
import { millsRatio } from '../../numerics/_scalar.ts'

const operators = M.defaultOperators()
const close = (a: number, b: number, tol: number): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}
const normalDensity = (x: number, v: number): number => Math.exp((-0.5 * x * x) / v) / Math.sqrt(2 * Math.PI * v)

/** Gaussian-vector prior of variance 4 and rows of probit or logistic regression on it. */
function rowsGraph(rows: number[][], labels: number[], op: string): M.CompiledGraph {
  const d = (rows[0] as number[]).length
  return M.compileGraph(
    {
      variables: [
        { name: 'w', family: { kind: 'gaussian-mv', size: d } },
        ...labels.map((l, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: l })),
      ],
      factors: [
        M.gaussianMvPrior('w', zeroArray(d), toFlat(mul(eye(d), 4))),
        ...rows.map((a, i): M.FactorSpec => ({ operator: op, args: [`y${i}`, 'w'], params: { a } })),
      ],
    },
    operators,
  )
}

const posterior = (g: M.CompiledGraph, state: M.MpState): { mean: number[]; covariance: number[][] } => {
  const w = M.marginalsOf(g, state)['w'] as Record<string, never>
  return { mean: w['mean'] as unknown as number[], covariance: w['covariance'] as unknown as number[][] }
}

describe('regression row operators', () => {
  it('probit-row reproduces the Gaussian-latent EP of pgm, posterior and evidence', () => {
    const { x, y } = probitRows(10)
    const g = rowsGraph(x, y, 'probit-row')
    const s: M.EngineSettings = { ...M.epDefaults, projection: M.momentProjection, tol: 1e-12, maxSweeps: 500 }
    const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    const ref = gaussianEp(
      {
        mean: [0, 0, 0],
        covariance: [
          [4, 0, 0],
          [0, 4, 0],
          [0, 0, 4],
        ],
      },
      x.map((row, i) => ({ kind: 'probit' as const, weights: row, label: y[i] === 1 ? (1 as const) : (-1 as const) })),
      { tol: 1e-13, maxIter: 500 },
    )
    const p = posterior(g, r.state)
    for (let i = 0; i < 3; i++) {
      close(p.mean[i] as number, ref.mean[i] as number, 1e-7)
      for (let j = 0; j < 3; j++)
        close((p.covariance[i] as number[])[j] as number, (ref.covariance[i] as number[])[j] as number, 1e-7)
    }
    close(M.evidenceOf(g, s, r.state), ref.logMarginalLikelihood, 1e-7)
  })

  it('one row is matched exactly: the posterior of the weights and the evidence are the brute-force ones, for both links', () => {
    const a = [1, 0.8]
    const prior = (w: readonly number[]): number => normalDensity(w[0] as number, 4) * normalDensity(w[1] as number, 4)
    for (const op of ['probit-row', 'logistic-row']) {
      const g = rowsGraph([a], [1], op)
      const s: M.EngineSettings = { ...M.epDefaults, projection: M.momentProjection, tol: 1e-12 }
      const r = runAlgorithm(M.messagePassingAlgorithm(g, s), { maxSteps: 50 })
      // Phi(z) = phi(z) / (phi(z) / Phi(z)): the normal density over the Mills ratio of the engine's own helpers.
      const lik = op === 'probit-row' ? (z: number): number => normalDensity(z, 1) / millsRatio(z) : sigmoidOf
      const f = (w: readonly number[]): number =>
        prior(w) * lik((a[0] as number) * (w[0] as number) + (a[1] as number) * (w[1] as number))
      const box: [number, number][] = [
        [-12, 12],
        [-12, 12],
      ]
      const z = integrateBox(f, box, 12, 16)
      const m0 = integrateBox((w) => (w[0] as number) * f(w), box, 12, 16) / z
      const m1 = integrateBox((w) => (w[1] as number) * f(w), box, 12, 16) / z
      const p = posterior(g, r.state)
      close(p.mean[0] as number, m0, 1e-4)
      close(p.mean[1] as number, m1, 1e-4)
      close(M.evidenceOf(g, s, r.state), Math.log(z), 1e-4)
    }
  })

  it('refuses a row of the wrong length, an unobserved outcome, and a link without an EP rule', () => {
    expect(() =>
      M.compileGraph(
        {
          variables: [
            { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
            { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
          ],
          factors: [{ operator: 'probit-row', args: ['y', 'w'], params: { a: [1, 2] } }],
        },
        operators,
      ),
    ).toThrow(/length/)
    expect(() =>
      M.compileGraph(
        {
          variables: [
            { name: 'w', family: { kind: 'gaussian-mv', size: 2 } },
            { name: 'y', family: { kind: 'bernoulli' } },
          ],
          factors: [{ operator: 'probit-row', args: ['y', 'w'], params: { a: [1, 2] } }],
        },
        operators,
      ),
    ).toThrow(/observed/)
    const { ep: _ep, ...withoutEp } = M.probit
    expect(() => M.glmRow('none', withoutEp, 'x')).toThrow(/no EP rule/)
  })

  it('probit-row takes a power: power EP on the rows reproduces the power EP of pgm', () => {
    const { x, y } = probitRows(8)
    const g = M.withPowers(rowsGraph(x, y, 'probit-row'), (f) => (f.operator.power === true ? 0.5 : 1))
    const r = runAlgorithm(
      M.messagePassingAlgorithm(g, { ...M.epDefaults, projection: M.momentProjection, tol: 1e-11, maxSweeps: 1000 }),
      { maxSteps: 1000 },
    )
    expect(r.status).toBe('converged')
    const ref = gaussianEp(
      {
        mean: [0, 0, 0],
        covariance: [
          [4, 0, 0],
          [0, 4, 0],
          [0, 0, 4],
        ],
      },
      x.map((row, i) => ({ kind: 'probit' as const, weights: row, label: y[i] === 1 ? (1 as const) : (-1 as const) })),
      { power: 0.5, tol: 1e-12, maxIter: 2000, quadraturePoints: 96 },
    )
    const p = posterior(g, r.state)
    for (let i = 0; i < 3; i++) close(p.mean[i] as number, ref.mean[i] as number, 2e-5)
  })
})
