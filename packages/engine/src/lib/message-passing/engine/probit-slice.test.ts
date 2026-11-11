import { describe, expect, it } from 'vitest'
import { integrateBox } from '../../testing/message-passing.ts'
import { logNdtr } from '../numerics/special/index.ts'
import { gaussianEp } from '../pgm/index.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'

const operators = M.defaultOperators()

/** Forty rows: an intercept and two covariates, labels drawn from a probit model with known weights. */
function dataset(): { x: number[][]; y: number[]; truth: number[] } {
  const k = key(2026)
  const z = toFlat(normal(foldIn(k, 1), [40, 2]))
  const u = toFlat(uniform(foldIn(k, 2), [40]))
  const truth = [0.3, 1.2, -0.8]
  const x = Array.from({ length: 40 }, (_, i) => [1, z[2 * i] as number, z[2 * i + 1] as number])
  const normalCdf = (v: number): number => Math.exp(Number(toFlat(logNdtr(v))[0]))
  const y = x.map((row, i) =>
    (u[i] as number) <
    normalCdf(
      (row[0] as number) * (truth[0] as number) +
        (row[1] as number) * (truth[1] as number) +
        (row[2] as number) * (truth[2] as number),
    )
      ? 1
      : 0,
  )
  return { x, y, truth }
}

/** The posterior over three weights on an adaptive grid: a coarse pass over the prior's range, then two passes over the region the posterior occupies. */
function gridPosterior(
  x: number[][],
  y: number[],
  priorSd: number,
): { mean: number[]; sd: number[]; logEvidence: number } {
  const n = x.length
  const logPosterior = (points: number[][]): Float64Array => {
    const zs = new Float64Array(points.length * n)
    for (const [p, w] of points.entries()) {
      for (const [i, row] of x.entries()) {
        const t =
          (row[0] as number) * (w[0] as number) +
          (row[1] as number) * (w[1] as number) +
          (row[2] as number) * (w[2] as number)
        zs[p * n + i] = y[i] === 1 ? t : -t
      }
    }
    const logs = toFlat(logNdtr(fromStorage(zs, [zs.length], 'float64')))
    return Float64Array.from(points, (w, p) => {
      let s = -0.5 * (w.reduce((a, v) => a + v * v, 0) / priorSd ** 2) - 1.5 * Math.log(2 * Math.PI * priorSd ** 2)
      for (let i = 0; i < n; i++) s += logs[p * n + i] as number
      return s
    })
  }
  let lo = [-4 * priorSd, -4 * priorSd, -4 * priorSd]
  let hi = [4 * priorSd, 4 * priorSd, 4 * priorSd]
  let result = { mean: [0, 0, 0], sd: [1, 1, 1], logEvidence: 0 }
  for (const m of [21, 31, 31]) {
    const axes = [0, 1, 2].map((d) =>
      Array.from({ length: m }, (_, i) => (lo[d] as number) + (((hi[d] as number) - (lo[d] as number)) * i) / (m - 1)),
    )
    const cell = [0, 1, 2].reduce((a, d) => a * (((hi[d] as number) - (lo[d] as number)) / (m - 1)), 1)
    const points: number[][] = []
    for (const a of axes[0] as number[])
      for (const b of axes[1] as number[]) for (const c of axes[2] as number[]) points.push([a, b, c])
    const lp = logPosterior(points)
    const top = Math.max(...lp)
    let z = 0
    const m1 = [0, 0, 0]
    const m2 = [0, 0, 0]
    lp.forEach((v, p) => {
      const w = Math.exp(v - top)
      z += w
      for (let d = 0; d < 3; d++) {
        const c = (points[p] as number[])[d] as number
        m1[d] = (m1[d] as number) + w * c
        m2[d] = (m2[d] as number) + w * c * c
      }
    })
    const mean = m1.map((v) => v / z)
    const sd = m2.map((v, d) => Math.sqrt(v / z - (mean[d] as number) ** 2))
    result = { mean, sd, logEvidence: top + Math.log(z * cell) }
    lo = mean.map((v, d) => v - 7 * (sd[d] as number))
    hi = mean.map((v, d) => v + 7 * (sd[d] as number))
  }
  return result
}

describe('the probit regression slice: EP on a hand-built factor graph against an exact grid', () => {
  const { x, y } = dataset()
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
      ...x.map((_, i): M.VariableSpec => ({ name: `s${i}`, family: { kind: 'gaussian' } })),
      ...x.map((_, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: y[i] as number })),
    ],
    factors: [
      M.gaussianMvPrior('w', [0, 0, 0], [4, 0, 0, 0, 4, 0, 0, 0, 4]),
      ...x.flatMap((row, i): M.FactorSpec[] => [
        { operator: 'dot-constant', args: [`s${i}`, 'w'], params: { a: row } },
        { operator: 'probit', args: [`y${i}`, `s${i}`] },
      ]),
    ],
  }
  const g = M.compileGraph(spec, operators)
  it('EP means are within 0.05 posterior standard deviations of the grid, and the evidence is close', () => {
    const settings: M.EngineSettings = {
      projection: M.momentProjection,
      schedule: 'serial',
      granularity: 'sweep',
      damping: 0,
      improper: 'damp',
      tol: 1e-10,
      maxSweeps: 200,
    }
    const r = runAlgorithm(M.messagePassingAlgorithm(g, settings), { maxSteps: 200 })
    expect(r.status).toBe('converged')
    const grid = gridPosterior(x, y, 2)
    const w = M.marginalsOf(g, r.state)['w'] as Record<string, never>
    const mean = w['mean'] as unknown as number[]
    const variance = w['variance'] as unknown as number[]
    for (let d = 0; d < 3; d++) {
      expect(Math.abs((mean[d] as number) - (grid.mean[d] as number)) / (grid.sd[d] as number)).toBeLessThan(0.05)
      expect(Math.sqrt(variance[d] as number) / (grid.sd[d] as number)).toBeGreaterThan(0.9)
      expect(Math.sqrt(variance[d] as number) / (grid.sd[d] as number)).toBeLessThan(1.1)
    }
    expect(Math.abs(M.evidenceOf(g, settings, r.state) - grid.logEvidence)).toBeLessThan(0.05)
    expect(r.state.improper.events).toBe(0)
  })
})

describe('the clutter problem (Minka 2001): EP with numeric sites against the exact one-dimensional posterior', () => {
  const w = 0.25
  const data = [1.8, 2.3, 2.0, 1.6, 2.4, 2.1, 1.9, 2.6, 1.7, 2.2, -3.1, 5.2, 0.4, -1.5]
  const density = (theta: number, yi: number): number =>
    ((1 - w) * Math.exp(-0.5 * (yi - theta) ** 2)) / Math.sqrt(2 * Math.PI) +
    (w * Math.exp((-0.5 * yi * yi) / 10)) / Math.sqrt(20 * Math.PI)
  const site = M.numericOperator({
    name: 'clutter',
    arity: 2,
    logDensity: ([yi, t]) => Math.log(density(t as number, yi as number)),
  })
  const spec: M.FactorGraphSpec = {
    variables: [
      { name: 'theta', family: { kind: 'gaussian' } },
      ...data.map((v, i): M.VariableSpec => ({ name: `y${i}`, family: { kind: 'gaussian' }, observed: v })),
    ],
    factors: [
      M.gaussianPrior('theta', 0, 100),
      ...data.map((_, i): M.FactorSpec => ({ operator: 'clutter', args: [`y${i}`, 'theta'] })),
    ],
  }
  const g = M.compileGraph(spec, { ...operators, clutter: site })
  // The exact posterior mean and standard deviation by quadrature.
  const post = (t: number): number => Math.exp((-0.5 * t * t) / 100) * data.reduce((p, yi) => p * density(t, yi), 1)
  const z = integrateBox((p) => post(p[0] as number), [[-12, 12]], 80, 12)
  const mean = integrateBox((p) => (p[0] as number) * post(p[0] as number), [[-12, 12]], 80, 12) / z
  const sd = Math.sqrt(
    integrateBox((p) => ((p[0] as number) - mean) ** 2 * post(p[0] as number), [[-12, 12]], 80, 12) / z,
  )
  it('reaches the fixed point of the Gaussian-latent EP of pgm with custom sites, near the exact posterior', () => {
    const r = runAlgorithm(M.epAlgorithm(g, { tol: 1e-9, damping: 0.3 }), { maxSteps: 300 })
    expect(r.status).toBe('converged')
    const m = M.marginalsOf(g, r.state)['theta'] as Record<string, never>
    const ref = gaussianEp(
      { mean: [0], covariance: [[100]] },
      data.map((yi) => ({
        kind: 'custom' as const,
        weights: [1],
        logLikelihood: (t: number) => Math.log(density(t, yi)),
      })),
      { tol: 1e-10, maxIter: 500, quadraturePoints: 128 },
    )
    expect(Math.abs((m['mean'] as unknown as number) - (ref.mean[0] as number))).toBeLessThan(2e-3)
    expect(
      Math.abs((m['variance'] as unknown as number) - ((ref.covariance[0] as number[])[0] as number)),
    ).toBeLessThan(2e-3)
    // EP is an approximation here: within half a posterior standard deviation of the exact mean, and a similar spread.
    expect(Math.abs((m['mean'] as unknown as number) - mean) / sd).toBeLessThan(0.3)
    expect(Math.sqrt(m['variance'] as unknown as number) / sd).toBeGreaterThan(0.8)
    expect(Math.sqrt(m['variance'] as unknown as number) / sd).toBeLessThan(1.2)
  })
  it('flooding and serial sweeps reach the same fixed point, with nothing to repair', () => {
    const serial = runAlgorithm(M.epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 300 })
    const flooding = runAlgorithm(M.epAlgorithm(g, { schedule: 'flooding', tol: 1e-10, maxSweeps: 400 }), {
      maxSteps: 400,
    })
    expect(flooding.status).toBe('converged')
    const a = M.marginalsOf(g, serial.state)['theta']
    const b = M.marginalsOf(g, flooding.state)['theta']
    expect(Math.abs((a?.['mean'] as number) - (b?.['mean'] as number))).toBeLessThan(1e-6)
    expect(flooding.state.improper.events).toBe(0)
  })
})
