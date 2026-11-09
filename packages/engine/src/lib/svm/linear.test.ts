import { describe, expect, it } from 'vitest'
import { solverConfigOf } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, stateFromJson, stateToJson } from '../trace/index.ts'
import { linearSvc, linearSvcSolvers, linearSvr, linearSvrSolvers } from './linear.ts'

const svcSolver = (config: object): never => solverConfigOf(linearSvcSolvers, config as never) as never
const svrSolver = (config: object): never => solverConfigOf(linearSvrSolvers, config as never) as never

const gauss = (seed: number, n: number, d: number): number[][] => {
  const z = normal(key(seed), [n * d]).data
  return Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => z[i * d + j] ?? 0))
}

/** k overlapping blobs in the plane. */
const blobs = (k: number, perClass: number, seed: number): { x: number[][]; y: number[] } => {
  const noise = gauss(seed, k * perClass, 2)
  const centres = [
    [0, 0],
    [3, 0],
    [0, 3],
    [3, 3],
  ]
  const y = Array.from({ length: k * perClass }, (_, i) => i % k)
  return { x: noise.map((r, i) => r.map((v, j) => 0.8 * v + (centres[y[i] ?? 0]?.[j] ?? 0))), y }
}

const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] ?? 0))).toBeLessThan(tol)
}

describe('linearSvc', () => {
  const data = blobs(2, 30, 1)
  const exact = { tol: 1e-10, maxIter: 20_000 }

  it('every solver of the slot reaches the same weights for the squared hinge', () => {
    const fit = (solver: object) => linearSvc.with({ ...exact, solver: svcSolver(solver) }).fit(data).params
    const dcd = fit({ kind: 'dcd' })
    for (const kind of ['newton', 'lbfgs', 'bfgs'])
      close(
        [...(fit({ kind, gradTol: 1e-10 }).coef[0] ?? []), ...fit({ kind, gradTol: 1e-10 }).intercept],
        [...(dcd.coef[0] ?? []), ...dcd.intercept],
        1e-5,
      )
    expect(dcd.converged).toBe(true)
  })

  it('the hinge needs dual coordinate descent; the L1 penalty needs OWL-QN, which the default becomes', () => {
    const hinge = linearSvc.with({ loss: 'hinge', ...exact })
    expect(hinge.fit(data).params.converged).toBe(true)
    expect(() => linearSvc.with({ loss: 'hinge', solver: svcSolver({ kind: 'lbfgs' }) }).fit(data)).toThrow(
      /not smooth/,
    )
    expect(() => linearSvc.with({ penalty: 'l1', loss: 'hinge' }).fit(data)).toThrow(/squared hinge only/)
    expect(() => linearSvc.with({ penalty: 'l1', solver: svcSolver({ kind: 'lbfgs' }) }).fit(data)).toThrow(/owlqn/)
    expect(() => linearSvc.with({ solver: svcSolver({ kind: 'owlqn' }) }).fit(data)).toThrow(/L1 penalty/)
    expect(linearSvc.with({ penalty: 'l1', C: 1, tol: 1e-8 }).fit(data).score(data)).toBeGreaterThan(0.9)
  })

  it('the L1 penalty gives exact zeros on irrelevant features, the L2 penalty does not', () => {
    const noise = gauss(5, 60, 8)
    const x = noise.map((r, i) => [
      ...r.slice(0, 6),
      (i % 2 === 0 ? 2.5 : -2.5) + 0.3 * (r[6] ?? 0),
      (i % 2 === 0 ? 1 : -1) * 0.5 + 0.3 * (r[7] ?? 0),
    ])
    const y = Array.from({ length: 60 }, (_, i) => i % 2)
    const l1 = linearSvc.with({ penalty: 'l1', C: 0.05, tol: 1e-10 }).fit({ x, y }).params.coef[0] ?? []
    const l2 = linearSvc.with({ C: 0.05, tol: 1e-10 }).fit({ x, y }).params.coef[0] ?? []
    expect(l1.slice(0, 6).filter((v) => v === 0).length).toBeGreaterThan(2)
    expect(l2.filter((v) => v === 0).length).toBe(0)
    expect(Math.abs(l1[6] ?? 0)).toBeGreaterThan(0.1)
  })

  it('more than two classes are one-versus-rest, one problem after another on the tape', () => {
    const three = blobs(3, 25, 2)
    const f = linearSvc.with({ C: 1, tol: 1e-6 }).fit(three, { logs: 'all' })
    expect(f.params.classes).toEqual([0, 1, 2])
    expect(f.params.coef.length).toBe(3)
    expect(f.score(three)).toBeGreaterThan(0.85)
    expect(f.decisionFunction(three.x).shape).toEqual([75, 3])
    const problem = metricSeries(f.run, 'solver/problem').values
    expect(problem[0]).toBe(0)
    expect(problem[problem.length - 1]).toBe(2)
    expect(problem.every((v, i) => i === 0 || v >= (problem[i - 1] ?? 0))).toBe(true)
    // Two classes: one row of weights, positive for the second class.
    const two = linearSvc.fit(data)
    expect(two.params.coef.length).toBe(1)
    expect(two.decisionFunction(data.x).shape).toEqual([60])
  })

  it('class weights, sample weights and the intercept scaling act on the fit', () => {
    const base = linearSvc.with({ tol: 1e-10 }).fit(data).params
    const keepRows = data.y.map((v, i) => (v === 0 || i < 14 ? i : -1)).filter((i) => i >= 0)
    const skewed = { x: keepRows.map((i) => data.x[i] ?? []), y: keepRows.map((i) => data.y[i] ?? 0) }
    const balanced = linearSvc.with({ tol: 1e-10, classWeight: 'balanced' }).fit(skewed).params
    const plain = linearSvc.with({ tol: 1e-10 }).fit(skewed).params
    expect(balanced.coef).not.toEqual(plain.coef)
    // A zero sample weight removes the row.
    const weights = data.y.map((_, i) => (i < 5 ? 0 : 1))
    const dropped = linearSvc.with({ tol: 1e-10 }).fit({ x: data.x.slice(5), y: data.y.slice(5) }).params
    close(linearSvc.with({ tol: 1e-10 }).fit({ ...data, weights }).params.coef[0] ?? [], dropped.coef[0] ?? [], 1e-7)
    expect(linearSvc.with({ tol: 1e-10, interceptScaling: 10 }).fit(data).params.intercept).not.toEqual(base.intercept)
    expect(linearSvc.with({ fitIntercept: false }).fit(data).params.intercept).toEqual([0])
  })

  it('a session plays the dual coordinate descent: the gap closes, the model at a step is available', () => {
    const s = linearSvc.with({ tol: 1e-9 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    const objective = metricSeries(s.run(), 'solver/objective').values
    expect(objective[objective.length - 1] ?? 0).toBeLessThanOrEqual((objective[0] ?? 0) + 1e-9)
    expect(s.fittedAt(1).params.coef).not.toEqual(s.fittedAt(s.t).params.coef)
    expect((s.probe(s.t, 'model/predictions') as number[]).length).toBe(60)
    const state = s.state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('names its errors', () => {
    expect(() => linearSvc.with({ C: 0 }).fit(data)).toThrow(/outside the method's space/)
    expect(() => linearSvc.with({ interceptScaling: 0 }).fit(data)).toThrow(/interceptScaling/)
    expect(() => linearSvc.fit({ x: [[0], [1]], y: [1, 1] })).toThrow(/two classes/)
  })
})

describe('linearSvr', () => {
  const x = gauss(7, 60, 2)
  const y = x.map((r, i) => 2 * (r[0] ?? 0) - (r[1] ?? 0) + 1 + 0.05 * Math.sin(i))

  it('recovers a linear function and scores near 1', () => {
    const f = linearSvr.with({ C: 100, tol: 1e-9, maxIter: 20_000 }).fit({ x, y })
    close(f.params.coef, [2, -1], 0.1)
    expect(f.params.intercept).toBeCloseTo(1, 1)
    expect(f.score({ x, y })).toBeGreaterThan(0.98)
    expect(toFlat(f.predict([[0, 0]]))[0]).toBeCloseTo(f.params.intercept, 12)
  })

  it('the smooth solvers of the slot reach the dual coordinate descent weights on the squared loss', () => {
    const h = { C: 10, epsilon: 0.1, loss: 'squaredEpsilonInsensitive', tol: 1e-10, maxIter: 20_000 } as const
    const dcd = linearSvr.with(h).fit({ x, y }).params
    for (const kind of ['newton', 'lbfgs']) {
      const f = linearSvr.with({ ...h, solver: svrSolver({ kind, gradTol: 1e-10 }) }).fit({ x, y }).params
      close([...f.coef, f.intercept], [...dcd.coef, dcd.intercept], 1e-5)
    }
    expect(() => linearSvr.with({ solver: svrSolver({ kind: 'lbfgs' }) }).fit({ x, y })).toThrow(/not smooth/)
  })

  it('a tube wider than the targets leaves nothing to fit, and the dual objective closes on the primal', () => {
    const wide = linearSvr.with({ epsilon: 10, tol: 1e-10, fitIntercept: false }).fit({ x, y }, { logs: 'all' })
    expect(wide.params.coef.every((v) => Math.abs(v) < 1e-9)).toBe(true)
    const f = linearSvr.with({ C: 5, epsilon: 0.1, tol: 1e-9, maxIter: 20_000 }).fit({ x, y }, { logs: 'all' })
    expect(f.params.converged).toBe(true)
    const objective = metricSeries(f.run, 'solver/objective').values
    expect(objective[objective.length - 1] ?? 0).toBeLessThan(objective[0] ?? 0)
  })

  it('names its errors', () => {
    expect(() => linearSvr.with({ epsilon: -1 }).fit({ x, y })).toThrow(/epsilon must be non-negative/)
    expect(() => linearSvr.with({ C: -1 }).fit({ x, y })).toThrow(/outside the method's space/)
  })
})
