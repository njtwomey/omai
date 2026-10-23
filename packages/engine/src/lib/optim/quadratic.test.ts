import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

function programOf(inputs: readonly unknown[]): O.QuadraticProgram {
  const [P, q, G, h, A, b, lb, ub] = inputs.map((x) => decode(x) as Tensor) as Tensor[]
  return {
    P: P as Tensor,
    q: q as Tensor,
    ...((G as Tensor).shape[0] === 0 ? {} : { G: G as Tensor, h: h as Tensor }),
    ...((A as Tensor).shape[0] === 0 ? {} : { A: A as Tensor, b: b as Tensor }),
    lb: lb as Tensor,
    ub: ub as Tensor,
  }
}

const fixture = loadFixture('optim/quadratic')

describe('quadprog against cvxpy (Clarabel)', () => {
  for (const c of fixture.cases) {
    for (const [name, method, tol] of [
      ['active set', O.qpActiveSet, 1e-7],
      ['interior point', O.qpInteriorPoint, 1e-6],
    ] as const) {
      it(`${c.name} by ${name}`, () => {
        const out = c.output as Out
        const r = O.quadprog(programOf(c.inputs), { method })
        expect(r.status).toBe('optimal')
        for (const k of ['x', 'z', 'y', 'zLower', 'zUpper'] as const)
          expectMatches(r[k], out[k], tol, tol, `${c.name} ${k}`)
        expectMatches(r.fun, out['fun'], tol, tol, `${c.name} fun`)
        expect(r.kkt.stationarity).toBeLessThan(1e-7)
        expect(r.kkt.complementarity).toBeLessThan(1e-7)
      })
    }
  }

  it('box by projected Newton', () => {
    const c = fixture.cases.find((k) => k.name === 'box')
    if (c === undefined) throw new Error('missing box case')
    const [P, q, , , , , lb, ub] = c.inputs.map((x) => decode(x) as Tensor) as Tensor[]
    const r = O.boxQp(P as Tensor, q as Tensor, lb as Tensor, ub as Tensor)
    const out = c.output as Out
    expect(r.converged).toBe(true)
    expectMatches(r.x, out['x'], 1e-8, 1e-8, 'box x')
    expectMatches(r.zLower, out['zLower'], 1e-7, 1e-7, 'box zLower')
    expectMatches(r.zUpper, out['zUpper'], 1e-7, 1e-7, 'box zUpper')
  })
})

describe('active set', () => {
  const nw: O.QuadraticProgram = {
    P: [
      [2, 0],
      [0, 2],
    ],
    q: [-2, -5],
    G: [
      [-1, 2],
      [1, 2],
      [1, -2],
    ],
    h: [2, 6, 2],
    lb: [0, 0],
  }

  it('follows Nocedal and Wright example 16.4 from (2, 0)', () => {
    const qp = O.qpArrays(nw)
    const rec = recorder()
    const r = runAlgorithm(O.qpActiveSetAlgorithm(qp, { x0: [2, 0] }), {
      maxSteps: 20,
      tracer: rec,
      policy: tracePolicy({ messages: true, tensors: true, detail: 'full', roles: ['param'] }),
    })
    expect(r.state.solution.status).toBe('optimal')
    const changes = messages(rec.run(), 'workingSet').map((m) => m.payload as { added: number; removed: number })
    // The book's path: drop constraint 3, step to (1, 0), drop 5, step to (1, 1.5) where constraint 1 blocks, then stop.
    expect(changes.map((c) => [c.added, c.removed])).toEqual([
      [-1, 2],
      [-1, -1],
      [-1, 4],
      [0, -1],
      [-1, -1],
    ])
    expect(Array.from(r.state.x).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([1.4, 1.7])
    expect(metricSeries(rec.run(), 'objective').values.at(-1)).toBeCloseTo(-6.45, 12)
  })

  it('detects infeasible and unbounded programs', () => {
    const infeasible = O.quadprog({ P: [[1]], q: [0], G: [[1], [-1]], h: [-1, -1] })
    expect(infeasible.status).toBe('infeasible')
    const unbounded = O.quadprog({ P: [[0]], q: [-1], lb: [0] })
    expect(unbounded.status).toBe('unbounded')
  })

  it('exposes number arrays for reuse', () => {
    const qp: O.QpArrays = {
      n: 2,
      P: new Float64Array([1, 0, 0, 1]),
      q: new Float64Array([-1, -1]),
      G: { data: new Float64Array([1, 1]), rows: 1, cols: 2 },
      h: new Float64Array([1]),
      A: { data: new Float64Array(0), rows: 0, cols: 2 },
      b: new Float64Array(0),
    }
    const sol = O.qpActiveSet(qp)
    expect(Array.from(sol.x)).toEqual([0.5, 0.5])
    expect(Array.from(sol.z)).toEqual([0.5])
    expect(sol.active).toEqual([0])
    const residual = O.kktResiduals(qp, sol.x, sol.y, sol.z)
    expect(Object.values(residual).every((v) => v < 1e-12)).toBe(true)
  })
})

describe('interior point and box QP', () => {
  it('drives the residuals and mu to the tolerance', () => {
    const qp = O.qpArrays({
      P: [
        [4, 1],
        [1, 2],
      ],
      q: [1, 1],
      G: [
        [-1, 0],
        [0, -1],
      ],
      h: [0, 0],
      A: [[1, 1]],
      b: [1],
    })
    const rec = recorder()
    runAlgorithm(O.qpInteriorPointAlgorithm(qp), { maxSteps: 50, tracer: rec })
    const mu = metricSeries(rec.run(), 'mu').values
    expect(mu.at(-1)).toBeLessThan(1e-9)
    expect(mu[0]).toBeGreaterThan(mu.at(-1) ?? 0)
  })

  it('projected Newton handles one-sided bounds and records active sets', () => {
    const r = O.boxQp(
      [
        [2, 1],
        [1, 2],
      ],
      [-6, 0],
      [-Infinity, 0],
      [2, Infinity],
    )
    expect(toFlat(r.x)).toEqual([2, 0])
    expect(toFlat(r.zUpper)[0]).toBeCloseTo(2, 12)
    expect(toFlat(r.zLower)[1]).toBeCloseTo(2, 12)
    const rec = recorder()
    O.boxQp([[1]], [-3], [0], [1], { tracer: rec })
    expect(metricSeries(rec.run(), 'active').values.at(-1)).toBe(1)
  })
})
