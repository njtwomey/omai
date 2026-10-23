import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { checkPrimitive, grad } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { fromStorage, sum, type Tensor, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import { metricSeries, recorder, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

const fixture = {
  cases: ['optim/transport', 'optim/wasserstein', 'optim/gromov'].flatMap((f) => loadFixture(f).cases),
}
const cases = (fn: string) =>
  fixture.cases.filter((c) => c.fn === fn).map((c) => ({ ...c, inputs: c.inputs.map((v) => decode(v) as Tensor) }))
type PlanOut = { readonly plan: unknown; readonly cost?: number; readonly distance?: number }

describe('exact transport against POT', () => {
  for (const c of cases('emd')) {
    it(c.name, () => {
      const [a, b, cost] = c.inputs as [Tensor, Tensor, Tensor]
      const r = O.emd(a, b, cost)
      const out = c.output as PlanOut
      expectMatches(r.plan, out.plan, c.rtol, c.atol, c.name)
      expect(Math.abs(r.cost - (out.cost ?? Number.NaN))).toBeLessThan(1e-12)
      // Strong duality and dual feasibility.
      const u = toFlat(r.u)
      const v = toFlat(r.v)
      const av = toFlat(a)
      const bv = toFlat(b)
      const dual = av.reduce((s, x, i) => s + x * (u[i] ?? 0), 0) + bv.reduce((s, x, j) => s + x * (v[j] ?? 0), 0)
      expect(Math.abs(dual - r.cost)).toBeLessThan(1e-12)
      const m = toArray(cost) as number[][]
      for (const [i, row] of m.entries())
        for (const [j, mij] of row.entries()) expect(mij - (u[i] ?? 0) - (v[j] ?? 0)).toBeGreaterThan(-1e-12)
    })
  }

  it('rejects unequal masses and traces the pivots', () => {
    expect(() => O.emd([1], [0.5], [[0]])).toThrow()
    const rec = recorder()
    O.emd(
      [0.2, 0.8],
      [0.5, 0.5],
      [
        [1, 2],
        [0, 4],
      ],
      { tracer: rec, policy: tracePolicy({}) },
    )
    const costs = metricSeries(rec.run(), 'cost').values
    expect(costs.every((v, i) => i === 0 || v <= (costs[i - 1] ?? Infinity) + 1e-15)).toBe(true)
  })
})

describe('entropic and unbalanced transport against POT', () => {
  for (const c of cases('sinkhorn')) {
    it(c.name, () => {
      const [a, b, cost] = c.inputs as [Tensor, Tensor, Tensor]
      const method = (c.params['method'] as 'standard' | undefined) ?? 'log'
      const r = O.sinkhorn(a, b, cost, c.params['reg'] as number, { method, tol: 1e-14, maxIterations: 100000 })
      expect(r.converged).toBe(true)
      expectMatches(r.plan, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('unbalanced')) {
    it(c.name, () => {
      const [a, b, cost] = c.inputs as [Tensor, Tensor, Tensor]
      const r = O.sinkhornUnbalanced(a, b, cost, c.params['reg'] as number, c.params['regM'] as number, {
        tol: 1e-14,
        maxIterations: 100000,
      })
      expectMatches(r.plan, c.output, c.rtol, c.atol, c.name)
    })
  }

  it('the entropic loss has the plan and potentials as derivatives', () => {
    const [a, b, cost] = (cases('sinkhorn')[1]?.inputs ?? []) as [Tensor, Tensor, Tensor]
    const f = (m: Value): Value => O.entropicTransportLoss(a, b, m, 0.5)
    const report = checkPrimitive(f, [cost], { rtol: 1e-5, atol: 1e-7 })
    expect(report.failures).toEqual([])
    const r = O.sinkhorn(a, b, cost, 0.5, { tol: 1e-13, maxIterations: 100000 })
    expect(toFlat(grad(f)(cost))).toEqual(toFlat(r.plan).map((v) => expect.closeTo(v, 10) as unknown as number))
    // Mass-preserving perturbations of a: the directional derivative is <f, da>.
    const da = tensor([0.01, -0.01, 0, 0, 0.005, -0.005])
    const plus = O.entropicTransportLoss(
      fromStorage(
        Float64Array.from(toFlat(a), (v, i) => v + 1e-6 * (toFlat(da)[i] ?? 0)),
        [6],
        'float64',
      ),
      b,
      cost,
      0.5,
    )
    const minus = O.entropicTransportLoss(
      fromStorage(
        Float64Array.from(toFlat(a), (v, i) => v - 1e-6 * (toFlat(da)[i] ?? 0)),
        [6],
        'float64',
      ),
      b,
      cost,
      0.5,
    )
    const fd = ((toFlat(plus)[0] ?? 0) - (toFlat(minus)[0] ?? 0)) / 2e-6
    const fda = toFlat(r.f).reduce((s, v, i) => s + v * (toFlat(da)[i] ?? 0), 0)
    expect(Math.abs(fd - fda)).toBeLessThan(1e-6)
  })

  it('small reg approaches the exact plan; unbalanced with huge regM approaches Sinkhorn', () => {
    const [a, b, cost] = (cases('emd')[0]?.inputs ?? []) as [Tensor, Tensor, Tensor]
    const exact = O.emd(a, b, cost)
    const soft = O.sinkhorn(a, b, cost, 0.01, { maxIterations: 50000 })
    expect(Math.abs(soft.cost - exact.cost)).toBeLessThan(0.05)
    const balanced = O.sinkhorn(a, b, cost, 0.5, { tol: 1e-12 })
    const relaxed = O.sinkhornUnbalanced(a, b, cost, 0.5, 1e9, {
      reference: 'uniform',
      tol: 1e-13,
      maxIterations: 100000,
    })
    expectMatches(relaxed.plan, { dtype: 'float64', shape: [5, 7], data: toFlat(balanced.plan) }, 1e-6, 1e-9, 'relaxed')
  })
})

describe('Wasserstein on the line, sliced, Gromov and barycentres against POT', () => {
  for (const c of cases('wasserstein1d')) {
    it(c.name, () => {
      const [u, v, uw, vw] = c.inputs as [Tensor, Tensor, Tensor | undefined, Tensor | undefined]
      const r = O.wasserstein1d(u, v, {
        p: c.params['p'] as number,
        ...(uw === undefined ? {} : { uWeights: uw }),
        ...(vw === undefined ? {} : { vWeights: vw }),
      })
      expectMatches(r, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('sliced')) {
    it(c.name, () => {
      const [x, y, projections] = c.inputs as [Tensor, Tensor, Tensor]
      expectMatches(
        O.slicedWasserstein(x, y, { projections, p: c.params['p'] as number }),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }

  it('Wasserstein distances are differentiable in the positions; random slices are reproducible', () => {
    const report = checkPrimitive((u) => O.wasserstein1d(u, [0.5, 2, 3.3], { p: 2 }), [tensor([0.1, 1.2, 2.9, 4])])
    expect(report.failures).toEqual([])
    const x = normal(key(1), [6, 3])
    const y = normal(key(2), [5, 3])
    const s1 = toFlat(O.slicedWasserstein(x, y, { key: key(3), nProjections: 20 }))
    expect(toFlat(O.slicedWasserstein(x, y, { key: key(3), nProjections: 20 }))).toEqual(s1)
    expect(checkPrimitive((v) => O.slicedWasserstein(v, y, { key: key(3), nProjections: 5 }), [x]).failures).toEqual([])
  })

  for (const c of cases('gromov')) {
    it(c.name, () => {
      const [c1, c2, p, q] = c.inputs as [Tensor, Tensor, Tensor, Tensor]
      const r = O.gromovWasserstein(c1, c2, p, q, { tolRel: 1e-12, tolAbs: 1e-14 })
      const out = c.output as PlanOut
      expectMatches(r.plan, out.plan, c.rtol, c.atol, c.name)
      expect(Math.abs(r.distance - (out.distance ?? Number.NaN))).toBeLessThan(1e-9)
    })
  }
  for (const c of cases('entropicGromov')) {
    it(c.name, () => {
      const [c1, c2, p, q] = c.inputs as [Tensor, Tensor, Tensor, Tensor]
      const r = O.entropicGromovWasserstein(c1, c2, p, q, c.params['epsilon'] as number, {
        tol: 1e-13,
        maxIterations: 100000,
      })
      const out = c.output as PlanOut
      expectMatches(r.plan, out.plan, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('barycenter')) {
    it(c.name, () => {
      const [a, cost, w] = c.inputs as [Tensor, Tensor, Tensor]
      const r = O.sinkhornBarycenter(a, cost, c.params['reg'] as number, {
        weights: toFlat(w),
        tol: 1e-15,
        maxIterations: 100000,
      })
      expectMatches(r.barycenter, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('freeSupport')) {
    it(c.name, () => {
      const k = c.params['measures'] as number
      const [init, ...rest] = c.inputs as [Tensor, ...Tensor[]]
      const r = O.freeSupportBarycenter(rest.slice(0, k), rest.slice(k), init, { tol: 1e-14, maxIterations: 200 })
      expectMatches(r.support, c.output, c.rtol, c.atol, c.name)
      expect(sum(r.support).shape).toEqual([])
    })
  }
})
