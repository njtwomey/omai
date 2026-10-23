import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { valueAndGrad } from '../autodiff/index.ts'
import { add, mul, pow, square, stack, sub, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { metricSeries, recorder, tensorSeries, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

// 0.5 sum a (p - c)^2 + 0.05 sum p^4, the objective of the torch fixtures.
const objective =
  (a: Value, c: Value) =>
  (p: Value): Value =>
    add(mul(sum(mul(a, square(sub(p, c)))), 0.5), mul(sum(pow(p, 4)), 0.05))

const FACTORIES: Readonly<Record<string, (o: Readonly<Record<string, unknown>>) => O.GradientTransformation>> = {
  sgd: (o) => O.sgd(o as unknown as O.SgdOptions),
  adam: (o) => O.adam(o as unknown as O.AdamOptions),
  adamw: (o) => O.adamw(o as unknown as O.AdamOptions),
  rmsprop: (o) => O.rmsprop(o as unknown as O.RmspropOptions),
  adagrad: (o) => O.adagrad(o as unknown as O.AdagradOptions),
}

describe('optimisers against torch.optim', () => {
  for (const c of loadFixture('optim/optimisers').cases) {
    it(c.name, () => {
      const [p0, a, cc] = c.inputs.map((x) => decode(x) as Tensor) as [Tensor, Tensor, Tensor]
      const opt = (FACTORIES[c.fn] as (o: Readonly<Record<string, unknown>>) => O.GradientTransformation)(c.params)
      const vg = valueAndGrad(objective(a, cc))
      let params: Value = p0
      let state = opt.init(params)
      const path: Value[] = []
      for (let t = 0; t < (c.params['steps'] as number); t++) {
        const { updates, state: next } = opt.update(vg(params).grad, state, params)
        params = O.applyUpdates(params, updates) as Value
        state = next
        path.push(params)
      }
      expectMatches(stack(path), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('transformations', () => {
  it('clip by global norm across a tree', () => {
    const g = { a: tensor([3, 0]), b: tensor([4]) }
    const out = O.clipByGlobalNorm(1).update(g, undefined).updates as typeof g
    ;[...toFlat(out.a), ...toFlat(out.b)].forEach((v, i) => {
      expect(v).toBeCloseTo([0.6, 0, 0.8][i] ?? Number.NaN, 15)
    })
    expect(O.clipByGlobalNorm(10).update(g, undefined).updates).toBe(g)
  })

  it('schedules follow their formulas', () => {
    expect(O.cosineDecay(2, 10, 0.1)(10)).toBeCloseTo(0.2, 12)
    expect(O.exponentialDecay(1, 0.5, 4, true)(7)).toBe(0.5)
    const w = O.warmupCosineDecay({ start: 0, peak: 1, end: 0.1, warmupSteps: 5, totalSteps: 15 })
    expect([w(0), w(5), w(15)].map((v) => Math.round(v * 1e12) / 1e12)).toEqual([0, 1, 0.1])
    expect(O.piecewiseConstant([2, 4], [1, 2, 3])(4)).toBe(3)
  })

  it('take schedules as learning rates', () => {
    const t = O.scaleByLearningRate((step) => 1 / (step + 1))
    let s = t.init(tensor([0]))
    const seen: number[] = []
    for (let i = 0; i < 3; i++) {
      const r = t.update(tensor([1]), s)
      seen.push(toFlat(r.updates as Value)[0] ?? 0)
      s = r.state
    }
    expect(seen).toEqual([-1, -0.5, -1 / 3])
  })
})

describe('minimisation', () => {
  it('converges on a tree of parameters and reports scipy-style results', () => {
    const f = (p: unknown): Value => {
      const { w, b } = p as { w: Value; b: Value }
      return add(sum(square(sub(w, tensor([1, -2])))), square(sub(b, 3)))
    }
    const r = O.minimize(
      f,
      { w: tensor([0, 0]), b: tensor(0) },
      { method: O.sgd({ learningRate: 0.3 }), maxSteps: 500, gradTol: 1e-9 },
    )
    expect(r.converged).toBe(true)
    const x = r.x as { w: Value; b: Value }
    expect(toFlat(x.w).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([1, -2])
    expect(r.fun).toBeLessThan(1e-15)
  })

  it('records the objective, gradient norm, parameters and gradients by policy', () => {
    const rec = recorder()
    O.minimize((p) => sum(square(p as Value)), tensor([2, -1]), {
      method: O.adam({ learningRate: 0.1 }),
      maxSteps: 20,
      tracer: rec,
      policy: tracePolicy({ tensors: true, roles: ['param', 'grad'], detail: 'full' }),
    })
    const run = rec.run()
    const loss = metricSeries(run, 'loss').values
    expect(loss).toHaveLength(21)
    expect((loss.at(-1) ?? Infinity) < (loss[0] ?? 0)).toBe(true)
    expect(tensorSeries(run, 'params.x')[0]?.values).toEqual([2, -1])
    expect(tensorSeries(run, 'grad.x')[0]?.values).toEqual([4, -2])
  })
})
