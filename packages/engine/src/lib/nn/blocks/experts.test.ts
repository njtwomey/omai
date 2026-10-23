import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, sum, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { Linear } from '../module/layers.ts'
import {
  expertChoiceRouting,
  importanceLoss,
  loadBalancingLoss,
  MixtureOfExperts,
  noisyTopKRouting,
  routerZLoss,
  softmaxRouting,
  switchRouting,
  topKRouting,
} from './experts.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('routing', () => {
  it('top-k keeps k experts per token with renormalised weights', () => {
    const r = topKRouting(n(0, [5, 4]), 2)
    const d = toArray(r.dispatch) as number[][]
    for (const row of d) expect(row.reduce((a, b) => a + b, 0)).toBe(2)
    for (const s of toFlat(sum(r.weights, -1))) expect(s).toBeCloseTo(1, 12)
    expect(() => topKRouting(n(0, [2, 3]), 4)).toThrow()
  })
  it('noisy top-k needs a key in training and is top-k in evaluation', () => {
    const z = n(1, [3, 4])
    expect(() => noisyTopKRouting(z, z, 1)).toThrow()
    expect(toFlat(noisyTopKRouting(z, z, 1, { training: false }).dispatch)).toEqual(toFlat(topKRouting(z, 1).dispatch))
    expect(noisyTopKRouting(z, z, 2, { key: key(2) }).dispatch.shape).toEqual([3, 4])
  })
  it('switch routing drops tokens over capacity and expert choice fills every expert', () => {
    const s = switchRouting(n(3, [8, 2]), 1)
    for (const load of toFlat(sum(s.dispatch, 0))) expect(load).toBeLessThanOrEqual(4)
    const e = expertChoiceRouting(n(4, [6, 3]), 2)
    expect(toFlat(sum(e.dispatch, 0))).toEqual([2, 2, 2])
  })
  it('auxiliary losses', () => {
    expect(
      item(
        loadBalancingLoss(
          softmaxRouting([
            [0, 0],
            [0, 0],
          ]),
        ),
      ),
    ).toBeCloseTo(1, 12)
    // All tokens to one expert is the worst balance.
    expect(
      item(
        loadBalancingLoss(
          topKRouting(
            [
              [5, 0],
              [5, 0],
            ],
            1,
          ),
        ),
      ),
    ).toBeGreaterThan(1.5)
    expect(item(routerZLoss([[0, 0, 0]]))).toBeCloseTo(Math.log(3) ** 2, 12)
    expect(
      item(
        importanceLoss(
          topKRouting(
            [
              [5, 0],
              [5, 0],
            ],
            1,
          ),
        ),
      ),
    ).toBeGreaterThan(0)
    expect(checkPrimitive((z) => routerZLoss(z), [n(5, [3, 2])]).failures).toEqual([])
  })
})

describe('mixture of experts', () => {
  it('sparse dispatch equals the dense weighted sum of experts', () => {
    const moe = new MixtureOfExperts(4, 3, { k: 2, hidden: 6, key: key(6) })
    const x = n(7, [2, 3, 4])
    const r = moe.forward(x)
    const tokens = toArray(x) as number[][][]
    const w = toArray(r.routing.weights) as number[][]
    const experts = moe.part<import('../module/containers.ts').ModuleList>('experts').modules
    const flat = tokens.flat()
    const want = flat.map((tok, i) => {
      const acc = new Array<number>(4).fill(0)
      for (const [j, e] of experts.entries()) {
        const y = toFlat(e.forward([tok]))
        for (let c = 0; c < 4; c++) acc[c] = (acc[c] ?? 0) + (w[i]?.[j] ?? 0) * (y[c] ?? 0)
      }
      return acc
    })
    const got = toFlat(r.output)
    for (const [i, v] of want.flat().entries()) expect(got[i]).toBeCloseTo(v, 12)
    expect(r.load.reduce((a, b) => a + b, 0)).toBe(12)
    expect(item(r.balanceLoss)).toBeGreaterThan(0)
  })
  it('every routing kind runs, with custom experts', () => {
    const x = n(8, [5, 4])
    for (const routing of ['softmax', 'switch', 'expert-choice', 'noisy-topk'] as const) {
      const moe = new MixtureOfExperts(4, 2, { routing, gated: false, hidden: 4, key: key(9) })
      expect(moe.forward(x, { key: key(10) }).output.shape).toEqual([5, 4])
    }
    const custom = new MixtureOfExperts(4, 2, { expert: (i) => new Linear(4, 4, { key: key(20 + i) }), key: key(11) })
    expect(Object.keys(custom.stateDict())).toContain('experts.1.weight')
  })
  it('passes the gradient conformance check', () => {
    const moe = new MixtureOfExperts(2, 2, { k: 1, hidden: 3, key: key(12) })
    expect(checkPrimitive((x) => moe.apply(moe.params, x).output, [n(13, [3, 2])], { rtol: 1e-5 }).failures).toEqual([])
  })
})
