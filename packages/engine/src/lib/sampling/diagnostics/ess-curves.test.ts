import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { ess } from './convergence.ts'
import { chainEssCurves } from './ess-curves.ts'

const iid = (seed: number, n: number): number[] => Array.from(normal(key(seed), [n]).data)

function ar1(seed: number, n: number, rho: number): number[] {
  let x = 0
  return iid(seed, n).map((v) => {
    x = rho * x + Math.sqrt(1 - rho * rho) * v
    return x
  })
}

describe('chainEssCurves', () => {
  it('has, at every point, the library ESS of the same draws', () => {
    const chains = [ar1(1, 300, 0.5), ar1(2, 300, 0.5)]
    const c = chainEssCurves({ x: chains }, { probs: [0.25, 0.9], points: 4 })['x']
    expect(c?.quantile.probs).toEqual([0.25, 0.9])
    expect(c?.quantile.ess[1]).toBeCloseTo(ess(chains, { method: 'quantile', prob: 0.9 }), 10)
    expect(c?.evolution.draws.at(-1)).toBe(300)
    expect(c?.evolution.bulk.at(-1)).toBeCloseTo(ess(chains, { method: 'bulk' }), 10)
    expect(c?.evolution.tail.at(-1)).toBeCloseTo(ess(chains, { method: 'tail' }), 10)
    const m = c?.evolution.draws[1] ?? 0
    expect(c?.evolution.bulk[1]).toBeCloseTo(
      ess(
        chains.map((r) => r.slice(0, m)),
        { method: 'bulk' },
      ),
      10,
    )
  })

  it('grows with the draws for independent chains and stays low for sticky ones', () => {
    const good = chainEssCurves({ x: [iid(3, 400), iid(4, 400)] })['x']
    const bad = chainEssCurves({ x: [ar1(3, 400, 0.97), ar1(4, 400, 0.97)] })['x']
    const grow = (e: number[]): number => (e.at(-1) ?? 0) / (e[0] ?? 1)
    expect(grow([...(good?.evolution.bulk ?? [])])).toBeGreaterThan(5)
    expect(good?.evolution.bulk.at(-1)).toBeGreaterThan(300)
    expect(bad?.evolution.bulk.at(-1)).toBeLessThan(80)
    expect(good?.quantile.probs).toHaveLength(19)
  })

  it('leaves out a quantity with too few draws and points where the ESS is not defined', () => {
    const out = chainEssCurves({
      short: [[1, 2, 3]],
      flat: [new Array<number>(50).fill(1), new Array<number>(50).fill(1)],
    })
    expect(out['short']).toBeUndefined()
    expect(out['flat']?.quantile.ess.every(Number.isFinite)).toBe(true)
  })
})
