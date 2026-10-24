import { describe, expect, it } from 'vitest'
import { recorder } from '../../trace/index.ts'
import {
  backwardInduction,
  bestResponse,
  cfr,
  cfrPlus,
  expectedReturns,
  exploitability,
  gameTree,
  kuhnEquilibrium,
  kuhnPoker,
  leducPoker,
  mccfr,
  payoffMatrices,
  solveSequenceForm,
  solveZeroSum,
  type TreeSpec,
  toNormalForm,
  treeGame,
  uniformPolicy,
} from '../index.ts'

/** Rosenthal's centipede game with four moves: take ends it, pass doubles the pot for the other player. */
function centipede(): TreeSpec {
  const payoffs = [
    [1, 0],
    [0, 2],
    [3, 1],
    [2, 4],
  ]
  const node = (k: number): TreeSpec =>
    k === 4
      ? { returns: [4, 3] }
      : {
          player: k % 2,
          actions: [
            { label: 'take', child: { returns: payoffs[k] ?? [0, 0] } },
            { label: 'pass', child: node(k + 1) },
          ],
        }
  return node(0)
}

describe('perfect-information games', () => {
  it('backward induction takes at the first node of the centipede game', () => {
    const g = treeGame('centipede', 2, centipede())
    const r = backwardInduction(g)
    expect(r.values).toEqual([1, 0])
    expect(r.policy['0@']).toEqual([1, 0])
    // Every player takes wherever it moves.
    for (const probs of Object.values(r.policy)) expect(probs).toEqual([1, 0])
    expect(exploitability(gameTree(g), r.policy)).toBeCloseTo(0, 12)
  })

  it('takes expectations at chance nodes', () => {
    const g = treeGame('gamble', 1, {
      player: 0,
      actions: [
        { label: 'safe', child: { returns: [1] } },
        {
          label: 'risky',
          child: {
            chance: [
              { p: 0.3, child: { returns: [5] } },
              { p: 0.7, child: { returns: [0] } },
            ],
          },
        },
      ],
    })
    expect(backwardInduction(g).values[0]).toBeCloseTo(1.5, 12)
  })

  it('rejects games with imperfect information', () => {
    expect(() => backwardInduction(kuhnPoker())).toThrow(/perfect information/)
  })
})

describe('Kuhn poker', () => {
  const tree = gameTree(kuhnPoker())

  it('has the equilibrium family of value -1/18', () => {
    for (const alpha of [0, 1 / 6, 1 / 3]) {
      const policy = kuhnEquilibrium(alpha)
      expect(expectedReturns(tree, policy)[0]).toBeCloseTo(-1 / 18, 12)
      expect(exploitability(tree, policy)).toBeLessThan(1e-12)
    }
    expect(() => kuhnEquilibrium(0.5)).toThrow()
  })

  it('its normal form has the same value', () => {
    const nf = toNormalForm(tree)
    expect(nf.shape).toEqual([64, 64])
    expect(solveZeroSum(payoffMatrices(nf)[0]).value).toBeCloseTo(-1 / 18, 9)
  })

  it('the sequence-form LP and CFR+ agree on the value', () => {
    const lp = solveSequenceForm(tree)
    expect(lp.values[0]).toBeCloseTo(-1 / 18, 9)
    expect(lp.values[1]).toBeCloseTo(1 / 18, 9)
    const r = cfrPlus(tree, { iterations: 300 })
    expect(r.exploitability).toBeLessThan(1e-3)
    expect(expectedReturns(tree, r.averagePolicy)[0]).toBeCloseTo(-1 / 18, 3)
  })

  it('best responses beat the uniform policy', () => {
    const u = uniformPolicy(tree)
    const b = bestResponse(tree, u, 1)
    expect(b.value).toBeGreaterThan(b.onPolicyValue)
    expect(Object.keys(b.policy).every((k) => /[pb]/.test(k))).toBe(true)
  })

  it('CFR variants and simultaneous updates converge, tracing exploitability', () => {
    for (const variant of ['cfr', 'cfr+', 'linear', 'discounted'] as const) {
      for (const alternating of [true, false]) {
        const rec = recorder()
        const r = cfr(tree, { iterations: 150, variant, alternating, tracer: rec })
        expect(r.exploitability).toBeLessThan(0.03)
        const values = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'exploitability' ? [e.value] : []))
        expect(values.length).toBe(151)
        expect(values[values.length - 1]).toBeLessThan(values[1] ?? 0)
      }
    }
    expect(() => cfr(tree, { variant: 'other' as 'cfr' })).toThrow()
  })

  it('Monte Carlo CFR converges with either sampling scheme and is reproducible', () => {
    const ext = mccfr(tree, { iterations: 4000, seed: 3 })
    expect(ext.exploitability).toBeLessThan(0.03)
    expect(mccfr(tree, { iterations: 4000, seed: 3 }).averagePolicy).toEqual(ext.averagePolicy)
    const out = mccfr(tree, { iterations: 20_000, sampling: 'outcome', seed: 4 })
    expect(out.exploitability).toBeLessThan(0.05)
    expect(() => mccfr(tree, { exploration: 0 })).toThrow()
  })
})

describe('Leduc poker', () => {
  it('CFR+ makes progress in a few iterations', () => {
    const tree = gameTree(leducPoker())
    const start = exploitability(tree, uniformPolicy(tree))
    const r = cfrPlus(tree, { iterations: 10 })
    expect(r.exploitability).toBeLessThan(start / 3)
  })
})
