import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../../random/index.ts'
import {
  backwardSample,
  type ChainConstraints,
  type ChainModel,
  constrainChain,
  constrainedPosteriorDecode,
  constrainedViterbi,
  forwardBackward,
  posteriorDecode,
  viterbi,
  viterbiKBest,
} from './chain.ts'

/** A random chain of T steps and K states with Gaussian log potentials, homogeneous or one transition per step. */
function randomChain(seed: number, T: number, K: number, perStep = false): ChainModel {
  const g = (k: number, n: number): number[] => Array.from(normal(key(seed * 10 + k), [n]).data)
  const rows = (v: number[], offset: number): number[][] =>
    Array.from({ length: K }, (_, i) => v.slice(offset + i * K, offset + (i + 1) * K))
  const em = g(1, T * K)
  const tr = g(2, K * K * Math.max(T - 1, 1))
  return {
    logInitial: g(3, K),
    logTransition: perStep ? Array.from({ length: Math.max(T - 1, 1) }, (_, t) => rows(tr, t * K * K)) : rows(tr, 0),
    logEmission: Array.from({ length: T }, (_, t) => em.slice(t * K, (t + 1) * K)),
  }
}

function transitionAt(chain: ChainModel, t: number): readonly (readonly number[])[] {
  const tr = chain.logTransition
  return (
    Array.isArray(tr[0]?.[0]) ? (tr as readonly (readonly (readonly number[])[])[])[t] : tr
  ) as readonly (readonly number[])[]
}

/** Every path with its score, by enumeration, best first (ties by path order). */
function enumerate(chain: ChainModel): { path: number[]; score: number }[] {
  const K = chain.logInitial.length
  const T = chain.logEmission.length
  const out: { path: number[]; score: number }[] = []
  const walk = (path: number[], score: number): void => {
    const t = path.length
    if (t === T) {
      out.push({ path, score })
      return
    }
    for (let k = 0; k < K; k++) {
      const s =
        score +
        (chain.logEmission[t]?.[k] ?? 0) +
        (t === 0 ? (chain.logInitial[k] ?? 0) : (transitionAt(chain, t - 1)[path[t - 1] ?? 0]?.[k] ?? 0))
      walk([...path, k], s)
    }
  }
  walk([], 0)
  return out.sort((a, b) => b.score - a.score)
}

const feasible = (path: readonly number[], c: ChainConstraints): boolean =>
  path.every((k, t) => {
    const here = c.positions?.[t]
    if (here !== undefined && !here.includes(k)) return false
    if (t === 0 && c.start !== undefined && c.start[k] !== true) return false
    if (t === path.length - 1 && c.end !== undefined && c.end[k] !== true) return false
    return t === 0 || c.transitions === undefined || c.transitions[path[t - 1] ?? 0]?.[k] === true
  })

describe('viterbiKBest', () => {
  for (const perStep of [false, true]) {
    it(`matches enumeration of all paths (${perStep ? 'per-step' : 'homogeneous'} transitions)`, () => {
      for (let seed = 1; seed <= 6; seed++) {
        const T = 2 + (seed % 4)
        const K = 2 + (seed % 3)
        const chain = randomChain(seed, T, K, perStep)
        const all = enumerate(chain)
        const k = Math.min(7, all.length)
        const got = viterbiKBest(chain, k)
        expect(got).toHaveLength(k)
        for (const [i, r] of got.entries()) {
          expect(r.logProbability).toBeCloseTo(all[i]?.score ?? 0, 10)
          const truth = all.find((a) => a.path.join() === r.path.join())
          expect(truth?.score).toBeCloseTo(r.logProbability, 10)
        }
        // no path twice, and in decreasing order
        expect(new Set(got.map((r) => r.path.join())).size).toBe(k)
        for (let i = 1; i < k; i++)
          expect(got[i - 1]?.logProbability ?? 0).toBeGreaterThanOrEqual(got[i]?.logProbability ?? 0)
      }
    })
  }

  it('starts with the Viterbi path and score', () => {
    for (let seed = 1; seed <= 8; seed++) {
      const chain = randomChain(seed, 6, 4)
      const best = viterbi(chain)
      const first = viterbiKBest(chain, 5)[0]
      expect(first?.path).toEqual(best.path)
      expect(first?.logProbability).toBe(best.logProbability)
    }
  })

  it('returns every path when k exceeds their number, and k = 1 is Viterbi', () => {
    const chain = randomChain(3, 3, 2)
    expect(viterbiKBest(chain, 100)).toHaveLength(8)
    expect(viterbiKBest(chain, 1)[0]?.path).toEqual(viterbi(chain).path)
  })

  it('leaves out paths of zero potential and handles an empty chain', () => {
    const chain: ChainModel = {
      logInitial: [0, 0],
      logTransition: [
        [0, Number.NEGATIVE_INFINITY],
        [Number.NEGATIVE_INFINITY, 0],
      ],
      logEmission: [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
    }
    expect(viterbiKBest(chain, 5).map((r) => r.path)).toEqual([
      [0, 0, 0],
      [1, 1, 1],
    ])
    expect(viterbiKBest({ ...chain, logEmission: [] }, 3)).toEqual([{ path: [], logProbability: 0 }])
    expect(() => viterbiKBest(chain, 0)).toThrow(/positive/)
  })

  it('the probabilities of all paths sum to one with the normaliser', () => {
    const chain = randomChain(5, 4, 3)
    const logZ = forwardBackward(chain, { method: 'log' }).logLikelihood
    const total = viterbiKBest(chain, 81).reduce((s, r) => s + Math.exp(r.logProbability - logZ), 0)
    expect(total).toBeCloseTo(1, 10)
  })
})

describe('constrained decoding', () => {
  const constraintsFor = (seed: number, K: number, T: number): ChainConstraints => {
    const u = Array.from(uniform(key(900 + seed), [K * K + 2 * K + T * K]).data)
    const flag = (i: number, p: number): boolean => (u[i] ?? 0) < p
    const transitions = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => flag(i * K + j, 0.7)))
    const start = Array.from({ length: K }, (_, k) => flag(K * K + k, 0.8) || k === 0)
    const end = Array.from({ length: K }, (_, k) => flag(K * K + K + k, 0.8) || k === 0)
    const positions = Array.from({ length: T }, (_, t) => {
      const ks = Array.from({ length: K }, (_, k) => k).filter((k) => flag(K * K + 2 * K + t * K + k, 0.85))
      return t % 2 === 0 && ks.length > 0 ? ks : undefined
    })
    return { transitions, start, end, positions }
  }

  it('constrained Viterbi is the best of the allowed paths by enumeration', () => {
    let checked = 0
    for (let seed = 1; seed <= 25; seed++) {
      const T = 3 + (seed % 3)
      const K = 3
      const chain = randomChain(seed, T, K, seed % 2 === 0)
      const c = constraintsFor(seed, K, T)
      const allowed = enumerate(chain).filter((a) => feasible(a.path, c))
      if (allowed.length === 0) {
        expect(() => constrainedViterbi(chain, c)).toThrow(/zero probability/)
        continue
      }
      checked++
      const got = constrainedViterbi(chain, c)
      expect(feasible(got.path, c)).toBe(true)
      expect(got.logProbability).toBeCloseTo(allowed[0]?.score ?? 0, 10)
      // k-best under constraints are the allowed paths in order
      const kb = viterbiKBest(constrainChain(chain, c), 4)
      for (const [i, r] of kb.entries()) expect(r.logProbability).toBeCloseTo(allowed[i]?.score ?? 0, 10)
      expect(kb).toHaveLength(Math.min(4, allowed.length))
    }
    expect(checked).toBeGreaterThan(8)
  })

  it('no constraints leaves the chain as it was', () => {
    const chain = randomChain(2, 4, 3)
    expect(constrainedViterbi(chain, {})).toEqual(viterbi(chain))
    expect(constrainChain(chain, {})).toEqual(chain)
  })

  it('rejects masks of the wrong size', () => {
    const chain = randomChain(2, 3, 3)
    expect(() => constrainChain(chain, { transitions: [[true]] })).toThrow(/3 x 3/)
    expect(() => constrainChain(chain, { start: [true] })).toThrow(/3 flags/)
  })

  it('the constrained distribution is the original conditioned on feasibility', () => {
    const chain = randomChain(4, 4, 3)
    const c = constraintsFor(4, 3, 4)
    const allowed = enumerate(chain).filter((a) => feasible(a.path, c))
    const logZ = Math.log(allowed.reduce((s, a) => s + Math.exp(a.score), 0))
    expect(forwardBackward(constrainChain(chain, c), { method: 'log' }).logLikelihood).toBeCloseTo(logZ, 10)
  })

  it('constrained posterior decoding is the allowed path of largest expected accuracy', () => {
    let checked = 0
    for (let seed = 1; seed <= 25; seed++) {
      const T = 3 + (seed % 3)
      const K = 3
      const chain = randomChain(seed, T, K)
      const c = constraintsFor(seed, K, T)
      const allowed = enumerate(chain).filter((a) => feasible(a.path, c))
      if (allowed.length === 0) continue
      checked++
      const logZ = Math.log(allowed.reduce((s, a) => s + Math.exp(a.score), 0))
      // brute force: expected number of correct labels of each allowed path under the conditioned distribution
      const marginal = Array.from({ length: T }, () => new Array<number>(K).fill(0))
      for (const a of allowed)
        for (const [t, k] of a.path.entries()) {
          const row = marginal[t] as number[]
          row[k] = (row[k] ?? 0) + Math.exp(a.score - logZ)
        }
      const expected = (path: readonly number[]): number => path.reduce((s, k, t) => s + (marginal[t]?.[k] ?? 0), 0)
      const best = Math.max(...allowed.map((a) => expected(a.path)))
      const got = constrainedPosteriorDecode(chain, c)
      expect(feasible(got.path, c)).toBe(true)
      expect(expected(got.path)).toBeCloseTo(best, 9)
      expect(got.expectedCorrect).toBeCloseTo(best, 9)
      for (const [t, row] of got.posteriors.entries())
        for (const [k, p] of row.entries()) expect(p).toBeCloseTo(marginal[t]?.[k] ?? 0, 9)
    }
    expect(checked).toBeGreaterThan(8)
  })

  it('without constraints it is posterior decoding', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const chain = randomChain(seed, 5, 3)
      expect(constrainedPosteriorDecode(chain).path).toEqual(posteriorDecode(chain))
    }
  })

  it('differs from both posterior decoding (which breaks the constraint) and constrained Viterbi', () => {
    // each position prefers its own label but the move 1 -> 0 is forbidden
    const chain: ChainModel = {
      logInitial: [0, 0],
      logTransition: [
        [0, 0],
        [0, 0],
      ],
      logEmission: [
        [0, 1],
        [1, 0],
      ],
    }
    expect(posteriorDecode(chain)).toEqual([1, 0])
    const c: ChainConstraints = {
      transitions: [
        [true, true],
        [false, true],
      ],
    }
    const got = constrainedPosteriorDecode(chain, c)
    expect(feasible(got.path, c)).toBe(true)
    // the feasible paths [0, 0], [0, 1] and [1, 1] have expected accuracy 1, 1.155 and 1: the mixed one wins, where
    // Viterbi (scores 1, 0 and 1) takes a uniform one
    expect(got.path).toEqual([0, 1])
    expect(constrainedViterbi(chain, c).path).toEqual([0, 0])
  })

  it('sampling from the constrained chain only ever draws feasible paths', () => {
    const chain = randomChain(7, 5, 3)
    const c = constraintsFor(7, 3, 5)
    if (enumerate(chain).some((a) => feasible(a.path, c)))
      for (const path of backwardSample(constrainChain(chain, c), key(1), 40)) expect(feasible(path, c)).toBe(true)
  })
})
