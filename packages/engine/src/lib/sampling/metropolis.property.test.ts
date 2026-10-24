import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key, uniform } from '../random/index.ts'
import { sampleChain } from './chains.ts'
import { metropolisAcceptance, metropolisHastings } from './metropolis.ts'
import { numericTarget } from './target.ts'

const weights = fc.array(fc.double({ min: 0.05, max: 5, noNaN: true }), { minLength: 2, maxLength: 6 })

/** A random stochastic proposal matrix with every entry positive. */
function proposalMatrix(raw: readonly number[], k: number): number[][] {
  return Array.from({ length: k }, (_, i) => {
    const row = Array.from({ length: k }, (_, j) => 0.1 + (raw[(i * k + j) % raw.length] ?? 1))
    const s = row.reduce((a, b) => a + b, 0)
    return row.map((v) => v / s)
  })
}

/** The Metropolis-Hastings transition matrix of a target and a proposal matrix. */
function transitionMatrix(pi: readonly number[], q: readonly (readonly number[])[]): number[][] {
  const k = pi.length
  return Array.from({ length: k }, (_, i) => {
    const row = Array.from({ length: k }, (_, j) =>
      i === j
        ? 0
        : (q[i]?.[j] ?? 0) *
          metropolisAcceptance(
            Math.log(pi[i] ?? 0),
            Math.log(pi[j] ?? 0),
            Math.log(q[i]?.[j] ?? 0),
            Math.log(q[j]?.[i] ?? 0),
          ),
    )
    row[i] = 1 - row.reduce((a, b) => a + b, 0)
    return row
  })
}

describe('Metropolis-Hastings on small discrete targets', () => {
  it('satisfies detailed balance pi_i P_ij = pi_j P_ji for any positive proposal', () => {
    fc.assert(
      fc.property(
        weights,
        fc.array(fc.double({ min: 0, max: 3, noNaN: true }), { minLength: 1, maxLength: 36 }),
        (w, raw) => {
          const total = w.reduce((a, b) => a + b, 0)
          const pi = w.map((v) => v / total)
          const p = transitionMatrix(pi, proposalMatrix(raw, pi.length))
          for (let i = 0; i < pi.length; i++) {
            expect(p[i]?.[i] ?? -1).toBeGreaterThanOrEqual(-1e-12)
            for (let j = 0; j < pi.length; j++) {
              expect((pi[i] ?? 0) * (p[i]?.[j] ?? 0)).toBeCloseTo((pi[j] ?? 0) * (p[j]?.[i] ?? 0), 12)
            }
          }
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('leaves the target stationary: pi P = pi', () => {
    fc.assert(
      fc.property(weights, (w) => {
        const total = w.reduce((a, b) => a + b, 0)
        const pi = w.map((v) => v / total)
        const k = pi.length
        // An asymmetric proposal: move one up with probability 0.7, one down with 0.3 (cyclically).
        const q = Array.from({ length: k }, (_, i) =>
          Array.from({ length: k }, (_, j) =>
            k === 2 ? (i === j ? 0 : 1) : j === (i + 1) % k ? 0.7 : j === (i + k - 1) % k ? 0.3 : 0,
          ),
        )
        const p = transitionMatrix(
          pi,
          q.map((r) => r.map((v) => Math.max(v, 1e-300))),
        )
        for (let j = 0; j < k; j++) {
          const s = pi.reduce((acc, v, i) => acc + v * (p[i]?.[j] ?? 0), 0)
          expect(s).toBeCloseTo(pi[j] ?? 0, 10)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('visits states in proportion to the target when run as a chain', () => {
    const pi = [0.1, 0.2, 0.3, 0.4]
    const t = numericTarget((x) => Math.log(pi[x[0] ?? 0] ?? 0), 1)
    // A symmetric random walk on the cycle {0, 1, 2, 3}.
    const k = metropolisHastings(t, (kk, x) => ({
      proposal: [((x[0] ?? 0) + ((uniform(kk, [1]).data[0] ?? 0) < 0.5 ? 1 : 3)) % 4],
      logForward: 0,
      logBackward: 0,
    }))
    const n = 20000
    const r = sampleChain(k, [0], { key: key(0), numDraws: n })
    const freq = [0, 0, 0, 0]
    for (const v of r.draws) freq[v] = (freq[v] ?? 0) + 1
    // A generous band: the chain is autocorrelated, so allow 6 iid standard errors.
    freq.forEach((f, i) => {
      expect(Math.abs(f / n - (pi[i] ?? 0))).toBeLessThan(6 * Math.sqrt(((pi[i] ?? 0) * (1 - (pi[i] ?? 0))) / n) * 2)
    })
  })
})
