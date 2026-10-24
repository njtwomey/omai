import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { expandDims, eye, matmul, mul, softmax, sub, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { ctmcStationary, ctmcTransition } from './ctmc.ts'
import {
  absorption,
  distributionAfter,
  isReversible,
  spectralGap,
  stationaryDistribution,
  timeReversal,
} from './markov-chains.ts'

/** A random positive stochastic matrix of size 2..5 (rows normalised from positive weights). */
const chain = fc
  .integer({ min: 2, max: 5 })
  .chain((n) =>
    fc.array(fc.array(fc.double({ min: 0.05, max: 1, noNaN: true }), { minLength: n, maxLength: n }), {
      minLength: n,
      maxLength: n,
    }),
  )
  .map((w) => w.map((r) => r.map((v) => v / r.reduce((a, b) => a + b, 0))))

/** A random walk on a weighted undirected graph: reversible by construction. */
const walk = fc
  .integer({ min: 2, max: 5 })
  .chain((n) =>
    fc
      .array(fc.double({ min: 0.05, max: 1, noNaN: true }), { minLength: n * n, maxLength: n * n })
      .map((w) => ({ n, w })),
  )
  .map(({ n, w }) => {
    const sym = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (w[i * n + j] ?? 0) + (w[j * n + i] ?? 0)),
    )
    return sym.map((r) => r.map((v) => v / r.reduce((a, b) => a + b, 0)))
  })

/** A random generator: positive off-diagonal rates, rows summing to zero. */
const generator = fc
  .integer({ min: 2, max: 4 })
  .chain((n) =>
    fc.array(fc.array(fc.double({ min: 0.1, max: 2, noNaN: true }), { minLength: n, maxLength: n }), {
      minLength: n,
      maxLength: n,
    }),
  )
  .map((w) =>
    w.map((r, i) => {
      const off = r.map((v, j) => (i === j ? 0 : v))
      return off.map((v, j) => (i === j ? -off.reduce((a, b) => a + b, 0) : v))
    }),
  )

const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0))).toBeLessThan(tol)
}

describe('discrete-time chains (properties)', () => {
  it('the stationary law is a fixed point of P, sums to one and is positive', () => {
    fc.assert(
      fc.property(chain, (p) => {
        const pi = toFlat(stationaryDistribution(p))
        close(toFlat(distributionAfter(p, pi, 1)), pi, 1e-10)
        expect(Math.abs(pi.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-12)
        expect(pi.every((v) => v > 0)).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('random walks on weighted graphs satisfy detailed balance and equal their time reversal', () => {
    fc.assert(
      fc.property(walk, (p) => {
        expect(isReversible(p)).toBe(true)
        close(toFlat(timeReversal(p)), p.flat(), 1e-10)
        const g = spectralGap(p)
        expect(g.gap).toBeGreaterThan(0)
        expect(g.absoluteGap).toBeLessThanOrEqual(g.gap + 1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('the time reversal has the same stationary law', () => {
    fc.assert(
      fc.property(chain, (p) => {
        close(toFlat(stationaryDistribution(timeReversal(p))), toFlat(stationaryDistribution(p)), 1e-9)
      }),
      { numRuns: RUNS },
    )
  })

  it('absorption probabilities from each transient state sum to one', () => {
    fc.assert(
      fc.property(chain, (q) => {
        // make the last state absorbing
        const n = q.length
        const p = q.map((r, i) => (i === n - 1 ? r.map((_, j) => (j === n - 1 ? 1 : 0)) : r))
        const a = absorption(p)
        close(toFlat(a.probabilities), new Array<number>(n - 1).fill(1), 1e-10)
        expect(toFlat(a.expectedSteps).every((t) => t >= 1)).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('the stationary law differentiates in the transition probabilities', () => {
    const f = (logits: Value): Value => stationaryDistribution(softmax(logits, -1))
    expectConformant(f, [
      tensor([
        [0.1, 0.5, -0.3],
        [0.2, -0.4, 0.7],
        [1, 0, 0.3],
      ]),
    ])
  })
})

describe('continuous-time chains (properties)', () => {
  it('satisfy Chapman-Kolmogorov, the Kolmogorov equations and stationarity', () => {
    fc.assert(
      fc.property(
        generator,
        fc.double({ min: 0.05, max: 1, noNaN: true }),
        fc.double({ min: 0.05, max: 1, noNaN: true }),
        (q, s, t) => {
          const ps = ctmcTransition(q, s)
          const pt = ctmcTransition(q, t)
          close(toFlat(ctmcTransition(q, s + t)), toFlat(matmul(ps, pt)), 1e-10)
          // rows of P(t) are probability vectors
          const n = q.length
          const flat = toFlat(pt)
          for (let i = 0; i < n; i++) {
            expect(Math.abs(flat.slice(i * n, (i + 1) * n).reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-12)
          }
          // forward and backward equations: P'(t) = P(t) Q = Q P(t), by central differences
          const h = 1e-5
          const derivative = toFlat(sub(ctmcTransition(q, t + h), ctmcTransition(q, t - h))).map((v) => v / (2 * h))
          close(derivative, toFlat(matmul(pt, q)), 1e-6)
          close(derivative, toFlat(matmul(q, pt)), 1e-6)
          // the stationary law is invariant under P(t)
          const pi = ctmcStationary(q)
          close(toFlat(matmul(pi, pt)), toFlat(pi), 1e-10)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('the transition matrix differentiates in time and rates', () => {
    expectConformant(
      (t: Value) =>
        ctmcTransition(
          [
            [-1, 1],
            [2, -2],
          ],
          t,
        ),
      [tensor(0.7)],
    )
    // rates on the off-diagonal, the diagonal set so that rows sum to zero
    const off = tensor([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ])
    const fromRates = (w: Value): Value => {
      const r = mul(w, off)
      return sub(r, mul(eye(3), expandDims(sum(r, -1), -1)))
    }
    expectConformant(
      (w: Value) => ctmcStationary(fromRates(w)),
      [
        tensor([
          [0, 1.2, 0.3],
          [0.5, 0, 2],
          [0.7, 0.4, 0],
        ]),
      ],
    )
  })
})
