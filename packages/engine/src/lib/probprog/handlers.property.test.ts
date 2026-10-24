import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { logSumExpOf } from '../base/index.ts'
import { bernoulli, categorical, normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { add, item, mul, take, tensor } from '../tensor/index.ts'
import {
  condition,
  enumerate,
  logJoint,
  type Model,
  mask,
  observe,
  plate,
  sample,
  scale,
  seed,
  siteLogProbs,
  substitute,
  traceOf,
} from './index.ts'

const finite = (lo: number, hi: number): fc.Arbitrary<number> => fc.double({ min: lo, max: hi, noNaN: true })

/** A small hierarchical program with the given data and prior scale. */
function program(y: readonly number[], s: number): Model {
  return () => {
    const mu = sample('mu', normal(0, s))
    const tau = sample('tau', normal(0, 1))
    plate('data', y.length, () => observe('y', normal(mu, mul(tau, tau)), y))
  }
}

describe('handler properties', () => {
  it('the log joint is the sum of the site log densities', () => {
    fc.assert(
      fc.property(
        fc.array(finite(-3, 3), { minLength: 1, maxLength: 5 }),
        finite(0.5, 3),
        finite(-2, 2),
        finite(0.4, 2),
        (y, s, mu, tau) => {
          const m = program(y, s)
          const t = traceOf(substitute(m, { mu, tau }), undefined)
          const total = Object.values(siteLogProbs(t)).reduce((a, v) => a + item(v), 0)
          expect(item(logJoint(m, undefined, { mu, tau }).logJoint)).toBeCloseTo(total, 10)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('conditioning and substituting give the same density; scales multiply and compose', () => {
    fc.assert(
      fc.property(
        fc.array(finite(-3, 3), { minLength: 1, maxLength: 4 }),
        finite(-2, 2),
        finite(0.4, 2),
        finite(0.1, 3),
        finite(0.1, 3),
        (y, mu, tau, a, b) => {
          const m = program(y, 1.5)
          const base = item(logJoint(m, undefined, { mu, tau }).logJoint)
          expect(item(logJoint(condition(m, { mu }), undefined, { tau }).logJoint)).toBeCloseTo(base, 10)
          expect(item(logJoint(scale(scale(m, a), b), undefined, { mu, tau }).logJoint)).toBeCloseTo(a * b * base, 8)
          expect(item(logJoint(mask(m, true), undefined, { mu, tau }).logJoint)).toBeCloseTo(base, 10)
          expect(item(logJoint(mask(m, false), undefined, { mu, tau }).logJoint)).toBe(0)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('seeded programs are reproducible, and different keys draw differently', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), (k) => {
        const m = program([0.1], 1)
        const a = traceOf(seed(m, key(k)), undefined).sites.get('mu')
        const b = traceOf(seed(m, key(k)), undefined).sites.get('mu')
        const c = traceOf(seed(m, key(k + 1)), undefined).sites.get('mu')
        expect(a?.type === 'sample' ? item(a.value) : 0).toBe(b?.type === 'sample' ? item(b.value) : 1)
        expect(a?.type === 'sample' ? item(a.value) : 0).not.toBe(c?.type === 'sample' ? item(c.value) : 0)
      }),
      { numRuns: RUNS },
    )
  })

  it('enumeration equals brute-force marginalisation', () => {
    fc.assert(
      fc.property(
        fc.array(finite(0.05, 0.95), { minLength: 2, maxLength: 2 }),
        fc.array(finite(0.05, 1), { minLength: 3, maxLength: 3 }),
        fc.array(finite(-2, 2), { minLength: 1, maxLength: 3 }),
        (flip, w, y) => {
          // c ~ Bernoulli(0.4); z_i ~ Categorical(w) per point; y_i ~ N(z_i + 2 c, 1); x ~ Bernoulli(flip[c]).
          const total = w.reduce((a, b) => a + b, 0)
          const probs = w.map((v) => v / total)
          const model: Model = () => {
            const c = sample('c', bernoulli({ probs: 0.4 }))
            observe('x', bernoulli({ probs: take(tensor(flip), c) }), 1)
            plate('data', y.length, () => {
              const z = sample('z', categorical({ probs }))
              observe('y', normal(add(z, mul(c, 2)), 1), y)
            })
          }
          const logN = (x: number, m: number): number => -0.5 * (x - m) ** 2 - 0.5 * Math.log(2 * Math.PI)
          const terms: number[] = []
          for (const c of [0, 1]) {
            let lp = Math.log(c === 1 ? 0.4 : 0.6) + Math.log(flip[c] ?? 0)
            for (const x of y) lp += logSumExpOf(probs.map((p, k) => Math.log(p) + logN(x, k + 2 * c)))
            terms.push(lp)
          }
          expect(item(logJoint(enumerate(model), undefined).logJoint)).toBeCloseTo(logSumExpOf(terms), 9)
        },
      ),
      { numRuns: RUNS },
    )
  })
})
