import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { families } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type GlmProblem, glmSolvers } from './glm.ts'

const respond = {
  binomial: { family: families.binomial(), link: families.logit(), y: (u: number) => (u > 0 ? 1 : 0) },
  poisson: { family: families.poisson(), link: families.log(), y: (u: number) => Math.floor(3 * Math.abs(u)) },
  gamma: { family: families.gamma(families.log()), link: families.log(), y: (u: number) => 0.2 + Math.abs(u) },
} as const

describe('the penalised GLM objective along IRLS', () => {
  it('never increases from one step to the next, for any design, family and penalty', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('binomial', 'poisson', 'gamma'),
        // A design entry is 0 or at least 1e-3 in size. fast-check also draws denormal-sized entries (1e-79, 5e-14), and a
        // column of those is a column of zeros to a double but not exactly: unpenalised, IRLS sends its coefficient to
        // 1e78, the step halving (30 halvings, a factor 1e-9) cannot undo that, and the objective rises by about 1e-9
        // (seeds 1257696618 and 913203801). A column of exact zeros stays covered.
        fc.array(
          fc.double({ min: -2, max: 2, noNaN: true }).map((v) => (Math.abs(v) < 1e-3 ? 0 : v)),
          { minLength: 24, maxLength: 24 },
        ),
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 12, maxLength: 12 }),
        fc.constantFrom(0, 0.01, 0.5),
        (name, x, u, l2) => {
          const { family, link, y } = respond[name]
          const problem: GlmProblem = {
            x,
            n: 12,
            d: 2,
            y: u.map(y),
            family,
            link,
            fitIntercept: true,
            penalty: { l2 },
          }
          const algorithm = glmSolvers.make(problem, { kind: 'irls' }, key(0))
          let state = algorithm.init(key(0))
          let previous = Number.POSITIVE_INFINITY
          for (let t = 0; t < 12; t++) {
            state = algorithm.step(state, { t } as never)
            const value = (state as { value: number }).value
            expect(Number.isFinite(value)).toBe(true)
            // The first step has nothing to compare with; after it the halving keeps the objective from rising.
            expect(value).toBeLessThanOrEqual(previous + 1e-9 * (1 + Math.abs(previous)))
            previous = value
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('has a gradient that vanishes where IRLS stops', () => {
    const problem: GlmProblem = {
      x: Array.from({ length: 40 }, (_, k) => Math.sin(k * 1.3)),
      n: 20,
      d: 2,
      y: Array.from({ length: 20 }, (_, i) => (Math.sin(i * 2.1) > 0 ? 1 : 0)),
      family: families.binomial(),
      link: families.logit(),
      fitIntercept: true,
      penalty: { l2: 0.1 },
    }
    const run = runAlgorithm(glmSolvers.make(problem, { kind: 'irls' }, key(0)), { maxSteps: 50 })
    const grad = (run.state as { grad: Float64Array }).grad
    expect(Math.max(...Array.from(grad, Math.abs))).toBeLessThan(1e-8)
  })
})
