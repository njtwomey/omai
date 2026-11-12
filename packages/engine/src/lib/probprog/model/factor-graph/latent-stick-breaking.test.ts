// The stick-breaking transform of the model layer (`ops.stickBreaking`): its value over named dimensions, and the truncated
// Dirichlet-process mixture under expectation propagation, which reads it as one `stick-breaking` factor per label. The reference is
// the exact quadrature of the stick-breaking mixture benchmark (the three breaks on a grid, the labels summed out).
import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from '../../../tensor/index.ts'
import { session } from '../../../trace/index.ts'
import { epSpecMethod } from '../../inference/index.ts'
import { analysisOf } from '../../inference/model-input.ts'
import { stickBreakingMixture } from '../../models/index.ts'
import { toFactorGraph } from '../index.ts'
import { defaultOperators, operatorShape } from '../operators.ts'

describe('the stickBreaking operator', () => {
  const table = defaultOperators()
  const op = table['stickBreaking']

  it('turns break fractions into weights that sum to one: one more weight than breaks, as an event axis', () => {
    const out = op?.evaluate([{ value: tensor([0.5, 0.5, 0.4]), dims: ['stick'] }], {})
    expect(Array.from(toFlat(out?.value ?? tensor(0)))).toEqual([0.5, 0.25, 0.4 * 0.25, 0.6 * 0.25])
    expect(out?.dims).toEqual([])
    expect(operatorShape(table, 'stickBreaking', [{ dims: ['group', 'stick'], eventRank: 0 }])).toEqual({
      dims: ['group'],
      eventRank: 1,
    })
  })

  it('works along the last dimension of a matrix of breaks, one weight vector per row', () => {
    const out = op?.evaluate(
      [
        {
          value: tensor([
            [0.2, 0.5],
            [1, 0.5],
          ]),
          dims: ['group', 'stick'],
        },
      ],
      {},
    )
    expect(out?.value.shape).toEqual([2, 3])
    expect(Array.from(toFlat(out?.value ?? tensor(0)))).toEqual([0.2, 0.4, 0.4, 1, 0, 0])
  })
})

describe('the truncated Dirichlet-process mixture under EP', () => {
  const bound = stickBreakingMixture.bound()
  const truth = stickBreakingMixture.truth(undefined, 'full')

  it('compiles to one stick-breaking factor per label, on the label and the three Beta breaks', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    const factors = c.graph?.factors.filter((f) => f.operator === 'stick-breaking') ?? []
    expect(factors.length).toBe(30)
    expect(factors[0]?.args).toEqual(['z[0]', 'v[0]', 'v[1]', 'v[2]'])
  })

  it('agrees with the quadrature of the breaks: means within a tenth of a sd, spreads within 15 percent', () => {
    const method = epSpecMethod(bound)
    expect(method.check?.(analysisOf(bound)).applies).toBe(true)
    const run = session(method, { steps: 500, mode: 'eager', config: { tol: 1e-9 } })
    const post = method.posterior?.(run.state, run.config)
    expect(post?.status).toBe('converged')
    for (const k of [0, 1, 2]) {
      const t = truth.marginals[`v[${k}]`]
      const row = post?.summary().find((r) => r.name === `v[${k}]`)
      if (t === undefined || row === undefined) throw new Error(`no v[${k}]`)
      expect(Math.abs((row.mean as number) - t.mean) / t.sd).toBeLessThan(0.1)
      expect(Math.abs((row.sd as number) / t.sd - 1)).toBeLessThan(0.15)
    }
    // The log evidence of the loopy graph is within a fifth of a nat of the quadrature's.
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.3)
  })
})
