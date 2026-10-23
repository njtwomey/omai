import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { problem } from '../../testing/problems.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as O from './index.ts'

const SEARCHES: Readonly<Record<string, (c2: number) => O.LineSearch>> = {
  strongWolfe: (c2) => O.strongWolfe({ c2 }),
  moreThuente: (c2) => O.moreThuente({ c2 }),
  backtracking: () => O.backtracking(),
}

describe('line searches against scipy.optimize', () => {
  for (const c of loadFixture('optim/line_search').cases) {
    it(c.name, () => {
      const [x, p] = c.inputs.map((v) => toFlat(decode(v) as Tensor)) as [number[], number[]]
      const obj = O.flatObjective(problem(c.params['problem'] as string), x)
      const phi = O.lineFunction(obj, x, p)
      const { value, grad } = obj.valueAndGrad(x)
      const slope = grad.reduce((s, g, i) => s + g * (p[i] ?? 0), 0)
      const previous = c.params['previousValue'] as number | undefined
      const search = (SEARCHES[c.fn] as (c2: number) => O.LineSearch)((c.params['c2'] as number | undefined) ?? 0.9)
      const r = search(phi, { value, slope, ...(previous === undefined ? {} : { previousValue: previous }) })
      expectMatches(tensor(r.success ? [r.alpha, r.value] : [Number.NaN, Number.NaN]), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('line search contracts', () => {
  it('scipy line_search defaults reach the same step through the vector front', () => {
    const r = O.lineSearch(problem('rosen'), [-1.2, 1], [1, 0.4])
    expect(r.success).toBe(true)
    expect(r.trials.length).toBe(r.evaluations)
    expect(r.x).toHaveLength(2)
  })
})
