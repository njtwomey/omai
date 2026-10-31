import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { flatObjective } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { gloveProblem } from './glove.ts'

describe('embedding/glove against numpy', () => {
  for (const c of loadFixture('embedding/glove').cases) {
    it(c.name, () => {
      const p = gloveProblem(
        c.inputs[0] as string[][],
        {
          dimension: c.params['dim'] as number,
          window: c.params['window'] as number,
          xmax: c.params['xmax'] as number,
          alpha: c.params['alpha'] as number,
        },
        key(0),
      )
      expect(p.words).toEqual(c.params['words'])
      const theta = Float64Array.from(toFlat(decode(c.inputs[1]) as never))
      const expected = Array.from(toFlat(decode(c.output) as never))
      const { value, grad } = flatObjective(p.objective, theta).valueAndGrad(theta)
      const got = [value, ...grad]
      expect(got.length).toBe(expected.length)
      for (const [i, e] of expected.entries())
        expect(Math.abs((got[i] ?? 0) - e)).toBeLessThanOrEqual(c.atol + c.rtol * Math.abs(e))
    })
  }
})
