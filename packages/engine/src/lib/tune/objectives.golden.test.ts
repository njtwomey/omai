import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { xy, xyClasses } from '../data/index.ts'
import { defaultFolds } from './index.ts'

const fixture = loadFixture('tune/objectives')

describe('default folds against scikit-learn', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const k = Number(c.params['k'])
      if (c.fn === 'kFold') {
        const n = Number(c.params['n'])
        const data = xy(
          Array.from({ length: n }, (_, i) => [i]),
          new Array<number>(n).fill(0),
        )
        expect(defaultFolds(data, k).map((f) => f.test)).toEqual(c.output)
        return
      }
      const labels = c.inputs[0] as number[]
      const data = xyClasses(
        labels.map((_, i) => [i]),
        labels,
      )
      expect(defaultFolds(data, k).map((f) => f.test)).toEqual(c.output)
    })
  }
})
