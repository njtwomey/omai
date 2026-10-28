import { learnGolden, logisticRegression, num } from '../../testing/learn.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { nestedDichotomies, oneVsOne, oneVsRest, outputCode } from './multiclass.ts'

const rowsOf = (t: Tensor): number[][] => toArray(t) as number[][]

learnGolden('multiclass reductions against sklearn.multiclass', 'learn/multiclass', {
  ovr: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = oneVsRest(logisticRegression({ C: num(p, 'C') })).fit({ x, y })
    return {
      predict: f.predict(q),
      decision: f.decisionFunction(q),
      proba: f.predictProba?.(q) ?? q,
    }
  },
  ovo: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = oneVsOne(logisticRegression({ C: num(p, 'C') })).fit({ x, y })
    return { predict: f.predict(q), decision: f.decisionFunction(q) }
  },
  ecoc: (inputs, p) => {
    const [x, y, q, book] = inputs as [Tensor, Tensor, Tensor, Tensor]
    const f = outputCode(logisticRegression({ C: num(p, 'C') }), { codeBook: rowsOf(book) }).fit({ x, y })
    return { predict: f.predict(q), decision: f.decisionFunction(q) }
  },
  nested: (inputs, p) => {
    const [x, y, q, hierarchy] = inputs as [Tensor, Tensor, Tensor, Tensor]
    // The hierarchy arrives as [nodes, 2, width] with -1 padding.
    const flat = toFlat(hierarchy)
    const [nodes = 0, , width = 0] = hierarchy.shape
    const side = (n: number, s: number): number[] =>
      Array.from(flat.slice((n * 2 + s) * width, (n * 2 + s + 1) * width)).filter((v) => v >= 0)
    const tree = Array.from({ length: nodes }, (_, n) => ({ left: side(n, 0), right: side(n, 1) }))
    const f = nestedDichotomies(logisticRegression({ C: num(p, 'C') }), { hierarchy: tree }).fit({ x, y })
    return { predict: f.predict(q), proba: (f.predictProba as NonNullable<typeof f.predictProba>)(q) }
  },
})
