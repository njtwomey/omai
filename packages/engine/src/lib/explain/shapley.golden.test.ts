import { learnGolden, leastSquares, num, rows } from '../../testing/learn.ts'
import { pipeline } from '../learn/compose.ts'
import { requireVerb } from '../learn/estimator.ts'
import { polynomialFeatures } from '../learn/preprocessing/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  ensembleTreeShap,
  hierarchyFromLinkage,
  interventionalValue,
  kernelShap,
  linearShap,
  owenValues,
  shapleyValues,
  type TreeModel,
  treeShap,
  treeShapInteractions,
} from './shapley.ts'

const polyFit = (x: Tensor, y: Tensor) =>
  pipeline([
    ['poly', polynomialFeatures()],
    ['lr', leastSquares()],
  ]).fit({ x, y })

function treeFrom(arrays: readonly Tensor[], offset = 0): TreeModel {
  const a = (i: number): number[] => toFlat(arrays[offset + i] as Tensor)
  return { childrenLeft: a(0), childrenRight: a(1), feature: a(2), threshold: a(3), value: a(4), cover: a(5) }
}

learnGolden('Shapley explanations against shap', 'explain/shapley', {
  kernel: ([x, y, xe, bg]) => {
    const f = pipeline([
      ['poly', polynomialFeatures()],
      ['lr', leastSquares()],
    ]).fit({ x: x as Tensor, y: y as Tensor })
    const predict = requireVerb(f, 'predict', 'test')
    const r = kernelShap(predict, xe as Tensor, bg as Tensor)
    // Exact enumeration of the same game agrees as well.
    const rows = toFlat(xe as Tensor)
    const exact = shapleyValues(interventionalValue(predict, rows.slice(0, 4), bg as Tensor), 4)
    if (exact.some((v, j) => Math.abs(v - (r.values[0]?.[j] ?? 0)) > 1e-9)) throw new Error('kernel and exact disagree')
    return { values: r.values, base: r.baseValue }
  },
  linear: ([, , xe, bg, coef, intercept]) => {
    const r = linearShap(toFlat(coef as Tensor), toFlat(intercept as Tensor)[0] ?? 0, xe as Tensor, bg as Tensor)
    return { values: r.values, base: r.baseValue }
  },
  exact: ([x, y, xe, bg]) => {
    const predict = requireVerb(polyFit(x as Tensor, y as Tensor), 'predict', 'test')
    const values = rows(xe as Tensor).map((row) => shapleyValues(interventionalValue(predict, row, bg as Tensor), 4))
    const base = interventionalValue(predict, rows(xe as Tensor)[0] ?? [], bg as Tensor)([false, false, false, false])
    return { values, base: values.map(() => base) }
  },
  owen: ([x, y, xe, bg, z]) => {
    const predict = requireVerb(polyFit(x as Tensor, y as Tensor), 'predict', 'test')
    const h = hierarchyFromLinkage(rows(z as Tensor), 4)
    const values = rows(xe as Tensor).map((row) => owenValues(interventionalValue(predict, row, bg as Tensor), h))
    const base = interventionalValue(predict, rows(xe as Tensor)[0] ?? [], bg as Tensor)([false, false, false, false])
    return { values, base: values.map(() => base) }
  },
  tree: (inputs) => {
    const tree = treeFrom(inputs)
    const xe = inputs[6] as Tensor
    const r = treeShap(tree, xe)
    return { values: r.values, base: r.baseValue, interactions: treeShapInteractions(tree, xe).map((m) => m.flat()) }
  },
  forest: (inputs, p) => {
    const n = num(p, 'trees')
    const trees = Array.from({ length: n }, (_, t) => treeFrom(inputs, 6 * t))
    const r = ensembleTreeShap(trees, inputs[6 * n] as Tensor)
    return { values: r.values, base: r.baseValue }
  },
})
