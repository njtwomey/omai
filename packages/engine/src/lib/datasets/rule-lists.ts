/**
 * Classification data with a planted rule list: categorical features coded as integers, labels given by the first of a
 * few short conjunctive rules that matches a row (a default class otherwise), with a chosen share of labels flipped. The
 * list, the default class, the noise and the exact accuracy of the planted list are the truth, so a rule learner can be
 * checked against the list it should find.
 */

import { type ClassificationData, type DatasetGenerator, xyClasses } from '../data/index.ts'
import { foldIn, key as makeKey, permutation, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { labels, matrix, stamp } from './_util.ts'

/** A literal of a planted rule: feature `feature` equals the category `value`. */
export type PlantedLiteral = { readonly feature: number; readonly value: number }

/** A planted rule: a conjunction of equalities and the class it assigns. */
export type PlantedRule = { readonly literals: readonly PlantedLiteral[]; readonly label: number }

/**
 * Categorical rows labelled by a planted decision list. Each of `rules` rules tests one or two features for given
 * categories (alternating), assigns a class different from the default, and the first matching rule decides a row;
 * rows no rule matches get the default class. A share `noise` of labels is then moved to a uniformly chosen other class.
 * The list depends on `listSeed` alone, so samples drawn with different keys come from the same list.
 * The truth has the list (`rules`), the `defaultClass`, the `noise`, the noise-free labels (`clean`), the exact
 * `bayesAccuracy` (1 minus the noise) and `predict`, the planted list as a function of a row.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { plantedRuleList } from 'mdml-engine/datasets'
 * generate(plantedRuleList, { knobs: { n: 50, rules: 2, noise: 0 } }).data.classes // => ['0', '1']
 * ```
 */
export const plantedRuleList: DatasetGenerator<ClassificationData> = {
  name: 'plantedRuleList',
  kind: 'classification',
  description: 'Categorical rows labelled by a planted decision list, with label noise and the list as the truth.',
  knobs: dict({
    n: int(2, 1_000_000, { label: 'samples', description: 'The number of rows.', default: 400 }),
    features: int(2, 40, { label: 'features', description: 'The number of categorical features.', default: 6 }),
    levels: int(2, 10, { label: 'categories', description: 'The number of categories of each feature.', default: 3 }),
    rules: int(1, 8, { label: 'rules', description: 'The number of rules in the planted list.', default: 3 }),
    classes: int(2, 6, { label: 'classes', description: 'The number of classes.', default: 2 }),
    noise: real(0, 1, {
      label: 'label noise',
      description: 'The share of labels moved to another class.',
      default: 0.05,
    }),
    listSeed: int(0, 1_000_000, {
      label: 'list seed',
      description: 'Seeds the planted list, so samples with different keys share it.',
      default: 0,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['features'] as number
    const levels = knobs['levels'] as number
    const nRules = knobs['rules'] as number
    const nClasses = knobs['classes'] as number
    const noise = knobs['noise'] as number
    const defaultClass = 0
    const rules: PlantedRule[] = []
    const listKey = makeKey(knobs['listSeed'] as number)
    for (let r = 0; r < nRules; r++) {
      const size = Math.min(d, 1 + (r % 2))
      const features = Array.from(permutation(foldIn(foldIn(listKey, 1), r), d).data.subarray(0, size))
      const values = uniform(foldIn(foldIn(listKey, 2), r), [size]).data
      rules.push({
        literals: features.map((feature, i) => ({ feature, value: Math.floor((values[i] ?? 0) * levels) })),
        label: 1 + (r % (nClasses - 1)),
      })
    }
    const draws = uniform(foldIn(k, 3), [n, d]).data
    const x = new Float64Array(n * d)
    for (let i = 0; i < n * d; i++) x[i] = Math.floor((draws[i] ?? 0) * levels)
    const predict = (row: ArrayLike<number>): number => {
      for (const rule of rules) if (rule.literals.every((l) => row[l.feature] === l.value)) return rule.label
      return defaultClass
    }
    const clean = Array.from({ length: n }, (_, i) => predict(x.subarray(i * d, (i + 1) * d)))
    const flip = uniform(foldIn(k, 4), [n, 2]).data
    const y = clean.map((c, i) =>
      (flip[2 * i] ?? 1) < noise ? (c + 1 + Math.floor((flip[2 * i + 1] ?? 0) * (nClasses - 1))) % nClasses : c,
    )
    return {
      data: xyClasses(matrix(x, n, d), labels(y), {
        ...stamp('plantedRuleList', plantedRuleList.description, knobs),
        classes: Array.from({ length: nClasses }, (_, c) => String(c)),
      }),
      truth: { extra: { rules, defaultClass, noise, clean, bayesAccuracy: 1 - noise, predict } },
    }
  },
}
