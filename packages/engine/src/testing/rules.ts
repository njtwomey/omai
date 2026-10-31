// Shared inputs of the rule-learning tests: the contact lenses table of Cendrowska (1987) and a planted rule list.
import { type ClassificationData, generate } from '../lib/data/index.ts'
import { plantedRuleList } from '../lib/datasets/index.ts'
import { key } from '../lib/random/index.ts'

/** The contact lenses data as category codes: age (young, pre-presbyopic, presbyopic), spectacle prescription (myope, hypermetrope), astigmatic (no, yes), tear production (reduced, normal); classes soft 0, hard 1, none 2. */
export const CONTACT_LENSES: { x: number[][]; y: number[]; nominal: number[] } = {
  x: [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [2, 0, 0, 0],
    [2, 0, 0, 1],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [2, 1, 0, 0],
    [2, 1, 0, 1],
    [2, 1, 1, 0],
    [2, 1, 1, 1],
  ],
  y: [2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2],
  nominal: [0, 1, 2, 3],
}

/** A planted rule list sample (6 categorical features of 3 levels, 3 rules, 2 classes) with its truth. */
export function plantedRules(
  n: number,
  noise: number,
  seed = 1,
): {
  data: ClassificationData
  nominal: number[]
  rules: { literals: { feature: number; value: number }[]; label: number }[]
  predict: (row: ArrayLike<number>) => number
  clean: number[]
} {
  const { data, truth } = generate(plantedRuleList, {
    key: key(seed),
    knobs: { n, rules: 3, features: 6, levels: 3, classes: 2, noise },
  })
  const extra = truth.extra ?? {}
  return {
    data: data,
    nominal: [0, 1, 2, 3, 4, 5],
    rules: extra['rules'] as { literals: { feature: number; value: number }[]; label: number }[],
    predict: extra['predict'] as (row: ArrayLike<number>) => number,
    clean: extra['clean'] as number[],
  }
}
