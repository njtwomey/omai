/**
 * Rule sets that are not found by covering: OneR (Holte 1993), the single feature whose per-value rules make the fewest
 * errors, with numeric features cut into buckets of at least six rows of the majority class; the decision table of
 * Kohavi (1995), a lookup of class counts by the values of a subset of features chosen by leave-one-out accuracy; and
 * the rules of a decision tree, one per leaf, from the conjunction of the splits on its path. All are sets of mutually
 * exclusive rules with a default; OneR and the decision table are traced algorithms whose step evaluates one feature or
 * one subset.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import type { FittedTree } from '../trees/index.ts'
import { countClasses, type RuleData, ruleData } from './_table.ts'
import { argmaxOf, type Literal, type Rule, type RuleModel } from './rule.ts'

const allRows = (rd: RuleData): Int32Array => Int32Array.from({ length: rd.n }, (_, i) => i)

// ------------------------------------------------------------------------------------------------ OneR

/** Options of OneR. */
export interface OneROptions {
  /** The fewest rows of the majority class a numeric bucket must hold before it closes (default 6). */
  readonly minBucket?: number
  /** Indices of features whose values are category codes (default none). */
  readonly nominal?: readonly number[]
}

/** The rules of one feature and their training errors. */
export type OneRCandidate = { readonly feature: number; readonly errors: number; readonly rules: readonly Rule[] }

/** The state of OneR: the features examined so far. */
export type OneRState = { readonly candidates: readonly OneRCandidate[]; readonly best: number; readonly done: boolean }

/** The rules of one feature: a rule per category, or per bucket of a numeric feature. */
function featureRules(rd: RuleData, j: number, minBucket: number): Rule[] {
  const rows = Array.from(allRows(rd)).filter((i) => !Number.isNaN(rd.x[i * rd.d + j] as number))
  const value = (i: number): number => rd.x[i * rd.d + j] as number
  const make = (members: readonly number[], literals: readonly Literal[]): Rule => {
    const counts = countClasses(rd, members)
    return { literals: [...literals], label: argmaxOf(counts), counts }
  }
  if (rd.nominal[j] === true) {
    const levels = [...new Set(rows.map(value))].sort((a, b) => a - b)
    return levels.map((v) =>
      make(
        rows.filter((i) => value(i) === v),
        [{ feature: j, op: '==', value: v }],
      ),
    )
  }
  const sorted = [...rows].sort((a, b) => value(a) - value(b))
  // Holte's buckets: close when the majority class has minBucket rows and the next row differs in class and value.
  const buckets: number[][] = []
  let cur: number[] = []
  const counts = new Array<number>(rd.k).fill(0)
  sorted.forEach((i, t) => {
    cur.push(i)
    counts[rd.y[i] as number] = (counts[rd.y[i] as number] ?? 0) + 1
    const top = argmaxOf(counts)
    const next = sorted[t + 1]
    if ((counts[top] ?? 0) >= minBucket && next !== undefined && rd.y[next] !== top && value(next) !== value(i)) {
      buckets.push(cur)
      cur = []
      counts.fill(0)
    }
  })
  if (cur.length > 0) buckets.push(cur)
  // Merge neighbours with the same majority class.
  const merged: number[][] = []
  for (const b of buckets) {
    const last = merged[merged.length - 1]
    if (last !== undefined && argmaxOf(countClasses(rd, last)) === argmaxOf(countClasses(rd, b))) last.push(...b)
    else merged.push([...b])
  }
  return merged.map((b, t) => {
    const prev = merged[t - 1]
    const lo = prev === undefined ? undefined : (value(prev[prev.length - 1] as number) + value(b[0] as number)) / 2
    const next = merged[t + 1]
    const hi = next === undefined ? undefined : (value(b[b.length - 1] as number) + value(next[0] as number)) / 2
    const literals: Literal[] = []
    if (lo !== undefined) literals.push({ feature: j, op: '>', value: lo })
    if (hi !== undefined) literals.push({ feature: j, op: '<=', value: hi })
    return make(b, literals)
  })
}

const ONER_SCHEMA: TraceSchema = channels({
  metrics: {
    errors: { description: 'training errors of the feature just examined', direction: 'lower' },
    bestErrors: { description: 'training errors of the best feature so far', direction: 'lower' },
    rules: { description: 'rules of the feature just examined' },
  },
  messages: {
    feature: { description: 'a feature examined', payload: { feature: 'index', errors: 'training errors' } },
  },
})

/**
 * OneR as a traced algorithm, one feature per step: the rules that feature's values give (the majority class of each
 * category or numeric bucket) and their training errors, with the best feature so far.
 *
 * @param data - Features and class labels.
 * @param options - The bucket size and nominal features.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { oneRAlgorithm } from 'omai-engine/rules'
 * oneRAlgorithm({ x: [[0], [1]], y: [0, 1] }).name // => 'one-r'
 * ```
 */
export function oneRAlgorithm(data: LearnData, options: OneROptions = {}): Algorithm<OneRState> {
  const rd = ruleData(data, options.nominal ?? [], 'oneR')
  const minBucket = options.minBucket ?? 6
  return {
    name: 'one-r',
    config: { minBucket },
    schema: ONER_SCHEMA,
    init: () => ({ candidates: [], best: -1, done: rd.d === 0 }),
    step: (s) => {
      const j = s.candidates.length
      const rules = featureRules(rd, j, minBucket)
      const errors = rules.reduce((sum, r) => sum + r.counts.reduce((a, v) => a + v, 0) - (r.counts[r.label] ?? 0), 0)
      const candidates = [...s.candidates, { feature: j, errors, rules }]
      const best = s.best < 0 || errors < (candidates[s.best] as OneRCandidate).errors ? j : s.best
      return { candidates, best, done: j + 1 >= rd.d }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => {
      const last = s.candidates[s.candidates.length - 1]
      return {
        metrics: {
          errors: last?.errors ?? 0,
          bestErrors: s.best < 0 ? 0 : (s.candidates[s.best] as OneRCandidate).errors,
          rules: last?.rules.length ?? 0,
        },
        messages:
          last === undefined ? [] : [{ kind: 'feature', payload: { feature: last.feature, errors: last.errors } }],
      }
    },
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as OneRState,
  }
}

/**
 * The model of a OneR state: the best feature's rules as a set, the default the training class counts.
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The options the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { oneRAlgorithm, oneRModel } from 'omai-engine/rules'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const data = { x: [[0, 5], [0, 6], [1, 5], [1, 6]], y: [0, 0, 1, 1] }
 * const { state } = runAlgorithm(oneRAlgorithm(data, { nominal: [0, 1] }), { maxSteps: 5 })
 * oneRModel(data, state, { nominal: [0, 1] }).rules.length // => 2
 * ```
 */
export function oneRModel(data: LearnData, s: OneRState, options: OneROptions = {}): RuleModel {
  const rd = ruleData(data, options.nominal ?? [], 'oneR')
  return {
    kind: 'set',
    classes: rd.classes,
    nFeatures: rd.d,
    rules: s.best < 0 ? [] : [...(s.candidates[s.best] as OneRCandidate).rules],
    defaultCounts: countClasses(rd, allRows(rd)),
  }
}

/**
 * OneR: the one feature whose rules make the fewest training errors.
 *
 * @param data - Features and class labels.
 * @param options - The bucket size, nominal features and tracing.
 * @returns The model and the index of the chosen feature.
 * @example
 * ```ts
 * import { fitOneR } from 'omai-engine/rules'
 * const data = { x: [[0, 5], [0, 6], [1, 5], [1, 6]], y: [0, 0, 1, 1] }
 * fitOneR(data, { nominal: [0, 1] }).feature // => 0
 * ```
 */
export function fitOneR(
  data: LearnData,
  options: OneROptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): { readonly model: RuleModel; readonly feature: number } {
  const { tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(oneRAlgorithm(data, rest), { ...run, maxSteps: 100_000 })
  return { model: oneRModel(data, state, rest), feature: state.best }
}

// ------------------------------------------------------------------------------------------------ decision table

/** Options of the decision table. */
export interface DecisionTableOptions {
  /** Equal-frequency bins for numeric features (default 5). */
  readonly bins?: number
  /** Indices of features whose values are category codes (default none). */
  readonly nominal?: readonly number[]
  /** The most features in the table (default all). */
  readonly maxFeatures?: number
}

/** The state of the decision table's forward search over feature subsets. */
export type DecisionTableState = {
  /** The features in the table. */
  readonly subset: readonly number[]
  /** The leave-one-out accuracy of the table with those features. */
  readonly accuracy: number
  /** The accuracy of each feature tried at the last step (-1 for features already in). */
  readonly tried: readonly number[]
  readonly done: boolean
}

/** Bin index of every row for every feature: the category code for nominal features, an equal-frequency bin otherwise. */
function binned(rd: RuleData, bins: number): { codes: Int32Array[]; cuts: number[][] } {
  const cuts: number[][] = []
  const codes = Array.from({ length: rd.d }, (_, j) => {
    const values = Array.from({ length: rd.n }, (_, i) => rd.x[i * rd.d + j] as number)
    if (rd.nominal[j] === true) {
      cuts.push([])
      return Int32Array.from(values, (v) => (Number.isNaN(v) ? -1 : v))
    }
    const sorted = values.filter((v) => !Number.isNaN(v)).sort((a, b) => a - b)
    const c: number[] = []
    for (let b = 1; b < bins; b++) {
      const v = sorted[Math.floor((b * sorted.length) / bins)]
      if (v !== undefined && (c.length === 0 || v > (c[c.length - 1] as number))) c.push(v)
    }
    // A cut is the midpoint between its value and the previous distinct value.
    const mids = c.map((v) => {
      const prev = [...sorted].reverse().find((u) => u < v)
      return prev === undefined ? v : (prev + v) / 2
    })
    cuts.push(mids)
    return Int32Array.from(values, (v) => (Number.isNaN(v) ? -1 : mids.filter((m) => v > m).length))
  })
  return { codes, cuts }
}

/** Leave-one-out accuracy of the table keyed by `subset`; a row whose cell is empty without it falls back to the class prior. */
function looAccuracy(rd: RuleData, codes: readonly Int32Array[], subset: readonly number[]): number {
  const cells = new Map<string, number[]>()
  const key = (i: number): string => subset.map((j) => codes[j]?.[i]).join(',')
  for (let i = 0; i < rd.n; i++) {
    const k = key(i)
    const c = cells.get(k) ?? new Array<number>(rd.k).fill(0)
    c[rd.y[i] as number] = (c[rd.y[i] as number] ?? 0) + 1
    cells.set(k, c)
  }
  const prior = countClasses(rd, allRows(rd))
  let right = 0
  for (let i = 0; i < rd.n; i++) {
    const c = [...(cells.get(key(i)) as number[])]
    c[rd.y[i] as number] = (c[rd.y[i] as number] ?? 0) - 1
    const guess = c.some((v) => v > 0) ? argmaxOf(c) : argmaxOf(prior.map((v, t) => (t === rd.y[i] ? v - 1 : v)))
    if (guess === rd.y[i]) right++
  }
  return right / Math.max(1, rd.n)
}

const TABLE_SCHEMA: TraceSchema = channels({
  metrics: {
    accuracy: { description: 'leave-one-out accuracy of the table', direction: 'higher' },
    features: { description: 'features in the table' },
    cells: { description: 'cells in the table' },
  },
  messages: {
    add: {
      description: 'a feature added to the table',
      payload: { feature: 'index', accuracy: 'leave-one-out accuracy' },
    },
  },
})

/**
 * The decision table majority classifier (Kohavi 1995) as a traced algorithm: a forward search over feature subsets
 * where each step adds the feature that most raises the table's leave-one-out accuracy, stopping when none does.
 *
 * @param data - Features and class labels.
 * @param options - Bins, nominal features and the table's size limit.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { decisionTableAlgorithm } from 'omai-engine/rules'
 * decisionTableAlgorithm({ x: [[0], [1]], y: [0, 1] }).name // => 'decision-table'
 * ```
 */
export function decisionTableAlgorithm(
  data: LearnData,
  options: DecisionTableOptions = {},
): Algorithm<DecisionTableState> {
  const rd = ruleData(data, options.nominal ?? [], 'decisionTable')
  const { codes } = binned(rd, options.bins ?? 5)
  const maxFeatures = options.maxFeatures ?? rd.d
  return {
    name: 'decision-table',
    config: { bins: options.bins ?? 5, maxFeatures },
    schema: TABLE_SCHEMA,
    init: () => ({ subset: [], accuracy: looAccuracy(rd, codes, []), tried: [], done: rd.d === 0 }),
    step: (s) => {
      const tried = Array.from({ length: rd.d }, (_, j) =>
        s.subset.includes(j)
          ? -1
          : looAccuracy(
              rd,
              codes,
              [...s.subset, j].sort((a, b) => a - b),
            ),
      )
      let best = -1
      for (let j = 0; j < rd.d; j++)
        if (!s.subset.includes(j) && (best < 0 || (tried[j] as number) > (tried[best] as number))) best = j
      if (best < 0 || (tried[best] as number) <= s.accuracy + 1e-12) return { ...s, tried, done: true }
      const subset = [...s.subset, best].sort((a, b) => a - b)
      return { subset, accuracy: tried[best] as number, tried, done: subset.length >= maxFeatures }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        accuracy: s.accuracy,
        features: s.subset.length,
        cells: new Set(Array.from({ length: rd.n }, (_, i) => s.subset.map((j) => codes[j]?.[i]).join(','))).size,
      },
      messages:
        s.subset.length === 0
          ? []
          : [{ kind: 'add', payload: { feature: s.subset[s.subset.length - 1] as number, accuracy: s.accuracy } }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as DecisionTableState,
  }
}

/**
 * The model of a decision-table state: one rule per occupied cell (the conjunction of its features' bins), with the
 * cell's class counts, and the training class counts as the default for rows in an empty cell.
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The options the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { decisionTableModel } from 'omai-engine/rules'
 * const data = { x: [[0], [0], [1], [1]], y: [0, 0, 1, 1] }
 * decisionTableModel(data, { subset: [0], accuracy: 1, tried: [], done: true }, { nominal: [0] }).rules.length // => 2
 * ```
 */
export function decisionTableModel(
  data: LearnData,
  s: DecisionTableState,
  options: DecisionTableOptions = {},
): RuleModel {
  const rd = ruleData(data, options.nominal ?? [], 'decisionTable')
  const { codes, cuts } = binned(rd, options.bins ?? 5)
  const cells = new Map<string, { rows: number[]; code: number[] }>()
  for (let i = 0; i < rd.n; i++) {
    const code = s.subset.map((j) => codes[j]?.[i] as number)
    if (code.some((c) => c < 0)) continue
    const k = code.join(',')
    const cell = cells.get(k) ?? { rows: [], code }
    cell.rows.push(i)
    cells.set(k, cell)
  }
  const rules: Rule[] = [...cells.values()]
    .sort((a, b) => (a.code.join(',') < b.code.join(',') ? -1 : 1))
    .map((cell) => {
      const literals: Literal[] = []
      s.subset.forEach((j, t) => {
        const c = cell.code[t] as number
        if (rd.nominal[j] === true) literals.push({ feature: j, op: '==', value: c })
        else {
          const cut = cuts[j] as number[]
          if (c > 0) literals.push({ feature: j, op: '>', value: cut[c - 1] as number })
          if (c < cut.length) literals.push({ feature: j, op: '<=', value: cut[c] as number })
        }
      })
      const counts = countClasses(rd, cell.rows)
      return { literals, label: argmaxOf(counts), counts }
    })
  return { kind: 'set', classes: rd.classes, nFeatures: rd.d, rules, defaultCounts: countClasses(rd, allRows(rd)) }
}

/**
 * The decision table majority classifier.
 *
 * @param data - Features and class labels.
 * @param options - Bins, nominal features, size limit and tracing.
 * @returns The model and the features chosen.
 * @example
 * ```ts
 * import { fitDecisionTable } from 'omai-engine/rules'
 * const data = { x: [[0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0]], y: [0, 0, 1, 1, 0, 1] }
 * fitDecisionTable(data, { nominal: [0, 1] }).features // => [0]
 * ```
 */
export function fitDecisionTable(
  data: LearnData,
  options: DecisionTableOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): { readonly model: RuleModel; readonly features: readonly number[] } {
  const { tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(decisionTableAlgorithm(data, rest), { ...run, maxSteps: 1000 })
  return { model: decisionTableModel(data, state, rest), features: state.subset }
}

// ------------------------------------------------------------------------------------------------ rules from trees

/**
 * The rules of a classification tree, one per leaf: the conjunction of the splits on the path (redundant bounds on the
 * same feature merged to the tightest), the leaf's class counts, the predicted class its majority. The rules are
 * mutually exclusive and cover every row, so the set predicts exactly as the tree does.
 *
 * @param tree - A classification tree with numeric splits.
 * @returns The rule model.
 * @throws OmaiError with code `value` for a regression tree or a categorical split.
 * @example
 * ```ts
 * import { treeRuleSet } from 'omai-engine/rules'
 * import { decisionTreeClassifier } from 'omai-engine/trees'
 * const fitted = decisionTreeClassifier().fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] })
 * treeRuleSet(fitted.params).rules.map((r) => r.label) // => [0, 1]
 * ```
 */
export function treeRuleSet(tree: FittedTree): RuleModel {
  if (tree.task !== 'classification') throw new OmaiError('value', 'treeRuleSet: a classification tree is required')
  const rules: Rule[] = []
  const walk = (node: number, path: readonly Literal[]): void => {
    const left = tree.left[node] ?? -1
    if (left < 0) {
      const counts = tree.value[node] ?? []
      // Merge bounds on the same feature and side to the tightest.
      const tight = new Map<string, Literal>()
      for (const l of path) {
        const k = `${l.feature}${l.op}`
        const prev = tight.get(k)
        if (prev === undefined || (l.op === '<=' ? l.value < prev.value : l.value > prev.value)) tight.set(k, l)
      }
      rules.push({ literals: [...tight.values()], label: argmaxOf(counts), counts: [...counts] })
      return
    }
    if ((tree.categories[node] ?? []).length > 0)
      throw new OmaiError('value', 'treeRuleSet: categorical splits are not supported')
    const f = tree.feature[node] as number
    const t = tree.threshold[node] as number
    walk(left, [...path, { feature: f, op: '<=', value: t }])
    walk(tree.right[node] as number, [...path, { feature: f, op: '>', value: t }])
  }
  walk(0, [])
  const total = (tree.value[0] ?? []).map((v) => v)
  return { kind: 'set', classes: tree.classes, nFeatures: tree.nFeatures, rules, defaultCounts: total }
}
