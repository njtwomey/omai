/**
 * Rule lists over pre-mined antecedents: the conjunctions of literals that hold on enough rows (found with Eclat from
 * `patterns`) are the vocabulary, and a list is an ordered selection of them with a default. CORELS (Angelino et al.
 * 2017) searches the prefixes of lists by branch and bound for the one minimising misclassification plus a penalty per
 * rule, certifying optimality when the queue empties; its bounds are the objective bound, the look-ahead, the minimum
 * support, the equivalent-points bound and the permutation bound. Bayesian rule lists (Letham et al. 2015) put a prior
 * on the list's length and the rules' sizes and a Dirichlet-multinomial likelihood on the classes each rule captures,
 * and sample lists by Metropolis-Hastings with add, remove and swap moves. Each is a traced algorithm: one prefix
 * expanded, or one proposal, per step.
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { lgamma } from '../numerics/special/index.ts'
import { eclat } from '../patterns/index.ts'
import { foldIn, uniform } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { conflicts, type LiteralPool, literalPool, type RuleData, ruleData } from './_table.ts'
import { argmaxOf, type Literal, type Rule, type RuleModel } from './rule.ts'

/** Options of the antecedent miner. */
export interface AntecedentOptions {
  /** The smallest share of the rows an antecedent covers (default 0.05). */
  readonly minSupport?: number
  /** The most literals in an antecedent (default 2). */
  readonly maxCardinality?: number
  /** The most thresholds per numeric feature (default 6). */
  readonly maxThresholds?: number
  /** Indices of features whose values are category codes (default none). */
  readonly nominal?: readonly number[]
}

/** The pre-mined antecedents: conjunctions of literals with the rows each covers. */
export type Antecedents = { readonly literals: readonly (readonly Literal[])[]; readonly masks: readonly Uint8Array[] }

const MINED = new WeakMap<RuleData, Map<string, { antecedents: Antecedents; pool: LiteralPool }>>()

function mine(rd: RuleData, o: AntecedentOptions): { antecedents: Antecedents; pool: LiteralPool } {
  const id = JSON.stringify([o.minSupport, o.maxCardinality, o.maxThresholds])
  const hit = MINED.get(rd)?.get(id)
  if (hit !== undefined) return hit
  const mined = mineUncached(rd, o)
  const forData = MINED.get(rd) ?? new Map()
  forData.set(id, mined)
  MINED.set(rd, forData)
  return mined
}

function mineUncached(rd: RuleData, o: AntecedentOptions): { antecedents: Antecedents; pool: LiteralPool } {
  const pool = literalPool(rd, o.maxThresholds ?? 6)
  const transactions: number[][] = Array.from({ length: rd.n }, () => [])
  pool.masks.forEach((m, l) => {
    for (let i = 0; i < rd.n; i++) if (m[i] === 1) transactions[i]?.push(l)
  })
  const sets = eclat(transactions, { minSupport: o.minSupport ?? 0.05, maxLength: o.maxCardinality ?? 2 })
  const literals: Literal[][] = []
  const masks: Uint8Array[] = []
  const seen = new Set<string>()
  for (const s of sets) {
    const items = [...s.items]
    if (items.some((l, i) => conflicts(pool, items.slice(0, i), l))) continue
    const mask = new Uint8Array(rd.n).fill(1)
    for (const l of items) for (let i = 0; i < rd.n; i++) if ((pool.masks[l] as Uint8Array)[i] !== 1) mask[i] = 0
    const id = mask.join('')
    if (seen.has(id)) continue
    seen.add(id)
    literals.push(items.map((l) => pool.literals[l] as Literal))
    masks.push(mask)
  }
  return { antecedents: { literals, masks }, pool }
}

/**
 * The antecedents a rule list is built from: every conjunction of up to `maxCardinality` literals covering at least
 * `minSupport` of the rows, found by frequent itemset mining over the literals each row satisfies, with antecedents that
 * cover exactly the same rows as a shorter one dropped.
 *
 * @param data - Features and class labels.
 * @param options - Support, cardinality, thresholds and nominal features.
 * @returns The antecedents and their coverages.
 * @example
 * ```ts
 * import { mineAntecedents } from 'mdml-engine/rules'
 * mineAntecedents({ x: [[0], [0], [1], [1]], y: [0, 0, 1, 1] }, { minSupport: 0.5, maxCardinality: 1 }).literals.length // => 2
 * ```
 */
export function mineAntecedents(data: LearnData, options: AntecedentOptions = {}): Antecedents {
  return mine(ruleData(data, options.nominal ?? [], 'mineAntecedents'), options).antecedents
}

/** The classes captured by each rule of a list (a row goes to the first antecedent it satisfies), and the rows left. */
function capture(
  rd: RuleData,
  masks: readonly Uint8Array[],
  list: readonly number[],
): { counts: number[][]; left: number[] } {
  const left = new Uint8Array(rd.n).fill(1)
  const counts = list.map((a) => {
    const c = new Array<number>(rd.k).fill(0)
    const mask = masks[a] as Uint8Array
    for (let i = 0; i < rd.n; i++) {
      if (left[i] === 1 && mask[i] === 1) {
        c[rd.y[i] as number] = (c[rd.y[i] as number] ?? 0) + 1
        left[i] = 0
      }
    }
    return c
  })
  const rest = new Array<number>(rd.k).fill(0)
  for (let i = 0; i < rd.n; i++) if (left[i] === 1) rest[rd.y[i] as number] = (rest[rd.y[i] as number] ?? 0) + 1
  return { counts, left: rest }
}

function listModel(rd: RuleData, a: Antecedents, list: readonly number[]): RuleModel {
  const { counts, left } = capture(rd, a.masks, list)
  const rules: Rule[] = list.map((ant, j) => ({
    literals: [...(a.literals[ant] as readonly Literal[])],
    label: argmaxOf(counts[j] as number[]),
    counts: counts[j] as number[],
  }))
  return { kind: 'list', classes: rd.classes, nFeatures: rd.d, rules, defaultCounts: left }
}

// ------------------------------------------------------------------------------------------------ CORELS

/** Options of CORELS. */
export interface CorelsOptions extends AntecedentOptions {
  /** The penalty per rule in the objective `misclassification rate + lambda * rules` (default 0.01). */
  readonly lambda?: number
  /** The most prefixes to evaluate before giving up the certificate (default 200000). */
  readonly maxNodes?: number
}

type Node = { readonly prefix: readonly number[]; readonly bound: number }

/** The state of CORELS. */
export type CorelsState = {
  /** Prefixes waiting to be expanded, with their lower bounds. */
  readonly queue: readonly Node[]
  /** The best list found. */
  readonly best: readonly number[]
  readonly objective: number
  /** The smallest lower bound still in the queue (the objective cannot fall below it). */
  readonly lowerBound: number
  readonly evaluated: number
  readonly pruned: number
  readonly expanded: number
  /** Best objective per set of rules in a prefix, for the permutation bound. */
  readonly seen: Readonly<Record<string, number>>
  readonly done: boolean
  /** Whether the search finished with an empty queue: the list is provably optimal. */
  readonly certified: boolean
  readonly current: readonly number[]
}

const CORELS_SCHEMA: TraceSchema = channels({
  metrics: {
    objective: { description: 'the objective of the incumbent list', direction: 'lower' },
    lowerBound: { description: 'the smallest bound among prefixes still to expand', direction: 'higher' },
    nodes: { description: 'prefixes evaluated' },
    pruned: { description: 'prefixes pruned by a bound' },
    queue: { description: 'prefixes waiting' },
    length: { description: 'rules in the incumbent list' },
  },
  messages: { expand: { description: 'a prefix expanded', payload: { prefix: 'antecedent indices' } } },
})

/**
 * CORELS-style certifiably optimal rule lists as a traced algorithm, one prefix expanded per step. A prefix's objective
 * is its mistakes on the rows it captures plus the default rule's mistakes on the rest, over the number of rows, plus
 * `lambda` per rule; its lower bound is the prefix's own mistakes, `lambda` per rule and the mistakes the remaining
 * rows must incur among identical rows. Children whose bound (plus one more rule) cannot beat the incumbent are pruned,
 * rules that capture too few rows to pay for themselves are not tried, and a prefix whose set of rules was reached
 * before by a permutation with an objective at least as good is dropped. When the queue is empty the incumbent is optimal
 * among lists over the antecedents.
 *
 * @param data - Features and class labels.
 * @param options - The penalty, the node budget and the antecedent mining options.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a negative lambda.
 * @example
 * ```ts
 * import { corelsAlgorithm } from 'mdml-engine/rules'
 * corelsAlgorithm({ x: [[0], [1]], y: [0, 1] }).name // => 'corels'
 * ```
 */
export function corelsAlgorithm(data: LearnData, options: CorelsOptions = {}): Algorithm<CorelsState> {
  const rd = ruleData(data, options.nominal ?? [], 'corels')
  const lambda = options.lambda ?? 0.01
  if (!(lambda >= 0)) throw new MdmlError('value', 'corels: lambda must be non-negative')
  const maxNodes = options.maxNodes ?? 200_000
  const { antecedents } = mine(rd, options)
  const A = antecedents.masks.length
  const n = rd.n
  // Identical rows can never be told apart by a list, so the minority of each class of identical rows is certain error.
  const group = new Int32Array(n)
  const ids = new Map<string, number>()
  for (let i = 0; i < n; i++) {
    const key = Array.from(rd.x.subarray(i * rd.d, (i + 1) * rd.d)).join(',')
    group[i] = ids.get(key) ?? ids.set(key, ids.size).get(key) ?? 0
  }
  const groups = ids.size
  const groupCounts = new Int32Array(groups * rd.k)
  /** The state after a prefix: rows left, mistakes on the captured rows, and (for a child) the rows its last rule captured. */
  type Walk = { readonly left: Uint8Array; readonly mistakes: number; readonly captured: number }
  const root: Walk = { left: new Uint8Array(n).fill(1), mistakes: 0, captured: 0 }
  const walkTo = (prefix: readonly number[]): Walk => {
    let w = root
    for (const a of prefix) w = extendWalk(w, a)
    return w
  }
  const extendWalk = (w: Walk, a: number): Walk => {
    const mask = antecedents.masks[a] as Uint8Array
    const left = new Uint8Array(w.left)
    const c = new Array<number>(rd.k).fill(0)
    let m = 0
    for (let i = 0; i < n; i++) {
      if (left[i] === 1 && mask[i] === 1) {
        c[rd.y[i] as number] = (c[rd.y[i] as number] ?? 0) + 1
        left[i] = 0
        m++
      }
    }
    return { left, mistakes: w.mistakes + m - Math.max(...c), captured: m }
  }
  const score = (w: Walk, length: number): { objective: number; bound: number } => {
    const rest = new Array<number>(rd.k).fill(0)
    groupCounts.fill(0)
    for (let i = 0; i < n; i++) {
      if (w.left[i] !== 1) continue
      const y = rd.y[i] as number
      rest[y] = (rest[y] ?? 0) + 1
      const slot = (group[i] as number) * rd.k + y
      groupCounts[slot] = (groupCounts[slot] ?? 0) + 1
    }
    const restRows = rest.reduce((sum, v) => sum + v, 0)
    let unavoidable = 0
    for (let g = 0; g < groups; g++) {
      let total = 0
      let top = 0
      for (let c = 0; c < rd.k; c++) {
        const v = groupCounts[g * rd.k + c] ?? 0
        total += v
        if (v > top) top = v
      }
      unavoidable += total - top
    }
    return {
      objective: (w.mistakes + restRows - Math.max(...rest)) / n + lambda * length,
      bound: (w.mistakes + unavoidable) / n + lambda * length,
    }
  }
  return {
    name: 'corels',
    config: {
      lambda,
      maxNodes,
      minSupport: options.minSupport ?? 0.05,
      maxCardinality: options.maxCardinality ?? 2,
      antecedents: A,
    },
    schema: CORELS_SCHEMA,
    init: () => {
      const e = score(root, 0)
      return {
        queue: [{ prefix: [], bound: e.bound }],
        best: [],
        objective: e.objective,
        lowerBound: e.bound,
        evaluated: 1,
        pruned: 0,
        expanded: 0,
        seen: {},
        done: n === 0,
        certified: false,
        current: [],
      }
    },
    step: (s) => {
      if (s.queue.length === 0) return { ...s, done: true, certified: true }
      let pick = 0
      s.queue.forEach((q, i) => {
        const p = s.queue[pick] as Node
        if (q.bound < p.bound || (q.bound === p.bound && q.prefix.length < p.prefix.length)) pick = i
      })
      const node = s.queue[pick] as Node
      const queue = s.queue.filter((_, i) => i !== pick)
      let best = s.best
      let objective = s.objective
      let evaluated = s.evaluated
      let pruned = s.pruned
      const seen: Record<string, number> = { ...s.seen }
      const inPrefix = new Set(node.prefix)
      const parent = walkTo(node.prefix)
      for (let a = 0; a < A && evaluated < maxNodes; a++) {
        if (inPrefix.has(a)) continue
        const prefix = [...node.prefix, a]
        const child = extendWalk(parent, a)
        const e = { ...score(child, prefix.length), captured: child.captured }
        evaluated++
        // A rule in an optimal list captures more than 2 lambda n rows (Angelino et al. 2017, the minimum support bound).
        if (e.captured <= 2 * lambda * n && lambda > 0) {
          pruned++
          continue
        }
        const key = [...prefix].sort((x, y) => x - y).join(',')
        const before = seen[key]
        if (before !== undefined && before <= e.objective + 1e-12) {
          pruned++
          continue
        }
        seen[key] = e.objective
        if (e.objective < objective - 1e-12) {
          objective = e.objective
          best = prefix
        }
        if (e.bound + lambda >= objective - 1e-12) {
          pruned++
          continue
        }
        queue.push({ prefix, bound: e.bound })
      }
      const live = queue.filter((q) => q.bound + lambda < objective - 1e-12)
      pruned += queue.length - live.length
      const lowerBound = live.length === 0 ? objective : Math.min(...live.map((q) => q.bound))
      const exhausted = evaluated >= maxNodes
      return {
        queue: exhausted ? [] : live,
        best,
        objective,
        lowerBound,
        evaluated,
        pruned,
        expanded: s.expanded + 1,
        seen,
        done: live.length === 0 || exhausted,
        certified: live.length === 0 && !exhausted,
        current: node.prefix,
      }
    },
    done: (s) => (s.done ? (s.certified ? 'converged' : 'stopped') : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        lowerBound: s.lowerBound,
        nodes: s.evaluated,
        pruned: s.pruned,
        queue: s.queue.length,
        length: s.best.length,
      },
      messages: s.expanded === 0 ? [] : [{ kind: 'expand', payload: { prefix: [...s.current] } }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as CorelsState,
  }
}

/**
 * The rule list of a CORELS state: the incumbent prefix with its default rule.
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The options the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { corelsAlgorithm, corelsModel } from 'mdml-engine/rules'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = { x: [[0], [0], [0], [1], [1], [1]], y: [0, 0, 0, 1, 1, 1] }
 * const o = { minSupport: 0.3, maxCardinality: 1 }
 * const { state } = runAlgorithm(corelsAlgorithm(data, o), { maxSteps: 100 })
 * corelsModel(data, state, o).rules.length // => 1
 * ```
 */
export function corelsModel(data: LearnData, s: CorelsState, options: CorelsOptions = {}): RuleModel {
  const rd = ruleData(data, options.nominal ?? [], 'corels')
  return listModel(rd, mine(rd, options).antecedents, s.best)
}

/**
 * Certifiably optimal rule list by CORELS-style branch and bound.
 *
 * @param data - Features and class labels.
 * @param options - Penalty, budget, antecedent mining and tracing.
 * @returns The model and whether optimality was certified.
 * @throws MdmlError with code `value` for a negative lambda.
 * @example
 * ```ts
 * import { fitCorels, modelText } from 'mdml-engine/rules'
 * const data = { x: [[0], [0], [0], [1], [1], [1]], y: [0, 0, 0, 1, 1, 1] }
 * const r = fitCorels(data, { minSupport: 0.3, maxCardinality: 1 })
 * [r.certified, modelText(r.model).split('\n').length] // => [true, 2]
 * ```
 */
export function fitCorels(
  data: LearnData,
  options: CorelsOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): { readonly model: RuleModel; readonly certified: boolean; readonly objective: number; readonly nodes: number } {
  const { tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(corelsAlgorithm(data, rest), { ...run, maxSteps: 1_000_000 })
  return {
    model: corelsModel(data, state, rest),
    certified: state.certified,
    objective: state.objective,
    nodes: state.evaluated,
  }
}

// ------------------------------------------------------------------------------------------------ Bayesian rule lists

/** Options of Bayesian rule lists. */
export interface BayesianRuleListOptions extends AntecedentOptions {
  /** The prior mean number of rules (default 3). */
  readonly listLength?: number
  /** The prior mean number of literals in a rule (default 1). */
  readonly ruleSize?: number
  /** The Dirichlet prior on each rule's class probabilities (default 1). */
  readonly alpha?: number
}

/** The state of the Bayesian rule list sampler. */
export type BayesianRuleListState = {
  /** The current list: antecedent indices in order. */
  readonly list: readonly number[]
  readonly logPosterior: number
  /** The most probable list visited. */
  readonly best: readonly number[]
  readonly bestLogPosterior: number
  readonly proposals: number
  readonly accepted: number
  /** What the last proposal was: add, remove, swap, or none. */
  readonly move: string
  readonly acceptedLast: boolean
}

const BRL_SCHEMA: TraceSchema = channels({
  metrics: {
    logPosterior: { description: 'log posterior of the current list (up to a constant)', direction: 'higher' },
    bestLogPosterior: { description: 'log posterior of the best list visited', direction: 'higher' },
    length: { description: 'rules in the current list' },
    accepted: { description: 'proposals accepted so far' },
  },
  messages: {
    move: { description: 'a proposal', payload: { move: 'add, remove or swap', accepted: 'whether it was accepted' } },
  },
})

const LG = new Map<number, number>()
const lg = (x: number): number => {
  let v = LG.get(x)
  if (v === undefined) {
    v = item(lgamma(x))
    LG.set(x, v)
  }
  return v
}
const logPoisson = (k: number, mean: number): number => k * Math.log(mean) - mean - lg(k + 1)

/** The log posterior (up to a constant) of lists of the pre-mined antecedents, and what it was built from. */
function posteriorOf(
  data: LearnData,
  options: BayesianRuleListOptions,
): { A: number; logPosterior: (list: readonly number[]) => number; config: Json } {
  const rd = ruleData(data, options.nominal ?? [], 'bayesianRuleList')
  const lambda = options.listLength ?? 3
  const eta = options.ruleSize ?? 1
  const alpha = options.alpha ?? 1
  if (!(lambda > 0 && eta > 0 && alpha > 0)) throw new MdmlError('value', 'bayesianRuleList: priors must be positive')
  const { antecedents } = mine(rd, options)
  const A = antecedents.masks.length
  const maxCard = Math.max(1, ...antecedents.literals.map((l) => l.length))
  // Truncated Poisson on the size, with the number of antecedents of each size for the uniform choice.
  const bySize = new Array<number>(maxCard + 1).fill(0)
  for (const l of antecedents.literals) bySize[l.length] = (bySize[l.length] ?? 0) + 1
  let normaliser = 0
  for (let c = 1; c <= maxCard; c++) if ((bySize[c] ?? 0) > 0) normaliser += Math.exp(logPoisson(c, eta))
  const logPrior = (list: readonly number[]): number => {
    let lp = logPoisson(list.length, lambda)
    for (const a of list) {
      const c = (antecedents.literals[a] as readonly Literal[]).length
      lp += logPoisson(c, eta) - Math.log(normaliser) - Math.log(bySize[c] ?? 1)
    }
    return lp
  }
  const logLikelihood = (list: readonly number[]): number => {
    const { counts, left } = capture(rd, antecedents.masks, list)
    let ll = 0
    for (const c of [...counts, left]) {
      const N = c.reduce((sum, v) => sum + v, 0)
      ll += lg(rd.k * alpha) - lg(rd.k * alpha + N)
      for (const v of c) ll += lg(alpha + v) - lg(alpha)
    }
    return ll
  }
  return {
    A,
    logPosterior: (list) => logPrior(list) + logLikelihood(list),
    config: { listLength: lambda, ruleSize: eta, alpha, antecedents: A },
  }
}

/**
 * The log posterior, up to an additive constant, of a rule list given as indices into the pre-mined antecedents (see
 * {@link mineAntecedents}): the log prior (truncated Poisson on the length and on each rule's size, uniform among
 * antecedents of a size) plus the log Dirichlet-multinomial marginal likelihood of the classes each rule captures. It is
 * the density the sampler of {@link bayesianRuleListAlgorithm} targets.
 *
 * @param data - Features and class labels.
 * @param list - The antecedent indices in order.
 * @param options - The prior and the antecedent mining options.
 * @returns The log posterior.
 * @throws MdmlError with code `value` for non-positive prior means.
 * @example
 * ```ts
 * import { ruleListLogPosterior } from 'mdml-engine/rules'
 * const data = { x: [[0], [0], [1], [1]], y: [0, 0, 1, 1] }
 * const o = { nominal: [0], minSupport: 0.3, maxCardinality: 1 }
 * ruleListLogPosterior(data, [0], o) > ruleListLogPosterior(data, [], o) // => true
 * ```
 */
export function ruleListLogPosterior(
  data: LearnData,
  list: readonly number[],
  options: BayesianRuleListOptions = {},
): number {
  return posteriorOf(data, options).logPosterior(list)
}

/**
 * Bayesian rule lists (Letham, Rudin, McCormick and Madigan 2015) sampled by Metropolis-Hastings as a traced algorithm,
 * one proposal per step. A list is an ordered selection of the pre-mined antecedents; its prior multiplies a truncated
 * Poisson on the length, a truncated Poisson on each rule's size and a uniform choice among antecedents of that size,
 * and its likelihood is the Dirichlet-multinomial marginal of the classes each rule captures from the rows earlier rules
 * left. A proposal adds an antecedent at a random position, removes a rule, or swaps two, with the Hastings correction.
 *
 * @param data - Features and class labels.
 * @param options - The prior's means, the Dirichlet parameter and the antecedent mining options.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for non-positive prior means.
 * @example
 * ```ts
 * import { bayesianRuleListAlgorithm } from 'mdml-engine/rules'
 * bayesianRuleListAlgorithm({ x: [[0], [1]], y: [0, 1] }).name // => 'bayesian-rule-list'
 * ```
 */
export function bayesianRuleListAlgorithm(
  data: LearnData,
  options: BayesianRuleListOptions = {},
): Algorithm<BayesianRuleListState> {
  const { A, logPosterior, config } = posteriorOf(data, options)
  return {
    name: 'bayesian-rule-list',
    config,
    schema: BRL_SCHEMA,
    init: () => {
      const lp = logPosterior([])
      return {
        list: [],
        logPosterior: lp,
        best: [],
        bestLogPosterior: lp,
        proposals: 0,
        accepted: 0,
        move: 'none',
        acceptedLast: false,
      }
    },
    step: (s, ctx) => {
      const u = uniform(foldIn(ctx.key, 7), [5]).data
      const m = s.list.length
      const kind = Math.min(2, Math.floor((u[0] ?? 0) * 3))
      let proposal: number[] | null = null
      let logRatio = 0
      let move = 'none'
      if (kind === 0 && m < A) {
        move = 'add'
        const free: number[] = []
        for (let a = 0; a < A; a++) if (!s.list.includes(a)) free.push(a)
        const a = free[Math.floor((u[1] ?? 0) * free.length)] as number
        const pos = Math.floor((u[2] ?? 0) * (m + 1))
        proposal = [...s.list.slice(0, pos), a, ...s.list.slice(pos)]
        logRatio = Math.log(A - m)
      } else if (kind === 1 && m > 0) {
        move = 'remove'
        const pos = Math.floor((u[2] ?? 0) * m)
        proposal = s.list.filter((_, i) => i !== pos)
        logRatio = -Math.log(A - m + 1)
      } else if (kind === 2 && m > 1) {
        move = 'swap'
        const i = Math.floor((u[1] ?? 0) * m)
        let j = Math.floor((u[2] ?? 0) * (m - 1))
        if (j >= i) j++
        proposal = [...s.list]
        proposal[i] = s.list[j] as number
        proposal[j] = s.list[i] as number
      }
      let list = s.list
      let lp = s.logPosterior
      let accepted = false
      if (proposal !== null) {
        const lpNew = logPosterior(proposal)
        if (Math.log(Math.max(u[3] ?? 0, 1e-300)) < lpNew - s.logPosterior + logRatio) {
          list = proposal
          lp = lpNew
          accepted = true
        }
      }
      return {
        list,
        logPosterior: lp,
        best: lp > s.bestLogPosterior ? list : s.best,
        bestLogPosterior: Math.max(lp, s.bestLogPosterior),
        proposals: s.proposals + 1,
        accepted: s.accepted + (accepted ? 1 : 0),
        move,
        acceptedLast: accepted,
      }
    },
    observe: (s) => ({
      metrics: {
        logPosterior: s.logPosterior,
        bestLogPosterior: s.bestLogPosterior,
        length: s.list.length,
        accepted: s.accepted,
      },
      messages: s.proposals === 0 ? [] : [{ kind: 'move', payload: { move: s.move, accepted: s.acceptedLast } }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as BayesianRuleListState,
  }
}

/**
 * The most probable list the sampler has visited, as a model whose rule probabilities are the posterior means.
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The options the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { bayesianRuleListAlgorithm, bayesianRuleListModel } from 'mdml-engine/rules'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = { x: [[0], [0], [0], [1], [1], [1]], y: [0, 0, 0, 1, 1, 1] }
 * const o = { minSupport: 0.3, maxCardinality: 1 }
 * const { state } = runAlgorithm(bayesianRuleListAlgorithm(data, o), { maxSteps: 200, seed: 1 })
 * bayesianRuleListModel(data, state, o).classes // => [0, 1]
 * ```
 */
export function bayesianRuleListModel(
  data: LearnData,
  s: BayesianRuleListState,
  options: BayesianRuleListOptions = {},
): RuleModel {
  const rd = ruleData(data, options.nominal ?? [], 'bayesianRuleList')
  return listModel(rd, mine(rd, options).antecedents, s.best)
}

/**
 * A Bayesian rule list: the highest-posterior list among those sampled in `steps` Metropolis-Hastings proposals.
 *
 * @param data - Features and class labels.
 * @param options - The prior, antecedent mining, `steps` (default 2000), a seed and tracing.
 * @returns The model.
 * @throws MdmlError with code `value` for non-positive prior means.
 * @example
 * ```ts
 * import { fitBayesianRuleList } from 'mdml-engine/rules'
 * const data = { x: [[0], [0], [0], [0], [1], [1], [1], [1]], y: [0, 0, 0, 0, 1, 1, 1, 1] }
 * fitBayesianRuleList(data, { minSupport: 0.3, maxCardinality: 1, seed: 2 }).rules.length > 0 // => true
 * ```
 */
export function fitBayesianRuleList(
  data: LearnData,
  options: BayesianRuleListOptions & { readonly steps?: number; readonly seed?: number } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>
    > = {},
): RuleModel {
  const { steps, seed, tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(seed === undefined ? {} : { seed }),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(bayesianRuleListAlgorithm(data, rest), { ...run, maxSteps: steps ?? 2000 })
  return bayesianRuleListModel(data, state, rest)
}
