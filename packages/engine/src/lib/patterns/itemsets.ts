/**
 * Frequent itemset mining on market-basket transactions: Apriori (level-wise candidates joined and pruned by the
 * downward closure), FP-growth (a prefix tree of the transactions mined by conditional trees) and Eclat (vertical
 * tid-lists intersected depth first), each also as a traced algorithm whose step is a level, a transaction or a
 * conditional tree, and a prefix class. The three return exactly the same itemsets with the same counts on every
 * input, which an exhaustive enumeration checks. Closed and maximal itemsets are filtered from a complete result.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'

/** A basket: the integer ids of the items in it (order and repeats do not matter). */
export type Transaction = readonly number[]

/** An itemset and the number of transactions that contain all of it. */
export interface FrequentItemset {
  /** The items, sorted ascending. */
  readonly items: readonly number[]
  readonly count: number
}

/** Options shared by the itemset miners. */
export interface ItemsetOptions {
  /** The smallest support as a fraction of the transactions, in (0, 1]. */
  readonly minSupport: number
  /** The largest itemset (default no limit). */
  readonly maxLength?: number
}

/** Tracing options of the miners that run an algorithm. */
export type ItemsetRunOptions = ItemsetOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>>

/**
 * The smallest count that meets a support fraction: `ceil(minSupport * n)`, with a tolerance so that 0.3 of 10
 * transactions is 3 and not 4.
 *
 * @param n - The number of transactions.
 * @param minSupport - The support fraction in (0, 1].
 * @returns The count.
 * @throws OmaiError with code `value` when the fraction is outside (0, 1].
 * @example
 * ```ts
 * import { minCount } from 'omai-engine/patterns'
 * [minCount(10, 0.3), minCount(7, 0.5)] // => [3, 4]
 * ```
 */
export function minCount(n: number, minSupport: number): number {
  if (!(minSupport > 0 && minSupport <= 1)) throw new OmaiError('value', `minSupport = ${minSupport} must be in (0, 1]`)
  return Math.max(1, Math.ceil(minSupport * n - 1e-9))
}

/**
 * Baskets with repeats removed and items sorted, the form every miner works on.
 *
 * @param transactions - Baskets of item ids.
 * @returns The normalised baskets.
 * @throws OmaiError with code `value` for an item that is not a non-negative integer.
 * @example
 * ```ts
 * import { normaliseTransactions } from 'omai-engine/patterns'
 * normaliseTransactions([[3, 1, 3], []]) // => [[1, 3], []]
 * ```
 */
export function normaliseTransactions(transactions: readonly Transaction[]): number[][] {
  return transactions.map((t) => {
    for (const i of t) if (!(Number.isInteger(i) && i >= 0)) throw new OmaiError('value', `item ${i} is not an id`)
    return [...new Set(t)].sort((a, b) => a - b)
  })
}

/**
 * The number of baskets that contain every item of an itemset.
 *
 * @param transactions - The baskets.
 * @param itemset - The items.
 * @returns The count.
 * @example
 * ```ts
 * import { supportCount } from 'omai-engine/patterns'
 * supportCount([[1, 2, 3], [1, 2], [2, 3]], [1, 2]) // => 2
 * ```
 */
export function supportCount(transactions: readonly Transaction[], itemset: readonly number[]): number {
  let c = 0
  for (const t of transactions) if (itemset.every((i) => t.includes(i))) c++
  return c
}

const byItems = (a: FrequentItemset, b: FrequentItemset): number => {
  if (a.items.length !== b.items.length) return a.items.length - b.items.length
  for (let i = 0; i < a.items.length; i++) {
    const d = (a.items[i] as number) - (b.items[i] as number)
    if (d !== 0) return d
  }
  return 0
}

/**
 * Itemsets in canonical order: by size, then lexicographically, items sorted in each. Two miners agree when their
 * canonical lists are equal.
 *
 * @param itemsets - Itemsets in any order.
 * @returns A sorted copy.
 * @example
 * ```ts
 * import { canonicalItemsets } from 'omai-engine/patterns'
 * canonicalItemsets([{ items: [2, 1], count: 1 }, { items: [0], count: 4 }]).map((s) => s.items.join()) // => ['0', '1,2']
 * ```
 */
export function canonicalItemsets(itemsets: readonly FrequentItemset[]): FrequentItemset[] {
  return itemsets.map((s) => ({ items: [...s.items].sort((a, b) => a - b), count: s.count })).sort(byItems)
}

const keyOf = (items: readonly number[]): string => items.join(',')

/**
 * Every frequent itemset by enumerating all subsets of the item universe and counting them: exponential in the number
 * of distinct items, the reference the three miners are checked against.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold and the largest itemset.
 * @returns The frequent itemsets in canonical order.
 * @throws OmaiError with code `value` for more than 22 distinct items.
 * @example
 * ```ts
 * import { exhaustiveItemsets } from 'omai-engine/patterns'
 * exhaustiveItemsets([[0, 1], [0, 1], [0]], { minSupport: 0.6 }).map((s) => `${s.items}:${s.count}`) // => ['0:3', '1:2', '0,1:2']
 * ```
 */
export function exhaustiveItemsets(transactions: readonly Transaction[], options: ItemsetOptions): FrequentItemset[] {
  const tx = normaliseTransactions(transactions)
  const need = minCount(tx.length, options.minSupport)
  const universe = [...new Set(tx.flat())].sort((a, b) => a - b)
  if (universe.length > 22) throw new OmaiError('value', 'exhaustiveItemsets: more than 22 distinct items')
  const max = options.maxLength ?? universe.length
  const out: FrequentItemset[] = []
  for (let mask = 1; mask < 1 << universe.length; mask++) {
    const items = universe.filter((_, b) => (mask >> b) & 1)
    if (items.length > max) continue
    const count = supportCount(tx, items)
    if (count >= need) out.push({ items, count })
  }
  return out.sort(byItems)
}

// ------------------------------------------------------------------------------------------------ Apriori

/** The state of {@link aprioriAlgorithm}: the level just counted and everything frequent so far. */
export interface AprioriState {
  /** The size of the itemsets just counted (0 before the first step). */
  readonly k: number
  /** The frequent itemsets of size k. */
  readonly level: readonly FrequentItemset[]
  /** Every frequent itemset found. */
  readonly frequent: readonly FrequentItemset[]
  /** Candidates generated at this level after the join, before the downward-closure prune. */
  readonly joined: number
  /** Candidates removed by the downward-closure prune. */
  readonly pruned: number
  /** Candidates counted against the baskets. */
  readonly candidates: number
  readonly done: boolean
}

const APRIORI_SCHEMA: TraceSchema = channels({
  metrics: {
    k: { description: 'the size of the itemsets counted at this level' },
    candidates: { description: 'candidates counted after joining and pruning' },
    frequent: { description: 'frequent itemsets at this level' },
    pruned: { description: 'candidates removed because a subset was infrequent' },
    total: { description: 'frequent itemsets so far' },
  },
})

/** Join the frequent (k-1)-itemsets sharing a (k-2)-prefix, then drop the candidates with an infrequent subset. */
function aprioriGenerate(level: readonly FrequentItemset[]): { candidates: number[][]; joined: number } {
  const known = new Set(level.map((s) => keyOf(s.items)))
  const sorted = level.map((s) => s.items).sort((a, b) => keyOrder(a, b))
  const candidates: number[][] = []
  let joined = 0
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i] as readonly number[]
    for (let j = i + 1; j < sorted.length; j++) {
      const b = sorted[j] as readonly number[]
      const m = a.length - 1
      let same = true
      for (let p = 0; p < m; p++) if (a[p] !== b[p]) same = false
      if (!same) break
      const c = [...a, b[m] as number]
      joined++
      let ok = true
      for (let drop = 0; drop < c.length - 2 && ok; drop++) {
        ok = known.has(keyOf(c.filter((_, q) => q !== drop)))
      }
      if (ok) candidates.push(c)
    }
  }
  return { candidates, joined }
}

function keyOrder(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = (a[i] as number) - (b[i] as number)
    if (d !== 0) return d
  }
  return a.length - b.length
}

function countCandidates(tx: readonly (readonly number[])[], candidates: readonly (readonly number[])[]): number[] {
  const counts = new Array<number>(candidates.length).fill(0)
  for (const t of tx) {
    const present = new Set(t)
    for (let c = 0; c < candidates.length; c++) {
      if ((candidates[c] as readonly number[]).every((i) => present.has(i))) counts[c] = (counts[c] ?? 0) + 1
    }
  }
  return counts
}

/**
 * Apriori (Agrawal and Srikant 1994) as a traced algorithm, one level per step: the singletons are counted first, then
 * each step joins the frequent (k-1)-itemsets that share a prefix into k-candidates, prunes those with an infrequent
 * subset (the downward closure), counts the rest against the baskets and keeps the frequent ones.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold and the largest itemset.
 * @returns The algorithm; its state holds the level, the candidates and pruned counts, and every itemset found.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { aprioriAlgorithm } from 'omai-engine/patterns'
 * aprioriAlgorithm([[0, 1], [0, 1]], { minSupport: 0.5 }).name // => 'apriori'
 * ```
 */
export function aprioriAlgorithm(
  transactions: readonly Transaction[],
  options: ItemsetOptions,
): Algorithm<AprioriState> {
  const tx = normaliseTransactions(transactions)
  const need = minCount(tx.length, options.minSupport)
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  return {
    name: 'apriori',
    config: {
      minSupport: options.minSupport,
      minCount: need,
      ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    },
    schema: APRIORI_SCHEMA,
    init: () => ({ k: 0, level: [], frequent: [], joined: 0, pruned: 0, candidates: 0, done: false }),
    step: (s) => {
      const k = s.k + 1
      let candidates: number[][]
      let joined = 0
      if (k === 1) {
        const items = [...new Set(tx.flat())].sort((a, b) => a - b)
        candidates = items.map((i) => [i])
        joined = candidates.length
      } else {
        ;({ candidates, joined } = aprioriGenerate(s.level))
      }
      const counts = countCandidates(tx, candidates)
      const level: FrequentItemset[] = []
      candidates.forEach((c, i) => {
        if ((counts[i] ?? 0) >= need) level.push({ items: c, count: counts[i] as number })
      })
      const frequent = [...s.frequent, ...level]
      return {
        k,
        level,
        frequent,
        joined,
        pruned: joined - candidates.length,
        candidates: candidates.length,
        done: level.length === 0 || k >= maxLength,
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        k: s.k,
        candidates: s.candidates,
        frequent: s.level.length,
        pruned: s.pruned,
        total: s.frequent.length,
      },
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as AprioriState,
  }
}

/**
 * Frequent itemsets by Apriori.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold, the largest itemset and tracing.
 * @returns The frequent itemsets in canonical order.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @index function Frequent itemsets by Apriori; FP-growth and Eclat beside it.
 * @example
 * ```ts
 * import { apriori } from 'omai-engine/patterns'
 * apriori([[0, 1, 2], [0, 1], [0, 2], [1, 2]], { minSupport: 0.5 }).map((s) => `${s.items}:${s.count}`) // => ['0:3', '1:3', '2:3', '0,1:2', '0,2:2', '1,2:2']
 * ```
 */
export function apriori(transactions: readonly Transaction[], options: ItemsetRunOptions): FrequentItemset[] {
  const { minSupport, maxLength, ...run } = options
  const alg = aprioriAlgorithm(transactions, { minSupport, ...(maxLength === undefined ? {} : { maxLength }) })
  const { state } = runAlgorithm(alg, { ...run, maxSteps: 100_000 })
  return canonicalItemsets(state.frequent)
}

// ------------------------------------------------------------------------------------------------ FP-growth

/** A prefix tree of transactions: node 0 is the root; a path from the root is an itemset prefix. */
export interface FpTree {
  readonly item: number[]
  readonly count: number[]
  readonly parent: number[]
  /** `childOf[`${node}:${item}`]` is the child of `node` labelled `item`. */
  readonly childOf: Record<string, number>
  /** The nodes labelled by each item (the header table's node links). */
  readonly nodes: Record<string, number[]>
}

/** A conditional pattern base entry: a prefix path and how many baskets share it. */
interface WeightedPath {
  readonly path: readonly number[]
  readonly count: number
}

interface MiningTask {
  readonly suffix: readonly number[]
  readonly base: readonly WeightedPath[]
}

/** The state of {@link fpGrowthAlgorithm}. */
export interface FpGrowthState {
  readonly phase: 'count' | 'build' | 'mine' | 'done'
  /** The next transaction to insert. */
  readonly next: number
  /** The frequent items, most frequent first (ties by id): the order paths are inserted in. */
  readonly order: readonly number[]
  readonly tree: FpTree
  /** Whether the full tree has been mined. */
  readonly rootMined: boolean
  /** Conditional pattern bases still to mine. */
  readonly tasks: readonly MiningTask[]
  readonly frequent: readonly FrequentItemset[]
  /** The suffix whose conditional tree was mined in the last step (empty for the full tree). */
  readonly suffix: readonly number[]
  /** The nodes of the conditional tree just mined. */
  readonly treeSize: number
}

function emptyTree(): FpTree {
  return { item: [-1], count: [0], parent: [-1], childOf: {}, nodes: {} }
}

/** Insert a path (already filtered and ordered) with a count into a tree, sharing prefixes. */
function insertPath(tree: FpTree, path: readonly number[], count: number): void {
  let node = 0
  for (const it of path) {
    const k = `${node}:${it}`
    let child = tree.childOf[k]
    if (child === undefined) {
      child = tree.item.length
      tree.item.push(it)
      tree.count.push(0)
      tree.parent.push(node)
      tree.childOf[k] = child
      const list = tree.nodes[String(it)]
      if (list === undefined) tree.nodes[String(it)] = [child]
      else list.push(child)
    }
    tree.count[child] = (tree.count[child] ?? 0) + count
    node = child
  }
}

/** Build a tree from weighted paths, keeping the items that meet the count, most frequent first. */
function treeOfBase(base: readonly WeightedPath[], need: number): FpTree {
  const counts = new Map<number, number>()
  for (const p of base) for (const i of p.path) counts.set(i, (counts.get(i) ?? 0) + p.count)
  const rank = new Map<number, number>()
  ;[...counts.entries()]
    .filter(([, c]) => c >= need)
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .forEach(([i], r) => {
      rank.set(i, r)
    })
  const tree = emptyTree()
  for (const p of base) {
    const path = p.path.filter((i) => rank.has(i)).sort((a, b) => (rank.get(a) as number) - (rank.get(b) as number))
    if (path.length > 0) insertPath(tree, path, p.count)
  }
  return tree
}

/** Mine one tree: every item in it extends the suffix; its conditional pattern base is queued for later. */
function mineTree(
  tree: FpTree,
  suffix: readonly number[],
  maxLength: number,
): { found: FrequentItemset[]; tasks: MiningTask[] } {
  const found: FrequentItemset[] = []
  const tasks: MiningTask[] = []
  const items = Object.keys(tree.nodes)
    .map(Number)
    .map((it) => ({ it, total: (tree.nodes[String(it)] ?? []).reduce((s, n) => s + (tree.count[n] ?? 0), 0) }))
    .sort((a, b) => a.total - b.total || b.it - a.it)
  for (const { it, total } of items) {
    const next = [it, ...suffix]
    if (next.length > maxLength) continue
    found.push({ items: [...next].sort((a, b) => a - b), count: total })
    if (next.length >= maxLength) continue
    const base: WeightedPath[] = []
    for (const n of tree.nodes[String(it)] ?? []) {
      const path: number[] = []
      for (let p = tree.parent[n] as number; p > 0; p = tree.parent[p] as number) path.push(tree.item[p] as number)
      if (path.length > 0) base.push({ path: path.reverse(), count: tree.count[n] as number })
    }
    if (base.length > 0) tasks.push({ suffix: next, base })
  }
  return { found, tasks }
}

const FP_SCHEMA: TraceSchema = channels({
  metrics: {
    nodes: { description: 'nodes in the tree being built or the conditional tree being mined' },
    inserted: { description: 'transactions inserted into the full tree' },
    frequent: { description: 'frequent itemsets found so far' },
    pending: { description: 'conditional pattern bases waiting to be mined' },
  },
  messages: {
    mine: { description: 'a conditional tree mined', payload: { suffix: 'the suffix items', nodes: 'tree size' } },
  },
})

/**
 * FP-growth (Han, Pei and Yin 2000) as a traced algorithm: one pass counts the items, then each step inserts one
 * transaction (items filtered to the frequent ones and ordered by descending count) into the FP-tree, and then each
 * step mines one conditional FP-tree: it reports the itemsets that extend the tree's suffix by one item and queues the
 * conditional pattern base of each.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold and the largest itemset.
 * @returns The algorithm; the state holds the tree, the mining tasks and the itemsets found.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { fpGrowthAlgorithm } from 'omai-engine/patterns'
 * fpGrowthAlgorithm([[0, 1], [0, 1]], { minSupport: 0.5 }).name // => 'fp-growth'
 * ```
 */
export function fpGrowthAlgorithm(
  transactions: readonly Transaction[],
  options: ItemsetOptions,
): Algorithm<FpGrowthState> {
  const tx = normaliseTransactions(transactions)
  const need = minCount(tx.length, options.minSupport)
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  const clone = (s: FpGrowthState): FpGrowthState => ({
    ...s,
    tree: {
      item: [...s.tree.item],
      count: [...s.tree.count],
      parent: [...s.tree.parent],
      childOf: { ...s.tree.childOf },
      nodes: Object.fromEntries(Object.entries(s.tree.nodes).map(([k, v]) => [k, [...v]])),
    },
  })
  return {
    name: 'fp-growth',
    config: {
      minSupport: options.minSupport,
      minCount: need,
      ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    },
    schema: FP_SCHEMA,
    init: () => ({
      phase: 'count',
      next: 0,
      order: [],
      tree: emptyTree(),
      rootMined: false,
      tasks: [],
      frequent: [],
      suffix: [],
      treeSize: 0,
    }),
    step: (prev) => {
      if (prev.phase === 'count') {
        const counts = new Map<number, number>()
        for (const t of tx) for (const i of t) counts.set(i, (counts.get(i) ?? 0) + 1)
        const order = [...counts.entries()]
          .filter(([, c]) => c >= need)
          .sort((a, b) => b[1] - a[1] || a[0] - b[0])
          .map(([i]) => i)
        return { ...prev, order, phase: order.length === 0 ? 'done' : 'build' }
      }
      const s = clone(prev)
      if (s.phase === 'build') {
        const rank = new Map(s.order.map((i, r) => [i, r]))
        const t = tx[s.next] ?? []
        const path = t.filter((i) => rank.has(i)).sort((a, b) => (rank.get(a) as number) - (rank.get(b) as number))
        insertPath(s.tree, path, 1)
        const next = s.next + 1
        return { ...s, next, phase: next >= tx.length ? 'mine' : 'build', treeSize: s.tree.item.length - 1 }
      }
      if (s.phase === 'mine') {
        let tree = s.tree
        let suffix: readonly number[] = []
        let tasks = [...s.tasks]
        if (s.rootMined) {
          const task = tasks.pop() as MiningTask
          suffix = task.suffix
          tree = treeOfBase(task.base, need)
        }
        const { found, tasks: more } = mineTree(tree, suffix, maxLength)
        tasks = [...tasks, ...more]
        return {
          ...s,
          rootMined: true,
          tasks,
          frequent: [...s.frequent, ...found],
          suffix,
          treeSize: tree.item.length - 1,
          phase: tasks.length === 0 ? 'done' : 'mine',
        }
      }
      return s
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { nodes: s.treeSize, inserted: s.next, frequent: s.frequent.length, pending: s.tasks.length },
      messages:
        s.phase === 'mine' || (s.phase === 'done' && s.rootMined)
          ? [{ kind: 'mine', payload: { suffix: [...s.suffix], nodes: s.treeSize } }]
          : [],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as FpGrowthState,
  }
}

/**
 * Frequent itemsets by FP-growth.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold, the largest itemset and tracing.
 * @returns The frequent itemsets in canonical order.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { fpGrowth } from 'omai-engine/patterns'
 * fpGrowth([[0, 1, 2], [0, 1], [0, 2], [1, 2]], { minSupport: 0.5 }).length // => 6
 * ```
 */
export function fpGrowth(transactions: readonly Transaction[], options: ItemsetRunOptions): FrequentItemset[] {
  const { minSupport, maxLength, ...run } = options
  const alg = fpGrowthAlgorithm(transactions, { minSupport, ...(maxLength === undefined ? {} : { maxLength }) })
  const { state } = runAlgorithm(alg, { ...run, maxSteps: 10_000_000 })
  return canonicalItemsets(state.frequent)
}

// ------------------------------------------------------------------------------------------------ Eclat

interface TidItem {
  readonly item: number
  readonly tids: readonly number[]
}

interface EclatClass {
  readonly prefix: readonly number[]
  readonly extensions: readonly TidItem[]
}

/** The state of {@link eclatAlgorithm}. */
export interface EclatState {
  /** Prefix classes still to expand. */
  readonly stack: readonly EclatClass[]
  readonly frequent: readonly FrequentItemset[]
  /** The prefix expanded in the last step. */
  readonly prefix: readonly number[]
  /** Tid-list intersections computed in the last step. */
  readonly intersections: number
  readonly started: boolean
}

function intersectSorted(a: readonly number[], b: readonly number[]): number[] {
  const out: number[] = []
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    const x = a[i] as number
    const y = b[j] as number
    if (x === y) {
      out.push(x)
      i++
      j++
    } else if (x < y) i++
    else j++
  }
  return out
}

const ECLAT_SCHEMA: TraceSchema = channels({
  metrics: {
    pending: { description: 'prefix classes waiting to be expanded' },
    frequent: { description: 'frequent itemsets found so far' },
    intersections: { description: 'tid-list intersections in this step' },
  },
  messages: { expand: { description: 'a prefix class expanded', payload: { prefix: 'the prefix items' } } },
})

/**
 * Eclat (Zaki 2000) as a traced algorithm: the baskets are turned into one tid-list per item, then each step expands
 * one prefix class, intersecting the tid-lists of its extensions pairwise to find the frequent itemsets one item
 * longer and queuing the classes they head.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold and the largest itemset.
 * @returns The algorithm; the state holds the stack of classes and the itemsets found.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { eclatAlgorithm } from 'omai-engine/patterns'
 * eclatAlgorithm([[0, 1], [0, 1]], { minSupport: 0.5 }).name // => 'eclat'
 * ```
 */
export function eclatAlgorithm(transactions: readonly Transaction[], options: ItemsetOptions): Algorithm<EclatState> {
  const tx = normaliseTransactions(transactions)
  const need = minCount(tx.length, options.minSupport)
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  return {
    name: 'eclat',
    config: {
      minSupport: options.minSupport,
      minCount: need,
      ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    },
    schema: ECLAT_SCHEMA,
    init: () => ({ stack: [], frequent: [], prefix: [], intersections: 0, started: false }),
    step: (s) => {
      if (!s.started) {
        const tids = new Map<number, number[]>()
        tx.forEach((t, id) => {
          for (const i of t) {
            const list = tids.get(i)
            if (list === undefined) tids.set(i, [id])
            else list.push(id)
          }
        })
        const extensions: TidItem[] = [...tids.entries()]
          .filter(([, l]) => l.length >= need)
          .sort((a, b) => a[0] - b[0])
          .map(([item, list]) => ({ item, tids: list }))
        const frequent = extensions.map((e) => ({ items: [e.item], count: e.tids.length }))
        return {
          stack: extensions.length > 1 && maxLength > 1 ? [{ prefix: [], extensions }] : [],
          frequent: maxLength >= 1 ? frequent : [],
          prefix: [],
          intersections: 0,
          started: true,
        }
      }
      const stack = [...s.stack]
      const cls = stack.pop() as EclatClass
      const frequent = [...s.frequent]
      let intersections = 0
      cls.extensions.forEach((a, ai) => {
        const prefix = [...cls.prefix, a.item]
        if (prefix.length + 1 > maxLength) return
        const extensions: TidItem[] = []
        for (let bi = ai + 1; bi < cls.extensions.length; bi++) {
          const b = cls.extensions[bi] as TidItem
          const tids = intersectSorted(a.tids, b.tids)
          intersections++
          if (tids.length >= need) {
            extensions.push({ item: b.item, tids })
            frequent.push({ items: [...prefix, b.item], count: tids.length })
          }
        }
        if (extensions.length > 1) stack.push({ prefix, extensions })
      })
      return { stack, frequent, prefix: cls.prefix, intersections, started: true }
    },
    done: (s) => (s.started && s.stack.length === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { pending: s.stack.length, frequent: s.frequent.length, intersections: s.intersections },
      messages: s.started ? [{ kind: 'expand', payload: { prefix: [...s.prefix] } }] : [],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as EclatState,
  }
}

/**
 * Frequent itemsets by Eclat.
 *
 * @param transactions - The baskets.
 * @param options - The support threshold, the largest itemset and tracing.
 * @returns The frequent itemsets in canonical order.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { eclat } from 'omai-engine/patterns'
 * eclat([[0, 1, 2], [0, 1], [0, 2], [1, 2]], { minSupport: 0.5 }).length // => 6
 * ```
 */
export function eclat(transactions: readonly Transaction[], options: ItemsetRunOptions): FrequentItemset[] {
  const { minSupport, maxLength, ...run } = options
  const alg = eclatAlgorithm(transactions, { minSupport, ...(maxLength === undefined ? {} : { maxLength }) })
  const { state } = runAlgorithm(alg, { ...run, maxSteps: 10_000_000 })
  return canonicalItemsets(state.frequent)
}

// ------------------------------------------------------------------------------------------------ closed and maximal

/**
 * The closed itemsets of a complete frequent-itemset list: those with no proper superset of the same count. They
 * determine every frequent itemset's count (the count of an itemset is the largest count of a closed superset).
 *
 * @param itemsets - All the frequent itemsets (from any miner, with no `maxLength`).
 * @returns The closed itemsets in canonical order.
 * @example
 * ```ts
 * import { closedItemsets } from 'omai-engine/patterns'
 * const all = [{ items: [0], count: 3 }, { items: [1], count: 2 }, { items: [0, 1], count: 2 }]
 * closedItemsets(all).map((s) => s.items.join()) // => ['0', '0,1']
 * ```
 */
export function closedItemsets(itemsets: readonly FrequentItemset[]): FrequentItemset[] {
  const covered = new Set<string>()
  const counts = new Map(itemsets.map((s) => [keyOf(s.items), s.count]))
  for (const s of itemsets) {
    for (let drop = 0; drop < s.items.length; drop++) {
      const sub = keyOf(s.items.filter((_, p) => p !== drop))
      if (counts.get(sub) === s.count) covered.add(sub)
    }
  }
  return canonicalItemsets(itemsets.filter((s) => !covered.has(keyOf(s.items))))
}

/**
 * The maximal itemsets of a complete frequent-itemset list: those with no frequent proper superset.
 *
 * @param itemsets - All the frequent itemsets.
 * @returns The maximal itemsets in canonical order.
 * @example
 * ```ts
 * import { maximalItemsets } from 'omai-engine/patterns'
 * const all = [{ items: [0], count: 3 }, { items: [1], count: 2 }, { items: [0, 1], count: 2 }]
 * maximalItemsets(all).map((s) => s.items.join()) // => ['0,1']
 * ```
 */
export function maximalItemsets(itemsets: readonly FrequentItemset[]): FrequentItemset[] {
  const covered = new Set<string>()
  for (const s of itemsets) {
    for (let drop = 0; drop < s.items.length; drop++) covered.add(keyOf(s.items.filter((_, p) => p !== drop)))
  }
  return canonicalItemsets(itemsets.filter((s) => !covered.has(keyOf(s.items))))
}
