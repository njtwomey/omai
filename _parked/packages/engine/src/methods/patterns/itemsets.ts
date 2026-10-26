/**
 * Frequent itemset mining, as in mlxtend's `apriori` and `fpgrowth`: Apriori played level by level over the itemset
 * lattice (join frequent itemsets into larger candidates, prune those with an infrequent subset, count the rest), and
 * FP-growth played transaction by transaction as its prefix tree grows. Also Han and Kamber's textbook baskets and
 * generated baskets with planted associations.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, uniform } from '../../lib/random/index.ts'

/** Transactions: each a set of items. */
export type Transactions = readonly (readonly string[])[]

/** A frequent itemset: its items (sorted), the transactions containing it and that count's share. */
export interface Itemset {
  readonly items: readonly string[]
  readonly count: number
  readonly support: number
}

/** Han and Kamber's nine transactions over items I1 to I5 (the worked example of Apriori and FP-growth). */
export const TEXTBOOK_BASKETS: Transactions = [
  ['I1', 'I2', 'I5'],
  ['I2', 'I4'],
  ['I2', 'I3'],
  ['I1', 'I2', 'I4'],
  ['I1', 'I3'],
  ['I2', 'I3'],
  ['I1', 'I3'],
  ['I1', 'I2', 'I3', 'I5'],
  ['I1', 'I2', 'I3'],
]

const GROCERIES = ['bread', 'milk', 'eggs', 'butter', 'beer', 'nappies', 'cola', 'crisps', 'cheese', 'apples']

/**
 * Shopping baskets with planted associations: bread with butter, nappies with beer, crisps with cola, plus noise.
 *
 * @param key - The key.
 * @param n - The number of baskets (default 60).
 * @returns The transactions.
 */
export function marketBaskets(key: Key, n = 60): Transactions {
  const u = Array.from(uniform(key, [n, GROCERIES.length + 3]).data)
  const at = (i: number, j: number): number => u[i * (GROCERIES.length + 3) + j] ?? 0
  return Array.from({ length: n }, (_, i) => {
    const basket = new Set<string>()
    GROCERIES.forEach((g, j) => {
      if (at(i, j) < 0.18) basket.add(g)
    })
    if (at(i, GROCERIES.length) < 0.45) {
      basket.add('bread')
      if (at(i, GROCERIES.length + 1) < 0.7) basket.add('butter')
      if (at(i, GROCERIES.length + 2) < 0.5) basket.add('milk')
    }
    if (at(i, GROCERIES.length + 1) > 0.72) {
      basket.add('nappies')
      if (at(i, GROCERIES.length + 2) > 0.3) basket.add('beer')
    }
    if (at(i, GROCERIES.length + 2) > 0.8) basket.add('crisps').add('cola')
    if (basket.size === 0) basket.add(GROCERIES[i % GROCERIES.length] as string)
    return [...basket].sort()
  })
}

/** Options of the itemset miners. */
export interface ItemsetOptions {
  /** The smallest support kept, a share of the transactions (default 0.2). */
  readonly minSupport?: number
  /** The largest itemset size (default unlimited). */
  readonly maxLength?: number
}

/** One level of Apriori: the candidates of size k, those pruned unseen, and those frequent. */
export interface AprioriLevel {
  readonly k: number
  readonly candidates: number
  /** Candidates with an infrequent subset, dropped without counting. */
  readonly pruned: number
  readonly frequent: number
}

/** What the itemset miners return. */
export interface ItemsetResult {
  readonly itemsets: readonly Itemset[]
  readonly transactions: number
  readonly minSupport: number
}

const keyOf = (items: readonly string[]): string => items.join('\u0001')

function checkOptions(transactions: Transactions, options: ItemsetOptions, name: string): number {
  const minSupport = options.minSupport ?? 0.2
  if (!(minSupport > 0 && minSupport <= 1)) throw new MdmlError('value', `${name}: minSupport must be in (0, 1]`)
  if (transactions.length === 0) throw new MdmlError('value', `${name}: no transactions`)
  return Math.ceil(minSupport * transactions.length - 1e-9)
}

// Short codes for items (A, B, C, ... in order of frequency) so lattice nodes stay legible.
function codesOf(items: readonly string[]): Map<string, string> {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return new Map(items.map((it, i) => [it, i < letters.length ? (letters[i] as string) : `#${i}`]))
}

/**
 * Apriori (Agrawal and Srikant 1994): count the single items, then repeatedly join the frequent itemsets of size k into
 * candidates of size k + 1, prune those with an infrequent subset unseen, and count the rest in one pass. Recorded
 * level by level over the lattice of every candidate: generated, pruned by the Apriori property, then counted
 * frequent or infrequent, with their support counts.
 *
 * @param transactions - The transactions.
 * @param options - Minimum support and maximum length.
 * @returns The frequent itemsets with the levels' counts, and a `boardSteps` record of the lattice.
 */
export function apriori(
  transactions: Transactions,
  options: ItemsetOptions = {},
): Explained<ItemsetResult & { readonly levels: readonly AprioriLevel[] }> {
  const minCount = checkOptions(transactions, options, 'apriori')
  const sets = transactions.map((t) => new Set(t))
  const countOf = (items: readonly string[]): number => sets.filter((s) => items.every((i) => s.has(i))).length
  const singles = [...new Set(transactions.flat())]
  const single = new Map(singles.map((i) => [i, countOf([i])]))
  singles.sort((a, b) => (single.get(b) ?? 0) - (single.get(a) ?? 0) || (a < b ? -1 : 1))
  const rank = new Map(singles.map((s, i) => [s, i]))
  const order = (items: readonly string[]): string[] =>
    [...items].sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0))
  // The lattice: every candidate ever generated, by level, with its parents (frequent subsets one smaller).
  const nodes: { items: string[]; level: number }[] = []
  const nodeOf = new Map<string, number>()
  const edges: [number, number][] = []
  const levels: AprioriLevel[] = []
  const prunedAt = new Map<number, number>()
  type Snap = { status: Map<number, number>; counts: Map<number, number>; caption: string; level: number }
  const snaps: Snap[] = []
  const status = new Map<number, number>()
  const counts = new Map<number, number>()
  const snap = (caption: string, level: number): void => {
    snaps.push({ status: new Map(status), counts: new Map(counts), caption, level })
  }
  const CANDIDATE = 1
  const PRUNED = 2
  const FREQUENT = 3
  const INFREQUENT = 4
  const add = (items: string[], level: number): number => {
    const i = nodes.length
    nodes.push({ items, level })
    nodeOf.set(keyOf(items), i)
    return i
  }
  let current: string[][] = singles.map((s) => [s])
  for (const c of current) status.set(add(c, 1), CANDIDATE)
  snap(`level 1: ${current.length} single items`, 1)
  const frequent: Itemset[] = []
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  for (let k = 1; current.length > 0 && k <= maxLength; k++) {
    // Count the candidates of this level.
    const kept: string[][] = []
    for (const c of current) {
      const i = nodeOf.get(keyOf(c)) as number
      const n = k === 1 ? (single.get(c[0] as string) ?? 0) : countOf(c)
      counts.set(i, n)
      status.set(i, n >= minCount ? FREQUENT : INFREQUENT)
      if (n >= minCount) {
        kept.push(c)
        frequent.push({ items: [...c].sort(), count: n, support: n / transactions.length })
      }
    }
    levels.push({ k, candidates: current.length, pruned: prunedAt.get(k) ?? 0, frequent: kept.length })
    snap(`level ${k}: count ${current.length} candidates, ${kept.length} frequent`, k)
    if (k >= maxLength) break
    // Join: frequent k-itemsets sharing their first k - 1 items (in frequency order).
    const known = new Set(kept.map(keyOf))
    const next: string[][] = []
    let pruned = 0
    for (let a = 0; a < kept.length; a++)
      for (let b = a + 1; b < kept.length; b++) {
        const x = kept[a] as string[]
        const y = kept[b] as string[]
        if (x.slice(0, -1).some((v, i) => v !== y[i])) continue
        const items = order([...x, y[y.length - 1] as string])
        if (nodeOf.has(keyOf(items))) continue
        const subsets = items.map((_, drop) => items.filter((_, j) => j !== drop))
        const i = add(items, k + 1)
        for (const s of subsets) {
          const p = nodeOf.get(keyOf(order(s)))
          if (p !== undefined) edges.push([p, i])
        }
        if (subsets.every((s) => known.has(keyOf(order(s))))) {
          next.push(items)
          status.set(i, CANDIDATE)
        } else {
          pruned += 1
          status.set(i, PRUNED)
        }
      }
    prunedAt.set(k + 1, pruned)
    if (next.length + pruned > 0)
      snap(`level ${k + 1}: join ${next.length + pruned} candidates, prune ${pruned} with an infrequent subset`, k + 1)
    current = next
  }
  const codes = codesOf(singles)
  const width = Math.max(
    ...[...new Set(nodes.map((n) => n.level))].map((l) => nodes.filter((n) => n.level === l).length),
  )
  const xs = new Map<number, number>()
  for (const l of new Set(nodes.map((n) => n.level))) {
    const row = nodes.map((n, i) => [n, i] as const).filter(([n]) => n.level === l)
    for (const [j, [, i]] of row.entries()) xs.set(i, ((j + 0.5) * width) / row.length)
  }
  const rowIndex = new Map<number, number>()
  const crowded = new Set<number>()
  for (const l of new Set(nodes.map((n) => n.level))) {
    const row = nodes.flatMap((n, i) => (n.level === l ? [i] : []))
    for (const [j, i] of row.entries()) rowIndex.set(i, j)
    if (row.length > 12) crowded.add(l)
  }
  const dy = Math.max(1, (0.55 * width) / Math.max(1, Math.max(...nodes.map((n) => n.level)) - 1))
  const categories = ['unseen', 'candidate', 'pruned', 'frequent', 'infrequent']
  const record: BoardSteps = boardSteps({
    method: 'apriori',
    layout: {
      type: 'graph',
      nodes: nodes.map((n, i) => ({
        x: xs.get(i) ?? 0,
        // Crowded levels zigzag, so neighbours sit apart.
        y: (n.level - 1) * dy + (crowded.has(n.level) && (rowIndex.get(i) ?? 0) % 2 === 1 ? 0.3 * dy : 0),
        label: n.items.map((it) => codes.get(it) ?? it).join(''),
      })),
      edges,
    },
    categories,
    steps: snaps.map((s) => ({
      cells: nodes.map((_, i) => s.status.get(i) ?? 0),
      labels: nodes.map((n, i) => {
        const c = s.counts.get(i)
        const code = n.items.map((it) => codes.get(it) ?? it).join('')
        return c === undefined ? code : `${code}:${c}`
      }),
      caption: s.caption,
    })),
    metrics: [
      {
        name: 'candidates so far',
        values: snaps.map((s) => [...s.status.values()].filter((v) => v !== PRUNED).length),
      },
      {
        name: 'frequent so far',
        values: snaps.map((s) => [...s.status.values()].filter((v) => v === FREQUENT).length),
      },
    ],
    stepName: 'pass',
    summary: `${frequent.length} frequent itemsets at support ${minCount}/${transactions.length}; ${singles
      .map((s) => `${codes.get(s)} = ${s}`)
      .join(', ')}`,
  })
  return {
    itemsets: frequent,
    transactions: transactions.length,
    minSupport: minCount / transactions.length,
    levels,
    exposition: [record],
  }
}

interface FpNode {
  readonly item: string
  count: number
  readonly parent: number
  readonly children: Map<string, number>
}

// Every frequent itemset of a conditional pattern base (prefix paths with counts), each extended by the suffix.
function mineBase(
  base: readonly { readonly items: readonly string[]; readonly count: number }[],
  suffix: readonly string[],
  minCount: number,
  maxLength: number,
  out: { items: string[]; count: number }[],
): void {
  if (suffix.length >= maxLength) return
  const counts = new Map<string, number>()
  for (const p of base) for (const i of p.items) counts.set(i, (counts.get(i) ?? 0) + p.count)
  for (const [item, count] of counts) {
    if (count < minCount) continue
    const items = [item, ...suffix]
    out.push({ items, count })
    const conditional = base.flatMap((p) => {
      const at = p.items.indexOf(item)
      return at > 0 ? [{ items: p.items.slice(0, at), count: p.count }] : []
    })
    mineBase(conditional, items, minCount, maxLength, out)
  }
}

/**
 * FP-growth (Han, Pei and Yin 2000): order each transaction's frequent items by frequency and insert it into a prefix
 * tree (the FP-tree), then mine each item's conditional pattern base (the tree's prefix paths ending at it) recursively,
 * with no candidate generation. Recorded transaction by transaction as the tree grows (counts on its nodes), then item
 * by item as each one's prefix paths are mined.
 *
 * @param transactions - The transactions.
 * @param options - Minimum support and maximum length.
 * @returns The frequent itemsets, and a `boardSteps` record of the FP-tree.
 */
export function fpGrowth(transactions: Transactions, options: ItemsetOptions = {}): Explained<ItemsetResult> {
  const minCount = checkOptions(transactions, options, 'fpGrowth')
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  const single = new Map<string, number>()
  for (const t of transactions) for (const i of new Set(t)) single.set(i, (single.get(i) ?? 0) + 1)
  const items = [...single.entries()]
    .filter(([, c]) => c >= minCount)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([i]) => i)
  const rank = new Map(items.map((s, i) => [s, i]))
  const nodes: FpNode[] = [{ item: '', count: 0, parent: -1, children: new Map() }]
  const growth: { created: number; counts: number[]; path: number[]; caption: string }[] = []
  for (const [t, raw] of transactions.entries()) {
    const kept = [...new Set(raw)].filter((i) => rank.has(i)).sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0))
    let at = 0
    const path: number[] = []
    for (const i of kept) {
      const node = nodes[at] as FpNode
      let next = node.children.get(i)
      if (next === undefined) {
        next = nodes.length
        nodes.push({ item: i, count: 0, parent: at, children: new Map() })
        node.children.set(i, next)
      }
      ;(nodes[next] as FpNode).count += 1
      path.push(next)
      at = next
    }
    growth.push({
      created: nodes.length,
      counts: nodes.map((n) => n.count),
      path,
      caption: `insert transaction ${t + 1}: ${kept.join(', ') || '(no frequent items)'}`,
    })
  }
  // Mine each item, least frequent first, from its prefix paths.
  const found: { items: string[]; count: number }[] = []
  const mining: { item: string; nodes: number[]; prefixes: number[]; caption: string }[] = []
  for (const item of [...items].reverse()) {
    const at = nodes.flatMap((n, i) => (n.item === item ? [i] : []))
    const base: { items: string[]; count: number }[] = []
    const prefixes = new Set<number>()
    for (const i of at) {
      const prefix: string[] = []
      for (let p = (nodes[i] as FpNode).parent; p > 0; p = (nodes[p] as FpNode).parent) {
        prefix.unshift((nodes[p] as FpNode).item)
        prefixes.add(p)
      }
      if (prefix.length > 0) base.push({ items: prefix, count: (nodes[i] as FpNode).count })
    }
    const count = at.reduce((a, i) => a + (nodes[i] as FpNode).count, 0)
    const before = found.length
    found.push({ items: [item], count })
    mineBase(base, [item], minCount, maxLength, found)
    mining.push({
      item,
      nodes: at,
      prefixes: [...prefixes],
      caption: `mine ${item}: ${base.length} prefix paths, ${found.length - before} itemsets ending in it`,
    })
  }
  const codes = codesOf(items)
  // Tree layout: leaves spread evenly, parents over their children.
  const xs = new Array<number>(nodes.length).fill(0)
  const depth = new Array<number>(nodes.length).fill(0)
  let leaf = 0
  const place = (i: number, d: number): void => {
    depth[i] = d
    const kids = [...(nodes[i] as FpNode).children.values()]
    if (kids.length === 0) xs[i] = leaf++
    else {
      for (const k of kids) place(k, d + 1)
      xs[i] = ((xs[kids[0] as number] ?? 0) + (xs[kids[kids.length - 1] as number] ?? 0)) / 2
    }
  }
  place(0, 0)
  const dy = Math.max(1, (0.6 * Math.max(1, leaf)) / Math.max(1, ...depth))
  const code = (i: number): string => (i === 0 ? '∅' : (codes.get((nodes[i] as FpNode).item) ?? ''))
  const categories = ['unseen', 'visited', 'current', 'path']
  const steps = [
    ...growth.map((g) => {
      const on = new Set(g.path)
      return {
        cells: nodes.map((_, i) => (i === 0 ? 1 : on.has(i) ? 2 : i < g.created ? 1 : 0)),
        labels: nodes.map((_, i) => (i === 0 || i >= g.created ? code(i) : `${code(i)}:${g.counts[i]}`)),
        caption: g.caption,
      }
    }),
    ...mining.map((m) => {
      const on = new Set(m.nodes)
      const pre = new Set(m.prefixes)
      return {
        cells: nodes.map((_, i) => (on.has(i) ? 2 : pre.has(i) ? 3 : 1)),
        labels: nodes.map((n, i) => (i === 0 ? code(i) : `${code(i)}:${n.count}`)),
        caption: m.caption,
      }
    }),
  ]
  const record: BoardSteps = boardSteps({
    method: 'fp-growth',
    layout: {
      type: 'graph',
      nodes: nodes.map((_, i) => ({ x: xs[i] ?? 0, y: (depth[i] ?? 0) * dy, label: code(i) })),
      edges: nodes.flatMap((n, i) => (n.parent < 0 ? [] : [[n.parent, i] as [number, number]])),
    },
    categories,
    steps,
    metrics: [
      {
        name: 'tree nodes',
        values: steps.map((_, s) => (s < growth.length ? (growth[s]?.created ?? 1) - 1 : nodes.length - 1)),
      },
    ],
    stepName: 'step',
    summary: `${found.length} frequent itemsets; ${items.map((s) => `${codes.get(s)} = ${s}`).join(', ')}`,
  })
  const itemsets = found.map((f) => ({
    items: [...f.items].sort(),
    count: f.count,
    support: f.count / transactions.length,
  }))
  return {
    itemsets,
    transactions: transactions.length,
    minSupport: minCount / transactions.length,
    exposition: [record],
  }
}
