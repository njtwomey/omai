/**
 * The segmentation lattice of the Unigram model, as HF tokenizers' (and SentencePiece's) `Lattice`: one node per
 * vocabulary piece starting at each position, plus BOS and EOS; positions are code point indices.
 */

/** A lattice node. */
export interface LatticeNode {
  /** The vocabulary id (or the BOS / EOS ids). */
  readonly id: number
  readonly nodeId: number
  readonly pos: number
  readonly length: number
  readonly score: number
  prev: LatticeNode | undefined
  backtrace: number
}

/** A lattice over a sentence's code points. */
export interface Lattice {
  readonly chars: readonly string[]
  readonly nodes: LatticeNode[]
  readonly begin: LatticeNode[][]
  readonly end: LatticeNode[][]
}

function node(id: number, nodeId: number, pos: number, length: number, score: number): LatticeNode {
  return { id, nodeId, pos, length, score, prev: undefined, backtrace: 0 }
}

/** An empty lattice with BOS ending at 0 and EOS beginning at the end. */
export function lattice(sentence: string, bosId: number, eosId: number): Lattice {
  const chars = Array.from(sentence)
  const n = chars.length
  const begin: LatticeNode[][] = Array.from({ length: n + 1 }, () => [])
  const end: LatticeNode[][] = Array.from({ length: n + 1 }, () => [])
  const bos = node(bosId, 0, 0, 0, 0)
  const eos = node(eosId, 1, n, 0, 0)
  begin[n]?.push(eos)
  end[0]?.push(bos)
  return { chars, nodes: [bos, eos], begin, end }
}

/** Insert a node for a piece of `length` code points at `pos`. */
export function insertNode(l: Lattice, pos: number, length: number, score: number, id: number): void {
  const nd = node(id, l.nodes.length, pos, length, score)
  l.begin[pos]?.push(nd)
  l.end[pos + length]?.push(nd)
  l.nodes.push(nd)
}

/** The best path (HF's Lattice::viterbi): ties keep the first end node, in insertion order. */
export function viterbiPath(l: Lattice): LatticeNode[] {
  const n = l.chars.length
  for (let pos = 0; pos <= n; pos++) {
    const begins = l.begin[pos] ?? []
    if (begins.length === 0) return []
    for (const r of begins) {
      r.prev = undefined
      let best = 0
      let bestNode: LatticeNode | undefined
      for (const left of l.end[pos] ?? []) {
        const s = left.backtrace + r.score
        if (bestNode === undefined || s > best) {
          bestNode = left
          best = s
        }
      }
      if (bestNode === undefined) return []
      r.prev = bestNode
      r.backtrace = best
    }
  }
  const out: LatticeNode[] = []
  let cur = l.begin[n]?.[0]?.prev
  while (cur?.prev !== undefined) {
    out.push(cur)
    cur = cur.prev
  }
  return out.reverse()
}

/** One entry of a k-best list: the score of a path to a node and where it came from. */
interface Entry {
  readonly score: number
  readonly from: LatticeNode | undefined
  readonly rank: number
}

/**
 * The k best paths through the lattice in order of score (ties in enumeration order), by a k-best Viterbi recursion;
 * the first path is {@link viterbiPath}'s.
 */
export function kBestPaths(l: Lattice, k: number): LatticeNode[][] {
  if (k < 1) return []
  const n = l.chars.length
  const lists = new Map<number, Entry[]>()
  const bos = l.end[0]?.[0]
  if (bos === undefined) return []
  lists.set(bos.nodeId, [{ score: 0, from: undefined, rank: 0 }])
  for (let pos = 0; pos <= n; pos++) {
    for (const r of l.begin[pos] ?? []) {
      const cands: Entry[] = []
      for (const left of l.end[pos] ?? [])
        for (const [rank, e] of (lists.get(left.nodeId) ?? []).entries())
          cands.push({ score: e.score + r.score, from: left, rank })
      // A stable sort keeps the first of equal scores, as the strict comparison of the Viterbi recursion does.
      cands.sort((a, b) => b.score - a.score)
      lists.set(r.nodeId, cands.slice(0, k))
    }
  }
  const eos = l.begin[n]?.[0]
  if (eos === undefined) return []
  const out: LatticeNode[][] = []
  for (const e of lists.get(eos.nodeId) ?? []) {
    const path: LatticeNode[] = []
    let cur = e
    while (cur.from !== undefined && cur.from !== bos) {
      const at: LatticeNode = cur.from
      path.push(at)
      const next = lists.get(at.nodeId)?.[cur.rank]
      if (next === undefined) break
      cur = next
    }
    out.push(path.reverse())
  }
  return out
}

/** HF's log_sum_exp: the first term is taken as is (init mode), and terms 50 below the maximum are dropped. */
export function logSumExpHf(x: number, y: number, init: boolean): number {
  if (init) return y
  // The comparisons of HF's code, not Math.min / Math.max (they differ on NaN).
  // biome-ignore lint/complexity/useMathMinMax: keeps HF's NaN behaviour
  const lo = x > y ? y : x
  // biome-ignore lint/complexity/useMathMinMax: keeps HF's NaN behaviour
  const hi = x > y ? x : y
  if (hi > lo + 50) return hi
  return hi + Math.log(Math.exp(lo - hi) + 1)
}

/**
 * Add freq times the marginal of every piece node to `expected` (by vocabulary id) and return freq log Z (HF's
 * Lattice::populate_marginal, including its test of the first end node by vocabulary id).
 */
export function populateMarginal(l: Lattice, freq: number, expected: Float64Array): number {
  const n = l.chars.length
  const alpha = new Float64Array(l.nodes.length)
  const beta = new Float64Array(l.nodes.length)
  for (let pos = 0; pos <= n; pos++) {
    const ends = l.end[pos] ?? []
    const first = ends[0]
    for (const r of l.begin[pos] ?? [])
      for (const left of ends)
        alpha[r.nodeId] = logSumExpHf(
          alpha[r.nodeId] ?? 0,
          left.score + (alpha[left.nodeId] ?? 0),
          left.id === first?.id,
        )
  }
  for (let pos = n; pos >= 0; pos--) {
    const begins = l.begin[pos] ?? []
    const first = begins[0]
    for (const left of l.end[pos] ?? [])
      for (const r of begins)
        beta[left.nodeId] = logSumExpHf(beta[left.nodeId] ?? 0, r.score + (beta[r.nodeId] ?? 0), r.id === first?.id)
  }
  const eos = l.begin[n]?.[0]
  const z = eos === undefined ? 0 : (alpha[eos.nodeId] ?? 0)
  for (let pos = 0; pos < n; pos++) {
    for (const nd of l.begin[pos] ?? []) {
      const total = (alpha[nd.nodeId] ?? 0) + nd.score + (beta[nd.nodeId] ?? 0) - z
      expected[nd.id] = (expected[nd.id] ?? 0) + freq * Math.exp(total)
    }
  }
  return freq * z
}
