/**
 * Prefix codes for lossless source coding, where frequent symbols get short codewords: Kraft sums, prefix-freeness and
 * the Sardinas-Patterson test of unique decodability, expected length against the entropy bound, Huffman's optimal
 * binary and D-ary codes (also as a traced algorithm), canonical codes as DEFLATE uses, the Shannon and Shannon-Fano
 * codes, and encoding and decoding with any prefix code.
 */

import { heap, type Json, MdmlError, pairwiseSum } from '../base/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  contractSchema,
  defineContract,
  event,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { float } from './_shared.ts'

/** A code assigning each symbol 0..m-1 a codeword of digits in 0..arity-1 (an empty codeword: the symbol is unused). */
export interface PrefixCode {
  readonly arity: number
  readonly codewords: readonly (readonly number[])[]
}

function probabilities(p: TensorLike, name: string): number[] {
  const xs = toFlat(float(p))
  if (xs.length === 0 || xs.some((v) => !(v >= 0)))
    throw new MdmlError('value', `${name}: probabilities must be non-negative and not empty`)
  const total = pairwiseSum(xs)
  if (!(total > 0)) throw new MdmlError('value', `${name}: probabilities sum to zero`)
  return xs.map((v) => v / total)
}

function checkArity(arity: number, name: string): void {
  if (!Number.isInteger(arity) || arity < 2) throw new MdmlError('value', `${name}: arity must be an integer >= 2`)
}

/**
 * The Kraft sum sum D^(-l_i) of codeword lengths. A prefix code (and, by McMillan's theorem, any uniquely decodable
 * code) has Kraft sum at most 1; conversely any lengths with sum at most 1 are those of some prefix code
 * ({@link canonicalCode} builds one). A complete code (a full code tree, such as Huffman's) has sum exactly 1.
 *
 * @param lengths - Codeword lengths (zero lengths are unused symbols and are skipped).
 * @param arity - The code alphabet size D (default 2).
 * @returns The Kraft sum.
 * @concept kraft-inequality
 * @example
 * ```ts
 * import { kraftSum } from 'mdml-engine/information'
 * kraftSum([1, 2, 3, 3]) // => 1
 * kraftSum([1, 1, 2]) // => 1.25
 * ```
 */
export function kraftSum(lengths: readonly number[], arity = 2): number {
  checkArity(arity, 'kraftSum')
  return pairwiseSum(lengths.filter((l) => l > 0).map((l) => arity ** -l))
}

const wordKey = (w: readonly number[]): string => w.join(',')

/**
 * Whether no codeword is a prefix of another (an instantaneous code). Empty codewords (unused symbols) are ignored.
 *
 * @param codewords - The codewords, as digit arrays.
 * @returns True for a prefix-free code.
 * @concept prefix-codes
 * @example
 * ```ts
 * import { isPrefixFree } from 'mdml-engine/information'
 * isPrefixFree([[0], [1, 0], [1, 1]]) // => true
 * isPrefixFree([[0], [0, 1]]) // => false
 * ```
 */
export function isPrefixFree(codewords: readonly (readonly number[])[]): boolean {
  const words = codewords.filter((w) => w.length > 0)
  const all = new Set(words.map(wordKey))
  if (all.size !== words.length) return false
  for (const w of words) for (let l = 1; l < w.length; l++) if (all.has(wordKey(w.slice(0, l)))) return false
  return true
}

/**
 * Whether a code is uniquely decodable, by the Sardinas-Patterson test: the sets of dangling suffixes are generated
 * until one contains a codeword (not decodable) or no new set appears (decodable). Prefix-free codes pass at once;
 * codes such as {0, 01} are decodable but not instantaneous.
 *
 * @param codewords - The codewords, as digit arrays (empty ones are ignored).
 * @returns True when every finite concatenation has a single parse.
 * @concept prefix-codes
 * @example
 * ```ts
 * import { isUniquelyDecodable } from 'mdml-engine/information'
 * isUniquelyDecodable([[0], [0, 1]]) // => true
 * isUniquelyDecodable([[0], [0, 1], [1, 0]]) // => false
 * ```
 */
export function isUniquelyDecodable(codewords: readonly (readonly number[])[]): boolean {
  const words = codewords.filter((w) => w.length > 0)
  const code = new Set(words.map(wordKey))
  if (code.size !== words.length) return false
  const suffixes = (a: readonly number[], b: readonly number[]): number[] | undefined =>
    a.length < b.length && a.every((d, i) => b[i] === d) ? b.slice(a.length) : undefined
  let current = new Map<string, number[]>()
  for (const a of words)
    for (const b of words) {
      const s = suffixes(a, b)
      if (s !== undefined) current.set(wordKey(s), s)
    }
  const seen = new Set<string>()
  while (current.size > 0) {
    for (const k of current.keys()) if (code.has(k)) return false
    const signature = [...current.keys()].sort().join('|')
    if (seen.has(signature)) return true
    seen.add(signature)
    const next = new Map<string, number[]>()
    for (const s of current.values())
      for (const w of words) {
        const a = suffixes(s, w)
        if (a !== undefined) next.set(wordKey(a), a)
        const b = suffixes(w, s)
        if (b !== undefined) next.set(wordKey(b), b)
      }
    current = next
  }
  return true
}

/**
 * The expected codeword length sum p_i l_i, to compare with the entropy bound H_D(p) <= L < H_D(p) + 1.
 *
 * @param code - A code, or its codeword lengths.
 * @param probs - Symbol probabilities (normalised here).
 * @returns The expected length in code digits per symbol.
 * @concept source-coding
 * @example
 * ```ts
 * import { expectedLength } from 'mdml-engine/information'
 * expectedLength([1, 2, 3, 3], [0.5, 0.25, 0.125, 0.125]) // => 1.75
 * ```
 */
export function expectedLength(code: PrefixCode | readonly number[], probs: TensorLike): number {
  const lengths = Array.isArray(code)
    ? (code as readonly number[])
    : (code as PrefixCode).codewords.map((w) => w.length)
  const p = probabilities(probs, 'expectedLength')
  if (p.length !== lengths.length) throw new MdmlError('shape', 'expectedLength: lengths and probabilities differ')
  return pairwiseSum(p.map((v, i) => v * (lengths[i] ?? 0)))
}

/** A node of a Huffman tree: a leaf (with its symbol, or a padding leaf for D-ary codes) or a merge of children. */
export interface HuffmanNode {
  readonly id: number
  readonly weight: number
  /** Children in digit order (empty for leaves). */
  readonly children: readonly number[]
  /** The symbol of a leaf; undefined for internal nodes and padding leaves. */
  readonly symbol?: number
}

/** The state of {@link huffmanAlgorithm}: the nodes so far and the queue of roots still to merge. */
export interface HuffmanState {
  readonly nodes: readonly HuffmanNode[]
  /** Node ids of the current forest's roots, in heap order. */
  readonly queue: readonly number[]
  /** The last merge (for the trace). */
  readonly last?: { readonly children: readonly number[]; readonly weight: number; readonly node: number }
}

/**
 * Huffman's construction as a traced algorithm: start from one leaf per symbol (plus zero-weight padding leaves so
 * that a D-ary tree is full) and, each step, merge the D lightest roots (ties broken by creation order) into a new root.
 * Each step records the merge as a message and the queue size as a metric.
 *
 * @param probs - Symbol probabilities or counts.
 * @param options - The code alphabet size (default 2).
 * @returns The algorithm; it converges when one root is left.
 * @concept huffman-coding
 * @example
 * ```ts
 * import { huffmanAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(huffmanAlgorithm([0.5, 0.25, 0.25]), { maxSteps: 10 }).steps // => 2
 * ```
 */
export function huffmanAlgorithm(
  probs: TensorLike,
  options: { readonly arity?: number } = {},
): Algorithm<HuffmanState> {
  const p = probabilities(probs, 'huffmanAlgorithm')
  const arity = options.arity ?? 2
  checkArity(arity, 'huffmanAlgorithm')
  const m = p.length
  const padding = m === 1 ? arity - 1 : (arity - 1 - ((m - 1) % (arity - 1))) % (arity - 1)
  const order = (nodes: readonly HuffmanNode[]) => (a: number, b: number) =>
    (nodes[a]?.weight ?? 0) - (nodes[b]?.weight ?? 0) || a - b
  return {
    name: 'huffman',
    config: { symbols: m, arity } as Json,
    schema: channels({
      metrics: {
        roots: { description: 'trees left in the forest' },
        weight: { description: 'weight of the node just made' },
      },
      messages: { merge: { description: 'a merge: the children (in digit order), their total weight, the new node' } },
    }),
    init: () => {
      const nodes: HuffmanNode[] = p.map((w, i) => ({ id: i, weight: w, children: [], symbol: i }))
      for (let j = 0; j < padding; j++) nodes.push({ id: m + j, weight: 0, children: [] })
      return {
        nodes,
        queue: heap(
          order(nodes),
          nodes.map((n) => n.id),
        ).toArray(),
      }
    },
    step: (s) => {
      const q = heap(order(s.nodes), s.queue)
      const children: number[] = []
      for (let j = 0; j < arity && q.size > 0; j++) children.push(q.pop() as number)
      const id = s.nodes.length
      const weight = pairwiseSum(children.map((c) => s.nodes[c]?.weight ?? 0))
      const nodes = [...s.nodes, { id, weight, children }]
      const next = heap(order(nodes), q.toArray())
      next.push(id)
      return { nodes, queue: next.toArray(), last: { children, weight, node: id } }
    },
    done: (s) => (s.queue.length <= 1 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { roots: s.queue.length, ...(s.last === undefined ? {} : { weight: s.last.weight }) },
      ...(s.last === undefined
        ? {}
        : {
            messages: [
              { kind: 'merge', payload: { children: [...s.last.children], weight: s.last.weight, node: s.last.node } },
            ],
          }),
    }),
  }
}

/** A Huffman tree: its nodes (leaves first, in symbol order, then padding leaves, then merges) and its root. */
export interface HuffmanTree {
  readonly nodes: readonly HuffmanNode[]
  readonly root: number
}

/**
 * Huffman's tree built directly, in O(n log n), from weights used as given (counts or probabilities, not normalised):
 * the same merges as {@link huffmanAlgorithm} (the D lightest roots, ties by creation order, children in digit order,
 * zero-weight padding leaves for a full D-ary tree), without the per-step trace. For large alphabets, such as the
 * vocabularies of a hierarchical softmax.
 *
 * @param weights - Non-negative symbol weights, at least one.
 * @param options - The code alphabet size (default 2).
 * @returns The tree: leaves `0..n-1` are the symbols, inner nodes follow in merge order, the root is last.
 * @throws MdmlError with code `value` for an empty or negative weight list or a bad arity.
 * @concept huffman-coding
 * @example
 * ```ts
 * import { huffmanTree } from 'mdml-engine/information'
 * const t = huffmanTree([5, 1, 1])
 * t.nodes.map((n) => n.children) // => [[], [], [], [1, 2], [3, 0]]
 * t.root // => 4
 * ```
 */
export function huffmanTree(weights: TensorLike, options: { readonly arity?: number } = {}): HuffmanTree {
  const w = toFlat(float(weights))
  if (w.length === 0 || w.some((v) => !(v >= 0)))
    throw new MdmlError('value', 'huffmanTree: weights must be non-negative and not empty')
  const arity = options.arity ?? 2
  checkArity(arity, 'huffmanTree')
  const m = w.length
  const padding = m === 1 ? arity - 1 : (arity - 1 - ((m - 1) % (arity - 1))) % (arity - 1)
  const nodes: HuffmanNode[] = w.map((v, i) => ({ id: i, weight: v, children: [], symbol: i }))
  for (let j = 0; j < padding; j++) nodes.push({ id: m + j, weight: 0, children: [] })
  const q = heap<number>(
    (a, b) => (nodes[a]?.weight ?? 0) - (nodes[b]?.weight ?? 0) || a - b,
    nodes.map((n) => n.id),
  )
  while (q.size > 1) {
    const children: number[] = []
    for (let j = 0; j < arity; j++) {
      const c = q.pop()
      if (c !== undefined) children.push(c)
    }
    const id = nodes.length
    nodes.push({ id, weight: pairwiseSum(children.map((c) => nodes[c]?.weight ?? 0)), children })
    q.push(id)
  }
  return { nodes, root: q.pop() ?? 0 }
}

/** A Huffman code with its tree. */
export interface HuffmanCode extends PrefixCode {
  readonly lengths: readonly number[]
  readonly nodes: readonly HuffmanNode[]
  readonly root: number
}

/**
 * An optimal (minimum expected length) prefix code by Huffman's algorithm, binary or D-ary, run as the traced
 * {@link huffmanAlgorithm}. Codeword digits follow the child order of each merge (the lightest child gets 0). A
 * single symbol gets a one-digit codeword.
 *
 * @param probs - Symbol probabilities or counts.
 * @param options - Arity, and a tracer and policy to record the construction.
 * @returns The codewords, their lengths and the tree.
 * @concept huffman-coding
 * @example
 * ```ts
 * import { huffman } from 'mdml-engine/information'
 * huffman([0.5, 0.25, 0.125, 0.125]).lengths // => [1, 2, 3, 3]
 * huffman([0.4, 0.3, 0.2, 0.1], { arity: 3 }).lengths // => [1, 1, 2, 2]
 * ```
 */
export function huffman(
  probs: TensorLike,
  options: { readonly arity?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): HuffmanCode {
  const { arity = 2, ...run } = options
  const algorithm = huffmanAlgorithm(probs, { arity })
  const m = toFlat(float(probs)).length
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: m + arity })
  const root = state.queue[0] ?? 0
  const codewords: number[][] = Array.from({ length: m }, () => [])
  const walk = (id: number, prefix: number[]): void => {
    const node = state.nodes[id]
    if (node === undefined) return
    if (node.children.length === 0) {
      if (node.symbol !== undefined) codewords[node.symbol] = prefix.length === 0 ? [0] : prefix
      return
    }
    node.children.forEach((c, digit) => {
      walk(c, [...prefix, digit])
    })
  }
  walk(root, [])
  // A lone symbol is merged with padding leaves; give it the first digit.
  if (m === 1) codewords[0] = [0]
  return { arity, codewords, lengths: codewords.map((w) => w.length), nodes: state.nodes, root }
}

/**
 * The canonical prefix code with given codeword lengths (as DEFLATE uses for its Huffman tables): symbols sorted by
 * length and then by index receive consecutive codewords, so the code is determined by its lengths alone.
 *
 * @param lengths - Codeword length of each symbol (0 marks an unused symbol).
 * @param options - The code alphabet size (default 2).
 * @returns The code.
 * @throws MdmlError with code `value` when the lengths violate the Kraft inequality.
 * @concept prefix-codes
 * @example
 * ```ts
 * import { canonicalCode } from 'mdml-engine/information'
 * canonicalCode([2, 1, 3, 3]).codewords // => [[1, 0], [0], [1, 1, 0], [1, 1, 1]]
 * ```
 */
export function canonicalCode(lengths: readonly number[], options: { readonly arity?: number } = {}): PrefixCode {
  const arity = options.arity ?? 2
  if (kraftSum(lengths, arity) > 1 + 1e-12)
    throw new MdmlError('value', 'canonicalCode: the lengths violate the Kraft inequality')
  const order = lengths
    .map((l, i) => [l, i] as const)
    .filter(([l]) => l > 0)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const codewords: number[][] = lengths.map(() => [])
  let code = 0
  let previous = 0
  order.forEach(([l, symbol], k) => {
    if (k > 0) code = (code + 1) * arity ** (l - previous)
    else code = 0
    previous = l
    const digits: number[] = []
    let v = code
    for (let j = 0; j < l; j++) {
      digits.unshift(v % arity)
      v = Math.floor(v / arity)
    }
    codewords[symbol] = digits
  })
  return { arity, codewords }
}

/**
 * Fano's binary code (the Shannon-Fano code of the textbooks): sort the symbols by decreasing probability, split the
 * list where the two halves' probabilities are closest, give one half 0 and the other 1, and recurse. Within one digit
 * of the entropy, but not always optimal.
 *
 * @param probs - Symbol probabilities or counts.
 * @returns The binary code.
 * @concept shannon-fano
 * @example
 * ```ts
 * import { shannonFano } from 'mdml-engine/information'
 * shannonFano([0.35, 0.17, 0.17, 0.16, 0.15]).codewords.map((w) => w.length) // => [2, 2, 2, 3, 3]
 * ```
 */
export function shannonFano(probs: TensorLike): PrefixCode {
  const p = probabilities(probs, 'shannonFano')
  const codewords: number[][] = p.map(() => [])
  const sorted = p.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1])
  const split = (items: readonly (readonly [number, number])[], prefix: number[]): void => {
    if (items.length === 1) {
      const [, symbol] = items[0] as readonly [number, number]
      codewords[symbol] = prefix.length === 0 ? [0] : prefix
      return
    }
    const total = pairwiseSum(items.map((it) => it[0]))
    let left = 0
    let best = 1
    let bestGap = Number.POSITIVE_INFINITY
    for (let k = 1; k < items.length; k++) {
      left += items[k - 1]?.[0] ?? 0
      const gap = Math.abs(total - 2 * left)
      if (gap < bestGap - 1e-15) {
        bestGap = gap
        best = k
      }
    }
    split(items.slice(0, best), [...prefix, 0])
    split(items.slice(best), [...prefix, 1])
  }
  split(sorted, [])
  return { arity: 2, codewords }
}

/**
 * Shannon's code: lengths ceil(-log_D p_i), and codewords the first l_i digits of the D-ary expansion of the cumulative
 * probability of the more probable symbols. Prefix-free with expected length below H_D(p) + 1. Symbols of probability
 * zero get no codeword.
 *
 * @param probs - Symbol probabilities or counts.
 * @param options - The code alphabet size (default 2).
 * @returns The code.
 * @concept shannon-fano
 * @example
 * ```ts
 * import { shannonCode } from 'mdml-engine/information'
 * shannonCode([0.5, 0.25, 0.25]).codewords // => [[0], [1, 0], [1, 1]]
 * ```
 */
export function shannonCode(probs: TensorLike, options: { readonly arity?: number } = {}): PrefixCode {
  const p = probabilities(probs, 'shannonCode')
  const arity = options.arity ?? 2
  checkArity(arity, 'shannonCode')
  const codewords: number[][] = p.map(() => [])
  const sorted = p.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1])
  let cumulative = 0
  for (const [v, symbol] of sorted) {
    if (v === 0) continue
    const l = Math.max(1, Math.ceil(-Math.log(v) / Math.log(arity) - 1e-12))
    const digits: number[] = []
    let f = cumulative
    for (let j = 0; j < l; j++) {
      f *= arity
      const d = Math.min(arity - 1, Math.floor(f + 1e-12))
      digits.push(d)
      f -= d
    }
    codewords[symbol] = digits
    cumulative += v
  }
  return { arity, codewords }
}

/**
 * Encode symbols with a prefix code: the concatenation of their codewords.
 *
 * @param code - A prefix code.
 * @param symbols - Symbols 0..m-1.
 * @returns The digits.
 * @concept prefix-codes
 * @example
 * ```ts
 * import { canonicalCode, prefixEncode } from 'mdml-engine/information'
 * prefixEncode(canonicalCode([1, 2, 2]), [0, 2, 1]) // => [0, 1, 1, 1, 0]
 * ```
 */
export function prefixEncode(code: PrefixCode, symbols: readonly number[]): number[] {
  const out: number[] = []
  for (const s of symbols) {
    const w = code.codewords[s]
    if (w === undefined || w.length === 0) throw new MdmlError('value', `prefixEncode: symbol ${s} has no codeword`)
    out.push(...w)
  }
  return out
}

/**
 * Decode digits with a prefix code by walking its code tree.
 *
 * @param code - A prefix-free code.
 * @param digits - The encoded digits.
 * @returns The symbols.
 * @throws MdmlError with code `value` for a code that is not prefix-free, an invalid digit sequence or a truncated
 *   final codeword.
 * @concept prefix-codes
 * @example
 * ```ts
 * import { canonicalCode, prefixDecode } from 'mdml-engine/information'
 * prefixDecode(canonicalCode([1, 2, 2]), [0, 1, 1, 1, 0]) // => [0, 2, 1]
 * ```
 */
export function prefixDecode(code: PrefixCode, digits: readonly number[]): number[] {
  if (!isPrefixFree(code.codewords)) throw new MdmlError('value', 'prefixDecode: the code is not prefix-free')
  interface Trie {
    symbol?: number
    next: Map<number, Trie>
  }
  const root: Trie = { next: new Map() }
  code.codewords.forEach((w, symbol) => {
    if (w.length === 0) return
    let t = root
    for (const d of w) {
      let n = t.next.get(d)
      if (n === undefined) {
        n = { next: new Map() }
        t.next.set(d, n)
      }
      t = n
    }
    t.symbol = symbol
  })
  const out: number[] = []
  let t = root
  for (const d of digits) {
    const n = t.next.get(d)
    if (n === undefined) throw new MdmlError('value', 'prefixDecode: the digits are not a sequence of codewords')
    if (n.symbol !== undefined) {
      out.push(n.symbol)
      t = root
    } else t = n
  }
  if (t !== root) throw new MdmlError('value', 'prefixDecode: the last codeword is truncated')
  return out
}

/** State of {@link prefixEncoderAlgorithm}. */
export interface PrefixEncoderState {
  /** Symbols coded. */
  readonly index: number
  /** The digits written. */
  readonly output: readonly number[]
  /** The information content of the symbols coded under the stated probabilities, in bits (0 without them). */
  readonly ideal: number
}

/** The paths of {@link prefixEncoderAlgorithm}. */
export type PrefixEncoderPath =
  | 'symbol/index'
  | 'code/length'
  | 'code/bits'
  | 'code/ideal'
  | 'code/output'
  | 'code/done'

/** The log contract of {@link prefixEncoderAlgorithm}. */
export const prefixEncoderContract: Contract<PrefixEncoderState, PrefixEncoderPath> = defineContract({
  'symbol/index': scalar({ description: 'symbols coded so far' }),
  'code/length': scalar({ description: 'the length of the codeword just written' }),
  'code/bits': scalar({ description: 'digits written so far' }),
  'code/ideal': scalar({
    description: 'the information content so far, the sum of -log2 p of the symbols (0 without probabilities)',
  }),
  'code/output': tensor({ role: 'output', shape: ['digits'], description: 'the code so far' }),
  'code/done': event({ context: { digits: 'the length of the finished code' } }),
})

/**
 * Encoding with a prefix code as a traced algorithm, one symbol per step: its codeword is appended, and the digits
 * written are tracked against the information content of the symbols under the source probabilities (a prefix code
 * spends a whole number of digits per symbol, so it falls behind the entropy by up to one digit per symbol, the gap
 * arithmetic coding closes). The output is exactly {@link prefixEncode}'s. The state serialises.
 *
 * @param code - The prefix code.
 * @param symbols - The symbols to encode.
 * @param probs - The source probabilities, for the information content (optional).
 * @returns The algorithm; it completes after one step per symbol.
 * @throws MdmlError with code `value` for a symbol without a codeword.
 * @concept huffman-coding
 * @example
 * ```ts
 * import { canonicalCode, prefixEncoderAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(prefixEncoderAlgorithm(canonicalCode([1, 2, 2]), [0, 2, 1]), { seed: 0, maxSteps: 10 }).state.output // => [0, 1, 1, 1, 0]
 * ```
 */
export function prefixEncoderAlgorithm(
  code: PrefixCode,
  symbols: readonly number[],
  probs?: TensorLike,
): Algorithm<PrefixEncoderState, PrefixEncoderPath> {
  const p = probs === undefined ? undefined : probabilities(probs, 'prefixEncoderAlgorithm')
  for (const s of symbols) {
    if ((code.codewords[s]?.length ?? 0) === 0)
      throw new MdmlError('value', `prefixEncoderAlgorithm: symbol ${s} has no codeword`)
  }
  return {
    name: 'information/prefix-encoder',
    config: { symbols: symbols.length, arity: code.arity },
    schema: contractSchema(prefixEncoderContract),
    init: () => ({ index: 0, output: [], ideal: 0 }),
    step: (s, ctx) => {
      const symbol = symbols[s.index] as number
      const word = code.codewords[symbol] as readonly number[]
      const next = {
        index: s.index + 1,
        output: [...s.output, ...word],
        ideal: s.ideal + (p === undefined ? 0 : -Math.log2(p[symbol] ?? 1)),
      }
      if (next.index >= symbols.length) ctx.emit('code/done', { digits: next.output.length })
      return next
    },
    done: (s) => (s.index >= symbols.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        'symbol/index': s.index,
        'code/length': s.index === 0 ? 0 : (code.codewords[symbols[s.index - 1] as number]?.length ?? 0),
        'code/bits': s.output.length,
        'code/ideal': s.ideal,
      },
      tensors: { 'code/output': { role: 'output', value: s.output, shape: [s.output.length] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PrefixEncoderState,
  }
}
