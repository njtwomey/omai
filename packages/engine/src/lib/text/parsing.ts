/**
 * Context-free grammars and chart parsing: grammars in nltk's notation (optionally probabilistic), parse trees and
 * their bracketed form, CYK parsing for grammars in Chomsky normal form, Earley parsing for any context-free grammar,
 * and conversion to Chomsky normal form.
 */

import { OmaiError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** A grammar symbol on a right-hand side: a nonterminal or a terminal word. */
export type GrammarSymbol = { readonly nonterminal: string } | { readonly terminal: string }

/** A production lhs -> rhs, with a probability for probabilistic grammars (1 otherwise). */
export interface Production {
  readonly lhs: string
  readonly rhs: readonly GrammarSymbol[]
  readonly probability: number
}

/** A context-free grammar: its start symbol and productions. */
export interface Grammar {
  readonly start: string
  readonly productions: readonly Production[]
}

/** A parse tree: a label and children (subtrees or words). */
export interface ParseTree {
  readonly label: string
  readonly children: readonly (ParseTree | string)[]
}

/**
 * Parse a grammar in nltk's notation (CFG.fromstring and PCFG.fromstring): lines `A -> B C | 'word'`, terminals in
 * single or double quotes, optional probabilities `[0.3]` after each alternative, `#` comments; the first left-hand
 * side is the start symbol. An empty alternative is an empty production.
 *
 * @param source - The grammar.
 * @returns The grammar.
 * @throws OmaiError with code `value` (with the line) for malformed rules or probabilities not summing to one.
 * @concept context-free-grammars
 * @example
 * ```ts
 * import { grammar } from 'omai-engine/text'
 * const g = grammar("S -> NP VP\nNP -> 'I' | Det N\nVP -> V NP\nDet -> 'a'\nN -> 'cat'\nV -> 'saw'")
 * [g.start, g.productions.length] // => ['S', 7]
 * ```
 */
export function grammar(source: string): Grammar {
  const productions: Production[] = []
  let start: string | undefined
  const probabilistic = /\[\s*[-+0-9.eE]+\s*\]/.test(source)
  for (const [k, raw] of source.split('\n').entries()) {
    const line = raw.replace(/#.*$/, '').trim()
    if (line.length === 0) continue
    const m = /^([^\s'"|[\]]+)\s*->(.*)$/.exec(line)
    if (m === null) throw new OmaiError('value', `grammar line ${k + 1}: expected A -> ...`, { line: k + 1 })
    const lhs = m[1] ?? ''
    start ??= lhs
    for (const alt of splitAlternatives(m[2] ?? '', k + 1)) {
      const { symbols, probability } = parseAlternative(alt, k + 1)
      if (probabilistic && probability === undefined)
        throw new OmaiError(
          'value',
          `grammar line ${k + 1}: a probabilistic grammar needs a probability on every rule`,
          { line: k + 1 },
        )
      productions.push({ lhs, rhs: symbols, probability: probability ?? 1 })
    }
  }
  if (start === undefined) throw new OmaiError('value', 'grammar: no productions')
  if (probabilistic) {
    const sums = new Map<string, number>()
    for (const p of productions) sums.set(p.lhs, (sums.get(p.lhs) ?? 0) + p.probability)
    for (const [lhs, s] of sums)
      if (Math.abs(s - 1) > 1e-6)
        throw new OmaiError('value', `grammar: the probabilities of ${lhs} sum to ${s}, not 1`)
  }
  return { start, productions }
}

function splitAlternatives(s: string, line: number): string[] {
  const out: string[] = []
  let cur = ''
  let quote: string | undefined
  for (const ch of s) {
    if (quote !== undefined) {
      cur += ch
      if (ch === quote) quote = undefined
    } else if (ch === "'" || ch === '"') {
      quote = ch
      cur += ch
    } else if (ch === '|') {
      out.push(cur)
      cur = ''
    } else cur += ch
  }
  if (quote !== undefined) throw new OmaiError('value', `grammar line ${line}: unterminated quote`, { line })
  out.push(cur)
  return out
}

function parseAlternative(s: string, line: number): { symbols: GrammarSymbol[]; probability?: number } {
  const symbols: GrammarSymbol[] = []
  let probability: number | undefined
  const re = /\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|\[\s*([-+0-9.eE]+)\s*\]|([^\s'"[\]]+))/gy
  let at = 0
  re.lastIndex = 0
  for (;;) {
    const m = re.exec(s)
    if (m === null || m[0].length === 0) break
    at = re.lastIndex
    if (m[1] !== undefined || m[2] !== undefined)
      symbols.push({ terminal: (m[1] ?? m[2] ?? '').replace(/\\(.)/g, '$1') })
    else if (m[3] !== undefined) probability = Number(m[3])
    else symbols.push({ nonterminal: m[4] ?? '' })
  }
  if (s.slice(at).trim().length > 0)
    throw new OmaiError('value', `grammar line ${line}: cannot read ${s.slice(at).trim()}`, { line })
  return probability === undefined ? { symbols } : { symbols, probability }
}

/**
 * A parse tree in nltk's bracketed form, `(S (NP I) (VP (V saw) (NP (Det a) (N cat))))`.
 *
 * @param tree - The tree.
 * @returns The bracketed string.
 * @concept context-free-grammars
 * @example
 * ```ts
 * import { treeToString } from 'omai-engine/text'
 * treeToString({ label: 'S', children: [{ label: 'NP', children: ['I'] }, 'ran'] }) // => '(S (NP I) ran)'
 * ```
 */
export function treeToString(tree: ParseTree): string {
  return `(${tree.label}${tree.children.map((c) => ` ${typeof c === 'string' ? c : treeToString(c)}`).join('')})`
}

/** Whether a grammar is in Chomsky normal form: A -> B C or A -> 'a'. */
function isCnf(g: Grammar): boolean {
  return g.productions.every(
    (p) =>
      (p.rhs.length === 2 && p.rhs.every((s) => 'nonterminal' in s)) ||
      (p.rhs.length === 1 && 'terminal' in (p.rhs[0] ?? { nonterminal: '' })),
  )
}

/** The paths the chart parsers log, as a type. */
export type ParsePath =
  | 'chart/span'
  | 'chart/cell'
  | 'chart/derivations'
  | 'earley/predict'
  | 'earley/scan'
  | 'earley/complete'
  | 'earley/column'

/**
 * What the chart parsers log. CYK logs, per span length, the span, the cells of that length that hold a nonterminal and
 * the derivations found in them; Earley logs, per item processed, the column it is in, and the items its predict, scan or
 * complete step added.
 */
export const parseContract: Contract<unknown, ParsePath> = defineContract({
  'chart/span': scalar({ description: 'the span length filled at this step (CYK)' }),
  'chart/cell': scalar({ description: 'the cells of this span length that hold at least one nonterminal (CYK)' }),
  'chart/derivations': scalar({ description: 'the derivations summed over the cells of this span length (CYK)' }),
  'earley/predict': scalar({ description: 'items added by predicting the nonterminal after the dot' }),
  'earley/scan': scalar({ description: 'items added by scanning the next word' }),
  'earley/complete': scalar({ description: 'items added by completing a finished item into its parents' }),
  'earley/column': scalar({ description: 'the chart column of the item processed' }),
})

/** The options of the parsers: the most trees to enumerate. */
export interface ParseOptions {
  /** The most trees to enumerate (default 1000). */
  readonly maxTrees?: number
}

/**
 * The state of {@link cykAlgorithm}: the derivation counts of every span, as a nonterminal-to-count record per cell
 * `cells[i][j]` (tokens `i` to `j`), and the next span length to fill. A step fills every cell of one length in place;
 * keep a copy of a state you want to look at later.
 */
export interface CykState {
  readonly n: number
  readonly cells: Record<string, number>[][]
  /** The next span length (1 fills the words from the lexical rules). */
  span: number
}

/**
 * CYK recognition (Cocke, Younger, Kasami) of a grammar in Chomsky normal form as a stepped algorithm, one span length
 * per step: length 1 from the lexical rules, then each longer span from every split point and every binary rule, the
 * derivations multiplying. It logs the span length, the cells that hold a nonterminal and the derivations in them, and
 * is done (`converged`) after the whole sentence. The state serialises; the parses of the state are read with
 * {@link cykResult}.
 *
 * @param g - The grammar (in CNF).
 * @param tokens - The words.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a grammar not in CNF.
 * @concept parsing
 * @example
 * ```ts
 * import { cykAlgorithm, grammar } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grammar("S -> NP VP\nVP -> V NP\nNP -> 'she' | 'fish'\nV -> 'eats'")
 * const run = runAlgorithm(cykAlgorithm(g, ['she', 'eats', 'fish']), { maxSteps: 10 })
 * [run.steps, run.state.cells[0]?.[3]] // => [3, { S: 1 }]
 * ```
 */
export function cykAlgorithm(g: Grammar, tokens: readonly string[]): Algorithm<CykState, ParsePath> {
  if (!isCnf(g)) throw new OmaiError('value', 'cyk: the grammar must be in Chomsky normal form')
  const n = tokens.length
  return {
    name: 'text/cyk',
    config: { n, productions: g.productions.length },
    init: () => ({
      n,
      cells: Array.from({ length: n + 1 }, () => Array.from({ length: n + 1 }, () => ({}))),
      span: 1,
    }),
    step: (s, ctx) => {
      const len = s.span
      const cell = (i: number, j: number): Record<string, number> => (s.cells[i] as Record<string, number>[])[j] ?? {}
      let filled = 0
      let derivations = 0
      for (let i = 0; i + len <= n; i++) {
        const j = i + len
        const here = cell(i, j)
        if (len === 1) {
          for (const p of g.productions)
            if (p.rhs.length === 1 && (p.rhs[0] as { terminal: string }).terminal === tokens[i])
              here[p.lhs] = (here[p.lhs] ?? 0) + 1
        } else {
          for (let k = i + 1; k < j; k++)
            for (const p of g.productions) {
              if (p.rhs.length !== 2) continue
              const b = cell(i, k)[(p.rhs[0] as { nonterminal: string }).nonterminal] ?? 0
              const c = cell(k, j)[(p.rhs[1] as { nonterminal: string }).nonterminal] ?? 0
              if (b > 0 && c > 0) here[p.lhs] = (here[p.lhs] ?? 0) + b * c
            }
        }
        const total = Object.values(here).reduce((acc, v) => acc + v, 0)
        if (total > 0) filled += 1
        derivations += total
      }
      s.span = len + 1
      ctx.log('chart/span', len)
      ctx.log('chart/cell', filled)
      ctx.log('chart/derivations', derivations)
      return s
    },
    done: (s) => (s.span > n ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CykState,
  }
}

/**
 * The outcome of a finished CYK state: whether the sentence is in the language, the number of parses and the parse
 * trees of the whole sentence from the start symbol (enumerated in production order, up to a cap).
 *
 * @param g - The grammar the state was run with.
 * @param tokens - The words.
 * @param state - A finished state of {@link cykAlgorithm}.
 * @param options - The most trees to enumerate (default 1000).
 * @returns Whether the sentence is recognised, the number of parses and the trees.
 * @concept parsing
 * @example
 * ```ts
 * import { cykAlgorithm, cykResult, grammar } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grammar("S -> NP VP\nVP -> V NP\nNP -> 'she' | 'fish'\nV -> 'eats'")
 * const tokens = ['she', 'eats', 'fish']
 * cykResult(g, tokens, runAlgorithm(cykAlgorithm(g, tokens), { maxSteps: 10 }).state).count // => 1
 * ```
 */
export function cykResult(
  g: Grammar,
  tokens: readonly string[],
  state: CykState,
  options: ParseOptions = {},
): { readonly recognised: boolean; readonly count: number; readonly trees: ParseTree[] } {
  const n = tokens.length
  if (n === 0) return { recognised: false, count: 0, trees: [] }
  const cell = (i: number, j: number): Record<string, number> => (state.cells[i] as Record<string, number>[])[j] ?? {}
  const count = cell(0, n)[g.start] ?? 0
  const limit = options.maxTrees ?? 1000
  const build = (sym: string, i: number, j: number, cap: number): ParseTree[] => {
    const out: ParseTree[] = []
    for (const p of g.productions) {
      if (p.lhs !== sym || out.length >= cap) continue
      if (p.rhs.length === 1) {
        if (j === i + 1 && (p.rhs[0] as { terminal: string }).terminal === tokens[i])
          out.push({ label: sym, children: [tokens[i] ?? ''] })
        continue
      }
      const [b, c] = p.rhs.map((x) => (x as { nonterminal: string }).nonterminal) as [string, string]
      for (let k = i + 1; k < j && out.length < cap; k++) {
        if (cell(i, k)[b] === undefined || cell(k, j)[c] === undefined) continue
        for (const l of build(b, i, k, cap))
          for (const r of build(c, k, j, cap - out.length))
            if (out.length < cap) out.push({ label: sym, children: [l, r] })
      }
    }
    return out
  }
  return { recognised: count > 0, count, trees: count > 0 ? build(g.start, 0, n, limit) : [] }
}

/**
 * CYK parsing (Cocke, Younger, Kasami) of a grammar in Chomsky normal form: the chart of nonterminals over every span,
 * and the parse trees of the whole sentence from the start symbol (enumerated in production order, up to a cap).
 * {@link cykAlgorithm} run to the end.
 *
 * @param g - The grammar (in CNF).
 * @param tokens - The words.
 * @param options - The most trees to enumerate (default 1000).
 * @returns Whether the sentence is in the language, the number of parses and the trees.
 * @throws OmaiError with code `value` for a grammar not in CNF.
 * @concept parsing
 * @example
 * ```ts
 * import { grammar, cyk } from 'omai-engine/text'
 * const g = grammar("S -> NP VP\nVP -> V NP\nNP -> 'she' | 'fish'\nV -> 'eats'")
 * cyk(g, ['she', 'eats', 'fish']).count // => 1
 * ```
 */
export function cyk(
  g: Grammar,
  tokens: readonly string[],
  options: ParseOptions = {},
): { readonly recognised: boolean; readonly count: number; readonly trees: ParseTree[] } {
  const run = runAlgorithm(cykAlgorithm(g, tokens), { maxSteps: Number.MAX_SAFE_INTEGER })
  return cykResult(g, tokens, run.state, options)
}

/** An Earley item: a production, a dot position and its origin. */
export interface EarleyItem {
  readonly p: number
  readonly dot: number
  readonly origin: number
}

/**
 * The state of {@link earleyAlgorithm}: the chart (the items of each column), a record of the items already in each
 * column (`p,dot,origin` keys), and the item to process next (column `k`, index `q`). A step adds items in place; keep a
 * copy of a state you want to look at later.
 */
export interface EarleyState {
  readonly chart: EarleyItem[][]
  readonly seen: Record<string, number>[]
  k: number
  q: number
}

/**
 * Earley parsing (Earley 1970) of any context-free grammar as a stepped algorithm, one item per step, with Aycock and
 * Horspool's handling of nullable symbols: the item at the front of the queue is either completed (its finished
 * constituent advances the dot of its origin column's items that were waiting for it), predicted (the rules of the
 * nonterminal after the dot join the column, and the item skips a nullable one) or scanned (the next word matched). It
 * logs the column and the items each operation added, and is done (`converged`) when the last column is exhausted. The
 * state serialises; the parses are read with {@link earleyResult}.
 *
 * @param g - The grammar.
 * @param tokens - The words.
 * @returns The algorithm.
 * @concept parsing
 * @example
 * ```ts
 * import { earleyAlgorithm, grammar } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grammar("S -> 'a' S | 'a'")
 * const run = runAlgorithm(earleyAlgorithm(g, ['a', 'a']), { maxSteps: 50 })
 * [run.status, run.state.chart.map((c) => c.length)] // => ['converged', [2, 4, 5]]
 * ```
 */
export function earleyAlgorithm(g: Grammar, tokens: readonly string[]): Algorithm<EarleyState, ParsePath> {
  const prods = g.productions
  const byLhs = new Map<string, number[]>()
  for (const [i, p] of prods.entries()) byLhs.set(p.lhs, [...(byLhs.get(p.lhs) ?? []), i])
  // Nullable nonterminals (fixed point).
  const nullable = new Set<string>()
  for (let changed = true; changed; ) {
    changed = false
    for (const p of prods)
      if (!nullable.has(p.lhs) && p.rhs.every((s) => 'nonterminal' in s && nullable.has(s.nonterminal))) {
        nullable.add(p.lhs)
        changed = true
      }
  }
  const n = tokens.length
  const add = (s: EarleyState, k: number, it: EarleyItem): boolean => {
    const key = `${it.p},${it.dot},${it.origin}`
    const seen = s.seen[k] as Record<string, number>
    if (seen[key] !== undefined) return false
    seen[key] = 1
    s.chart[k]?.push(it)
    return true
  }
  // Move to the next item to process: skip exhausted columns.
  const settle = (s: EarleyState): void => {
    while (s.q >= (s.chart[s.k]?.length ?? 0) && s.k < n) {
      s.k += 1
      s.q = 0
    }
  }
  return {
    name: 'text/earley',
    config: { n, productions: prods.length },
    init: () => {
      const s: EarleyState = {
        chart: Array.from({ length: n + 1 }, () => []),
        seen: Array.from({ length: n + 1 }, () => ({})),
        k: 0,
        q: 0,
      }
      for (const p of byLhs.get(g.start) ?? []) add(s, 0, { p, dot: 0, origin: 0 })
      settle(s)
      return s
    },
    step: (s, ctx) => {
      const k = s.k
      const it = s.chart[k]?.[s.q] as EarleyItem
      const prod = prods[it.p] as Production
      const next = prod.rhs[it.dot]
      let predicted = 0
      let scanned = 0
      let completed = 0
      if (next === undefined) {
        for (const parent of s.chart[it.origin] ?? []) {
          const pp = prods[parent.p] as Production
          const sym = pp.rhs[parent.dot]
          if (sym !== undefined && 'nonterminal' in sym && sym.nonterminal === prod.lhs)
            if (add(s, k, { ...parent, dot: parent.dot + 1 })) completed += 1
        }
      } else if ('nonterminal' in next) {
        for (const p of byLhs.get(next.nonterminal) ?? []) if (add(s, k, { p, dot: 0, origin: k })) predicted += 1
        if (nullable.has(next.nonterminal) && add(s, k, { ...it, dot: it.dot + 1 })) predicted += 1
      } else if (k < n && tokens[k] === next.terminal && add(s, k + 1, { ...it, dot: it.dot + 1 })) scanned += 1
      s.q += 1
      settle(s)
      ctx.log('earley/column', k)
      ctx.log('earley/predict', predicted)
      ctx.log('earley/scan', scanned)
      ctx.log('earley/complete', completed)
      return s
    },
    done: (s) => (s.k >= n && s.q >= (s.chart[n]?.length ?? 0) ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EarleyState,
  }
}

/**
 * The outcome of a finished Earley state: whether the sentence is in the language, the number of chart items and the
 * parse trees (in production order, up to a cap; cyclic grammars yield their acyclic derivations).
 *
 * @param g - The grammar the state was run with.
 * @param tokens - The words.
 * @param state - A finished state of {@link earleyAlgorithm}.
 * @param options - The most trees to enumerate (default 1000).
 * @returns Whether the sentence is recognised, the number of chart items and the trees.
 * @concept parsing
 * @example
 * ```ts
 * import { earleyAlgorithm, earleyResult, grammar } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grammar("S -> 'a' S | 'a'")
 * earleyResult(g, ['a', 'a'], runAlgorithm(earleyAlgorithm(g, ['a', 'a']), { maxSteps: 50 }).state).trees.length // => 1
 * ```
 */
export function earleyResult(
  g: Grammar,
  tokens: readonly string[],
  state: EarleyState,
  options: ParseOptions = {},
): { readonly recognised: boolean; readonly items: number; readonly trees: ParseTree[] } {
  const prods = g.productions
  const n = tokens.length
  const chart = state.chart
  const complete = (chart[n] ?? []).filter(
    (it) => it.origin === 0 && it.dot === (prods[it.p]?.rhs.length ?? 0) && prods[it.p]?.lhs === g.start,
  )
  const recognised = complete.length > 0
  const items = chart.reduce((sum, c) => sum + c.length, 0)
  if (!recognised) return { recognised, items, trees: [] }
  // Completed constituents by (symbol, start, end), for tree enumeration.
  const done = new Map<string, number[]>()
  for (let k = 0; k <= n; k++)
    for (const it of chart[k] ?? [])
      if (it.dot === (prods[it.p]?.rhs.length ?? 0)) {
        const key = `${prods[it.p]?.lhs}|${it.origin}|${k}`
        const list = done.get(key) ?? []
        if (!list.includes(it.p)) list.push(it.p)
        done.set(key, list)
      }
  for (const list of done.values()) list.sort((a, b) => a - b)
  const cap = options.maxTrees ?? 1000
  const trees = (sym: string, i: number, j: number, stack: ReadonlySet<string>): ParseTree[] => {
    const key = `${sym}|${i}|${j}`
    if (stack.has(key)) return []
    const inner = new Set(stack).add(key)
    const out: ParseTree[] = []
    for (const p of done.get(key) ?? []) {
      const rhs = (prods[p] as Production).rhs
      const seqs = (d: number, at: number): (ParseTree | string)[][] => {
        if (d === rhs.length) return at === j ? [[]] : []
        const sy = rhs[d] as GrammarSymbol
        const res: (ParseTree | string)[][] = []
        if ('terminal' in sy) {
          if (tokens[at] === sy.terminal) for (const rest of seqs(d + 1, at + 1)) res.push([sy.terminal, ...rest])
          return res
        }
        for (let e = at; e <= j && res.length < cap; e++) {
          if (!done.has(`${sy.nonterminal}|${at}|${e}`)) continue
          const rests = seqs(d + 1, e)
          if (rests.length === 0) continue
          for (const t of trees(sy.nonterminal, at, e, inner))
            for (const rest of rests) if (res.length < cap) res.push([t, ...rest])
        }
        return res
      }
      for (const children of seqs(0, i)) if (out.length < cap) out.push({ label: sym, children })
    }
    return out
  }
  return { recognised, items, trees: trees(g.start, 0, n, new Set()) }
}

/**
 * Earley parsing (Earley 1970) of any context-free grammar, with Aycock and Horspool's handling of nullable symbols:
 * predict, scan and complete over the chart, then enumerate the parse trees of the sentence (in production order,
 * up to a cap; cyclic grammars yield their acyclic derivations). {@link earleyAlgorithm} run to the end.
 *
 * @param g - The grammar.
 * @param tokens - The words.
 * @param options - The most trees to enumerate (default 1000).
 * @returns Whether the sentence is in the language, the number of chart items and the trees.
 * @concept parsing
 * @example
 * ```ts
 * import { grammar, earley, treeToString } from 'omai-engine/text'
 * const g = grammar("S -> NP VP\nNP -> Det N | NP PP | 'I'\nVP -> V NP | VP PP\nPP -> P NP\nDet -> 'the'\nN -> 'man' | 'telescope'\nV -> 'saw'\nP -> 'with'")
 * earley(g, 'I saw the man with the telescope'.split(' ')).trees.length // => 2
 * ```
 */
export function earley(
  g: Grammar,
  tokens: readonly string[],
  options: ParseOptions = {},
): { readonly recognised: boolean; readonly items: number; readonly trees: ParseTree[] } {
  const run = runAlgorithm(earleyAlgorithm(g, tokens), { maxSteps: Number.MAX_SAFE_INTEGER })
  return earleyResult(g, tokens, run.state, options)
}

/**
 * Convert a grammar without empty productions to Chomsky normal form: terminals in long rules are replaced by new nonterminals, long rules are binarised with new symbols
 * `A|<B-C>`, and unit productions A -> B are replaced by B's rules (probabilities multiplied).
 *
 * @param g - The grammar.
 * @returns An equivalent grammar in CNF (the same language; probabilities preserved for PCFGs).
 * @throws OmaiError with code `value` for empty productions.
 * @concept context-free-grammars
 * @example
 * ```ts
 * import { grammar, toChomskyNormalForm, cyk } from 'omai-engine/text'
 * const g = toChomskyNormalForm(grammar("S -> 'a' S 'b' | 'a' 'b'"))
 * cyk(g, ['a', 'a', 'b', 'b']).recognised // => true
 * ```
 */
export function toChomskyNormalForm(g: Grammar): Grammar {
  if (g.productions.some((p) => p.rhs.length === 0))
    throw new OmaiError('value', 'toChomskyNormalForm: empty productions are not supported')
  const out: Production[] = []
  const lexical = new Map<string, string>()
  const term = (t: string): string => {
    let nt = lexical.get(t)
    if (nt === undefined) {
      nt = `${t.toUpperCase()}_T`
      lexical.set(t, nt)
      out.push({ lhs: nt, rhs: [{ terminal: t }], probability: 1 })
    }
    return nt
  }
  const staged: Production[] = []
  for (const p of g.productions) {
    if (p.rhs.length === 1) {
      staged.push(p)
      continue
    }
    const symbols = p.rhs.map((s) => ('terminal' in s ? term(s.terminal) : s.nonterminal))
    let lhs = p.lhs
    let prob = p.probability
    let rest = symbols
    while (rest.length > 2) {
      const name = `${p.lhs}|<${rest.slice(1).join('-')}>`
      staged.push({ lhs, rhs: [{ nonterminal: rest[0] ?? '' }, { nonterminal: name }], probability: prob })
      lhs = name
      prob = 1
      rest = rest.slice(1)
    }
    staged.push({ lhs, rhs: rest.map((s) => ({ nonterminal: s })), probability: prob })
  }
  // Remove unit productions by closure.
  const units = staged.filter((p) => p.rhs.length === 1 && 'nonterminal' in (p.rhs[0] as GrammarSymbol))
  const others = [...out, ...staged.filter((p) => !units.includes(p))]
  const result = [...others]
  const closure = (lhs: string, sym: string, prob: number, seen: ReadonlySet<string>): void => {
    for (const q of others) if (q.lhs === sym) result.push({ lhs, rhs: q.rhs, probability: prob * q.probability })
    for (const u of units)
      if (u.lhs === sym) {
        const next = (u.rhs[0] as { nonterminal: string }).nonterminal
        if (!seen.has(next)) closure(lhs, next, prob * u.probability, new Set(seen).add(next))
      }
  }
  for (const u of units) {
    const sym = (u.rhs[0] as { nonterminal: string }).nonterminal
    closure(u.lhs, sym, u.probability, new Set([u.lhs, sym]))
  }
  // Merge duplicates (summing probabilities) and drop rules of symbols that are only reachable through units.
  const merged = new Map<string, Production>()
  for (const p of result) {
    const key = `${p.lhs}->${p.rhs.map((s) => ('terminal' in s ? `'${s.terminal}'` : s.nonterminal)).join(' ')}`
    const prev = merged.get(key)
    merged.set(key, prev === undefined ? p : { ...p, probability: prev.probability + p.probability })
  }
  return { start: g.start, productions: [...merged.values()] }
}
