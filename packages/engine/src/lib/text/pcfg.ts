/**
 * Probabilistic context-free grammars: the most probable (Viterbi) parse as nltk's ViterbiParser finds it, the total
 * probability of a sentence, the inside-outside algorithm giving expected production counts, and unsupervised grammar
 * estimation by EM with inside-outside as a traced algorithm.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, runAlgorithm, type TracePolicy, type Tracer } from '../trace/index.ts'
import type { Grammar, GrammarSymbol, ParseTree, Production } from './parsing.ts'

/** The best item of a span: its probability and how it was built. */
interface Best {
  readonly prob: number
  readonly p: number
  /** For each right-hand side symbol, the end of its span. */
  readonly ends: readonly number[]
}

/** The (max or sum) score of covering tokens[i, j) with the rhs of a production, over every split. */
function coverRhs(
  rhs: readonly GrammarSymbol[],
  tokens: readonly string[],
  i: number,
  j: number,
  table: (sym: string, a: number, b: number) => number,
  combine: 'max' | 'sum',
): { readonly score: number; readonly ends: number[] } {
  // f[d][e]: best score of rhs[0, d) over tokens[i, e), with back-pointers.
  const m = rhs.length
  const f = Array.from({ length: m + 1 }, () => new Float64Array(j - i + 1))
  const back = Array.from({ length: m + 1 }, () => new Int32Array(j - i + 1).fill(-1))
  ;(f[0] as Float64Array)[0] = 1
  for (let d = 0; d < m; d++) {
    const s = rhs[d] as GrammarSymbol
    for (let a = i; a <= j; a++) {
      const fa = f[d]?.[a - i] ?? 0
      if (fa === 0) continue
      if ('terminal' in s) {
        if (a < j && tokens[a] === s.terminal) {
          const v = fa
          const cur = f[d + 1]?.[a + 1 - i] ?? 0
          if (combine === 'sum') (f[d + 1] as Float64Array)[a + 1 - i] = cur + v
          else if (v > cur) {
            ;(f[d + 1] as Float64Array)[a + 1 - i] = v
            ;(back[d + 1] as Int32Array)[a + 1 - i] = a
          }
        }
        continue
      }
      for (let b = a; b <= j; b++) {
        const t = table(s.nonterminal, a, b)
        if (t === 0) continue
        const v = fa * t
        const cur = f[d + 1]?.[b - i] ?? 0
        if (combine === 'sum') (f[d + 1] as Float64Array)[b - i] = cur + v
        else if (v > cur) {
          ;(f[d + 1] as Float64Array)[b - i] = v
          ;(back[d + 1] as Int32Array)[b - i] = a
        }
      }
    }
  }
  const score = f[m]?.[j - i] ?? 0
  const ends: number[] = new Array<number>(m).fill(j)
  if (combine === 'max' && score > 0) {
    let e = j
    for (let d = m; d > 0; d--) {
      ends[d - 1] = e
      e = back[d]?.[e - i] ?? i
    }
  }
  return { score, ends }
}

/** The span chart of a PCFG over a sentence: best (Viterbi) or total (inside) probability of each symbol and span. */
function chart(
  g: Grammar,
  tokens: readonly string[],
  combine: 'max' | 'sum',
): { readonly value: (sym: string, i: number, j: number) => number; readonly best: Map<string, Best> } {
  const n = tokens.length
  const values = new Map<string, number>()
  const best = new Map<string, Best>()
  const value = (sym: string, i: number, j: number): number => values.get(`${sym}|${i}|${j}`) ?? 0
  for (let len = 0; len <= n; len++) {
    for (let i = 0; i + len <= n; i++) {
      const j = i + len
      // Unary and empty chains within a span: iterate to a fixed point (bounded; probabilities shrink along cycles).
      for (let round = 0; round < 200; round++) {
        let changed = false
        const next = new Map<string, number>()
        const nextBest = new Map<string, Best>()
        for (const [pi, p] of g.productions.entries()) {
          const { score, ends } = coverRhs(p.rhs, tokens, i, j, value, combine)
          const v = score * p.probability
          if (v === 0) continue
          const key = p.lhs
          if (combine === 'sum') next.set(key, (next.get(key) ?? 0) + v)
          else if (v > (next.get(key) ?? 0)) {
            next.set(key, v)
            nextBest.set(key, { prob: v, p: pi, ends })
          }
        }
        for (const [sym, v] of next) {
          const k = `${sym}|${i}|${j}`
          const old = values.get(k) ?? 0
          if (Math.abs(v - old) > 1e-300 && Math.abs(v - old) > 1e-15 * Math.abs(v)) changed = true
          values.set(k, v)
          const b = nextBest.get(sym)
          if (b !== undefined) best.set(k, b)
        }
        if (!changed) break
      }
    }
  }
  return { value, best }
}

/**
 * The most probable parse of a sentence under a PCFG (the Viterbi parse, as nltk's ViterbiParser finds it): a max-product
 * chart over spans for productions of any length, with unary chains resolved within each span.
 *
 * @param g - The probabilistic grammar.
 * @param tokens - The words.
 * @returns The best tree and its probability, or undefined when the sentence has no parse.
 * @concept probabilistic-parsing
 * @example
 * ```ts
 * import { grammar, viterbiParse, treeToString } from 'omai-engine/text'
 * const g = grammar("S -> NP VP [1.0]\nNP -> 'I' [0.6] | 'fish' [0.4]\nVP -> V NP [0.7] | 'fish' [0.3]\nV -> 'eat' [1.0]")
 * const r = viterbiParse(g, ['I', 'eat', 'fish'])
 * [r === undefined ? '' : treeToString(r.tree), r?.probability] // => ['(S (NP I) (VP (V eat) (NP fish)))', 0.168]
 * ```
 */
export function viterbiParse(
  g: Grammar,
  tokens: readonly string[],
): { readonly tree: ParseTree; readonly probability: number } | undefined {
  const { value, best } = chart(g, tokens, 'max')
  const probability = value(g.start, 0, tokens.length)
  if (probability === 0) return undefined
  const build = (sym: string, i: number, j: number, depth: number): ParseTree => {
    const b = best.get(`${sym}|${i}|${j}`)
    const p = b === undefined ? undefined : g.productions[b.p]
    if (b === undefined || p === undefined || depth > 10_000) return { label: sym, children: [] }
    const children: (ParseTree | string)[] = []
    let at = i
    for (const [d, s] of p.rhs.entries()) {
      const e = b.ends[d] ?? j
      children.push('terminal' in s ? s.terminal : build(s.nonterminal, at, e, depth + 1))
      at = e
    }
    return { label: sym, children }
  }
  return { tree: build(g.start, 0, tokens.length, 0), probability }
}

/**
 * The total probability of a sentence under a PCFG (the inside probability of the start symbol over the whole
 * sentence): the sum over all parses, as the sum of nltk's InsideChartParser parse probabilities.
 *
 * @param g - The probabilistic grammar.
 * @param tokens - The words.
 * @returns P(sentence).
 * @concept probabilistic-parsing
 * @example
 * ```ts
 * import { grammar, sentenceProbability } from 'omai-engine/text'
 * const g = grammar("S -> S S [0.4] | 'a' [0.6]")
 * sentenceProbability(g, ['a', 'a', 'a']) // => 0.06912
 * ```
 */
export function sentenceProbability(g: Grammar, tokens: readonly string[]): number {
  return chart(g, tokens, 'sum').value(g.start, 0, tokens.length)
}

/** Whether a grammar is binary or lexical (CNF), as inside-outside needs. */
function checkCnf(g: Grammar, name: string): void {
  for (const p of g.productions) {
    const ok =
      (p.rhs.length === 2 && p.rhs.every((s) => 'nonterminal' in s)) ||
      (p.rhs.length === 1 && 'terminal' in (p.rhs[0] as GrammarSymbol))
    if (!ok) throw new OmaiError('value', `${name}: the grammar must be in Chomsky normal form`)
  }
}

/**
 * Inside-outside (Baker 1979; Lari and Young 1990) for a PCFG in Chomsky normal form: the inside and outside
 * probabilities of every symbol and span, the sentence probability, and the expected count of each production.
 *
 * @param g - The grammar (CNF).
 * @param tokens - The words.
 * @returns The sentence probability and expected production counts (in production order; zero without a parse).
 * @throws OmaiError with code `value` for a grammar not in CNF.
 * @concept probabilistic-parsing
 * @example
 * ```ts
 * import { grammar, insideOutside } from 'omai-engine/text'
 * const g = grammar("S -> A B [1.0]\nA -> 'a' [1.0]\nB -> 'b' [1.0]")
 * insideOutside(g, ['a', 'b']).expectedCounts // => [1, 1, 1]
 * ```
 */
export function insideOutside(
  g: Grammar,
  tokens: readonly string[],
): { readonly probability: number; readonly expectedCounts: number[] } {
  checkCnf(g, 'insideOutside')
  const n = tokens.length
  const symbols = [...new Set(g.productions.map((p) => p.lhs))]
  const idx = new Map(symbols.map((s, i) => [s, i]))
  const S = symbols.length
  const at = (i: number, j: number, a: number): number => (i * (n + 1) + j) * S + a
  const inside = new Float64Array((n + 1) * (n + 1) * S)
  const outside = new Float64Array((n + 1) * (n + 1) * S)
  const binary = g.productions.map((p, pi) => ({ p, pi })).filter(({ p }) => p.rhs.length === 2)
  const lexical = g.productions.map((p, pi) => ({ p, pi })).filter(({ p }) => p.rhs.length === 1)
  const nt = (s: GrammarSymbol | undefined): number => idx.get((s as { nonterminal: string }).nonterminal) ?? -1
  const addTo = (t: Float64Array, k: number, v: number): void => {
    t[k] = (t[k] ?? 0) + v
  }
  for (let i = 0; i < n; i++)
    for (const { p } of lexical)
      if ((p.rhs[0] as { terminal: string }).terminal === tokens[i])
        addTo(inside, at(i, i + 1, idx.get(p.lhs) ?? 0), p.probability)
  for (let len = 2; len <= n; len++)
    for (let i = 0; i + len <= n; i++) {
      const j = i + len
      for (const { p } of binary) {
        const b = nt(p.rhs[0])
        const c = nt(p.rhs[1])
        if (b < 0 || c < 0) continue
        let s = 0
        for (let k = i + 1; k < j; k++) s += (inside[at(i, k, b)] ?? 0) * (inside[at(k, j, c)] ?? 0)
        addTo(inside, at(i, j, idx.get(p.lhs) ?? 0), p.probability * s)
      }
    }
  const start = idx.get(g.start)
  const probability = start === undefined || n === 0 ? 0 : (inside[at(0, n, start)] ?? 0)
  const counts = new Array<number>(g.productions.length).fill(0)
  if (probability === 0) return { probability, expectedCounts: counts }
  outside[at(0, n, start as number)] = 1
  for (let len = n; len >= 2; len--)
    for (let i = 0; i + len <= n; i++) {
      const j = i + len
      for (const { p, pi } of binary) {
        const a = idx.get(p.lhs) ?? 0
        const out = outside[at(i, j, a)] ?? 0
        if (out === 0) continue
        const b = nt(p.rhs[0])
        const c = nt(p.rhs[1])
        for (let k = i + 1; k < j; k++) {
          const ib = inside[at(i, k, b)] ?? 0
          const ic = inside[at(k, j, c)] ?? 0
          addTo(outside, at(i, k, b), out * p.probability * ic)
          addTo(outside, at(k, j, c), out * p.probability * ib)
          counts[pi] = (counts[pi] ?? 0) + (out * p.probability * ib * ic) / probability
        }
      }
    }
  for (let i = 0; i < n; i++)
    for (const { p, pi } of lexical)
      if ((p.rhs[0] as { terminal: string }).terminal === tokens[i]) {
        const a = idx.get(p.lhs) ?? 0
        counts[pi] = (counts[pi] ?? 0) + ((outside[at(i, i + 1, a)] ?? 0) * p.probability) / probability
      }
  return { probability, expectedCounts: counts }
}

/** The state of PCFG EM training: the grammar and the log-likelihood of the corpus under it. */
export interface PcfgEmState {
  readonly grammar: Grammar
  readonly logLikelihood: number
  readonly improvement: number
}

/**
 * Unsupervised PCFG estimation by EM with inside-outside (Lari and Young 1990) as an {@link Algorithm}: each step
 * re-estimates every production's probability as its expected count over the sum for its left-hand side, recording the
 * corpus log-likelihood, which never decreases; it stops when the improvement falls below `tol`.
 *
 * @param g - The starting grammar (CNF).
 * @param sentences - Token lists (each must have a parse).
 * @param options - Convergence tolerance (default 1e-8).
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a grammar not in CNF or a sentence without a parse.
 * @concept probabilistic-parsing
 * @example
 * ```ts
 * import { grammar, pcfgEmAlgorithm } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = grammar("S -> S S [0.5] | 'a' [0.5]")
 * const r = runAlgorithm(pcfgEmAlgorithm(g, [['a'], ['a', 'a']]), { maxSteps: 200 })
 * r.state.grammar.productions[0]?.probability // => 0.25
 * ```
 */
export function pcfgEmAlgorithm(
  g: Grammar,
  sentences: readonly (readonly string[])[],
  options: { readonly tol?: number } = {},
): Algorithm<PcfgEmState> {
  checkCnf(g, 'pcfgEm')
  const tol = options.tol ?? 1e-8
  const loglik = (gr: Grammar): number =>
    sentences.reduce((s, t) => {
      const p = insideOutside(gr, t).probability
      if (!(p > 0)) throw new OmaiError('value', `pcfgEm: no parse of ${t.join(' ')}`)
      return s + Math.log(p)
    }, 0)
  return {
    name: 'pcfgEm',
    config: { sentences: sentences.length, productions: g.productions.length } as Json,
    init: () => ({ grammar: g, logLikelihood: loglik(g), improvement: Number.POSITIVE_INFINITY }),
    step: (state) => {
      const totals = new Array<number>(state.grammar.productions.length).fill(0)
      for (const t of sentences)
        for (const [i, c] of insideOutside(state.grammar, t).expectedCounts.entries()) totals[i] = (totals[i] ?? 0) + c
      const byLhs = new Map<string, number>()
      for (const [i, p] of state.grammar.productions.entries())
        byLhs.set(p.lhs, (byLhs.get(p.lhs) ?? 0) + (totals[i] ?? 0))
      const productions: Production[] = state.grammar.productions.map((p, i) => {
        const z = byLhs.get(p.lhs) ?? 0
        return { ...p, probability: z > 0 ? (totals[i] ?? 0) / z : p.probability }
      })
      const grammar = { start: state.grammar.start, productions }
      const ll = loglik(grammar)
      return { grammar, logLikelihood: ll, improvement: ll - state.logLikelihood }
    },
    done: (s) => (s.improvement < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood } }),
  }
}

/**
 * Train a PCFG by EM ({@link pcfgEmAlgorithm}), with an iteration cap.
 *
 * @param g - The starting grammar (CNF).
 * @param sentences - Token lists.
 * @param options - Tolerance, the most iterations (default 100), tracer and policy.
 * @returns The trained grammar and its log-likelihood.
 * @concept probabilistic-parsing
 * @example
 * ```ts
 * import { grammar, trainPcfg } from 'omai-engine/text'
 * trainPcfg(grammar("S -> S S [0.5] | 'a' [0.5]"), [['a'], ['a', 'a']]).grammar.productions[1]?.probability // => 0.75
 * ```
 */
export function trainPcfg(
  g: Grammar,
  sentences: readonly (readonly string[])[],
  options: {
    readonly tol?: number
    readonly maxIter?: number
    readonly tracer?: Tracer
    readonly policy?: TracePolicy
  } = {},
): { readonly grammar: Grammar; readonly logLikelihood: number; readonly iterations: number } {
  const r = runAlgorithm(pcfgEmAlgorithm(g, sentences, options.tol === undefined ? {} : { tol: options.tol }), {
    maxSteps: options.maxIter ?? 100,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return { grammar: r.state.grammar, logLikelihood: r.state.logLikelihood, iterations: r.steps }
}
