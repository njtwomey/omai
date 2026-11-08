/**
 * Description logic ALC (attributive language with complements): concepts built from names, top, bottom, negation,
 * conjunction, disjunction and the quantifiers `some R C` and `all R C`, a parser for their text form, negation normal
 * form, and the tableau algorithm that decides concept satisfiability with respect to a TBox of general concept
 * inclusions. The tableau grows a completion graph one rule application per step (and, or with backtracking, all, some),
 * a clash (a concept and its negation, or bottom) closes a branch, and subset blocking stops the expansion of cyclic
 * ontologies. Subsumption is unsatisfiability of the difference, and a small ontology is classified into its hierarchy.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'

/** An ALC concept. */
export type Concept =
  | { readonly k: 'top' }
  | { readonly k: 'bottom' }
  | { readonly k: 'name'; readonly n: string }
  | { readonly k: 'not'; readonly c: Concept }
  | { readonly k: 'and'; readonly a: Concept; readonly b: Concept }
  | { readonly k: 'or'; readonly a: Concept; readonly b: Concept }
  | { readonly k: 'some'; readonly r: string; readonly c: Concept }
  | { readonly k: 'all'; readonly r: string; readonly c: Concept }

/**
 * A concept as text: names, `top`, `bottom`, `not C`, `C and D`, `C or D`, `some R C`, `all R C`, with parentheses;
 * `not` and the quantifiers bind tighter than `and`, which binds tighter than `or`.
 *
 * @param concept - The concept.
 * @returns The text, parenthesised where needed to parse back.
 * @example
 * ```ts
 * import { conceptText, parseConcept } from 'omai-engine/logic'
 * conceptText(parseConcept('Human and some hasChild (Human or not Cat)')) // => '(Human and some hasChild (Human or not Cat))'
 * ```
 */
export function conceptText(concept: Concept): string {
  switch (concept.k) {
    case 'top':
    case 'bottom':
      return concept.k
    case 'name':
      return concept.n
    case 'not':
      return `not ${wrap(concept.c)}`
    case 'and':
      return `(${conceptText(concept.a)} and ${conceptText(concept.b)})`
    case 'or':
      return `(${conceptText(concept.a)} or ${conceptText(concept.b)})`
    case 'some':
      return `some ${concept.r} ${wrap(concept.c)}`
    case 'all':
      return `all ${concept.r} ${wrap(concept.c)}`
  }
}

const wrap = (c: Concept): string => {
  const t = conceptText(c)
  return c.k === 'not' || c.k === 'some' || c.k === 'all' ? `(${t})` : t
}

/**
 * Parse the text form of a concept.
 *
 * @param text - The concept, e.g. `Human and some hasChild Human`.
 * @returns The concept.
 * @throws OmaiError with code `value` for malformed text.
 * @example
 * ```ts
 * import { conceptText, parseConcept } from 'omai-engine/logic'
 * conceptText(parseConcept('some R A or all R B')) // => '(some R A or all R B)'
 * ```
 */
export function parseConcept(text: string): Concept {
  const tokens = text.match(/\(|\)|[A-Za-z_][A-Za-z0-9_]*/g) ?? []
  let pos = 0
  const peek = (): string | undefined => tokens[pos]
  const take = (): string => {
    const t = tokens[pos++]
    if (t === undefined) throw new OmaiError('value', `unexpected end of concept '${text}'`)
    return t
  }
  const parseOr = (): Concept => {
    let left = parseAnd()
    while (peek() === 'or') {
      take()
      left = { k: 'or', a: left, b: parseAnd() }
    }
    return left
  }
  const parseAnd = (): Concept => {
    let left = parseUnary()
    while (peek() === 'and') {
      take()
      left = { k: 'and', a: left, b: parseUnary() }
    }
    return left
  }
  const parseUnary = (): Concept => {
    const t = take()
    if (t === 'not') return { k: 'not', c: parseUnary() }
    if (t === 'some' || t === 'all') {
      const r = take()
      return { k: t, r, c: parseUnary() }
    }
    if (t === '(') {
      const c = parseOr()
      if (take() !== ')') throw new OmaiError('value', `expected ')' in '${text}'`)
      return c
    }
    if (t === 'top' || t === 'bottom') return { k: t }
    if (t === ')' || t === 'and' || t === 'or') throw new OmaiError('value', `unexpected '${t}' in '${text}'`)
    return { k: 'name', n: t }
  }
  const c = parseOr()
  if (pos < tokens.length) throw new OmaiError('value', `unexpected '${tokens[pos]}' in '${text}'`)
  return c
}

/**
 * Negation normal form: negation pushed to the names by De Morgan and the quantifier dualities.
 *
 * @param concept - The concept.
 * @returns An equivalent concept in which `not` applies only to names.
 * @example
 * ```ts
 * import { conceptText, nnf, parseConcept } from 'omai-engine/logic'
 * conceptText(nnf(parseConcept('not (some R A and all R B)'))) // => '(all R (not A) or some R (not B))'
 * ```
 */
export function nnf(concept: Concept): Concept {
  const neg = (c: Concept): Concept => {
    switch (c.k) {
      case 'top':
        return { k: 'bottom' }
      case 'bottom':
        return { k: 'top' }
      case 'name':
        return { k: 'not', c }
      case 'not':
        return nnf(c.c)
      case 'and':
        return { k: 'or', a: neg(c.a), b: neg(c.b) }
      case 'or':
        return { k: 'and', a: neg(c.a), b: neg(c.b) }
      case 'some':
        return { k: 'all', r: c.r, c: neg(c.c) }
      case 'all':
        return { k: 'some', r: c.r, c: neg(c.c) }
    }
  }
  switch (concept.k) {
    case 'top':
    case 'bottom':
    case 'name':
      return concept
    case 'not':
      return neg(concept.c)
    case 'and':
    case 'or':
      return { k: concept.k, a: nnf(concept.a), b: nnf(concept.b) }
    case 'some':
    case 'all':
      return { k: concept.k, r: concept.r, c: nnf(concept.c) }
  }
}

/** A TBox: general concept inclusions `[sub, super]` (an equivalence is two). */
export type TBox = readonly (readonly [sub: string, sup: string])[]

/** The state of the tableau: the completion graph, the choice points and the verdict. */
export interface TableauState {
  /** Per node: its label (concepts in negation normal form, as text), its parent and whether it is blocked. */
  readonly nodes: readonly { readonly label: readonly string[]; readonly parent: number; readonly blocked: boolean }[]
  readonly edges: readonly (readonly [from: number, to: number, role: string])[]
  /** Saved alternatives of `or` rules: the graph before the choice and the disjunct still to try. */
  readonly choices: readonly {
    readonly nodes: TableauState['nodes']
    readonly edges: TableauState['edges']
    readonly node: number
    readonly concept: string
  }[]
  readonly status: 'running' | 'satisfiable' | 'unsatisfiable'
  /** The rule applied in the last step. */
  readonly rule: { readonly name: string; readonly node: number; readonly concept: string } | null
  readonly clashes: number
}

const TABLEAU_SCHEMA: TraceSchema = channels({
  metrics: {
    nodes: { description: 'nodes in the completion graph' },
    branches: { description: 'choice points still holding an untried disjunct' },
    clashes: { description: 'branches closed by a clash so far' },
    blocked: { description: 'nodes blocked by an ancestor' },
  },
  messages: {
    rule: {
      description: 'the rule applied',
      payload: { name: 'and, or, all, some, clash or backtrack', node: 'node', concept: 'concept' },
    },
  },
})

/**
 * The satisfiability tableau for ALC as a traced algorithm, one rule application per step. The root's label is the
 * concept plus the TBox internalised (every node's label also holds `not C or D` for each inclusion `C ⊑ D`). The `and`
 * rule adds both conjuncts, `all` pushes a filler along each edge, `or` tries a disjunct and saves the other as a choice
 * point, `some` makes a successor unless a successor already has the filler. A node is blocked when an ancestor's label
 * contains its own, and blocked nodes are not expanded. A clash backtracks to the last choice point; with none left the
 * concept is unsatisfiable, and when no rule applies the graph is a model.
 *
 * @param concept - The concept (text or parsed).
 * @param tbox - General concept inclusions as text pairs.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for malformed concepts.
 * @example
 * ```ts
 * import { alcTableauAlgorithm } from 'omai-engine/logic'
 * alcTableauAlgorithm('some R A and all R not A').name // => 'alc-tableau'
 * ```
 */
export function alcTableauAlgorithm(concept: string | Concept, tbox: TBox = []): Algorithm<TableauState> {
  const root = nnf(typeof concept === 'string' ? parseConcept(concept) : concept)
  const axioms = tbox.map(([sub, sup]) =>
    conceptText(nnf({ k: 'or', a: { k: 'not', c: parseConcept(sub) }, b: parseConcept(sup) })),
  )
  const cache = new Map<string, Concept>()
  const get = (text: string): Concept => {
    let c = cache.get(text)
    if (c === undefined) {
      c = parseConcept(text)
      cache.set(text, c)
    }
    return c
  }
  const norm = (c: Concept): string => conceptText(nnf(c))
  type Nodes = { label: string[]; parent: number; blocked: boolean }[]
  const withBlocking = (nodes: Nodes): Nodes => {
    const out = nodes.map((n) => ({ ...n, blocked: false }))
    out.forEach((n, i) => {
      for (let a = n.parent; a >= 0; a = out[a]?.parent ?? -1) {
        if (!(out[a]?.blocked ?? false) && n.label.every((c) => out[a]?.label.includes(c))) {
          n.blocked = true
          break
        }
      }
      void i
    })
    return out
  }
  const clashAt = (label: readonly string[]): boolean =>
    label.includes('bottom') ||
    label.some((t) => {
      const c = get(t)
      return c.k === 'name' && label.includes(`not ${c.n}`)
    })
  return {
    name: 'alc-tableau',
    config: { concept: conceptText(root), axioms: [...axioms] },
    schema: TABLEAU_SCHEMA,
    init: () => ({
      nodes: [{ label: [...new Set([conceptText(root), ...axioms])], parent: -1, blocked: false }],
      edges: [],
      choices: [],
      status: 'running',
      rule: null,
      clashes: 0,
    }),
    step: (s) => {
      if (s.status !== 'running') return s
      const nodes: Nodes = withBlocking(
        s.nodes.map((n) => ({ label: [...n.label], parent: n.parent, blocked: n.blocked })),
      )
      const edges = s.edges.map((e) => [...e] as [number, number, string])
      const backtrack = (clash: string): TableauState => {
        const choice = s.choices[s.choices.length - 1]
        if (choice === undefined)
          return {
            ...s,
            nodes,
            status: 'unsatisfiable',
            rule: { name: 'clash', node: 0, concept: clash },
            clashes: s.clashes + 1,
          }
        const restored: Nodes = choice.nodes.map((n) => ({ label: [...n.label], parent: n.parent, blocked: n.blocked }))
        restored[choice.node]?.label.push(choice.concept)
        return {
          nodes: restored,
          edges: choice.edges.map((e) => [...e] as [number, number, string]),
          choices: s.choices.slice(0, -1),
          status: 'running',
          rule: { name: 'backtrack', node: choice.node, concept: choice.concept },
          clashes: s.clashes + 1,
        }
      }
      for (let x = 0; x < nodes.length; x++) {
        const label = (nodes[x] as Nodes[number]).label
        if (clashAt(label)) return backtrack(label.find((t) => t === 'bottom') ?? label[0] ?? '')
      }
      const apply = (name: string, node: number, concept: string, next: Partial<TableauState>): TableauState => ({
        nodes,
        edges,
        choices: s.choices,
        status: 'running',
        rule: { name, node, concept },
        clashes: s.clashes,
        ...next,
      })
      // Deterministic rules first, then the choice rule.
      for (let x = 0; x < nodes.length; x++) {
        const n = nodes[x] as Nodes[number]
        if (n.blocked) continue
        for (const t of n.label) {
          const c = get(t)
          if (c.k === 'and') {
            const missing = [norm(c.a), norm(c.b)].filter((m) => !n.label.includes(m))
            if (missing.length > 0) {
              n.label.push(...missing)
              return apply('and', x, t, {})
            }
          } else if (c.k === 'all') {
            for (const [from, to, role] of edges) {
              const f = norm(c.c)
              if (from === x && role === c.r && !(nodes[to] as Nodes[number]).label.includes(f)) {
                ;(nodes[to] as Nodes[number]).label.push(f)
                return apply('all', x, t, {})
              }
            }
          }
        }
      }
      for (let x = 0; x < nodes.length; x++) {
        const n = nodes[x] as Nodes[number]
        if (n.blocked) continue
        for (const t of n.label) {
          const c = get(t)
          if (c.k === 'some') {
            const f = norm(c.c)
            const has = edges.some(
              ([from, to, role]) => from === x && role === c.r && (nodes[to] as Nodes[number]).label.includes(f),
            )
            if (!has) {
              nodes.push({ label: [...new Set([f, ...axioms])], parent: x, blocked: false })
              edges.push([x, nodes.length - 1, c.r])
              return apply('some', x, t, {})
            }
          }
        }
      }
      for (let x = 0; x < nodes.length; x++) {
        const n = nodes[x] as Nodes[number]
        if (n.blocked) continue
        for (const t of n.label) {
          const c = get(t)
          if (c.k === 'or') {
            const a = norm(c.a)
            const b = norm(c.b)
            if (!n.label.includes(a) && !n.label.includes(b)) {
              const saved = { nodes: s.nodes, edges: s.edges, node: x, concept: b }
              n.label.push(a)
              return apply('or', x, t, { choices: [...s.choices, saved] })
            }
          }
        }
      }
      return { ...s, nodes, edges, status: 'satisfiable', rule: null }
    },
    done: (s) => (s.status === 'running' ? undefined : 'converged'),
    observe: (s) => ({
      metrics: {
        nodes: s.nodes.length,
        branches: s.choices.length,
        clashes: s.clashes,
        blocked: s.nodes.filter((n) => n.blocked).length,
      },
      messages: s.rule === null ? [] : [{ kind: 'rule', payload: s.rule as unknown as Json }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as TableauState,
  }
}

/**
 * Whether a concept is satisfiable with respect to a TBox.
 *
 * @param concept - The concept as text or parsed.
 * @param tbox - General concept inclusions.
 * @param run - Tracing options.
 * @returns True when some model has an element in the concept.
 * @throws OmaiError with code `value` for malformed concepts.
 * @example
 * ```ts
 * import { conceptSatisfiable } from 'omai-engine/logic'
 * [conceptSatisfiable('some R A and all R not A'), conceptSatisfiable('some R A and all R B')] // => [false, true]
 * ```
 */
export function conceptSatisfiable(
  concept: string | Concept,
  tbox: TBox = [],
  run: Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): boolean {
  const { state } = runAlgorithm(alcTableauAlgorithm(concept, tbox), { ...run, maxSteps: 200_000 })
  if (state.status === 'running') throw new OmaiError('value', 'the tableau did not terminate within its step budget')
  return state.status === 'satisfiable'
}

/**
 * Whether `sup` conceptSubsumes `sub` (every `sub` is a `sup`) with respect to a TBox: the difference `sub and not sup` is
 * unsatisfiable.
 *
 * @param sup - The more general concept.
 * @param sub - The more specific concept.
 * @param tbox - General concept inclusions.
 * @returns True when `sub` is subsumed by `sup`.
 * @throws OmaiError with code `value` for malformed concepts.
 * @example
 * ```ts
 * import { conceptSubsumes } from 'omai-engine/logic'
 * [conceptSubsumes('some R A', 'some R (A and B)'), conceptSubsumes('some R (A and B)', 'some R A')] // => [true, false]
 * ```
 */
export function conceptSubsumes(sup: string, sub: string, tbox: TBox = []): boolean {
  return !conceptSatisfiable({ k: 'and', a: parseConcept(sub), b: { k: 'not', c: parseConcept(sup) } }, tbox)
}

/**
 * Classify the named concepts of a TBox into a hierarchy: for each name, its direct subsumers among the names (a
 * transitive reduction of the subsumption order, with equivalent names listed together under the first).
 *
 * @param names - The concept names to classifyConcepts.
 * @param tbox - General concept inclusions defining them.
 * @returns For each name, its direct parents.
 * @throws OmaiError with code `value` for malformed concepts.
 * @example
 * ```ts
 * import { classifyConcepts } from 'omai-engine/logic'
 * const tbox = [['Man', 'Human'], ['Woman', 'Human'], ['Mother', 'Woman'], ['Mother', 'some hasChild Human']]
 * classifyConcepts(['Human', 'Man', 'Woman', 'Mother'], tbox) // => { Human: [], Man: ['Human'], Woman: ['Human'], Mother: ['Woman'] }
 * ```
 */
export function classifyConcepts(names: readonly string[], tbox: TBox): Record<string, string[]> {
  const memo = new Map<string, boolean>()
  const subsumed = (sup: string, sub: string): boolean => {
    const key = `${sup}\u0000${sub}`
    let v = memo.get(key)
    if (v === undefined) {
      v = conceptSubsumes(sup, sub, tbox)
      memo.set(key, v)
    }
    return v
  }
  /** `a` is strictly above `b`: conceptSubsumes it and is not subsumed by it. */
  const above = (a: string, b: string): boolean => a !== b && subsumed(a, b) && !subsumed(b, a)
  const out: Record<string, string[]> = {}
  for (const b of names) {
    const supers = names.filter((a) => above(a, b))
    out[b] = supers.filter((a) => !supers.some((m) => above(a, m)))
  }
  return out
}
