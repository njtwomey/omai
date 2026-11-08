/**
 * Production systems, the architecture of classical expert systems: a working memory of ground facts and rules that
 * fire when all their conditions match (variables are bound by one-way matching against facts, `not(...)` conditions
 * require absence). Each cycle finds the conflict set of rule instantiations not yet fired (refraction), a conflict
 * resolution strategy (rule order, priority, recency, specificity) picks one, and firing adds and removes facts. Matching
 * is either recomputed from scratch or kept incrementally in a Rete-lite network: alpha memories of the facts each
 * condition pattern matches, and beta memories of the partial matches of each rule's leading conditions, updated as
 * facts are added and retracted. Both give the same conflict sets. The system is a traced algorithm: one rule firing per
 * step, with the working memory and the fired rule in the state.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { parseTerm } from './parse.ts'
import { type Term, termEquals, termToString } from './terms.ts'
import { applySubstitution, matchTerm, type Substitution } from './unify.ts'

/** A production rule: conditions (patterns; `not(P)` requires that nothing matches P) and the facts it adds and removes. */
export interface ProductionRule {
  readonly name: string
  readonly conditions: readonly Term[]
  readonly add: readonly Term[]
  readonly remove: readonly Term[]
  /** Higher fires first under the `priority` strategy (default 0). */
  readonly priority: number
  /** Stop the system after this rule fires. */
  readonly halt: boolean
}

/**
 * A production rule from text: conditions, then the facts to add and to remove (each a Prolog term, with variables
 * capitalised).
 *
 * @param name - The rule's name.
 * @param conditions - The conditions, e.g. `parent(X, Y)`, `not(married(X))`.
 * @param actions - Facts to `add`, facts to `remove`, a `priority` and whether to `halt`.
 * @returns The rule.
 * @example
 * ```ts
 * import { productionRule } from 'omai-engine/logic'
 * productionRule('grandparent', ['parent(X, Y)', 'parent(Y, Z)'], { add: ['grandparent(X, Z)'] }).conditions.length // => 2
 * ```
 */
export function productionRule(
  name: string,
  conditions: readonly string[],
  actions: {
    readonly add?: readonly string[]
    readonly remove?: readonly string[]
    readonly priority?: number
    readonly halt?: boolean
  } = {},
): ProductionRule {
  return {
    name,
    conditions: conditions.map((c) => parseTerm(c)),
    add: (actions.add ?? []).map((c) => parseTerm(c)),
    remove: (actions.remove ?? []).map((c) => parseTerm(c)),
    priority: actions.priority ?? 0,
    halt: actions.halt ?? false,
  }
}

/** A fact in working memory with the cycle-order timestamp it was added at. */
export interface WorkingElement {
  readonly id: number
  readonly fact: Term
}

/** A rule instantiation: the rule, the facts matched by its positive conditions and the variable bindings. */
export interface Activation {
  readonly rule: number
  readonly facts: readonly number[]
  readonly bindings: Readonly<Record<string, string>>
}

/** Conflict resolution strategies. */
export type ConflictStrategy = 'order' | 'priority' | 'recency' | 'specificity'

const isNot = (t: Term): t is Extract<Term, { kind: 'compound' }> =>
  t.kind === 'compound' && t.functor === 'not' && t.args.length === 1

const activationKey = (a: Activation): string => `${a.rule}:${a.facts.join(',')}`

function bindingsOf(s: Substitution): Record<string, string> {
  return Object.fromEntries(
    [...s.entries()].map(([k, v]) => [k, termToString(v)]).sort(([a], [b]) => ((a as string) < (b as string) ? -1 : 1)),
  )
}

/** Tokens for the positive conditions of a rule, by joining facts one condition at a time (the naive matcher). */
function matchRule(rule: ProductionRule, ri: number, memory: readonly WorkingElement[]): Activation[] {
  const positives = rule.conditions.filter((c) => !isNot(c))
  const negatives = rule.conditions.filter(isNot)
  let tokens: { s: Substitution; facts: number[] }[] = [{ s: new Map(), facts: [] }]
  for (const c of positives) {
    const next: { s: Substitution; facts: number[] }[] = []
    for (const t of tokens) {
      for (const w of memory) {
        const m = matchTerm(applySubstitution(c, t.s), w.fact, t.s)
        if (m !== null) next.push({ s: m, facts: [...t.facts, w.id] })
      }
    }
    tokens = next
  }
  return tokens
    .filter((t) =>
      negatives.every(
        (n) =>
          !memory.some(
            (w) =>
              matchTerm(
                applySubstitution((n as Extract<Term, { kind: 'compound' }>).args[0] as Term, t.s),
                w.fact,
                t.s,
              ) !== null,
          ),
      ),
    )
    .map((t) => ({ rule: ri, facts: t.facts, bindings: bindingsOf(t.s) }))
}

/**
 * The Rete-lite matcher: for each rule, an alpha memory per positive condition (the facts matching its pattern) and a
 * beta memory per prefix of the conditions (the partial matches, with their bindings). Adding a fact extends the beta
 * memories through the joins it can join with; retracting removes every token that used it.
 */
export class ReteNetwork {
  readonly #rules: readonly ProductionRule[]
  readonly #alpha: Map<string, number[]>[] = []
  readonly #beta: { s: Substitution; facts: number[] }[][][] = []
  readonly #facts = new Map<number, Term>()
  /** Joins attempted since the network was built: the work incremental matching does. */
  joins = 0

  constructor(rules: readonly ProductionRule[]) {
    this.#rules = rules
    for (const r of rules) {
      const positives = r.conditions.filter((c) => !isNot(c))
      this.#alpha.push(new Map(positives.map((_, i) => [String(i), []])))
      this.#beta.push(positives.map(() => []))
    }
  }

  /** Facts in the network. */
  get size(): number {
    return this.#facts.size
  }

  /**
   * Add a fact, joining it into every rule's memories.
   *
   * @param id - The fact's identifier.
   * @param fact - The ground fact.
   */
  add(id: number, fact: Term): void {
    this.#facts.set(id, fact)
    this.#rules.forEach((r, ri) => {
      const positives = r.conditions.filter((c) => !isNot(c))
      positives.forEach((c, level) => {
        if (matchTerm(c, fact) === null) return
        this.#alpha[ri]?.get(String(level))?.push(id)
        // Right activation: join the new fact with the tokens of the previous level, then pass new tokens down.
        const left =
          level === 0 ? [{ s: new Map() as Substitution, facts: [] as number[] }] : (this.#beta[ri]?.[level - 1] ?? [])
        let fresh: { s: Substitution; facts: number[] }[] = []
        for (const t of left) {
          this.joins++
          const m = matchTerm(applySubstitution(c, t.s), fact, t.s)
          if (m !== null) fresh.push({ s: m, facts: [...t.facts, id] })
        }
        for (let l = level; l < positives.length; l++) {
          this.#beta[ri]?.[l]?.push(...fresh)
          if (l + 1 >= positives.length) break
          const next: { s: Substitution; facts: number[] }[] = []
          for (const t of fresh) {
            for (const wid of this.#alpha[ri]?.get(String(l + 1)) ?? []) {
              this.joins++
              if (wid === id && l + 1 === level) continue
              const m = matchTerm(applySubstitution(positives[l + 1] as Term, t.s), this.#facts.get(wid) as Term, t.s)
              if (m !== null) next.push({ s: m, facts: [...t.facts, wid] })
            }
          }
          fresh = next
        }
      })
    })
  }

  /**
   * Retract a fact: remove it from the alpha memories and every token that matched it.
   *
   * @param id - The fact's identifier.
   */
  remove(id: number): void {
    this.#facts.delete(id)
    for (const alpha of this.#alpha)
      for (const [k, ids] of alpha)
        alpha.set(
          k,
          ids.filter((x) => x !== id),
        )
    for (const levels of this.#beta) {
      for (let l = 0; l < levels.length; l++) levels[l] = (levels[l] ?? []).filter((t) => !t.facts.includes(id))
    }
  }

  /**
   * The activations of every rule: the complete tokens of each rule that pass its `not` conditions.
   *
   * @returns The activations, rule by rule.
   */
  activations(): Activation[] {
    const out: Activation[] = []
    this.#rules.forEach((r, ri) => {
      const positives = r.conditions.filter((c) => !isNot(c))
      const negatives = r.conditions.filter(isNot)
      const complete =
        positives.length === 0
          ? [{ s: new Map() as Substitution, facts: [] as number[] }]
          : (this.#beta[ri]?.[positives.length - 1] ?? [])
      for (const t of complete) {
        const blocked = negatives.some((n) =>
          [...this.#facts.values()].some(
            (f) =>
              matchTerm(applySubstitution((n as Extract<Term, { kind: 'compound' }>).args[0] as Term, t.s), f, t.s) !==
              null,
          ),
        )
        if (!blocked) out.push({ rule: ri, facts: t.facts, bindings: bindingsOf(t.s) })
      }
    })
    return out
  }
}

/** The state of a production system. */
export interface ProductionState {
  readonly memory: readonly WorkingElement[]
  readonly nextId: number
  /** Activations already fired (refraction), by key. */
  readonly fired: readonly string[]
  readonly cycle: number
  readonly conflictSize: number
  /** What fired in the last cycle. */
  readonly last: {
    readonly rule: string
    readonly bindings: Readonly<Record<string, string>>
    readonly added: readonly string[]
    readonly removed: readonly string[]
  } | null
  readonly halted: boolean
}

/** Options of the production system. */
export interface ProductionOptions {
  /** How the conflict set is resolved (default `recency`). */
  readonly strategy?: ConflictStrategy
  /** Match from scratch each cycle (`naive`) or incrementally (`rete`, the default). */
  readonly matcher?: 'rete' | 'naive'
  /** Stop after this many firings (default 1000). */
  readonly maxFirings?: number
}

const PRODUCTION_SCHEMA: TraceSchema = channels({
  metrics: {
    matched: { description: 'activations in the conflict set before firing' },
    facts: { description: 'facts in working memory' },
    fired: { description: 'rules fired so far' },
  },
  messages: {
    fire: {
      description: 'the rule fired with its bindings',
      payload: { rule: 'name', bindings: 'variable bindings', added: 'facts added', removed: 'facts removed' },
    },
  },
})

/**
 * Order the conflict set by a strategy and return the activation that fires. `order` takes the first rule in the rule
 * base; `priority` the highest priority (then rule order); `recency` the activation matching the most recent fact (ties
 * compared on their next most recent facts, as OPS5's LEX); `specificity` the rule with the most conditions (then recency).
 *
 * @param conflict - The activations not yet fired.
 * @param rules - The rule base.
 * @param strategy - The strategy.
 * @returns The chosen activation, or undefined for an empty conflict set.
 * @example
 * ```ts
 * import { productionRule, resolveConflict } from 'omai-engine/logic'
 * const rules = [productionRule('a', ['p']), productionRule('b', ['p', 'q'])]
 * resolveConflict([{ rule: 0, facts: [0], bindings: {} }, { rule: 1, facts: [0, 1], bindings: {} }], rules, 'specificity')?.rule // => 1
 * ```
 */
export function resolveConflict(
  conflict: readonly Activation[],
  rules: readonly ProductionRule[],
  strategy: ConflictStrategy,
): Activation | undefined {
  const recency = (a: Activation, b: Activation): number => {
    const x = [...a.facts].sort((p, q) => q - p)
    const y = [...b.facts].sort((p, q) => q - p)
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return (y[i] as number) - (x[i] as number)
    return y.length - x.length
  }
  const specificity = (a: Activation): number => rules[a.rule]?.conditions.length ?? 0
  // Ties are broken by the order of the matched facts, so that every matcher fires the same instantiation.
  const byFacts = (a: Activation, b: Activation): number => {
    for (let i = 0; i < Math.min(a.facts.length, b.facts.length); i++)
      if (a.facts[i] !== b.facts[i]) return (a.facts[i] as number) - (b.facts[i] as number)
    return a.facts.length - b.facts.length
  }
  const compare = (a: Activation, b: Activation): number => byFirst(a, b) || byFacts(a, b)
  const byFirst = (a: Activation, b: Activation): number => {
    if (strategy === 'priority')
      return (rules[b.rule]?.priority ?? 0) - (rules[a.rule]?.priority ?? 0) || a.rule - b.rule || recency(a, b)
    if (strategy === 'recency') return recency(a, b) || a.rule - b.rule
    if (strategy === 'specificity') return specificity(b) - specificity(a) || recency(a, b) || a.rule - b.rule
    return a.rule - b.rule || recency(a, b)
  }
  return [...conflict].sort(compare)[0]
}

/**
 * A production system as a traced algorithm: each step builds the conflict set, resolves it, fires one rule instantiation
 * (adding and removing facts) and records the rule fired, its bindings and the working memory. It ends at quiescence (an
 * empty conflict set), when a rule with `halt` fires, or after `maxFirings`.
 *
 * @param rules - The rule base.
 * @param facts - The initial working memory, as text (ground terms).
 * @param options - Strategy, matcher and firing limit.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a fact containing variables.
 * @example
 * ```ts
 * import { productionRule, productionSystemAlgorithm } from 'omai-engine/logic'
 * productionSystemAlgorithm([productionRule('r', ['p'], { add: ['q'] })], ['p']).name // => 'production-system'
 * ```
 */
export function productionSystemAlgorithm(
  rules: readonly ProductionRule[],
  facts: readonly string[],
  options: ProductionOptions = {},
): Algorithm<ProductionState> {
  const strategy = options.strategy ?? 'recency'
  const maxFirings = options.maxFirings ?? 1000
  const matcher = options.matcher ?? 'rete'
  const initial = facts.map((f, id) => {
    const fact = parseTerm(f)
    if (/(^|[^A-Za-z0-9_])[A-Z_]/.test(termToString(fact)))
      throw new OmaiError('value', `production system: '${f}' is not ground`)
    return { id, fact }
  })
  // The Rete network follows the state; it is rebuilt when the state is not the one it last produced (a seek or a fork).
  let network: ReteNetwork | null = null
  let networkIds = ''
  const reteFor = (memory: readonly WorkingElement[]): ReteNetwork => {
    const ids = memory.map((w) => w.id).join(',')
    if (network === null || ids !== networkIds) {
      network = new ReteNetwork(rules)
      for (const w of memory) network.add(w.id, w.fact)
      networkIds = ids
    }
    return network
  }
  const conflictOf = (s: ProductionState): Activation[] => {
    const all =
      matcher === 'rete' ? reteFor(s.memory).activations() : rules.flatMap((r, ri) => matchRule(r, ri, s.memory))
    const done = new Set(s.fired)
    return all.filter((a) => !done.has(activationKey(a)))
  }
  return {
    name: 'production-system',
    config: { strategy, matcher, maxFirings },
    schema: PRODUCTION_SCHEMA,
    init: () => ({
      memory: initial,
      nextId: initial.length,
      fired: [],
      cycle: 0,
      conflictSize: 0,
      last: null,
      halted: false,
    }),
    step: (s) => {
      const conflict = conflictOf(s)
      const chosen = resolveConflict(conflict, rules, strategy)
      if (chosen === undefined) return { ...s, conflictSize: 0, halted: true, last: null }
      const rule = rules[chosen.rule] as ProductionRule
      const subst = new Map<string, Term>()
      for (const [k, v] of Object.entries(chosen.bindings)) subst.set(k, parseTerm(v))
      let memory = [...s.memory]
      let nextId = s.nextId
      const removed: string[] = []
      for (const r of rule.remove) {
        const target = applySubstitution(r, subst)
        const hit = memory.find((w) => termEquals(w.fact, target))
        if (hit !== undefined) {
          memory = memory.filter((w) => w.id !== hit.id)
          removed.push(termToString(target))
          if (matcher === 'rete') {
            const net = reteFor(s.memory)
            net.remove(hit.id)
            networkIds = memory.map((w) => w.id).join(',')
          }
        }
      }
      const added: string[] = []
      for (const a of rule.add) {
        const fact = applySubstitution(a, subst)
        if (memory.some((w) => termEquals(w.fact, fact))) continue
        memory.push({ id: nextId, fact })
        added.push(termToString(fact))
        if (matcher === 'rete' && network !== null) {
          network.add(nextId, fact)
          networkIds = memory.map((w) => w.id).join(',')
        }
        nextId++
      }
      return {
        memory,
        nextId,
        fired: [...s.fired, activationKey(chosen)],
        cycle: s.cycle + 1,
        conflictSize: conflict.length,
        last: { rule: rule.name, bindings: chosen.bindings, added, removed },
        halted: rule.halt || s.cycle + 1 >= maxFirings,
      }
    },
    done: (s) => (s.halted ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { matched: s.conflictSize, facts: s.memory.length, fired: s.cycle },
      messages: s.last === null ? [] : [{ kind: 'fire', payload: s.last as unknown as Json }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as ProductionState,
  }
}

/** The outcome of running a production system to the end. */
export interface ProductionResult {
  /** The final working memory as text, in the order the facts were added. */
  readonly facts: readonly string[]
  /** The rules fired, in order, with their bindings. */
  readonly firings: readonly { readonly rule: string; readonly bindings: Readonly<Record<string, string>> }[]
}

/**
 * Run a production system to quiescence.
 *
 * @param rules - The rule base.
 * @param facts - The initial facts.
 * @param options - Strategy, matcher, firing limit and tracing.
 * @returns The final facts and the firing sequence.
 * @throws OmaiError with code `value` for a fact containing variables.
 * @example
 * ```ts
 * import { productionRule, runProduction } from 'omai-engine/logic'
 * const rules = [productionRule('gp', ['parent(X, Y)', 'parent(Y, Z)'], { add: ['grandparent(X, Z)'] })]
 * runProduction(rules, ['parent(a, b)', 'parent(b, c)']).facts.at(-1) // => 'grandparent(a,c)'
 * ```
 */
export function runProduction(
  rules: readonly ProductionRule[],
  facts: readonly string[],
  options: ProductionOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): ProductionResult {
  const { tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const firings: { rule: string; bindings: Readonly<Record<string, string>> }[] = []
  const alg = productionSystemAlgorithm(rules, facts, rest)
  const traced: Algorithm<ProductionState> = {
    ...alg,
    step: (s, ctx) => {
      const next = alg.step(s, ctx)
      if (next.last !== null) firings.push({ rule: next.last.rule, bindings: next.last.bindings })
      return next
    },
  }
  const { state } = runAlgorithm(traced, { ...run, maxSteps: (rest.maxFirings ?? 1000) + 5 })
  return { facts: state.memory.map((w) => termToString(w.fact)), firings }
}
