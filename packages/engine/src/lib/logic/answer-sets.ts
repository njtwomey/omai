/**
 * Answer set programming: logic programs with default negation (`not p`) whose meanings are their stable models, the
 * sets of atoms that justify themselves under the Gelfond-Lifschitz reduct. Reading and grounding normal programs with
 * integrity constraints, the reduct and the stable-model check, and a guess-and-check search for all answer sets as a
 * traced algorithm.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { evaluationOrder, factKey, joinRule, Store } from './_relations.ts'
import {
  checkSafety,
  type DatalogRule,
  evaluateDatalog,
  type Fact,
  logicProgramOperators,
  rulesFromClauses,
} from './datalog.ts'
import { parseProgram } from './parse.ts'
import { type Clause, compound, conjuncts, indicator, type Term } from './terms.ts'

/**
 * A normal logic program for answer set semantics: rules `head :- body` whose bodies may use default negation
 * (`not p`), and integrity constraints `:- body` (no answer set satisfies a constraint's body).
 *
 * @concept answer-set-programming
 */
export interface NormalProgram {
  readonly rules: readonly DatalogRule[]
  readonly constraints: readonly DatalogRule[]
}

/** A ground rule: head (null for a constraint), positive body atoms and negated body atoms, all as fact texts. */
export interface GroundRule {
  readonly head: string | null
  readonly positive: readonly string[]
  readonly negative: readonly string[]
}

const FALSE = '$false'

/**
 * Read a normal program in answer set syntax (`p :- q, not r.`, constraints `:- p, q.`); every rule must be safe.
 *
 * @param source - Program text, or clauses (constraints then have the head `$false`).
 * @returns The program.
 * @throws OmaiError with code `value` for unsafe rules, function symbols or a syntax error.
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { normalProgram } from 'omai-engine/logic'
 * const p = normalProgram('a :- not b. b :- not a. :- a, c. c.')
 * [p.rules.length, p.constraints.length] // => [3, 1]
 * ```
 */
export function normalProgram(source: string | readonly Clause[]): NormalProgram {
  let clauses: Clause[]
  if (typeof source === 'string') {
    const parsed = parseProgram(source, { operators: logicProgramOperators() })
    clauses = [...parsed.clauses, ...parsed.directives.map((d) => ({ head: compound(FALSE, []), body: conjuncts(d) }))]
  } else clauses = [...source]
  const all = rulesFromClauses(clauses, 'normalProgram')
  for (const r of all) checkSafety(r, 'normalProgram')
  const isConstraint = (r: DatalogRule): boolean => r.head.kind === 'atom' && r.head.name === FALSE
  return { rules: all.filter((r) => !isConstraint(r)), constraints: all.filter(isConstraint) }
}

/**
 * Ground a normal program: instantiate every rule over the atoms that could possibly be true (the least model of the
 * program with its negative literals dropped), evaluating comparisons and dropping negated atoms that cannot be true.
 *
 * @param p - The program.
 * @returns The ground rules (constraints have a null head).
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { groundProgram, normalProgram } from 'omai-engine/logic'
 * groundProgram(normalProgram('n(1). n(2). p(X) :- n(X), not q(X). q(2).')).map((r) => r.negative) // => [[], [], [], ['q(2)'], []]
 * ```
 */
export function groundProgram(p: NormalProgram): GroundRule[] {
  const positive: DatalogRule[] = [...p.rules, ...p.constraints].map((r) => ({
    head: r.head,
    body: r.body.filter((l) => l.kind !== 'neg'),
  }))
  const upper = evaluateDatalog({
    rules: positive,
    strata: [[...new Set(positive.map((r) => indicator(r.head)))].sort()],
    edb: [],
    idb: [],
  })
  const possible = new Store()
  for (const fs of upper.model.values()) for (const f of fs) possible.add(f, null)
  const out: GroundRule[] = []
  const seen = new Set<string>()
  for (const rule of [...p.rules, ...p.constraints]) {
    joinRule(
      rule,
      evaluationOrder(rule),
      (_, pred) => possible.get(pred),
      () => false,
      (b, premises, negated) => {
        const headTerm = rule.head.kind === 'atom' && rule.head.name === FALSE ? null : instantiateHead(rule.head, b)
        const g: GroundRule = {
          head: headTerm === null ? null : factKey(headTerm),
          positive: [...new Set(premises.map(factKey))],
          negative: [...new Set(negated.filter((a) => possible.has(a)).map(factKey))],
        }
        const key = JSON.stringify(g)
        if (seen.has(key)) return
        seen.add(key)
        out.push(g)
      },
    )
  }
  return out
}

function instantiateHead(t: Term, b: ReadonlyMap<string, Term>): Fact {
  if (t.kind === 'var') return b.get(t.name) ?? t
  if (t.kind !== 'compound') return t
  return { kind: 'compound', functor: t.functor, args: t.args.map((a) => instantiateHead(a, b)) }
}

/**
 * The Gelfond-Lifschitz reduct of a ground program by a set of atoms: rules with a negated atom in the set are
 * deleted, and the remaining negative literals dropped, leaving a definite program.
 *
 * @param ground - The ground program.
 * @param atoms - The candidate set (fact texts).
 * @returns The reduct (constraints are kept with their positive bodies).
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { groundProgram, normalProgram, reduct } from 'omai-engine/logic'
 * const g = groundProgram(normalProgram('a :- not b. b :- not a.'))
 * reduct(g, new Set(['a'])) // => [{ head: 'a', positive: [], negative: [] }]
 * ```
 */
export function reduct(ground: readonly GroundRule[], atoms: ReadonlySet<string>): GroundRule[] {
  return ground
    .filter((r) => !r.negative.some((a) => atoms.has(a)))
    .map((r) => ({ head: r.head, positive: r.positive, negative: [] }))
}

/** The least model of a definite ground program (constraints ignored), by forward chaining. */
function leastModelOf(rules: readonly GroundRule[]): Set<string> {
  const model = new Set<string>()
  for (let changed = true; changed; ) {
    changed = false
    for (const r of rules) {
      if (r.head === null || model.has(r.head)) continue
      if (r.positive.every((a) => model.has(a))) {
        model.add(r.head)
        changed = true
      }
    }
  }
  return model
}

function violates(ground: readonly GroundRule[], model: ReadonlySet<string>): boolean {
  return ground.some(
    (r) => r.head === null && r.positive.every((a) => model.has(a)) && !r.negative.some((a) => model.has(a)),
  )
}

/**
 * Whether a set of atoms is a stable model (answer set): it equals the least model of the program's reduct by it,
 * and satisfies every integrity constraint.
 *
 * @param p - The program.
 * @param atoms - The candidate (fact texts).
 * @returns True for an answer set.
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { isStableModel, normalProgram } from 'omai-engine/logic'
 * const p = normalProgram('a :- not b. b :- not a.')
 * [isStableModel(p, ['a']), isStableModel(p, ['a', 'b']), isStableModel(p, [])] // => [true, false, false]
 * ```
 */
export function isStableModel(p: NormalProgram, atoms: readonly string[]): boolean {
  const ground = groundProgram(p)
  const set = new Set(atoms)
  const lm = leastModelOf(reduct(ground, set))
  return lm.size === set.size && [...set].every((a) => lm.has(a)) && !violates(ground, lm)
}

/** Options of {@link stableModels}. */
export interface StableModelOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Stop after this many answer sets (default all). */
  readonly maxModels?: number
  /** Refuse programs with more than this many negated ground atoms to guess over (default 20). */
  readonly maxGuessAtoms?: number
}

/** The observable state of the guess-and-check search. */
export interface StableModelState {
  readonly status: 'running' | 'done'
  /** The negated ground atoms guessed over. */
  readonly atoms: readonly string[]
  readonly candidate: number
  readonly candidates: number
  readonly models: readonly (readonly string[])[]
  readonly last: {
    readonly guess: readonly string[]
    readonly model: readonly string[]
    readonly stable: boolean
  } | null
}

interface StableMachine extends StableModelState {
  status: 'running' | 'done'
  candidate: number
  models: string[][]
  last: { guess: string[]; model: string[]; stable: boolean } | null
  ground: GroundRule[]
}

const ASP_SCHEMA: TraceSchema = channels({
  metrics: {
    candidate: { description: 'guesses checked' },
    models: { description: 'answer sets found' },
  },
  messages: {
    check: {
      description: 'a guess for the negated atoms and the least model of the reduct it gives',
      payload: { guess: 'atoms assumed true', model: 'least model of the reduct', stable: 'whether they agree' },
    },
  },
})

function stableMachine(p: NormalProgram, options: StableModelOptions): Algorithm<StableMachine> {
  const maxModels = options.maxModels ?? Number.POSITIVE_INFINITY
  const limit = options.maxGuessAtoms ?? 20
  return {
    name: 'stable-models',
    config: { maxGuessAtoms: limit },
    schema: ASP_SCHEMA,
    init: () => {
      const ground = groundProgram(p)
      const atoms = [...new Set(ground.flatMap((r) => r.negative))].sort()
      if (atoms.length > limit) {
        throw new OmaiError('value', `stableModels: ${atoms.length} negated atoms exceed maxGuessAtoms ${limit}`)
      }
      return { status: 'running', atoms, candidate: 0, candidates: 2 ** atoms.length, models: [], last: null, ground }
    },
    step: (s) => {
      const c = s.candidate
      const guess = s.atoms.filter((_, i) => ((c >> i) & 1) === 1)
      const lm = leastModelOf(reduct(s.ground, new Set(guess)))
      const stable = s.atoms.every((a, i) => lm.has(a) === (((c >> i) & 1) === 1)) && !violates(s.ground, lm)
      const model = [...lm].sort()
      s.last = { guess, model, stable }
      if (stable) s.models.push(model)
      s.candidate++
      if (s.candidate >= s.candidates || s.models.length >= maxModels) s.status = 'done'
      return s
    },
    done: (s) => (s.status === 'done' ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { candidate: s.candidate, models: s.models.length },
      messages: s.last === null ? [] : [{ kind: 'check', payload: s.last as unknown as Json }],
    }),
  }
}

/**
 * Stable models by guess and check as a traced {@link Algorithm}: each step guesses which negated ground atoms are
 * true, computes the least model of the reduct, and keeps it when it agrees with the guess and satisfies the
 * constraints. Exponential in the number of negated atoms; meant for small programs and for comparing answer set
 * semantics with Datalog's (a stratified program has exactly one stable model, its perfect model).
 *
 * @param p - The program.
 * @param options - Answer sets wanted and the guess limit.
 * @returns The algorithm.
 * @throws OmaiError with code `value` (on init) when there are more negated atoms than `maxGuessAtoms`.
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { normalProgram, stableModelsAlgorithm } from 'omai-engine/logic'
 * stableModelsAlgorithm(normalProgram('a.')).name // => 'stable-models'
 * ```
 */
export function stableModelsAlgorithm(p: NormalProgram, options: StableModelOptions = {}): Algorithm<StableModelState> {
  return stableMachine(p, options)
}

/** The outcome of {@link stableModels}. */
export interface StableModelsResult {
  /** The answer sets, each sorted, in order of the sorted atom lists. */
  readonly models: readonly (readonly string[])[]
  readonly candidates: number
  readonly ground: readonly GroundRule[]
}

/**
 * The answer sets (stable models) of a normal program.
 *
 * @param program - A program or its text.
 * @param options - Answer sets wanted, the guess limit and tracing.
 * @returns The answer sets, the number of guesses checked and the ground program.
 * @throws OmaiError with code `value` for an invalid program or too many negated atoms.
 * @concept answer-set-programming
 * @example
 * ```ts
 * import { stableModels } from 'omai-engine/logic'
 * stableModels('a :- not b. b :- not a.').models // => [['a'], ['b']]
 * stableModels('p :- not p.').models // => []
 * // Three-colouring a triangle: 6 answer sets.
 * const colours = 'node(1). node(2). node(3). edge(1,2). edge(2,3). edge(1,3). red(X) :- node(X), not green(X), not blue(X). green(X) :- node(X), not red(X), not blue(X). blue(X) :- node(X), not red(X), not green(X). :- edge(X,Y), red(X), red(Y). :- edge(X,Y), green(X), green(Y). :- edge(X,Y), blue(X), blue(Y).'
 * stableModels(colours).models.length // => 6
 * ```
 */
export function stableModels(program: NormalProgram | string, options: StableModelOptions = {}): StableModelsResult {
  const p = typeof program === 'string' ? normalProgram(program) : program
  const { maxModels: _m, maxGuessAtoms: _g, ...run } = options
  const { state } = runAlgorithm(stableMachine(p, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  const key = (m: readonly string[]): string => m.join('\u0000')
  const models = [...state.models].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0))
  return { models, candidates: state.candidate, ground: state.ground }
}

/**
 * Write a ground rule.
 *
 * @param r - The rule.
 * @returns Its text.
 * @example
 * ```ts
 * import { groundRuleToString } from 'omai-engine/logic'
 * groundRuleToString({ head: 'a', positive: ['b'], negative: ['c'] }) // => 'a :- b, not c.'
 * ```
 */
export function groundRuleToString(r: GroundRule): string {
  const body = [...r.positive, ...r.negative.map((a) => `not ${a}`)]
  const head = r.head ?? ''
  return body.length === 0 ? `${head}.` : `${head}${head === '' ? '' : ' '}:- ${body.join(', ')}.`
}
