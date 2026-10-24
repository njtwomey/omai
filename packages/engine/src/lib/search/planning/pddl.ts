import { WhiteboxError } from '../../base/index.ts'
import { type StripsAction, type StripsProblem, strips } from './strips.ts'

/** An s-expression: a symbol or a list. */
export type SExpr = string | readonly SExpr[]

/**
 * Parse s-expressions (PDDL's syntax): parentheses, symbols (lower-cased) and `;` comments.
 *
 * @param text - The source.
 * @returns The top-level expressions.
 * @throws WhiteboxError with code `value` on unbalanced parentheses.
 * @concept classical-planning
 * @example
 * ```ts
 * import { parseSExpr } from 'whitebox-engine/search'
 * parseSExpr('(define (domain d) ; comment\n (:requirements :strips))') // => [['define', ['domain', 'd'], [':requirements', ':strips']]]
 * ```
 */
export function parseSExpr(text: string): SExpr[] {
  const tokens = text
    .replace(/;[^\n]*/g, ' ')
    .replace(/([()])/g, ' $1 ')
    .split(/\s+/)
    .filter((t) => t.length > 0)
  const stack: SExpr[][] = [[]]
  for (const t of tokens) {
    if (t === '(') stack.push([])
    else if (t === ')') {
      const done = stack.pop()
      const parent = stack[stack.length - 1]
      if (done === undefined || parent === undefined) throw new WhiteboxError('value', 'parseSExpr: unbalanced ")"')
      parent.push(done)
    } else (stack[stack.length - 1] as SExpr[]).push(t.toLowerCase())
  }
  if (stack.length !== 1) throw new WhiteboxError('value', 'parseSExpr: unbalanced "("')
  return stack[0] as SExpr[]
}

/** A typed name: a variable or object with its type. */
export interface Typed {
  readonly name: string
  readonly type: string
}

/** An atom: a predicate applied to terms (variables start with `?`). */
export interface Atom {
  readonly predicate: string
  readonly args: readonly string[]
}

/** A literal of a precondition: an atom, or an equality, possibly negated. */
export interface Literal {
  readonly atom: Atom | { readonly predicate: '='; readonly args: readonly [string, string] }
  readonly positive: boolean
}

/** A PDDL action schema of the STRIPS subset. */
export interface ActionSchema {
  readonly name: string
  readonly parameters: readonly Typed[]
  readonly precondition: readonly Literal[]
  readonly add: readonly Atom[]
  readonly del: readonly Atom[]
}

/** A PDDL domain of the STRIPS subset (`:strips`, `:typing`, `:equality`). */
export interface PddlDomain {
  readonly name: string
  /** Each type with its parent type (`object` is the root). */
  readonly types: ReadonlyMap<string, string>
  readonly constants: readonly Typed[]
  readonly actions: readonly ActionSchema[]
}

/** A PDDL problem: objects, the initial atoms and a conjunctive goal. */
export interface PddlProblem {
  readonly name: string
  readonly domain: string
  readonly objects: readonly Typed[]
  readonly init: readonly Atom[]
  readonly goal: readonly Atom[]
}

const SUPPORTED = new Set([':strips', ':typing', ':equality'])

function list(x: SExpr | undefined, what: string): readonly SExpr[] {
  if (!Array.isArray(x)) throw new WhiteboxError('value', `pddl: expected a list for ${what}`)
  return x as readonly SExpr[]
}

function sym(x: SExpr | undefined, what: string): string {
  if (typeof x !== 'string') throw new WhiteboxError('value', `pddl: expected a symbol for ${what}`)
  return x
}

function typedList(items: readonly SExpr[]): Typed[] {
  const out: Typed[] = []
  let pending: string[] = []
  for (let i = 0; i < items.length; i++) {
    const t = sym(items[i], 'a typed list')
    if (t === '-') {
      const type = items[i + 1]
      if (Array.isArray(type)) throw new WhiteboxError('not-implemented', 'pddl: either types are not supported')
      for (const name of pending) out.push({ name, type: sym(type, 'a type') })
      pending = []
      i += 1
    } else pending.push(t)
  }
  for (const name of pending) out.push({ name, type: 'object' })
  return out
}

function atom(x: SExpr | undefined): Atom {
  const l = list(x, 'an atom')
  return { predicate: sym(l[0], 'a predicate'), args: l.slice(1).map((a) => sym(a, 'a term')) }
}

function conjunction(x: SExpr | undefined): SExpr[] {
  if (x === undefined) return []
  const l = list(x, 'a formula')
  if (l.length === 0) return []
  return l[0] === 'and' ? l.slice(1).map((y) => y) : [x]
}

function literal(x: SExpr): Literal {
  const l = list(x, 'a literal')
  if (l[0] === 'not') {
    const inner = atom(l[1])
    if (inner.predicate !== '=') {
      throw new WhiteboxError('not-implemented', 'pddl: negative preconditions are outside the STRIPS subset')
    }
    return { atom: { predicate: '=', args: [inner.args[0] ?? '', inner.args[1] ?? ''] }, positive: false }
  }
  const head = l[0]
  if (head === 'or' || head === 'forall' || head === 'exists' || head === 'imply' || head === 'when') {
    throw new WhiteboxError('not-implemented', `pddl: ${head} is outside the STRIPS subset`)
  }
  return { atom: atom(x), positive: true }
}

function section(def: readonly SExpr[], key: string): readonly SExpr[] | undefined {
  return def.find((d) => Array.isArray(d) && d[0] === key) as readonly SExpr[] | undefined
}

/**
 * Parse a PDDL domain in the STRIPS subset: `:requirements` among `:strips`, `:typing` and `:equality`, `:types`,
 * `:constants`, `:predicates` and actions with conjunctive preconditions (atoms and (in)equalities) and add and
 * delete effects.
 *
 * @param text - The domain source.
 * @returns The domain.
 * @throws WhiteboxError with code `not-implemented` for requirements or formulas outside the subset, or `value` for
 * malformed input.
 * @concept classical-planning
 * @example
 * ```ts
 * import { parseDomain } from 'whitebox-engine/search'
 * const d = parseDomain('(define (domain move) (:predicates (at ?x)) (:action go :parameters (?a ?b) :precondition (at ?a) :effect (and (at ?b) (not (at ?a)))))')
 * [d.name, d.actions[0]?.add[0]?.predicate] // => ['move', 'at']
 * ```
 */
export function parseDomain(text: string): PddlDomain {
  const def = list(parseSExpr(text)[0], 'define')
  if (def[0] !== 'define') throw new WhiteboxError('value', 'parseDomain: expected (define ...)')
  const name = sym(list(def[1], 'the domain name')[1], 'the domain name')
  for (const r of section(def, ':requirements')?.slice(1) ?? []) {
    if (!SUPPORTED.has(sym(r, 'a requirement'))) {
      throw new WhiteboxError('not-implemented', `parseDomain: requirement ${String(r)} is outside the STRIPS subset`)
    }
  }
  const types = new Map<string, string>([['object', 'object']])
  for (const t of typedList(section(def, ':types')?.slice(1) ?? [])) types.set(t.name, t.type)
  const constants = typedList(section(def, ':constants')?.slice(1) ?? [])
  const actions = def
    .filter((d) => Array.isArray(d) && d[0] === ':action')
    .map((d) => {
      const a = d as readonly SExpr[]
      const field = (k: string): SExpr | undefined => {
        const i = a.indexOf(k)
        return i < 0 ? undefined : a[i + 1]
      }
      const effects = conjunction(field(':effect'))
      return {
        name: sym(a[1], 'an action name'),
        parameters: typedList(list(field(':parameters') ?? [], 'parameters')),
        precondition: conjunction(field(':precondition')).map(literal),
        add: effects.filter((e) => list(e, 'an effect')[0] !== 'not').map(atom),
        del: effects.filter((e) => list(e, 'an effect')[0] === 'not').map((e) => atom(list(e, 'an effect')[1])),
      }
    })
  return { name, types, constants, actions }
}

/**
 * Parse a PDDL problem: `:objects` (typed), `:init` atoms and a conjunctive `:goal`.
 *
 * @param text - The problem source.
 * @returns The problem.
 * @throws WhiteboxError with code `value` for malformed input, or `not-implemented` for a goal outside the subset.
 * @concept classical-planning
 * @example
 * ```ts
 * import { parseProblem } from 'whitebox-engine/search'
 * const p = parseProblem('(define (problem p) (:domain move) (:objects home shop) (:init (at home)) (:goal (at shop)))')
 * [p.objects.length, p.goal[0]?.args] // => [2, ['shop']]
 * ```
 */
export function parseProblem(text: string): PddlProblem {
  const def = list(parseSExpr(text)[0], 'define')
  if (def[0] !== 'define') throw new WhiteboxError('value', 'parseProblem: expected (define ...)')
  const goal = conjunction(section(def, ':goal')?.[1]).map(literal)
  if (goal.some((g) => !g.positive || g.atom.predicate === '=')) {
    throw new WhiteboxError('not-implemented', 'parseProblem: goals must be conjunctions of atoms')
  }
  return {
    name: sym(list(def[1], 'the problem name')[1], 'the problem name'),
    domain: sym(section(def, ':domain')?.[1], 'the domain'),
    objects: typedList(section(def, ':objects')?.slice(1) ?? []),
    init: (section(def, ':init')?.slice(1) ?? []).map(atom),
    goal: goal.map((g) => g.atom as Atom),
  }
}

/**
 * A ground atom as a fact string in pyperplan's lisp notation.
 *
 * @param predicate - The predicate.
 * @param args - The objects.
 * @returns The fact, such as `(on a b)`.
 * @example
 * ```ts
 * import { fact } from 'whitebox-engine/search'
 * [fact('on', ['a', 'b']), fact('handempty', [])] // => ['(on a b)', '(handempty)']
 * ```
 */
export function fact(predicate: string, args: readonly string[]): string {
  return `(${[predicate, ...args].join(' ')})`
}

/**
 * Ground a PDDL domain and problem into a STRIPS problem: instantiate each action schema with objects of the
 * parameters' types, keep the instances whose static preconditions (on predicates no action changes) and
 * (in)equalities hold, and keep only actions reachable from the initial state ignoring deletes. Facts are strings such
 * as `(on a b)`, action names `(stack a b)`.
 *
 * @param domain - The domain.
 * @param problem - The problem.
 * @returns The STRIPS problem.
 * @throws WhiteboxError with code `value` when the problem names another domain or a type is unknown.
 * @concept classical-planning
 * @example
 * ```ts
 * import { ground, parseDomain, parseProblem } from 'whitebox-engine/search'
 * const d = parseDomain('(define (domain move) (:predicates (at ?x)) (:action go :parameters (?a ?b) :precondition (and (at ?a) (not (= ?a ?b))) :effect (and (at ?b) (not (at ?a)))))')
 * const p = parseProblem('(define (problem p) (:domain move) (:objects home shop) (:init (at home)) (:goal (at shop)))')
 * ground(d, p).actions.map((a) => a.name) // => ['(go home shop)', '(go shop home)']
 * ```
 */
export function ground(domain: PddlDomain, problem: PddlProblem): StripsProblem {
  if (problem.domain !== domain.name) {
    throw new WhiteboxError('value', `ground: the problem is for domain ${problem.domain}, not ${domain.name}`)
  }
  const objects = [...domain.constants, ...problem.objects]
  for (const t of [...objects, ...domain.actions.flatMap((a) => a.parameters)]) {
    if (!domain.types.has(t.type)) throw new WhiteboxError('value', `ground: unknown type ${t.type} of ${t.name}`)
  }
  const isA = (type: string, want: string): boolean => {
    for (let t = type, guard = 0; guard < 100; guard++) {
      if (t === want) return true
      const parent = domain.types.get(t)
      if (parent === undefined) throw new WhiteboxError('value', `ground: unknown type ${t}`)
      if (parent === t) return want === 'object'
      t = parent
    }
    return false
  }
  const ofType = (type: string): string[] => objects.filter((o) => isA(o.type, type)).map((o) => o.name)
  const changing = new Set(domain.actions.flatMap((a) => [...a.add, ...a.del].map((x) => x.predicate)))
  const init = new Set(problem.init.map((a) => fact(a.predicate, a.args)))
  const actions: StripsAction[] = []
  for (const schema of domain.actions) {
    const domains = schema.parameters.map((p) => ofType(p.type))
    const binding = new Map<string, string>()
    const value = (t: string): string => (t.startsWith('?') ? (binding.get(t) ?? t) : t)
    const instantiate = (i: number): void => {
      if (i === schema.parameters.length) {
        const pre: string[] = []
        for (const l of schema.precondition) {
          const args = l.atom.args.map(value)
          if (l.atom.predicate === '=') {
            if ((args[0] === args[1]) !== l.positive) return
            continue
          }
          const f = fact(l.atom.predicate, args)
          if (!changing.has(l.atom.predicate)) {
            if (!init.has(f)) return
            continue
          }
          pre.push(f)
        }
        const args = schema.parameters.map((p) => binding.get(p.name) ?? '')
        actions.push({
          name: fact(schema.name, args),
          pre,
          add: schema.add.map((a) => fact(a.predicate, a.args.map(value))),
          del: schema.del.map((a) => fact(a.predicate, a.args.map(value))),
        })
        return
      }
      const p = schema.parameters[i] as Typed
      for (const o of domains[i] ?? []) {
        binding.set(p.name, o)
        instantiate(i + 1)
      }
      binding.delete(p.name)
    }
    instantiate(0)
  }
  // Keep the actions reachable from the initial state when deletes are ignored.
  const reached = new Set(init)
  const keep = new Set<number>()
  for (let changed = true; changed; ) {
    changed = false
    actions.forEach((a, i) => {
      if (keep.has(i) || !a.pre.every((f) => reached.has(f))) return
      keep.add(i)
      for (const f of a.add) reached.add(f)
      changed = true
    })
  }
  const dynamic = (f: string): boolean => changing.has(f.slice(1, f.indexOf(' ') > 0 ? f.indexOf(' ') : -1))
  return strips(
    [...init].filter(dynamic),
    problem.goal.map((g) => fact(g.predicate, g.args)),
    actions.filter((_, i) => keep.has(i)),
  )
}

/**
 * Parse and ground a PDDL domain and problem in one call.
 *
 * @param domainText - The domain source.
 * @param problemText - The problem source.
 * @returns The STRIPS problem.
 * @throws WhiteboxError as {@link parseDomain}, {@link parseProblem} and {@link ground} do.
 * @concept classical-planning
 * @example
 * ```ts
 * import { pddl } from 'whitebox-engine/search'
 * const p = pddl('(define (domain move) (:action go :parameters (?a ?b) :precondition (at ?a) :effect (and (at ?b) (not (at ?a)))))', '(define (problem p) (:domain move) (:objects home shop) (:init (at home)) (:goal (at shop)))')
 * p.goal // => ['(at shop)']
 * ```
 */
export function pddl(domainText: string, problemText: string): StripsProblem {
  return ground(parseDomain(domainText), parseProblem(problemText))
}
