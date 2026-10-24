// Relations and joins shared by bottom-up Datalog evaluation and answer-set grounding: a fact store, matching of atoms
// against facts, rule bodies evaluated left to right with negations and comparisons as soon as they are ground.
import { WhiteboxError } from '../base/index.ts'
import type { DatalogLiteral, DatalogRule, Derivation, Fact } from './datalog.ts'
import { compareTerms, indicator, type Term, termToString, termVariables } from './terms.ts'

/** The text of a fact, its identity in a store. */
export const factKey = (f: Fact): string => termToString(f)

/** Ground facts by predicate, in insertion order, with the first derivation of each derived fact. */
export class Store {
  readonly byPred: Map<string, Fact[]> = new Map()
  readonly keys: Set<string> = new Set()
  readonly derivations: Map<string, Derivation> = new Map()

  has(f: Fact): boolean {
    return this.keys.has(factKey(f))
  }

  add(f: Fact, d: Derivation | null): boolean {
    const k = factKey(f)
    if (this.keys.has(k)) return false
    this.keys.add(k)
    const p = indicator(f)
    const rel = this.byPred.get(p)
    if (rel === undefined) this.byPred.set(p, [f])
    else rel.push(f)
    if (d !== null) this.derivations.set(k, d)
    return true
  }

  get(p: string): readonly Fact[] {
    return this.byPred.get(p) ?? []
  }

  get size(): number {
    return this.keys.size
  }
}

/** Variable bindings during a join. */
export type Bindings = Map<string, Term>

/** Match an atom against a ground fact, extending bindings (anonymous variables match anything). */
export function match(pattern: Term, fact: Fact, b: Bindings): Bindings | null {
  if (pattern.kind === 'atom') return fact.kind === 'atom' && fact.name === pattern.name ? b : null
  if (pattern.kind !== 'compound' || fact.kind !== 'compound') return null
  if (fact.functor !== pattern.functor || fact.args.length !== pattern.args.length) return null
  let out: Bindings | null = null
  for (let i = 0; i < pattern.args.length; i++) {
    const p = pattern.args[i] as Term
    const x = fact.args[i] as Term
    if (p.kind === 'var') {
      if (p.name === '_' || p.name.startsWith('_G')) continue
      const cur = (out ?? b).get(p.name)
      if (cur === undefined) {
        out ??= new Map(b)
        out.set(p.name, x)
      } else if (compareTerms(cur, x) !== 0) return null
    } else if (compareTerms(p, x) !== 0) return null
  }
  return out ?? b
}

/** Apply bindings to a term. */
export function instantiate(t: Term, b: Bindings): Term {
  if (t.kind === 'var') return b.get(t.name) ?? t
  if (t.kind !== 'compound') return t
  return { kind: 'compound', functor: t.functor, args: t.args.map((a) => instantiate(a, b)) }
}

/** Evaluate a built-in comparison of ground terms. */
export function compare(op: string, x: Term, y: Term): boolean {
  if (op === '=' || op === '==') return compareTerms(x, y) === 0
  if (op === '\\=' || op === '\\==') return compareTerms(x, y) !== 0
  if (op.startsWith('@')) {
    const c = compareTerms(x, y)
    return op === '@<' ? c < 0 : op === '@>' ? c > 0 : op === '@=<' ? c <= 0 : c >= 0
  }
  if (x.kind !== 'num' || y.kind !== 'num') {
    throw new WhiteboxError('value', `datalog: ${op} compares numbers, got ${termToString(x)} and ${termToString(y)}`)
  }
  return op === '<'
    ? x.value < y.value
    : op === '>'
      ? x.value > y.value
      : op === '=<'
        ? x.value <= y.value
        : x.value >= y.value
}

/**
 * The order in which a rule's body literals are evaluated: positive literals as written, each negation or comparison
 * as soon as its variables are bound.
 */
export function evaluationOrder(rule: DatalogRule): number[] {
  const order: number[] = []
  const bound = new Set<string>()
  const pending = rule.body.map((l, i) => ({ l, i })).filter(({ l }) => l.kind !== 'pos')
  const ready = (l: DatalogLiteral): boolean =>
    termVariables(l.kind === 'neg' ? l.atom : l.kind === 'cmp' ? [l.left, l.right] : l.atom).every(
      (v) => bound.has(v.name) || v.name.startsWith('_'),
    )
  const flush = (): void => {
    for (let k = 0; k < pending.length; ) {
      const x = pending[k] as { l: DatalogLiteral; i: number }
      if (ready(x.l)) {
        order.push(x.i)
        pending.splice(k, 1)
      } else k++
    }
  }
  flush()
  rule.body.forEach((l, i) => {
    if (l.kind !== 'pos') return
    order.push(i)
    for (const v of termVariables(l.atom)) bound.add(v.name)
    flush()
  })
  for (const x of pending) order.push(x.i)
  return order
}

/**
 * Evaluate a rule body with a chosen source relation per positive literal; `holds` decides negated atoms (a negation
 * succeeds when its atom does not hold). `emit` receives each solution's bindings, positive premises and negated atoms.
 */
export function joinRule(
  rule: DatalogRule,
  order: readonly number[],
  source: (literal: number, pred: string) => readonly Fact[],
  holds: (f: Fact) => boolean,
  emit: (b: Bindings, premises: readonly Fact[], negated: readonly Fact[]) => void,
): void {
  const premises: Fact[] = []
  const negated: Fact[] = []
  const go = (k: number, b: Bindings): void => {
    if (k === order.length) {
      emit(b, premises, negated)
      return
    }
    const i = order[k] as number
    const l = rule.body[i] as DatalogLiteral
    if (l.kind === 'pos') {
      for (const f of source(i, indicator(l.atom))) {
        const nb = match(l.atom, f, b)
        if (nb === null) continue
        premises.push(f)
        go(k + 1, nb)
        premises.pop()
      }
    } else if (l.kind === 'neg') {
      const a = instantiate(l.atom, b)
      if (holds(a)) return
      negated.push(a)
      go(k + 1, b)
      negated.pop()
    } else if (compare(l.op, instantiate(l.left, b), instantiate(l.right, b))) go(k + 1, b)
  }
  go(0, new Map())
}
