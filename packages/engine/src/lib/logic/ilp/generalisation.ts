import { WhiteboxError } from '../../base/index.ts'
import { type FolClause, type FolLiteral, subsumes } from '../first-order.ts'
import {
  type Clause,
  compareTerms,
  compound,
  indicator,
  type Term,
  termToString,
  termVariables,
  variable,
} from '../terms.ts'
import { renameTerm, type Substitution } from '../unify.ts'

/** A definite clause as literals: the head positive, the body atoms negative. */
function asLiterals(c: Clause): FolClause {
  return [{ positive: true, atom: c.head }, ...c.body.map((atom): FolLiteral => ({ positive: false, atom }))]
}

/**
 * Theta-subsumption of definite clauses (Plotkin): whether some substitution maps `c`'s head onto `d`'s head and each
 * of `c`'s body atoms onto a body atom of `d`. It is the generality order of ILP: if `c` subsumes `d` then `c` entails
 * `d`.
 *
 * @param c - The general clause.
 * @param d - The specific clause.
 * @returns The substitution, or null.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { thetaSubsumes } from 'whitebox-engine/logic/ilp'
 * import { parseClause } from 'whitebox-engine/logic'
 * thetaSubsumes(parseClause('p(X) :- q(X, Y).'), parseClause('p(a) :- q(a, b), r(b).')) === null // => false
 * thetaSubsumes(parseClause('p(X) :- q(X, X).'), parseClause('p(a) :- q(a, b).')) // => null
 * ```
 */
export function thetaSubsumes(c: Clause, d: Clause): Substitution | null {
  // Treat d's variables as constants by renaming them apart (matching never binds them).
  const dd: Clause = {
    head: renameTerm(d.head, (n) => `${n}$d`),
    body: d.body.map((b) => renameTerm(b, (n) => `${n}$d`)),
  }
  return subsumes(asLiterals(c), asLiterals(dd))
}

/**
 * Whether two clauses subsume each other (are equivalent under theta-subsumption).
 *
 * @param c - A clause.
 * @param d - Another.
 * @returns True when equivalent.
 * @example
 * ```ts
 * import { isThetaEquivalent } from 'whitebox-engine/logic/ilp'
 * import { parseClause } from 'whitebox-engine/logic'
 * isThetaEquivalent(parseClause('p(X) :- q(X, Y), q(X, Z).'), parseClause('p(A) :- q(A, B).')) // => true
 * ```
 */
export function isThetaEquivalent(c: Clause, d: Clause): boolean {
  return thetaSubsumes(c, d) !== null && thetaSubsumes(d, c) !== null
}

/** Shared table of anti-unification: each pair of differing terms maps to one variable. */
class LggTable {
  private readonly table = new Map<string, Term>()
  private next = 0

  constructor(private readonly prefix: string) {}

  of(a: Term, b: Term): Term {
    if (compareTerms(a, b) === 0) return a
    if (a.kind === 'compound' && b.kind === 'compound' && a.functor === b.functor && a.args.length === b.args.length) {
      return compound(
        a.functor,
        a.args.map((x, i) => this.of(x, b.args[i] as Term)),
      )
    }
    const key = `${termToString(a)}\u0000${termToString(b)}`
    let v = this.table.get(key)
    if (v === undefined) {
      v = variable(`${this.prefix}${this.next++}`)
      this.table.set(key, v)
    }
    return v
  }
}

/**
 * The least general generalisation of two terms (Plotkin's anti-unification): the most specific term of which both
 * are instances. Equal subterm pairs share one variable.
 *
 * @param a - A term.
 * @param b - Another.
 * @returns The generalisation (variables `G0`, `G1`, ...).
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { lgg } from 'whitebox-engine/logic/ilp'
 * import { parseTerm, termToString } from 'whitebox-engine/logic'
 * termToString(lgg(parseTerm('p(f(a), a, g(b))'), parseTerm('p(f(b), b, g(c))'))) // => 'p(f(G0),G0,g(G1))'
 * ```
 */
export function lgg(a: Term, b: Term): Term {
  return new LggTable('G').of(a, b)
}

/**
 * The least general generalisation of two definite clauses under theta-subsumption: the lgg of the heads (which must
 * share a predicate) and the lggs of every pair of body atoms with the same predicate, all with one shared variable
 * table. The result can be large; {@link reduceClause} removes redundant literals.
 *
 * @param c - A clause.
 * @param d - Another.
 * @returns The generalisation.
 * @throws WhiteboxError with code `value` when the heads have different predicates.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { lggClauses } from 'whitebox-engine/logic/ilp'
 * import { clauseToString, parseClause } from 'whitebox-engine/logic'
 * const c = lggClauses(parseClause('likes(ann, X) :- red(X), round(X).'), parseClause('likes(bob, Y) :- red(Y), soft(Y).'))
 * clauseToString(c) // => 'likes(G0,G1) :- red(G1).'
 * ```
 */
export function lggClauses(c: Clause, d: Clause): Clause {
  if (indicator(c.head) !== indicator(d.head)) {
    throw new WhiteboxError('value', `lggClauses: heads ${indicator(c.head)} and ${indicator(d.head)} differ`)
  }
  // Rename apart so that variables of the two clauses never coincide.
  const cc = renameClauseVars(c, 'L')
  const dd = renameClauseVars(d, 'R')
  const t = new LggTable('G')
  const head = t.of(cc.head, dd.head)
  const body: Term[] = []
  const seen = new Set<string>()
  for (const x of cc.body) {
    for (const y of dd.body) {
      if (indicator(x) !== indicator(y)) continue
      const g = t.of(x, y)
      const k = termToString(g)
      if (seen.has(k)) continue
      seen.add(k)
      body.push(g)
    }
  }
  return renumber({ head, body })
}

function renameClauseVars(c: Clause, prefix: string): Clause {
  return {
    head: renameTerm(c.head, (n) => `${prefix}${n}`),
    body: c.body.map((b) => renameTerm(b, (n) => `${prefix}${n}`)),
  }
}

/** Rename a clause's variables `G0`, `G1`, ... in order of first occurrence (head first). */
function renumber(c: Clause, prefix = 'G'): Clause {
  const names = new Map<string, string>()
  for (const v of termVariables([c.head, ...c.body])) names.set(v.name, `${prefix}${names.size}`)
  const r = (n: string): string => names.get(n) ?? n
  return { head: renameTerm(c.head, r), body: c.body.map((b) => renameTerm(b, r)) }
}

/**
 * Plotkin's clause reduction: repeatedly delete a body literal whose removal leaves an equivalent clause (the clause
 * still subsumes what remains), until none can go. The result is the reduced (smallest equivalent) clause.
 *
 * @param c - The clause.
 * @returns The reduced clause.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { reduceClause } from 'whitebox-engine/logic/ilp'
 * import { clauseToString, parseClause } from 'whitebox-engine/logic'
 * clauseToString(reduceClause(parseClause('p(X) :- q(X, Y), q(X, Z), r(Z), q(X, a).'))) // => 'p(X) :- q(X,Z), r(Z), q(X,a).'
 * ```
 */
export function reduceClause(c: Clause): Clause {
  let cur = c
  for (let changed = true; changed; ) {
    changed = false
    for (let i = 0; i < cur.body.length; i++) {
      const smaller: Clause = { head: cur.head, body: cur.body.filter((_, j) => j !== i) }
      if (thetaSubsumes(cur, smaller) !== null) {
        cur = smaller
        changed = true
        break
      }
    }
  }
  return cur
}

/** Options of {@link rlgg}. */
export interface RlggOptions {
  /** Keep only body literals linked to the head through shared variables (default true, as Golem does). */
  readonly connected?: boolean
  /** Reduce the result (default true). */
  readonly reduce?: boolean
  /** Refuse to build clauses with more body literals than this before reduction (default 20000). */
  readonly maxLiterals?: number
}

/**
 * The relative least general generalisation of ground examples with respect to ground background knowledge
 * (Plotkin, as used by Golem): the lgg of the clauses `e_i :- K` where `K` is the whole background, then (by default)
 * restricted to literals connected to the head and reduced.
 *
 * @param examples - Ground positive examples of one predicate.
 * @param background - Ground background facts.
 * @param options - Connectedness filtering, reduction and a size limit.
 * @returns The generalisation.
 * @throws WhiteboxError with code `value` for no examples, different predicates, or a clause over the size limit.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { rlgg } from 'whitebox-engine/logic/ilp'
 * import { clauseToString, parseTerm } from 'whitebox-engine/logic'
 * // Lavrač and Džeroski's daughter example.
 * const bk = ['parent(ann, mary)', 'parent(ann, tom)', 'parent(tom, eve)', 'parent(tom, ian)', 'female(ann)', 'female(mary)', 'female(eve)'].map((s) => parseTerm(s))
 * clauseToString(rlgg([parseTerm('daughter(mary, ann)'), parseTerm('daughter(eve, tom)')], bk)) // => 'daughter(G0,G1) :- parent(G1,G0), female(G0).'
 * ```
 */
export function rlgg(examples: readonly Term[], background: readonly Term[], options: RlggOptions = {}): Clause {
  const first = examples[0]
  if (first === undefined) throw new WhiteboxError('value', 'rlgg: no examples')
  const maxLiterals = options.maxLiterals ?? 20_000
  let acc: Clause = { head: first, body: [...background] }
  for (const e of examples.slice(1)) {
    const next = lggClauses(acc, { head: e, body: [...background] })
    if (next.body.length > maxLiterals) {
      throw new WhiteboxError('value', `rlgg: ${next.body.length} body literals exceed maxLiterals ${maxLiterals}`)
    }
    acc = next
  }
  if (options.connected ?? true) acc = headConnected(acc)
  if (options.reduce ?? true) acc = reduceClause(acc)
  return renumber(acc)
}

/**
 * Keep the body literals linked to the head by chains of shared variables (ground literals and literals over
 * unconnected variables are dropped).
 *
 * @param c - The clause.
 * @returns The clause restricted to head-connected literals.
 * @example
 * ```ts
 * import { headConnected } from 'whitebox-engine/logic/ilp'
 * import { clauseToString, parseClause } from 'whitebox-engine/logic'
 * clauseToString(headConnected(parseClause('p(X) :- q(X, Y), r(Y), s(Z), t(a).'))) // => 'p(X) :- q(X,Y), r(Y).'
 * ```
 */
export function headConnected(c: Clause): Clause {
  const linked = new Set(termVariables(c.head).map((v) => v.name))
  const keep = new Set<number>()
  for (let changed = true; changed; ) {
    changed = false
    c.body.forEach((b, i) => {
      if (keep.has(i)) return
      const vs = termVariables(b).map((v) => v.name)
      if (vs.some((v) => linked.has(v))) {
        keep.add(i)
        for (const v of vs) linked.add(v)
        changed = true
      }
    })
  }
  return { head: c.head, body: c.body.filter((_, i) => keep.has(i)) }
}
