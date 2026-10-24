// Deterministic and nondeterministic built-in predicates of the resolution engine (the control constructs, call/N,
// findall/bagof/setof and catch/throw live in the engine itself). A built-in returns true or false, or an iterable of
// alternatives, each a function that performs its unifications (and side effects) and reports success.
import { compareNumbers, evaluate } from './_arith.ts'
import { domainError, instantiationError, PrologThrow, permissionError, typeError } from './_errors.ts'
import { parseTerm } from './parse.ts'
import {
  atom,
  type Clause,
  clauseToTerm,
  compareTerms,
  compound,
  indicator,
  LIST_FUNCTOR,
  list,
  listItems,
  NIL,
  type Num,
  num,
  type Operators,
  type Term,
  termToClause,
  termToString,
  termVariables,
  type Var,
} from './terms.ts'
import type { Bindings } from './unify.ts'

/** The mutable database of a running query (arrays are replaced, never mutated, so running calls keep their view). */
export interface MutableDb {
  readonly preds: Map<string, readonly Clause[]>
  readonly dynamic: Set<string>
}

/** What a built-in can use of the engine. */
export interface Engine {
  readonly bindings: Bindings
  readonly db: MutableDb
  readonly operators: Operators
  readonly occursCheck: boolean
  unify(a: Term, b: Term): boolean
  deref(t: Term): Term
  resolve(t: Term): Term
  fresh(): Var
  write(text: string): void
  /** Names of built-in and control predicates (which clauses may not redefine). */
  isBuiltin(key: string): boolean
}

export type BuiltinResult = boolean | Iterable<() => boolean>
export type Builtin = (args: readonly Term[], e: Engine) => BuiltinResult

const TRUE = atom('true')

function text(t: Term, e: Engine, where: string, what = 'atom'): string {
  const x = e.deref(t)
  if (x.kind === 'var') return instantiationError(where)
  if (x.kind === 'atom') return x.name
  if (x.kind === 'num') return termToString(x)
  return typeError(what, x, where)
}

function intArg(t: Term, e: Engine, where: string): number {
  const x = e.deref(t)
  if (x.kind === 'var') return instantiationError(where)
  if (x.kind !== 'num' || x.float) return typeError('integer', x, where)
  return x.value
}

/** The elements of a proper list (after dereferencing), or an error. */
function properList(t: Term, e: Engine, where: string): Term[] {
  const out: Term[] = []
  let cur = e.deref(t)
  while (cur.kind === 'compound' && cur.functor === LIST_FUNCTOR && cur.args.length === 2) {
    out.push(cur.args[0] as Term)
    cur = e.deref(cur.args[1] as Term)
  }
  if (cur.kind === 'var') return instantiationError(where)
  if (!(cur.kind === 'atom' && cur.name === '[]')) return typeError('list', t, where)
  return out
}

/** Copy a term with fresh variables (sharing preserved). */
export function copyTerm(t: Term, e: Engine): Term {
  const names = new Map<string, Var>()
  const go = (x: Term): Term => {
    const y = e.deref(x)
    if (y.kind === 'var') {
      let v = names.get(y.name)
      if (v === undefined) {
        v = e.fresh()
        names.set(y.name, v)
      }
      return v
    }
    if (y.kind !== 'compound') return y
    return { kind: 'compound', functor: y.functor, args: y.args.map(go) }
  }
  return go(t)
}

function parseNumber(s: string): Num | null {
  const trimmed = s.trim()
  if (trimmed === '') return null
  try {
    const t = parseTerm(trimmed)
    return t.kind === 'num' ? t : null
  } catch {
    return null
  }
}

const codes = (s: string): Term => list([...s].map((c) => num(c.codePointAt(0) ?? 0)))
const chars = (s: string): Term => list([...s].map((c) => atom(c)))

function fromCodesOrChars(t: Term, e: Engine, where: string): string | null {
  const x = e.resolve(t)
  const items = listItems(x)
  if (items === null) return null
  let out = ''
  for (const it of items) {
    if (it.kind === 'var') return null
    if (it.kind === 'num' && !it.float) out += String.fromCodePoint(it.value)
    else if (it.kind === 'atom' && [...it.name].length === 1) out += it.name
    else return typeError('character_code', it, where)
  }
  return out
}

function compareBuiltin(order: (c: number) => boolean): Builtin {
  return ([a, b], e) => order(compareTerms(e.resolve(a as Term), e.resolve(b as Term)))
}

function arithCompare(order: (c: number) => boolean, where: string): Builtin {
  return ([a, b], e) =>
    order(
      compareNumbers(
        evaluate(a as Term, (t) => e.deref(t), where),
        evaluate(b as Term, (t) => e.deref(t), where),
      ),
    )
}

function typeCheck(test: (t: Term, e: Engine) => boolean): Builtin {
  return ([a], e) => test(e.deref(a as Term), e)
}

function* range(lo: number, hi: number): Generator<number> {
  for (let i = lo; i <= hi; i++) yield i
}

function clauseOf(t: Term, e: Engine, where: string): Clause {
  const c = termToClause(copyTerm(t, e))
  const h = e.deref(c.head)
  if (h.kind === 'var') return instantiationError(where)
  if (h.kind === 'num') return typeError('callable', h, where)
  for (const g of c.body) if (e.deref(g).kind === 'num') typeError('callable', g, where)
  const key = indicator(h)
  if (e.isBuiltin(key)) permissionError('modify', 'static_procedure', piTerm(key), where)
  return { head: h, body: c.body.map((g) => e.resolve(g)) }
}

function piTerm(key: string): Term {
  const i = key.lastIndexOf('/')
  return compound('/', [atom(key.slice(0, i)), num(Number(key.slice(i + 1)))])
}

function assertClause(front: boolean, where: string): Builtin {
  return ([t], e) => {
    const c = clauseOf(t as Term, e, where)
    const key = indicator(c.head)
    const old = e.db.preds.get(key) ?? []
    e.db.preds.set(key, front ? [c, ...old] : [...old, c])
    e.db.dynamic.add(key)
    return true
  }
}

function headAndBody(t: Term, e: Engine, where: string): [Term, Term] {
  const x = e.deref(t)
  if (x.kind === 'compound' && x.functor === ':-' && x.args.length === 2) {
    return [e.deref(x.args[0] as Term), x.args[1] as Term]
  }
  if (x.kind === 'var') return instantiationError(where)
  return [x, TRUE]
}

function renamed(c: Clause, e: Engine): Term {
  const names = new Map<string, Var>()
  const go = (x: Term): Term => {
    if (x.kind === 'var') {
      let v = names.get(x.name)
      if (v === undefined) {
        v = e.fresh()
        names.set(x.name, v)
      }
      return v
    }
    if (x.kind !== 'compound') return x
    return { kind: 'compound', functor: x.functor, args: x.args.map(go) }
  }
  return go(clauseToTerm(c))
}

function clauseParts(t: Term): [Term, Term] {
  if (t.kind === 'compound' && t.functor === ':-' && t.args.length === 2) return [t.args[0] as Term, t.args[1] as Term]
  return [t, TRUE]
}

function format(fmt: string, args: readonly Term[], e: Engine, where: string): string {
  let out = ''
  let k = 0
  const next = (): Term => {
    const a = args[k++]
    if (a === undefined) return fail(where, 'not enough arguments')
    return a
  }
  for (let i = 0; i < fmt.length; i++) {
    const c = fmt[i] as string
    if (c !== '~') {
      out += c
      continue
    }
    let j = i + 1
    let numArg = ''
    while (/[0-9]/.test(fmt[j] ?? '')) numArg += fmt[j++]
    if (fmt[j] === '*') {
      numArg = String(intArg(next(), e, where))
      j++
    }
    const d = fmt[j]
    i = j
    switch (d) {
      case 'w':
        out += termToString(e.resolve(next()), { quoted: false, operators: e.operators })
        break
      case 'p':
        out += termToString(e.resolve(next()), { quoted: true, operators: e.operators })
        break
      case 'q':
        out += termToString(e.resolve(next()), { quoted: true, operators: e.operators })
        break
      case 'a':
        out += text(next(), e, where)
        break
      case 'd': {
        const v = intArg(next(), e, where)
        const places = numArg === '' ? 0 : Number(numArg)
        if (places > 0) {
          const s = String(Math.abs(v)).padStart(places + 1, '0')
          out += `${v < 0 ? '-' : ''}${s.slice(0, -places)}.${s.slice(-places)}`
        } else out += String(v)
        break
      }
      case 'D': {
        const v = intArg(next(), e, where)
        out += String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',')
        break
      }
      case 'f':
      case 'e':
      case 'g': {
        const x = evaluate(next(), (t) => e.deref(t), where).value
        const places = numArg === '' ? 6 : Number(numArg)
        out +=
          d === 'f'
            ? x.toFixed(places)
            : d === 'e'
              ? x.toExponential(places).replace(/e([+-])(\d)$/, 'e$10$2')
              : String(x)
        break
      }
      case 's': {
        const s = fromCodesOrChars(next(), e, where)
        out += s ?? ''
        break
      }
      case 'c': {
        const code = intArg(next(), e, where)
        out += String.fromCodePoint(code).repeat(numArg === '' ? 1 : Number(numArg))
        break
      }
      case 'n':
        out += '\n'.repeat(numArg === '' ? 1 : Number(numArg))
        break
      case '~':
        out += '~'
        break
      case 'i':
        next()
        break
      case 't':
      case '|':
      case '+':
        // Column alignment is not modelled; fill characters and column stops are ignored.
        break
      default:
        return fail(where, `unknown directive ~${d ?? ''}`)
    }
  }
  if (k < args.length) return fail(where, 'too many arguments')
  return out
}

function fail(where: string, message: string): never {
  throw new PrologThrow(
    compound('error', [compound('format', [atom(message)]), compound('context', [atom(where), atom('_')])]),
  )
}

function lengthBuiltin([l, n]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'length/2'
  let cur = e.deref(l as Term)
  let count = 0
  while (cur.kind === 'compound' && cur.functor === LIST_FUNCTOR && cur.args.length === 2) {
    count++
    cur = e.deref(cur.args[1] as Term)
  }
  const nn = e.deref(n as Term)
  if (nn.kind !== 'var' && (nn.kind !== 'num' || nn.float)) return typeError('integer', nn, where)
  if (cur.kind === 'atom' && cur.name === '[]') return e.unify(nn, num(count))
  if (cur.kind !== 'var') return false
  const tail = cur
  if (nn.kind === 'num') {
    if (nn.value < 0) return domainError('not_less_than_zero', nn, where)
    if (nn.value < count) return false
    return e.unify(tail, list(Array.from({ length: nn.value - count }, () => e.fresh())))
  }
  // A partial list and an unbound length: enumerate the lengths count, count + 1, ...
  return (function* () {
    for (let k = 0; ; k++) {
      yield () => {
        const fresh = Array.from({ length: k }, () => e.fresh())
        return e.unify(tail, list(fresh)) && e.unify(nn, num(count + k))
      }
    }
  })()
}

function univ([t, l]: readonly Term[], e: Engine): BuiltinResult {
  const where = '=../2'
  const x = e.deref(t as Term)
  if (x.kind === 'compound') return e.unify(l as Term, list([atom(x.functor), ...x.args]))
  if (x.kind !== 'var') return e.unify(l as Term, list([x]))
  const items = properList(l as Term, e, where).map((i) => e.deref(i))
  const [h, ...rest] = items
  if (h === undefined) return domainError('non_empty_list', NIL, where)
  if (rest.length === 0) return e.unify(x, h)
  if (h.kind === 'var') return instantiationError(where)
  if (h.kind !== 'atom') return typeError('atom', h, where)
  return e.unify(x, compound(h.name, rest))
}

function functorBuiltin([t, n, a]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'functor/3'
  const x = e.deref(t as Term)
  if (x.kind === 'compound') return e.unify(n as Term, atom(x.functor)) && e.unify(a as Term, num(x.args.length))
  if (x.kind !== 'var') return e.unify(n as Term, x) && e.unify(a as Term, num(0))
  const name = e.deref(n as Term)
  const arity = intArg(a as Term, e, where)
  if (name.kind === 'var') return instantiationError(where)
  if (arity === 0) return e.unify(x, name)
  if (name.kind === 'compound') return typeError('atomic', name, where)
  if (name.kind !== 'atom') return typeError('atom', name, where)
  return e.unify(
    x,
    compound(
      name.name,
      Array.from({ length: arity }, () => e.fresh()),
    ),
  )
}

function argBuiltin([n, t, a]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'arg/3'
  const x = e.deref(t as Term)
  if (x.kind === 'var') return instantiationError(where)
  if (x.kind !== 'compound') return typeError('compound', x, where)
  const nn = e.deref(n as Term)
  if (nn.kind === 'num' && !nn.float) {
    const arg = x.args[nn.value - 1]
    return arg !== undefined && e.unify(a as Term, arg)
  }
  if (nn.kind !== 'var') return typeError('integer', nn, where)
  return x.args.map((arg, i) => () => e.unify(nn, num(i + 1)) && e.unify(a as Term, arg))
}

function atomConcat([a, b, c]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'atom_concat/3'
  const x = e.deref(a as Term)
  const y = e.deref(b as Term)
  if (x.kind !== 'var' && y.kind !== 'var') return e.unify(c as Term, atom(text(x, e, where) + text(y, e, where)))
  const whole = [...text(c as Term, e, where)]
  return Array.from(
    { length: whole.length + 1 },
    (_, i) => () => e.unify(x, atom(whole.slice(0, i).join(''))) && e.unify(y, atom(whole.slice(i).join(''))),
  )
}

function subAtom([a, b, l, af, s]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'sub_atom/5'
  const whole = [...text(a as Term, e, where)]
  const sub = e.deref(s as Term)
  const out: (() => boolean)[] = []
  if (sub.kind !== 'var') {
    const target = [...text(sub, e, where)]
    for (let i = 0; i + target.length <= whole.length; i++) {
      if (whole.slice(i, i + target.length).join('') === target.join('')) {
        out.push(
          () =>
            e.unify(b as Term, num(i)) &&
            e.unify(l as Term, num(target.length)) &&
            e.unify(af as Term, num(whole.length - i - target.length)),
        )
      }
    }
    return out
  }
  for (let i = 0; i <= whole.length; i++) {
    for (let len = 0; i + len <= whole.length; len++) {
      out.push(
        () =>
          e.unify(b as Term, num(i)) &&
          e.unify(l as Term, num(len)) &&
          e.unify(af as Term, num(whole.length - i - len)) &&
          e.unify(sub, atom(whole.slice(i, i + len).join(''))),
      )
    }
  }
  return out
}

function atomicListConcat(args: readonly Term[], e: Engine): BuiltinResult {
  const where = `atomic_list_concat/${args.length}`
  const [l, sepOrResult, result] = args
  const sep = args.length === 3 ? text(sepOrResult as Term, e, where) : ''
  const target = (args.length === 3 ? result : sepOrResult) as Term
  const items = listItems(e.resolve(l as Term))
  if (items?.every((i) => i.kind !== 'var')) {
    return e.unify(target, atom(items.map((i) => text(i, e, where)).join(sep)))
  }
  if (args.length === 3 && sep !== '') {
    const whole = text(target, e, where)
    return e.unify(l as Term, list(whole.split(sep).map((p) => atom(p))))
  }
  return instantiationError(where)
}

function sort4([k, o, l, s]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'sort/4'
  const key = intArg(k as Term, e, where)
  const order = text(o as Term, e, where)
  const items = properList(l as Term, e, where).map((i) => e.resolve(i))
  const keyOf = (t: Term): Term => {
    if (key === 0) return t
    if (t.kind !== 'compound') return typeError('compound', t, where)
    const a = t.args[key - 1]
    return a === undefined ? typeError('compound', t, where) : a
  }
  const desc = order === '@>' || order === '@>='
  const dedupe = order === '@<' || order === '@>'
  if (!['@<', '@>', '@=<', '@>='].includes(order)) return domainError('order', atom(order), where)
  const sorted = items
    .map((t, i) => ({ t, i, kk: keyOf(t) }))
    .sort((p, q) => (desc ? -1 : 1) * compareTerms(p.kk, q.kk) || p.i - q.i)
  const out: Term[] = []
  let last: Term | undefined
  for (const x of sorted) {
    if (dedupe && last !== undefined && compareTerms(last, x.kk) === 0) continue
    out.push(x.t)
    last = x.kk
  }
  return e.unify(s as Term, list(out))
}

function writeBuiltin(options: { quoted: boolean; ignoreOps?: boolean; newline?: boolean }): Builtin {
  return ([t], e) => {
    e.write(
      termToString(e.resolve(t as Term), {
        quoted: options.quoted,
        operators: e.operators,
        ...(options.ignoreOps === true ? { ignoreOps: true } : {}),
      }) + (options.newline === true ? '\n' : ''),
    )
    return true
  }
}

function termToAtom([t, a]: readonly Term[], e: Engine): BuiltinResult {
  const where = 'term_to_atom/2'
  const x = e.deref(t as Term)
  const y = e.deref(a as Term)
  if (y.kind !== 'var') {
    const parsed = parseTerm(text(y, e, where), { operators: e.operators })
    const names = new Map<string, Var>()
    const fresh = (z: Term): Term => {
      if (z.kind === 'var') {
        let v = names.get(z.name)
        if (v === undefined) {
          v = e.fresh()
          names.set(z.name, v)
        }
        return v
      }
      if (z.kind !== 'compound') return z
      return { kind: 'compound', functor: z.functor, args: z.args.map(fresh) }
    }
    return e.unify(x, fresh(parsed))
  }
  if (x.kind === 'var') return instantiationError(where)
  return e.unify(y, atom(termToString(e.resolve(x), { quoted: true, operators: e.operators })))
}

/** The built-in predicates by indicator. */
export const BUILTINS: ReadonlyMap<string, Builtin> = new Map<string, Builtin>([
  ['=/2', ([a, b], e) => e.unify(a as Term, b as Term)],
  [
    '\\=/2',
    ([a, b], e) => {
      const m = e.bindings.mark()
      const ok = e.unify(a as Term, b as Term)
      e.bindings.undo(m)
      return !ok
    },
  ],
  ['unify_with_occurs_check/2', ([a, b], e) => e.bindings.unify(a as Term, b as Term, true)],
  ['==/2', compareBuiltin((c) => c === 0)],
  ['\\==/2', compareBuiltin((c) => c !== 0)],
  ['@</2', compareBuiltin((c) => c < 0)],
  ['@>/2', compareBuiltin((c) => c > 0)],
  ['@=</2', compareBuiltin((c) => c <= 0)],
  ['@>=/2', compareBuiltin((c) => c >= 0)],
  [
    'compare/3',
    ([o, a, b], e) => {
      const c = compareTerms(e.resolve(a as Term), e.resolve(b as Term))
      return e.unify(o as Term, atom(c < 0 ? '<' : c > 0 ? '>' : '='))
    },
  ],
  [
    'subsumes_term/2',
    ([g, s], e) => {
      const m = e.bindings.mark()
      const specific = e.resolve(s as Term)
      const before = termVariables(specific).map((v) => v.name)
      const ok = e.unify(g as Term, s as Term)
      const after = e.resolve(specific)
      const unchanged = ok && compareTerms(after, specific) === 0 && before.length === termVariables(after).length
      e.bindings.undo(m)
      return unchanged
    },
  ],
  ['var/1', typeCheck((t) => t.kind === 'var')],
  ['nonvar/1', typeCheck((t) => t.kind !== 'var')],
  ['atom/1', typeCheck((t) => t.kind === 'atom')],
  ['number/1', typeCheck((t) => t.kind === 'num')],
  ['integer/1', typeCheck((t) => t.kind === 'num' && !t.float)],
  ['float/1', typeCheck((t) => t.kind === 'num' && t.float)],
  ['atomic/1', typeCheck((t) => t.kind === 'atom' || t.kind === 'num')],
  ['compound/1', typeCheck((t) => t.kind === 'compound')],
  ['callable/1', typeCheck((t) => t.kind === 'atom' || t.kind === 'compound')],
  ['is_list/1', typeCheck((t, e) => listItems(e.resolve(t)) !== null)],
  ['ground/1', typeCheck((t, e) => termVariables(e.resolve(t)).length === 0)],
  [
    'is/2',
    ([x, y], e) =>
      e.unify(
        x as Term,
        evaluate(y as Term, (t) => e.deref(t), 'is/2'),
      ),
  ],
  ['=:=/2', arithCompare((c) => c === 0, '=:=/2')],
  ['=\\=/2', arithCompare((c) => c !== 0, '=\\=/2')],
  ['</2', arithCompare((c) => c < 0, '</2')],
  ['>/2', arithCompare((c) => c > 0, '>/2')],
  ['=</2', arithCompare((c) => c <= 0, '=</2')],
  ['>=/2', arithCompare((c) => c >= 0, '>=/2')],
  [
    'succ/2',
    ([a, b], e) => {
      const where = 'succ/2'
      const x = e.deref(a as Term)
      if (x.kind !== 'var') {
        const v = intArg(x, e, where)
        if (v < 0) return typeError('not_less_than_zero', x, where)
        return e.unify(b as Term, num(v + 1))
      }
      const v = intArg(b as Term, e, where)
      if (v < 0) return typeError('not_less_than_zero', num(v), where)
      return v > 0 && e.unify(x, num(v - 1))
    },
  ],
  [
    'plus/3',
    ([a, b, c], e) => {
      const where = 'plus/3'
      const [x, y, z] = [e.deref(a as Term), e.deref(b as Term), e.deref(c as Term)]
      if (x.kind !== 'var' && y.kind !== 'var') return e.unify(z, num(intArg(x, e, where) + intArg(y, e, where)))
      if (x.kind !== 'var') return e.unify(y, num(intArg(z, e, where) - intArg(x, e, where)))
      if (y.kind !== 'var') return e.unify(x, num(intArg(z, e, where) - intArg(y, e, where)))
      return instantiationError(where)
    },
  ],
  [
    'between/3',
    ([lo, hi, x], e) => {
      const where = 'between/3'
      const l = intArg(lo as Term, e, where)
      const h = e.deref(hi as Term)
      const high =
        h.kind === 'atom' && (h.name === 'inf' || h.name === 'infinite')
          ? Number.POSITIVE_INFINITY
          : intArg(h, e, where)
      const v = e.deref(x as Term)
      if (v.kind !== 'var') {
        const n = intArg(v, e, where)
        return n >= l && n <= high
      }
      return (function* () {
        for (const i of range(l, high)) yield () => e.unify(v, num(i))
      })()
    },
  ],
  ['length/2', lengthBuiltin],
  ['functor/3', functorBuiltin],
  ['arg/3', argBuiltin],
  ['=../2', univ],
  ['copy_term/2', ([a, b], e) => e.unify(b as Term, copyTerm(a as Term, e))],
  ['term_variables/2', ([a, b], e) => e.unify(b as Term, list(termVariables(e.resolve(a as Term))))],
  [
    'atom_codes/2',
    ([a, l], e) => {
      const x = e.deref(a as Term)
      if (x.kind !== 'var') return e.unify(l as Term, codes(text(x, e, 'atom_codes/2')))
      const s = fromCodesOrChars(l as Term, e, 'atom_codes/2')
      return s === null ? instantiationError('atom_codes/2') : e.unify(x, atom(s))
    },
  ],
  [
    'atom_chars/2',
    ([a, l], e) => {
      const x = e.deref(a as Term)
      if (x.kind !== 'var') return e.unify(l as Term, chars(text(x, e, 'atom_chars/2')))
      const s = fromCodesOrChars(l as Term, e, 'atom_chars/2')
      return s === null ? instantiationError('atom_chars/2') : e.unify(x, atom(s))
    },
  ],
  [
    'char_code/2',
    ([a, c], e) => {
      const x = e.deref(a as Term)
      if (x.kind === 'atom') return e.unify(c as Term, num(x.name.codePointAt(0) ?? 0))
      return e.unify(x, atom(String.fromCodePoint(intArg(c as Term, e, 'char_code/2'))))
    },
  ],
  ['atom_length/2', ([a, n], e) => e.unify(n as Term, num([...text(a as Term, e, 'atom_length/2')].length))],
  ['atom_concat/3', atomConcat],
  ['sub_atom/5', subAtom],
  [
    'number_codes/2',
    ([a, l], e) => {
      const x = e.deref(a as Term)
      if (x.kind !== 'var') return e.unify(l as Term, codes(termToString(x)))
      const s = fromCodesOrChars(l as Term, e, 'number_codes/2')
      const n = s === null ? null : parseNumber(s)
      if (n === null) return fail('number_codes/2', 'illegal number')
      return e.unify(x, n)
    },
  ],
  [
    'atom_number/2',
    ([a, n], e) => {
      const x = e.deref(a as Term)
      if (x.kind === 'var') {
        const v = e.deref(n as Term)
        if (v.kind === 'var') return instantiationError('atom_number/2')
        return e.unify(x, atom(termToString(v)))
      }
      const parsed = parseNumber(text(x, e, 'atom_number/2'))
      return parsed !== null && e.unify(n as Term, parsed)
    },
  ],
  ['upcase_atom/2', ([a, b], e) => e.unify(b as Term, atom(text(a as Term, e, 'upcase_atom/2').toUpperCase()))],
  ['downcase_atom/2', ([a, b], e) => e.unify(b as Term, atom(text(a as Term, e, 'downcase_atom/2').toLowerCase()))],
  ['atomic_list_concat/2', atomicListConcat],
  ['atomic_list_concat/3', atomicListConcat],
  ['term_to_atom/2', termToAtom],
  [
    'msort/2',
    ([l, s], e) =>
      e.unify(
        s as Term,
        list(
          properList(l as Term, e, 'msort/2')
            .map((t) => e.resolve(t))
            .sort(compareTerms),
        ),
      ),
  ],
  [
    'sort/2',
    ([l, s], e) => {
      const sorted = properList(l as Term, e, 'sort/2')
        .map((t) => e.resolve(t))
        .sort(compareTerms)
      const out = sorted.filter((t, i) => i === 0 || compareTerms(sorted[i - 1] as Term, t) !== 0)
      return e.unify(s as Term, list(out))
    },
  ],
  ['sort/4', sort4],
  [
    'keysort/2',
    ([l, s], e) => {
      const where = 'keysort/2'
      const pairs = properList(l as Term, e, where).map((t) => {
        const p = e.resolve(t)
        if (p.kind === 'var') return instantiationError(where)
        if (p.kind !== 'compound' || p.functor !== '-' || p.args.length !== 2) return typeError('pair', p, where)
        return p
      })
      const sorted = pairs
        .map((p, i) => ({ p, i }))
        .sort((x, y) => compareTerms(x.p.args[0] as Term, y.p.args[0] as Term) || x.i - y.i)
        .map((x) => x.p)
      return e.unify(s as Term, list(sorted))
    },
  ],
  ['assert/1', assertClause(false, 'assert/1')],
  ['assertz/1', assertClause(false, 'assertz/1')],
  ['asserta/1', assertClause(true, 'asserta/1')],
  [
    'retract/1',
    ([t], e) => {
      const where = 'retract/1'
      const [h, body] = headAndBody(t as Term, e, where)
      if (h.kind === 'num') return typeError('callable', h, where)
      const key = indicator(h)
      if (e.isBuiltin(key)) return permissionError('modify', 'static_procedure', piTerm(key), where)
      const snapshot = e.db.preds.get(key) ?? []
      return snapshot.map((c) => () => {
        const current = e.db.preds.get(key) ?? []
        if (!current.includes(c)) return false
        const [ch, cb] = clauseParts(renamed(c, e))
        if (!(e.unify(h, ch) && e.unify(body, cb))) return false
        e.db.preds.set(
          key,
          current.filter((x) => x !== c),
        )
        return true
      })
    },
  ],
  [
    'retractall/1',
    ([t], e) => {
      const where = 'retractall/1'
      const h = e.deref(t as Term)
      if (h.kind === 'var') return instantiationError(where)
      if (h.kind === 'num') return typeError('callable', h, where)
      const key = indicator(h)
      if (e.isBuiltin(key)) return permissionError('modify', 'static_procedure', piTerm(key), where)
      const keep = (e.db.preds.get(key) ?? []).filter((c) => {
        const m = e.bindings.mark()
        const [ch] = clauseParts(renamed(c, e))
        const hit = e.unify(h, ch)
        e.bindings.undo(m)
        return !hit
      })
      e.db.preds.set(key, keep)
      e.db.dynamic.add(key)
      return true
    },
  ],
  [
    'clause/2',
    ([hd, b], e) => {
      const where = 'clause/2'
      const h = e.deref(hd as Term)
      if (h.kind === 'var') return instantiationError(where)
      if (h.kind === 'num') return typeError('callable', h, where)
      const key = indicator(h)
      if (e.isBuiltin(key)) return permissionError('access', 'private_procedure', piTerm(key), where)
      return (e.db.preds.get(key) ?? []).map((c) => () => {
        const [ch, cb] = clauseParts(renamed(c, e))
        return e.unify(h, ch) && e.unify(b as Term, cb)
      })
    },
  ],
  ['write/1', writeBuiltin({ quoted: false })],
  ['print/1', writeBuiltin({ quoted: true })],
  ['writeln/1', writeBuiltin({ quoted: false, newline: true })],
  ['writeq/1', writeBuiltin({ quoted: true })],
  ['write_canonical/1', writeBuiltin({ quoted: true, ignoreOps: true })],
  [
    'nl/0',
    (_, e) => {
      e.write('\n')
      return true
    },
  ],
  [
    'tab/1',
    ([n], e) => {
      e.write(' '.repeat(evaluate(n as Term, (t) => e.deref(t), 'tab/1').value))
      return true
    },
  ],
  [
    'format/1',
    ([f], e) => {
      e.write(format(formatText(f as Term, e), [], e, 'format/1'))
      return true
    },
  ],
  [
    'format/2',
    ([f, a], e) => {
      const args = e.resolve(a as Term)
      const items = listItems(args) ?? [args]
      e.write(format(formatText(f as Term, e), items, e, 'format/2'))
      return true
    },
  ],
  [
    'halt/0',
    () => {
      throw new PrologThrow(atom('$halt'))
    },
  ],
])

function formatText(f: Term, e: Engine): string {
  const x = e.deref(f)
  if (x.kind === 'atom') return x.name
  const s = fromCodesOrChars(x, e, 'format/2')
  if (s === null) return typeError('text', x, 'format/2')
  return s
}

/** Whether a term is a callable goal; otherwise the error to raise. */
export function checkCallable(t: Term, where: string): void {
  if (t.kind === 'var') instantiationError(where)
  if (t.kind === 'num') typeError('callable', t, where)
}

/** Build `G(Args..., Extra...)` for call/N. */
export function addArgs(goal: Term, extra: readonly Term[], where: string): Term {
  if (goal.kind === 'atom') return compound(goal.name, extra)
  if (goal.kind === 'compound') return compound(goal.functor, [...goal.args, ...extra])
  checkCallable(goal, where)
  return typeError('callable', goal, where)
}
