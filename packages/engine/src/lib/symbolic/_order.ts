// SymPy's printing order: `default_sort_key`, `as_ordered_terms` (lex order of monomials over sorted generators) and
// `as_ordered_factors`, plus `could_extract_minus_sign`. The printers use these so that output matches SymPy's
// `str()` and `latex()`; the canonical argument order inside nodes is the cheaper key order of _node.ts.
import { asCoeffMul, E, isNumeric, MINUS_ONE, makeMul, makePow, numFromRational, ONE } from './_canon.ts'
import { evalComplexConstant } from './_numeric.ts'
import { rat } from './_rational.ts'
import type { Expr } from './expr.ts'

/** A SymPy sort key: nested tuples of numbers and strings. @internal */
export type SortKey = number | string | readonly SortKey[]

/** Compare two sort keys as Python compares tuples. @internal */
export function compareKeys(a: SortKey, b: SortKey): number {
  if (typeof a === 'number' && typeof b === 'number') return a < b ? -1 : a > b ? 1 : 0
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.min(a.length, b.length)
    for (let i = 0; i < n; i++) {
      const c = compareKeys(a[i] as SortKey, b[i] as SortKey)
      if (c !== 0) return c
    }
    return a.length - b.length
  }
  const rank = (x: SortKey): number => (typeof x === 'number' ? 0 : typeof x === 'string' ? 1 : 2)
  return rank(a) - rank(b)
}

const FUNCTION_RANK: Readonly<Record<string, number>> = {
  exp: 10,
  log: 11,
  sin: 20,
  cos: 21,
  tan: 22,
  cot: 23,
  sinh: 30,
  cosh: 31,
  tanh: 32,
  coth: 33,
  conjugate: 40,
  re: 41,
  im: 42,
  arg: 43,
}

/** Names of the known (fixed-arity) functions; anything else is an undefined function, as in SymPy. @internal */
export const KNOWN_FUNCTIONS: ReadonlySet<string> = new Set([
  'exp',
  'log',
  'sin',
  'cos',
  'tan',
  'cot',
  'sec',
  'csc',
  'asin',
  'acos',
  'atan',
  'acot',
  'sinh',
  'cosh',
  'tanh',
  'coth',
  'asinh',
  'acosh',
  'atanh',
  'Abs',
  'sign',
  'floor',
  'ceiling',
  'erf',
  'gamma',
  'factorial',
  're',
  'im',
  'conjugate',
  'arg',
  'polygamma',
  'atan2',
])

const REL_CLASS: Readonly<Record<string, string>> = {
  '==': 'Equality',
  '!=': 'Unequality',
  '<': 'StrictLessThan',
  '<=': 'LessThan',
  '>': 'StrictGreaterThan',
  '>=': 'GreaterThan',
}

const NUMBER_CLASS: SortKey = [1, 0, 'Number']

function classKey(e: Expr): SortKey {
  switch (e.kind) {
    case 'num':
    case 'float':
      return NUMBER_CLASS
    case 'sym':
      return [2, 0, 'Symbol']
    case 'const':
      if (e.name === 'pi') return [2, 0, 'Pi']
      if (e.name === 'E') return [2, 0, 'Exp1']
      if (e.name === 'I') return [2, 0, 'ImaginaryUnit']
      if (e.name === 'zoo') return [2, 0, 'ComplexInfinity']
      if (e.name === 'true') return [5, 0, 'BooleanTrue']
      if (e.name === 'false') return [5, 0, 'BooleanFalse']
      return NUMBER_CLASS
    case 'add':
      return [3, 1, 'Add']
    case 'mul':
      return [3, 0, 'Mul']
    case 'pow':
      return [3, 2, 'Pow']
    case 'fn':
      return [4, FUNCTION_RANK[e.name] ?? (KNOWN_FUNCTIONS.has(e.name) ? 10000 : 0), e.name]
    case 'piecewise':
      return [4, 0, 'Piecewise']
    case 'rel':
      return [5, 0, REL_CLASS[e.op] ?? 'Relational']
    case 'logic':
      return [5, 0, e.op === 'and' ? 'And' : e.op === 'or' ? 'Or' : 'Not']
    case 'order':
      return [5, 0, 'Order']
    case 'integral':
      return [5, 0, 'Integral']
    case 'derivative':
      return [5, 0, 'Derivative']
  }
}

function numberValue(e: Expr): number {
  if (e.kind === 'num') return Number(e.n) / Number(e.d)
  if (e.kind === 'float') return e.value
  if (e.kind === 'const') return e.name === 'oo' ? Number.POSITIVE_INFINITY : e.name === '-oo' ? -Infinity : 0
  return 0
}

function isNumberClass(e: Expr): boolean {
  return isNumeric(e) || (e.kind === 'const' && (e.name === 'oo' || e.name === '-oo' || e.name === 'nan'))
}

function atomText(e: Expr): string {
  if (e.kind === 'sym') return e.name
  if (e.kind === 'const') return e.name === 'true' ? 'True' : e.name === 'false' ? 'False' : e.name
  if (e.kind === 'num') return e.d === 1n ? `${e.n}` : `${e.n}/${e.d}`
  if (e.kind === 'float') return `${e.value}`
  return ''
}

function tupleKey(items: readonly SortKey[]): SortKey {
  return [[5, 0, 'Tuple'], [items.length, items], sortKey(ONE), 1]
}

function argKeys(e: Expr): SortKey[] {
  switch (e.kind) {
    case 'add':
      return orderedTerms(e).map(sortKey)
    case 'mul':
      return orderedFactors(e).map(sortKey)
    case 'pow':
      return [sortKey(e.base), sortKey(e.exp)]
    case 'fn':
    case 'logic':
      return e.args.map(sortKey)
    case 'piecewise':
      return e.pieces.map(([v, c]) => [[5, 0, 'ExprCondPair'], [2, [sortKey(v), sortKey(c)]], sortKey(ONE), 1])
    case 'rel':
      return [sortKey(e.lhs), sortKey(e.rhs)]
    case 'order':
      return [sortKey(e.expr), tupleKey([sortKey(e.variable), sortKey(e.point)])]
    case 'integral':
      return [
        sortKey(e.expr),
        tupleKey(
          e.bounds === undefined
            ? [sortKey(e.variable)]
            : [sortKey(e.variable), sortKey(e.bounds[0]), sortKey(e.bounds[1])],
        ),
      ]
    case 'derivative':
      return [
        sortKey(e.expr),
        ...e.variables.map(([v, k]) => tupleKey([sortKey(v), sortKey(numFromRational(rat(BigInt(k))))])),
      ]
    default:
      return [atomText(e)]
  }
}

/** SymPy's `default_sort_key`. @internal */
export function sortKey(e: Expr): SortKey {
  if (isNumberClass(e)) return [NUMBER_CLASS, [0, []], [], numberValue(e)]
  const [c, rest] = asCoeffMul(e)
  let base = rest
  let exp: Expr = ONE
  if (rest.kind === 'pow') {
    base = rest.base
    exp = rest.exp
  }
  const atom = base.kind === 'sym' || base.kind === 'const' || isNumeric(base)
  const args = atom ? [atomText(base)] : argKeys(base)
  return [classKey(base), [args.length, args], sortKey(exp), numberValue(c)]
}

/** Sort expressions by SymPy's default sort key (stable). @internal */
export function sortBySympyKey(xs: readonly Expr[]): Expr[] {
  const keyed = xs.map((x) => ({ x, k: sortKey(x) }))
  keyed.sort((a, b) => compareKeys(a.k, b.k))
  return keyed.map((p) => p.x)
}

/** SymPy's `as_ordered_factors`: the factors of a product sorted by sort key. @internal */
export function orderedFactors(e: Expr): Expr[] {
  return e.kind === 'mul' ? sortBySympyKey(e.args) : [e]
}

/** Whether a factor is a number in SymPy's sense (no free symbols, evaluates to a complex number). */
function isNumberLike(e: Expr): boolean {
  return evalComplexConstant(e) !== undefined
}

/** SymPy's `decompose_power`: a symbolic base and an integer exponent. */
function decomposePower(f: Expr): [Expr, number] {
  let b: Expr
  let e: Expr
  if (f.kind === 'pow') {
    b = f.base
    e = f.exp
  } else if (f.kind === 'fn' && f.name === 'exp') {
    b = E
    e = f.args[0] as Expr
  } else return [f, 1]
  if (e.kind === 'num') {
    if (e.d !== 1n) return [makePow(b, numFromRational(rat(1n, e.d))), Number(e.n)]
    return [b, Number(e.n)]
  }
  if (e.kind === 'float') return [f, 1]
  const [c, tail] = asCoeffMul(e)
  if (c.kind !== 'num') return [f, 1]
  if (c.n === -1n && c.d === 1n) return [makePow(b, tail), -1]
  if (!(c.n === 1n && c.d === 1n)) {
    return [makePow(b, makeMul([numFromRational(rat(1n, c.d)), tail])), Number(c.n)]
  }
  return [f, 1]
}

/** SymPy's `as_ordered_terms` with the default (lex) order. @internal */
export function orderedTerms(e: Expr): Expr[] {
  if (e.kind !== 'add') return [e]
  const args = e.args
  if (args.length === 2) {
    const isNumberish = (x: Expr): boolean =>
      isNumberClass(x) || (x.kind === 'const' && (x.name === 'pi' || x.name === 'E'))
    const sorted = [...args].sort((a, b) => Number(!isNumberish(a)) - Number(!isNumberish(b)))
    const [a, b] = sorted as [Expr, Expr]
    if (isNumberish(a) && b.kind === 'mul' && b.args.length === 2) {
      const [m0] = b.args
      if (m0 !== undefined && isNumberClass(m0) && numberValue(m0) < 0) {
        const positive = a.kind === 'const' ? a.name !== '-oo' && a.name !== 'nan' : numberValue(a) > 0
        if (positive) return [a, b]
      }
    }
  }
  const gens = new Map<string, Expr>()
  const raw = args.map((t) => {
    const [c, rest] = asCoeffMul(t)
    let re = numberValue(c)
    let im = 0
    const cpart = new Map<string, number>()
    for (const f of rest.kind === 'mul' ? rest.args : rest.kind === 'num' ? [] : [rest]) {
      if (isNumberLike(f)) {
        const z = evalComplexConstant(f)
        if (z !== undefined) {
          const [zr, zi] = z
          const nr = re * zr - im * zi
          im = re * zi + im * zr
          re = nr
          continue
        }
      }
      const [base, k] = decomposePower(f)
      cpart.set(base.key, k)
      gens.set(base.key, base)
    }
    return { t, re, im, cpart }
  })
  const sortedGens = sortBySympyKey([...gens.values()])
  const keyed = raw.map((r) => {
    const monom = sortedGens.map((g) => -(r.cpart.get(g.key) ?? 0))
    const k: SortKey = [
      monom,
      [],
      [
        [r.im !== 0 ? 1 : 0, r.im],
        [r.re, r.im],
      ],
    ]
    return { t: r.t, k }
  })
  if (!args.some((t) => t.kind === 'order')) {
    keyed.sort((p, q) => compareKeys(p.k, q.k))
    return keyed.map((p) => p.t)
  }
  // With an O term SymPy lists the other terms in reverse (ascending powers, as a series) and the O term last.
  const terms = keyed.filter((p) => p.t.kind !== 'order').sort((p, q) => compareKeys(q.k, p.k))
  const orders = keyed.filter((p) => p.t.kind === 'order').sort((p, q) => compareKeys(q.k, p.k))
  return [...terms, ...orders].map((p) => p.t)
}

/** SymPy's `could_extract_minus_sign`. @internal */
export function couldExtractMinus(e: Expr): boolean {
  if (e.kind === 'num') return e.n < 0n
  if (e.kind === 'float') return e.value < 0
  if (e.kind === 'const') return e.name === '-oo'
  if (e.kind === 'mul') {
    const c = e.args[0] as Expr
    return isNumberClass(c) && numberValue(c) < 0
  }
  if (e.kind === 'add') {
    let negative = 0
    for (const a of e.args) if (couldExtractMinus(a)) negative++
    const positive = e.args.length - negative
    if (positive > negative) return false
    if (positive < negative) return true
    return compareKeys(sortKey(e), sortKey(makeMul([MINUS_ONE, e]))) < 0
  }
  return false
}
