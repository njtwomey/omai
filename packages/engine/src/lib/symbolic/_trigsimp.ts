// Trigonometric simplification in the spirit of SymPy's Fu-style `trigsimp`: a small beam search over rewrites
// (to sin and cos, Pythagorean matching, power reduction, product-to-sum, angle expansion, back to tan, cancel and
// factor), keeping the candidates with the fewest operations. Every candidate is checked numerically against the
// input, so a rewrite can only fail to simplify, never change the value.
import {
  asCoeffMul,
  isRat,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  mapChildren,
  numFromRational,
  ONE,
  TWO,
} from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { makeFn } from './_functions.ts'
import { complexity, numericallyEqual } from './_measure.ts'
import { children } from './_node.ts'
import { rat } from './_rational.ts'
import { cancel, together } from './_rational-functions.ts'
import type { Expr, Step } from './expr.ts'

const TRIG = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'sinh', 'cosh', 'tanh', 'coth'])
const half = numFromRational(rat(1n, 2n))
const sin = (a: Expr): Expr => makeFn('sin', [a])
const cos = (a: Expr): Expr => makeFn('cos', [a])
const neg = (a: Expr): Expr => makeMul([MINUS_ONE, a])

/** Whether an expression contains trigonometric or hyperbolic functions. @internal */
export function hasTrig(e: Expr): boolean {
  if (e.kind === 'fn' && TRIG.has(e.name)) return true
  return children(e).some(hasTrig)
}

function bottomUp(e: Expr, f: (x: Expr) => Expr): Expr {
  return f(mapChildren(e, (c) => bottomUp(c, f)))
}

/** tan, cot, sec, csc (and tanh, coth) in terms of sin and cos (sinh and cosh). */
function toSinCos(e: Expr): Expr {
  return bottomUp(e, (x) => {
    if (x.kind !== 'fn') return x
    const a = x.args[0] as Expr
    switch (x.name) {
      case 'tan':
        return makeMul([sin(a), makePow(cos(a), MINUS_ONE)])
      case 'cot':
        return makeMul([cos(a), makePow(sin(a), MINUS_ONE)])
      case 'sec':
        return makePow(cos(a), MINUS_ONE)
      case 'csc':
        return makePow(sin(a), MINUS_ONE)
      case 'tanh':
        return makeMul([makeFn('sinh', [a]), makePow(makeFn('cosh', [a]), MINUS_ONE)])
      case 'coth':
        return makeMul([makeFn('cosh', [a]), makePow(makeFn('sinh', [a]), MINUS_ONE)])
      default:
        return x
    }
  })
}

/** sin(a)^k / cos(a)^k back to tan(a)^k (and sinh/cosh to tanh). */
function toTan(e: Expr): Expr {
  return bottomUp(e, (x) => {
    if (x.kind !== 'mul') return x
    const powers = new Map<string, { arg: Expr; s: Expr; c: Expr; hyp: boolean }>()
    const rest: Expr[] = []
    for (const f of x.args) {
      const [b, k] = f.kind === 'pow' ? [f.base, f.exp] : [f, ONE]
      if (
        b.kind === 'fn' &&
        (b.name === 'sin' || b.name === 'cos' || b.name === 'sinh' || b.name === 'cosh') &&
        k.kind === 'num'
      ) {
        const hyp = b.name.endsWith('h')
        const key = `${hyp ? 'h' : 't'}${(b.args[0] as Expr).key}`
        const slot = powers.get(key) ?? {
          arg: b.args[0] as Expr,
          s: numFromRational(rat(0n)),
          c: numFromRational(rat(0n)),
          hyp,
        }
        if (b.name.startsWith('sin')) slot.s = makeAdd([slot.s, k])
        else slot.c = makeAdd([slot.c, k])
        powers.set(key, slot)
      } else rest.push(f)
    }
    const out: Expr[] = [...rest]
    for (const { arg, s, c, hyp } of powers.values()) {
      const S = makeFn(hyp ? 'sinh' : 'sin', [arg])
      const Cf = makeFn(hyp ? 'cosh' : 'cos', [arg])
      if (s.kind === 'num' && c.kind === 'num' && s.n > 0n && c.n < 0n && s.d === 1n && c.d === 1n) {
        // The smaller of the two powers (bigints, so not Math.min).
        let k = s.n
        if (-c.n < k) k = -c.n
        const kk = numFromRational(rat(k))
        out.push(makePow(makeFn(hyp ? 'tanh' : 'tan', [arg]), kk))
        out.push(makePow(S, numFromRational(rat(s.n - k))), makePow(Cf, numFromRational(rat(c.n + k))))
      } else out.push(makePow(S, s), makePow(Cf, c))
    }
    return makeMul(out)
  })
}

interface Square {
  readonly fn: string
  readonly arg: Expr
  readonly rest: Expr
  readonly coeff: Expr
}

/** Read c * f(a)^2 * rest for f in sin, cos, sinh, cosh. */
function asSquare(t: Expr): Square | undefined {
  const [c, body] = asCoeffMul(t)
  for (const f of body.kind === 'mul' ? body.args : [body]) {
    if (
      f.kind === 'pow' &&
      isRat(f.exp, 2n) &&
      f.base.kind === 'fn' &&
      ['sin', 'cos', 'sinh', 'cosh'].includes(f.base.name)
    ) {
      const rest = makeMul((body.kind === 'mul' ? body.args : [body]).filter((g) => g !== f))
      return { fn: f.base.name, arg: f.base.args[0] as Expr, rest, coeff: c }
    }
  }
  return undefined
}

const PARTNER: Readonly<Record<string, string>> = { sin: 'cos', cos: 'sin', sinh: 'cosh', cosh: 'sinh' }

function same(a: Expr, b: Expr): boolean {
  return a.key === b.key
}

/** One Pythagorean rewrite of the pair (s, t), or undefined. s is c f(a)^2 R. */
function pairRewrite(s: Square, t: Expr): Expr | undefined {
  const g = PARTNER[s.fn] as string
  const hyp = s.fn.endsWith('h')
  const minus = (x: Expr): Expr => makeMul([MINUS_ONE, x])
  const b = asSquare(t)
  if (b !== undefined && b.fn === g && same(b.arg, s.arg) && same(b.rest, s.rest)) {
    // sin^2 + cos^2 = 1; cosh^2 - sinh^2 = 1.
    if (!hyp && same(b.coeff, s.coeff)) return makeMul([s.coeff, s.rest])
    if (hyp && same(b.coeff, minus(s.coeff))) return makeMul([s.fn === 'cosh' ? s.coeff : b.coeff, s.rest])
    return undefined
  }
  const [tc, tb] = asCoeffMul(t)
  if (!same(tb, s.rest)) return undefined
  const sq = (name: string): Expr => makePow(makeFn(name, [s.arg]), TWO)
  // c R - c f^2 R = c g^2 R (trig).
  if (!hyp && same(tc, minus(s.coeff))) return makeMul([tc, s.rest, sq(g)])
  // c R + c sinh^2 R = c cosh^2 R; c cosh^2 R - c R = c sinh^2 R.
  if (hyp && s.fn === 'sinh' && same(tc, s.coeff)) return makeMul([tc, s.rest, sq('cosh')])
  if (hyp && s.fn === 'cosh' && same(tc, minus(s.coeff))) return makeMul([s.coeff, s.rest, sq('sinh')])
  return undefined
}

/**
 * Pythagorean matching inside every sum: c sin^2 R + c cos^2 R = c R, c R - c cos^2 R = c sin^2 R,
 * c R - c sin^2 R = c cos^2 R, and the hyperbolic forms of cosh^2 - sinh^2 = 1.
 */
function pythagorean(e: Expr): Expr {
  return bottomUp(e, (x) => {
    if (x.kind !== 'add') return x
    let terms = [...x.args]
    for (let pass = 0; pass < terms.length + 1; pass++) {
      let hit: [number, number, Expr] | undefined
      for (let i = 0; i < terms.length && hit === undefined; i++) {
        const s = asSquare(terms[i] as Expr)
        if (s === undefined) continue
        for (let j = 0; j < terms.length && hit === undefined; j++) {
          if (j === i) continue
          const r = pairRewrite(s, terms[j] as Expr)
          if (r !== undefined) hit = [i, j, r]
        }
      }
      if (hit === undefined) break
      const [i, j, r] = hit
      terms = [...terms.filter((_, k) => k !== i && k !== j), r]
    }
    return makeAdd(terms)
  })
}

/** sin^2 = (1 - cos 2a)/2 and cos^2 = (1 + cos 2a)/2 for even powers, then expand. */
/** @internal */
export function powerReduce(e: Expr): Expr {
  const r = bottomUp(e, (x) => {
    if (x.kind !== 'pow' || x.base.kind !== 'fn' || x.exp.kind !== 'num' || x.exp.d !== 1n || x.exp.n < 2n) return x
    if (x.base.name !== 'sin' && x.base.name !== 'cos') return x
    const a = x.base.args[0] as Expr
    const c2 = cos(makeMul([TWO, a]))
    const sq = makeMul([half, makeAdd([ONE, x.base.name === 'sin' ? neg(c2) : c2])])
    const k = x.exp.n / 2n
    const odd = x.exp.n % 2n === 1n
    return makeMul([makePow(sq, numFromRational(rat(k))), odd ? x.base : ONE])
  })
  return expandExpr(r)
}

/** Product to sum: sin a cos b = (sin(a+b) + sin(a-b))/2 and friends, on the first pair in each product. */
/** @internal */
export function productToSum(e: Expr): Expr {
  const r = bottomUp(e, (x) => {
    if (x.kind !== 'mul') return x
    const idx = x.args.findIndex((f) => f.kind === 'fn' && (f.name === 'sin' || f.name === 'cos'))
    if (idx < 0) return x
    const jdx = x.args.findIndex((f, j) => j > idx && f.kind === 'fn' && (f.name === 'sin' || f.name === 'cos'))
    if (jdx < 0) return x
    const f = x.args[idx] as Expr & { kind: 'fn' }
    const g = x.args[jdx] as Expr & { kind: 'fn' }
    const a = f.args[0] as Expr
    const b = g.args[0] as Expr
    const sum = makeAdd([a, b])
    const diff = makeAdd([a, neg(b)])
    let v: Expr
    if (f.name === 'sin' && g.name === 'sin') v = makeMul([half, makeAdd([cos(diff), neg(cos(sum))])])
    else if (f.name === 'cos' && g.name === 'cos') v = makeMul([half, makeAdd([cos(diff), cos(sum)])])
    else {
      const [s, c] = f.name === 'sin' ? [a, b] : [b, a]
      v = makeMul([half, makeAdd([sin(makeAdd([s, c])), sin(makeAdd([s, neg(c)]))])])
    }
    return makeMul([v, ...x.args.filter((_, k) => k !== idx && k !== jdx)])
  })
  return expandExpr(r)
}

/** sin and cos of sums and of small integer multiples in terms of the parts. */
function expandTrig(e: Expr): Expr {
  const r = bottomUp(e, (x) => {
    if (x.kind !== 'fn' || (x.name !== 'sin' && x.name !== 'cos')) return x
    const a = x.args[0] as Expr
    if (a.kind === 'add') {
      const [first, ...others] = a.args as [Expr, ...Expr[]]
      const rest = makeAdd(others)
      return x.name === 'sin'
        ? makeAdd([makeMul([sin(first), cos(rest)]), makeMul([cos(first), sin(rest)])])
        : makeAdd([makeMul([cos(first), cos(rest)]), neg(makeMul([sin(first), sin(rest)]))])
    }
    const [c, body] = asCoeffMul(a)
    if (c.kind === 'num' && c.d === 1n && c.n >= 2n && c.n <= 4n) {
      const m = makeMul([numFromRational(rat(c.n - 1n)), body])
      return x.name === 'sin'
        ? makeAdd([makeMul([sin(m), cos(body)]), makeMul([cos(m), sin(body)])])
        : makeAdd([makeMul([cos(m), cos(body)]), neg(makeMul([sin(m), sin(body)]))])
    }
    return x
  })
  return expandExpr(r)
}

const RULES: readonly [string, (e: Expr) => Expr][] = [
  ['rewrite in sin and cos', toSinCos],
  ['Pythagorean identity', pythagorean],
  ['power reduction', powerReduce],
  ['product to sum', productToSum],
  ['angle expansion', expandTrig],
  ['sin/cos to tan', toTan],
  ['combine over a common denominator', together],
  ['cancel', cancel],
  ['expand', (e) => expandExpr(e)],
]

/**
 * Simplify trigonometric and hyperbolic expressions by a beam search over the rewrites; records the chain of rules
 * that led to the result when `steps` is given.
 * @internal
 */
export function trigSimplifyExpr(e: Expr, steps?: Step[]): Expr {
  if (!hasTrig(e)) return e
  type Node = { e: Expr; path: Step[] }
  let beam: Node[] = [{ e, path: [] }]
  let best: Node = beam[0] as Node
  const seen = new Set<string>([e.key])
  for (let depth = 0; depth < 4; depth++) {
    const next: Node[] = []
    for (const node of beam) {
      for (const [rule, f] of RULES) {
        let out: Expr
        try {
          out = f(node.e)
        } catch {
          continue
        }
        if (seen.has(out.key)) continue
        seen.add(out.key)
        if (!numericallyEqual(out, e)) continue
        next.push({ e: out, path: [...node.path, { rule, before: node.e, after: out }] })
      }
    }
    if (next.length === 0) break
    next.sort((a, b) => complexity(a.e) - complexity(b.e))
    beam = next.slice(0, 6)
    const top = beam[0] as Node
    if (complexity(top.e) < complexity(best.e)) best = top
  }
  if (steps !== undefined) steps.push(...best.path)
  return best.e
}
