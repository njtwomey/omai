import { WhiteboxError } from '../base/index.ts'
import { isInfinite, MINUS_ONE, makeAdd, makeMul, makePow, NAN, NEG_INF, OO, ZERO, ZOO } from './_canon.ts'
import { makeFn } from './_functions.ts'
import { children, hasSymbol, rawFn } from './_node.ts'
import { type C, evalComplex, freeNames } from './_numeric.ts'
import { numerDenom, together } from './_rational-functions.ts'
import { LOG_T, seriesAt0 } from './_series.ts'
import { diff } from './differentiate.ts'
import { asExpr, type Derivation, type Expr, type ExprLike, type ExprSym, type Step, substitute, sym } from './expr.ts'

/** Direction of a limit: from above, from below, or both sides (which must agree). */
export type LimitDirection = '+' | '-' | '+-'

const T = sym('t_limit__', { positive: true })
const DISCONTINUOUS = new Set(['floor', 'ceiling', 'sign'])

function hasDiscontinuity(e: Expr): boolean {
  if (e.kind === 'fn' && DISCONTINUOUS.has(e.name)) return true
  if (e.kind === 'piecewise') return true
  return children(e).some(hasDiscontinuity)
}

function isFiniteExpr(e: Expr): boolean {
  return !isInfinite(e) && e.key !== NAN.key && !containsBad(e)
}

function containsBad(e: Expr): boolean {
  if (e.key === ZOO.key || e.key === NAN.key || e.key === OO.key || e.key === NEG_INF.key) return true
  return children(e).some(containsBad)
}

function frame(x: ExprSym, x0: Expr, dir: '+' | '-'): Expr {
  if (x0.key === OO.key) return makePow(T, MINUS_ONE)
  if (x0.key === NEG_INF.key) return makeMul([MINUS_ONE, makePow(T, MINUS_ONE)])
  void x
  return makeAdd([x0, dir === '+' ? T : makeMul([MINUS_ONE, T])])
}

/** Limit from the leading term c t^v of the series in t -> 0+. */
function fromSeries(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-', steps: Step[] | undefined): Expr {
  const inT = substitute(e, [[x, frame(x, x0, dir)]])
  const s = seriesAt0(inT, T, 1)
  const lead = s.c[0]
  let out: Expr
  if (lead === undefined || s.v > 0) out = ZERO
  else if (hasSymbol(lead, LOG_T)) {
    // lead = sum a_j L^j with L = log t -> -oo: the top power a_k L^k decides the sign; t^v (v <= 0) only helps.
    const k = maxPower(lead, LOG_T)
    const ak = diff(lead, [LOG_T, k])
    out = makeMul([makeFn('sign', [ak]), asExpr(k % 2 === 0 ? 1 : -1), OO])
  } else if (s.v === 0) out = lead
  else out = makeMul([makeFn('sign', [lead]), OO])
  if (steps !== undefined) {
    steps.push({ rule: `series expansion: leading term of order ${s.v}`, before: limitNode(e, x, x0, dir), after: out })
  }
  return out
}

function maxPower(e: Expr, s: ExprSym): number {
  let k = 0
  for (let j = 1; j <= 8; j++) {
    const dj = diff(e, [s, j])
    if (dj.kind === 'num' && dj.n === 0n) break
    k = j
  }
  return k
}

function limitNode(e: Expr, x: ExprSym, x0: Expr, dir: string): Expr {
  return rawFn('Limit', [e, x, x0, sym(dir === '+' ? 'right' : dir === '-' ? 'left' : 'both')])
}

const isZeroLimit = (v: Expr): boolean => v.kind === 'num' && v.n === 0n

function lHopital(
  n: Expr,
  d: Expr,
  e: Expr,
  x: ExprSym,
  x0: Expr,
  dir: '+' | '-',
  steps: Step[] | undefined,
  depth: number,
): Expr {
  const ln = oneSided(n, x, x0, dir, undefined, depth + 1)
  const ld = oneSided(d, x, x0, dir, undefined, depth + 1)
  if ((isZeroLimit(ln) && isZeroLimit(ld)) || (isInfinite(ln) && isInfinite(ld))) {
    const next = makeMul([diff(n, x), makePow(diff(d, x), MINUS_ONE)])
    if (steps !== undefined) {
      steps.push({ rule: "l'Hopital's rule", before: limitNode(e, x, x0, dir), after: limitNode(next, x, x0, dir) })
    }
    return oneSided(next, x, x0, dir, steps, depth + 1)
  }
  if (!isZeroLimit(ld) && isFiniteExpr(ld)) return makeMul([ln, makePow(ld, MINUS_ONE)])
  if (isZeroLimit(ld) && !isZeroLimit(ln))
    return makeMul([makeFn('sign', [makeMul([ln, sampleSign(d, x, x0, dir)])]), OO])
  return makeMul([ln, makePow(ld, MINUS_ONE)])
}

/** The sign of an expression just beside the point (from a sample), as an expression. */
function sampleSign(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-'): Expr {
  const v = sample(e, x, x0, dir, 1e-8)
  return asExpr(v === undefined || v === 0 ? 1 : Math.sign(v))
}

/** Limits of compound expressions from the limits of their parts, falling back to l'Hopital's rule. */
function compose(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-', steps: Step[] | undefined, depth: number): Expr {
  const lim = (y: Expr): Expr => oneSided(y, x, x0, dir, undefined, depth + 1)
  if (e.kind === 'fn' && e.args.length === 1) {
    const L = lim(e.args[0] as Expr)
    if (e.name === 'log' && isZeroLimit(L)) return NEG_INF
    if (DISCONTINUOUS.has(e.name)) {
      const z = evalComplex(L, new Map())
      if (z === undefined || Number.isInteger(z[0])) throw new WhiteboxError('not-implemented', 'limit: at a jump')
    }
    const v = makeFn(e.name, [L])
    if (isFiniteExpr(v) || v.key === OO.key || v.key === NEG_INF.key) {
      if (steps !== undefined)
        steps.push({ rule: `continuity of ${e.name}`, before: limitNode(e, x, x0, dir), after: v })
      return v
    }
    throw new WhiteboxError('not-implemented', `limit: ${e.name} has no limit here (it oscillates or is undefined)`)
  }
  if (e.kind === 'pow') {
    if (hasSymbol(e.exp, x)) {
      // f^g = exp(g log f).
      const inner = oneSided(makeMul([e.exp, makeFn('log', [e.base])]), x, x0, dir, steps, depth + 1)
      return makeFn('exp', [inner])
    }
    const Lb = lim(e.base)
    const v = makePow(Lb, e.exp)
    if (isFiniteExpr(v) || v.key === OO.key || v.key === NEG_INF.key) return v
    if (isZeroLimit(Lb)) return makeMul([makeFn('sign', [makePow(sampleSign(e.base, x, x0, dir), e.exp)]), OO])
    throw new WhiteboxError('not-implemented', 'limit: no method applies to this power')
  }
  if (e.kind === 'mul') {
    const Ls = e.args.map(lim)
    const zi = Ls.findIndex(isZeroLimit)
    const ii = Ls.findIndex(isInfinite)
    if (zi < 0 || ii < 0) {
      const v = makeMul(Ls)
      if (v.key !== NAN.key) return v
    }
    if (zi >= 0 && ii >= 0) {
      // 0 * oo: write f*g*rest as (g*rest)/(1/f), or as (f*rest)/(1/g) when that fails.
      const attempts = [zi, ii].map((k) => () => {
        const top = makeMul(e.args.filter((_, i) => i !== k))
        return lHopital(top, makePow(e.args[k] as Expr, MINUS_ONE), e, x, x0, dir, steps, depth)
      })
      for (const run of attempts) {
        try {
          return run()
        } catch (err) {
          if (!(err instanceof WhiteboxError)) throw err
        }
      }
    }
  }
  if (e.kind === 'add') {
    const Ls = e.args.map(lim)
    const v = makeAdd(Ls)
    if (v.key !== NAN.key) return v
  }
  const [n, d] = numerDenom(together(e))
  if (!hasSymbol(d, x)) throw new WhiteboxError('not-implemented', 'limit: no method applies')
  return lHopital(n, d, e, x, x0, dir, steps, depth)
}

function oneSided(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-', steps: Step[] | undefined, depth: number): Expr {
  if (depth > 12) throw new WhiteboxError('not-implemented', 'limit: no method terminated')
  if (!hasSymbol(e, x)) return e
  if (isFiniteExpr(x0) && !hasDiscontinuity(e)) {
    const direct = substitute(e, [[x, x0]])
    if (isFiniteExpr(direct)) {
      if (steps !== undefined)
        steps.push({ rule: 'direct substitution', before: limitNode(e, x, x0, dir), after: direct })
      return direct
    }
  }
  try {
    return fromSeries(e, x, x0, dir, steps)
  } catch (err) {
    if (!(err instanceof WhiteboxError) || (err.code !== 'not-implemented' && err.code !== 'convergence')) throw err
  }
  return compose(e, x, x0, dir, steps, depth)
}

function sample(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-', h: number): number | undefined {
  const xv = x0.key === OO.key ? 1 / h : x0.key === NEG_INF.key ? -1 / h : undefined
  const env = new Map<string, C>()
  if (xv !== undefined) env.set(x.name, [xv, 0])
  else {
    const z = evalComplex(x0, new Map())
    if (z === undefined) return undefined
    env.set(x.name, [z[0] + (dir === '+' ? h : -h), 0])
  }
  const v = evalComplex(e, env)
  if (v === undefined || Math.abs(v[1]) > 1e-9 * Math.max(1, Math.abs(v[0]))) return undefined
  return v[0]
}

/** Compare with values approaching the point; throws when they clearly contradict the symbolic result. */
function crossCheck(e: Expr, x: ExprSym, x0: Expr, dir: '+' | '-', value: Expr, steps: Step[] | undefined): void {
  if (freeNames(e).some((n) => n !== x.name) || freeNames(x0).length > 0) return
  // Moderate steps: close enough to see the limit, far enough that cancellation does not swamp the value.
  const v1 = sample(e, x, x0, dir, 1e-2)
  const v2 = sample(e, x, x0, dir, 1e-3)
  if (v1 === undefined || v2 === undefined || !Number.isFinite(v1) || !Number.isFinite(v2)) return
  const L = evalComplex(value, new Map())
  if (L === undefined || Math.abs(L[1]) > 0) return
  const target = L[0]
  let ok: boolean
  if (!Number.isFinite(target)) ok = Math.sign(v2) === Math.sign(target) && Math.abs(v2) >= Math.abs(v1) * 0.999
  else {
    const close = Math.abs(v2 - target) <= 1e-2 * Math.max(1, Math.abs(target))
    const trend = Math.abs(v2 - target) < Math.abs(v1 - target)
    ok = close || trend
  }
  if (steps !== undefined) {
    steps.push({
      rule: `numeric check: f = ${v1.toPrecision(6)}, ${v2.toPrecision(6)} near the point`,
      before: value,
      after: value,
    })
  }
  if (!ok) {
    throw new WhiteboxError(
      'convergence',
      `limit: the symbolic result ${value.key} disagrees with values ${v1}, ${v2} near the point`,
    )
  }
}

function limitImpl(e: Expr, x: ExprSym, x0v: ExprLike, dir: LimitDirection, steps: Step[] | undefined): Expr {
  const x0 = asExpr(x0v)
  if (dir === '+-' && !isInfinite(x0)) {
    const a = limitImpl(e, x, x0, '+', steps)
    const b = limitImpl(e, x, x0, '-', steps)
    if (a.key !== b.key) throw new WhiteboxError('value', `limit: the one-sided limits differ (${a.key} and ${b.key})`)
    return a
  }
  const side = dir === '-' ? '-' : '+'
  const side2 = x0.key === OO.key ? '-' : x0.key === NEG_INF.key ? '+' : side
  const value = oneSided(e, x, x0, side, steps, 0)
  crossCheck(e, x, x0, side2 === side ? side : side, value, steps)
  return value
}

/**
 * The limit of an expression as x tends to x0 (SymPy's `limit`), by direct substitution where the expression is
 * continuous, otherwise from the leading term of its series (Laurent series at poles, log terms), with l'Hopital's
 * rule for quotients the series cannot expand and exp(lim g log f) for powers. Approaching from the right is the
 * default, as in SymPy; `'+-'` requires both sides to agree. When the expression has no other symbols the result is
 * cross-checked against values near the point.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The point (a number, an expression, or `oo` / `-oo`).
 * @param dir - `'+'` (default), `'-'` or `'+-'`.
 * @returns The limit (possibly `oo`, `-oo`).
 * @throws WhiteboxError with code `value` when two-sided limits differ, `convergence` when the numeric check
 *   contradicts the result, `not-implemented` when no method applies.
 * @example
 * ```ts
 * import { limit, oo, parse, sym, toText } from 'whitebox-engine/symbolic'
 * const x = sym('x')
 * toText(limit(parse('sin(x)/x'), x, 0)) // => '1'
 * toText(limit(parse('(1 + 1/x)^x'), x, oo)) // => 'E'
 * toText(limit(parse('1/x'), x, 0, '-')) // => '-oo'
 * ```
 */
export function limit(e: Expr, x: ExprSym, x0: ExprLike, dir: LimitDirection = '+'): Expr {
  return limitImpl(e, x, x0, dir, undefined)
}

/**
 * {@link limit} with the steps taken (substitution, series leading term, l'Hopital, numeric check).
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The point.
 * @param dir - Direction.
 * @returns The limit and its steps.
 * @example
 * ```ts
 * import { limitSteps, parse, sym } from 'whitebox-engine/symbolic'
 * limitSteps(parse('(1 - cos(x))/x^2'), sym('x'), 0).steps[0]?.rule // => 'series expansion: leading term of order 0'
 * ```
 */
export function limitSteps(e: Expr, x: ExprSym, x0: ExprLike, dir: LimitDirection = '+'): Derivation {
  const steps: Step[] = []
  const result = limitImpl(e, x, x0, dir, steps)
  return { result, steps }
}
