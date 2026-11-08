// Rational-function rewrites: numerator/denominator splitting, `together`, `cancel` and partial fractions (`apart`,
// by undetermined coefficients over the rationals extended by any other symbols).
import { OmaiError } from '../base/index.ts'
import {
  isNumeric,
  isRat,
  keepCoeff,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  numFromRational,
  ONE,
  ratOf,
  ZERO,
} from './_canon.ts'
import { factorMultivariate } from './_factor.ts'
import { hasSymbol } from './_node.ts'
import {
  exprToP,
  findGensAll,
  type P,
  padd,
  pcoeffsIn,
  pconst,
  pdeg,
  pdivExact,
  pgcd,
  pIsZero,
  plcIn,
  pmul,
  pnumericContent,
  ppow,
  psub,
  pToExpr,
  pvar,
} from './_poly.ts'
import { blcm, R0, R1, rat, rdiv } from './_rational.ts'
import type { Expr, ExprSym } from './expr.ts'

/** Split a product into numerator and denominator (SymPy's `as_numer_denom` for products and powers). @internal */
export function numerDenom(e: Expr): [Expr, Expr] {
  const num: Expr[] = []
  const den: Expr[] = []
  for (const f of e.kind === 'mul' ? e.args : [e]) {
    if (f.kind === 'num' && f.d !== 1n) {
      num.push(numFromRational(rat(f.n)))
      den.push(numFromRational(rat(f.d)))
    } else if (f.kind === 'pow' && isNumeric(f.exp) && (f.exp.kind === 'num' ? f.exp.n < 0n : f.exp.value < 0)) {
      den.push(makePow(f.base, makeMul([MINUS_ONE, f.exp])))
    } else if (f.kind === 'pow' && f.exp.kind === 'mul' && isNumeric(f.exp.args[0] as Expr)) {
      const c = f.exp.args[0] as Expr
      if (c.kind === 'num' ? c.n < 0n : c.kind === 'float' && c.value < 0) {
        den.push(makePow(f.base, makeMul([MINUS_ONE, f.exp])))
      } else num.push(f)
    } else num.push(f)
  }
  return [makeMul(num), makeMul(den)]
}

interface Part {
  readonly num: Expr
  readonly dens: Map<string, { base: Expr; exp: Expr }>
  readonly scale: bigint
}

function parts(e: Expr): Part {
  const [n, d] = numerDenom(e)
  const dens = new Map<string, { base: Expr; exp: Expr }>()
  let scale = 1n
  for (const f of d.kind === 'mul' ? d.args : [d]) {
    if (f.kind === 'num') {
      scale *= f.n
      continue
    }
    const [base, exp] = f.kind === 'pow' ? [f.base, f.exp] : [f, ONE]
    const key = exp.kind === 'num' ? base.key : `${base.key}^${exp.key}`
    dens.set(key, exp.kind === 'num' ? { base, exp } : { base: f, exp: ONE })
  }
  return { num: n, dens, scale }
}

/**
 * Combine a sum over a common denominator without expanding (SymPy's `together`): powers of the same base share the
 * highest power, numeric denominators their lcm.
 * @internal
 */
export function together(e: Expr): Expr {
  if (e.kind === 'mul') return makeMul(e.args.map(together))
  if (e.kind === 'pow') return makePow(together(e.base), e.exp)
  if (e.kind !== 'add') return e
  const ps = e.args.map((t) => parts(together(t)))
  let lcm = 1n
  for (const p of ps) lcm = blcm(lcm, p.scale)
  const common = new Map<string, { base: Expr; exp: Expr }>()
  for (const p of ps) {
    for (const [k, v] of p.dens) {
      const old = common.get(k)
      if (
        old === undefined ||
        (old.exp.kind === 'num' &&
          v.exp.kind === 'num' &&
          ratOf(v.exp).n * ratOf(old.exp).d > ratOf(old.exp).n * ratOf(v.exp).d)
      ) {
        common.set(k, v)
      }
    }
  }
  if (common.size === 0 && lcm === 1n) return e
  const numerators = ps.map((p) => {
    const extra: Expr[] = [numFromRational(rat(lcm / p.scale))]
    for (const [k, v] of common) {
      const own = p.dens.get(k)
      extra.push(makePow(v.base, own === undefined ? v.exp : makeAdd([v.exp, makeMul([MINUS_ONE, own.exp])])))
    }
    return makeMul([p.num, ...extra])
  })
  const numerator = makeAdd(numerators)
  const denominator = makeMul([...common.values()].map((v) => makePow(v.base, makeMul([MINUS_ONE, v.exp]))))
  return keepCoeff(numFromRational(rat(1n, lcm)), makeMul([numerator, denominator]))
}

/**
 * Cancel common factors of a rational function: p/q with p and q expanded, coprime and primitive, and a rational
 * constant in front (SymPy's `cancel`).
 * @internal
 */
export function cancel(e: Expr): Expr {
  if (e.kind === 'fn' || e.kind === 'sym' || isNumeric(e) || e.kind === 'const') return e
  const t = together(e)
  const [n, d] = numerDenom(t)
  const gens = findGensAll([n, d])
  const pn = exprToP(n, gens)
  const pd = exprToP(d, gens)
  if (pn === undefined || pd === undefined || pIsZero(pd)) return t
  if (pIsZero(pn)) return ZERO
  const g = pgcd(pn, pd)
  const a = pdivExact(pn, g) ?? pn
  const b = pdivExact(pd, g) ?? pd
  const [ca, pa] = pnumericContent(a)
  const [cb, pb] = pnumericContent(b)
  const c = numFromRational(rdiv(ca, cb))
  const body = makeMul([pToExpr(pa, gens), makePow(pToExpr(pb, gens), MINUS_ONE)])
  // SymPy's cancel returns c*(p/q) evaluated: a constant distributes over a polynomial result.
  return makeMul([c, body])
}

/** Gaussian elimination over expressions (entries normalised by cancel); unique solution or an error. */
function solveSquare(rows: Expr[][], rhs: Expr[]): Expr[] {
  const n = rhs.length
  const m = rows.map((r, i) => [...r, rhs[i] as Expr])
  const isZero = (x: Expr): boolean => cancel(x).key === ZERO.key
  const width = (m[0]?.length ?? 1) - 1
  let r = 0
  const pivotCols: number[] = []
  for (let c = 0; c < width && r < n; c++) {
    let piv = -1
    for (let i = r; i < n; i++) {
      if (!isZero((m[i] as Expr[])[c] as Expr)) {
        piv = i
        break
      }
    }
    if (piv < 0) continue
    ;[m[r], m[piv]] = [m[piv] as Expr[], m[r] as Expr[]]
    const row = m[r] as Expr[]
    const inv = makePow(row[c] as Expr, MINUS_ONE)
    for (let j = 0; j <= width; j++) row[j] = cancel(makeMul([row[j] as Expr, inv]))
    for (let i = 0; i < n; i++) {
      if (i === r) continue
      const other = m[i] as Expr[]
      const f = other[c] as Expr
      if (isZero(f)) continue
      for (let j = 0; j <= width; j++)
        other[j] = cancel(makeAdd([other[j] as Expr, makeMul([MINUS_ONE, f, row[j] as Expr])]))
    }
    pivotCols.push(c)
    r++
  }
  if (pivotCols.length < width) throw new OmaiError('singular', 'apart: singular coefficient system')
  const out: Expr[] = new Array<Expr>(width).fill(ZERO)
  pivotCols.forEach((c, i) => {
    out[c] = (m[i] as Expr[])[width] as Expr
  })
  return out
}

/**
 * Partial fraction decomposition in x by undetermined coefficients: the polynomial part plus a sum of
 * N(x)/f(x)^j over the irreducible factors f of the denominator (over the rationals, with other symbols as
 * parameters), deg N < deg f.
 * @internal
 */
export function apart(e: Expr, x: ExprSym): Expr {
  const t = cancel(e)
  const [n, d] = numerDenom(t)
  if (!hasSymbol(d, x)) return t
  const others = findGensAll([n, d]).filter((g) => g.key !== x.key)
  if (others.some((g) => hasSymbol(g, x))) {
    throw new OmaiError('value', 'apart: the expression must be a rational function of x')
  }
  const gens = [x as Expr, ...others]
  const pn = exprToP(n, gens)
  const pdFull = exprToP(d, gens)
  if (pn === undefined || pdFull === undefined) throw new OmaiError('value', 'apart: not a rational function')
  // Factor the denominator; factors free of x join the constant in front.
  const [constant, factors] = factorMultivariate(pdFull)
  let front: Expr = numFromRational(constant)
  let pd = pconst(pdFull.n, R1)
  const xFactors: [P, number][] = []
  for (const [f, k] of factors) {
    if (pdeg(f, 0) <= 0) front = makeMul([front, makePow(pToExpr(f, gens), numFromRational(rat(BigInt(k))))])
    else {
      xFactors.push([f, k])
      pd = pmul(pd, ppow(f, k))
    }
  }
  // Polynomial part by pseudo-division in x: s * n = quotient * pd + remainder with s a power of lc(pd).
  const lc = plcIn(pd, 0)
  const db = pdeg(pd, 0)
  let remainder = pn
  let quotient = pconst(pd.n, R0)
  let s = pconst(pd.n, R1)
  while (!pIsZero(remainder) && pdeg(remainder, 0) >= db) {
    const t = pmul(plcIn(remainder, 0), ppow(pvar(pd.n, 0), pdeg(remainder, 0) - db))
    remainder = psub(pmul(lc, remainder), pmul(t, pd))
    quotient = padd(pmul(lc, quotient), t)
    s = pmul(lc, s)
  }
  const scale = makePow(makeMul([front, pToExpr(s, gens)]), MINUS_ONE)
  // Unknown numerator coefficients: deg f of them for each f^j, j <= k.
  const blocks: { f: P; j: number; deg: number; offset: number }[] = []
  let count = 0
  for (const [f, k] of xFactors) {
    const deg = pdeg(f, 0)
    for (let j = 1; j <= k; j++) {
      blocks.push({ f, j, deg, offset: count })
      count += deg
    }
  }
  const total = pdeg(pd, 0)
  const rows: Expr[][] = Array.from({ length: total }, () => new Array<Expr>(count).fill(ZERO))
  const rhs: Expr[] = new Array<Expr>(total).fill(ZERO)
  for (const [k, c] of pcoeffsIn(remainder, 0)) if (k < total) rhs[k] = pToExpr(c, gens)
  for (const b of blocks) {
    const cofactor = pdivExact(pd, ppow(b.f, b.j)) ?? pd
    for (let m = 0; m < b.deg; m++) {
      const term = pmul(cofactor, ppow(pvar(pd.n, 0), m))
      for (const [k, c] of pcoeffsIn(term, 0)) {
        if (k >= total) continue
        const row = rows[k] as Expr[]
        row[b.offset + m] = makeAdd([row[b.offset + m] as Expr, pToExpr(c, gens)])
      }
    }
  }
  const sol = solveSquare(rows, rhs)
  const terms: Expr[] = [makeMul([scale, pToExpr(quotient, gens)])]
  for (const b of blocks) {
    const coeffs = Array.from({ length: b.deg }, (_, m) => cancel(makeMul([scale, sol[b.offset + m] as Expr])))
    const numer = makeAdd(coeffs.map((c, m) => makeMul([c, makePow(x, numFromRational(rat(BigInt(m))))])))
    if (isRat(numer, 0n)) continue
    const den = makePow(pToExpr(b.f, gens), numFromRational(rat(BigInt(-b.j))))
    // A rational numerator is written as content times a primitive polynomial: -(x + 1)/(2*(x**2 + 1)), as SymPy.
    const nP = exprToP(cancel(numer), gens)
    if (nP !== undefined) {
      const [c, prim] = pnumericContent(nP)
      terms.push(makeMul([numFromRational(c), pToExpr(prim, gens), den]))
    } else terms.push(makeMul([cancel(numer), den]))
  }
  return makeAdd(terms)
}
