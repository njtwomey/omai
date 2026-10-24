// Automatic evaluation of known functions on construction, as SymPy's `eval` class methods do: special values at
// rational multiples of π, odd and even symmetry, shifts by multiples of π/2, inverse pairs, exp-log cancellation and
// exact values of Abs, sign, floor, gamma and factorial. Nothing here may touch _canon exports at module load
// (the modules import each other), so every table is built inside a function.
import { isInteger, isNegative, isNonnegative, isPositive, isReal } from './_assume.ts'
import {
  asCoeffMul,
  E,
  HALF,
  I,
  isNumeric,
  isRat,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  NAN,
  NEG_INF,
  numFromFloat,
  numFromRational,
  ONE,
  OO,
  PI,
  ratOf,
  TWO,
  ZERO,
  ZOO,
} from './_canon.ts'
import { rawFn } from './_node.ts'
import { evalComplexConstant } from './_numeric.ts'
import { couldExtractMinus } from './_order.ts'
import { R0, R1, type Rational, rabs, radd, rat, rcmp, rfloor, rmul, rsub } from './_rational.ts'
import type { Expr } from './expr.ts'

const q = (n: number, d = 1): Expr => numFromRational(rat(BigInt(n), BigInt(d)))
const sqrtOf = (n: number): Expr => makePow(q(n), HALF)
const neg = (e: Expr): Expr => makeMul([MINUS_ONE, e])

/** The rational r with e = r*π, or undefined. */
function piMultiple(e: Expr): Rational | undefined {
  if (e.key === PI.key) return R1
  if (e.kind === 'num' && e.n === 0n) return R0
  if (e.kind === 'mul' && e.args.length === 2 && e.args[0]?.kind === 'num' && e.args[1]?.key === PI.key) {
    return ratOf(e.args[0])
  }
  return undefined
}

/** Split e = rest + r*π. */
function splitPi(e: Expr): [Expr, Rational] {
  if (e.kind !== 'add') {
    const r = piMultiple(e)
    return r === undefined ? [e, R0] : [ZERO, r]
  }
  let r: Rational = R0
  const rest: Expr[] = []
  for (const t of e.args) {
    const m = piMultiple(t)
    if (m === undefined) rest.push(t)
    else r = radd(r, m)
  }
  return [makeAdd(rest), r]
}

/** r mod m in [0, m). */
function mod(r: Rational, m: Rational): Rational {
  const k = rfloor(rmul(r, { n: m.d, d: m.n }))
  return rsub(r, rmul(m, rat(k)))
}

function key(r: Rational): string {
  return `${r.n}/${r.d}`
}

function sinTable(): Record<string, Expr> {
  const s6 = sqrtOf(6)
  const s2 = sqrtOf(2)
  return {
    '0/1': ZERO,
    '1/12': makeMul([q(1, 4), makeAdd([s6, neg(s2)])]),
    '1/6': q(1, 2),
    '1/4': makeMul([q(1, 2), s2]),
    '1/3': makeMul([q(1, 2), sqrtOf(3)]),
    '5/12': makeMul([q(1, 4), makeAdd([s6, s2])]),
    '1/2': ONE,
  }
}

function tanTable(): Record<string, Expr> {
  const s3 = sqrtOf(3)
  return {
    '0/1': ZERO,
    '1/12': makeAdd([TWO, neg(s3)]),
    '1/6': makeMul([q(1, 3), s3]),
    '1/4': ONE,
    '1/3': s3,
    '5/12': makeAdd([TWO, s3]),
    '1/2': ZOO,
  }
}

/** sin(r*π) for rational r: exact for denominators 1, 2, 3, 4, 6 and 12, otherwise reduced into [0, π/2]. */
function sinPi(r: Rational): Expr {
  let a = mod(r, rat(2n))
  let sign = 1
  if (rcmp(a, R1) >= 0) {
    a = rsub(a, R1)
    sign = -1
  }
  if (rcmp(a, rat(1n, 2n)) > 0) a = rsub(R1, a)
  const v = sinTable()[key(a)] ?? rawFn('sin', [makeMul([numFromRational(a), PI])])
  return sign < 0 ? neg(v) : v
}

function cosPi(r: Rational): Expr {
  let a = mod(r, rat(2n))
  if (rcmp(a, R1) > 0) a = rsub(rat(2n), a)
  let sign = 1
  if (rcmp(a, rat(1n, 2n)) > 0) {
    a = rsub(R1, a)
    sign = -1
  }
  const v = sinTable()[key(rsub(rat(1n, 2n), a))] ?? rawFn('cos', [makeMul([numFromRational(a), PI])])
  return sign < 0 ? neg(v) : v
}

function tanPi(r: Rational): Expr {
  let a = mod(r, R1)
  let sign = 1
  if (rcmp(a, rat(1n, 2n)) > 0) {
    a = rsub(R1, a)
    sign = -1
  }
  const v = tanTable()[key(a)] ?? rawFn('tan', [makeMul([numFromRational(a), PI])])
  return sign < 0 && v.key !== ZOO.key ? neg(v) : v
}

function trig(name: 'sin' | 'cos' | 'tan' | 'cot', x: Expr): Expr | undefined {
  const odd = name !== 'cos'
  if (couldExtractMinus(x)) {
    const v = makeFn(name, [neg(x)])
    return odd ? neg(v) : v
  }
  const [rest, r] = splitPi(x)
  if (rest.kind === 'num' && rest.n === 0n) {
    if (name === 'sin') return sinPi(r)
    if (name === 'cos') return cosPi(r)
    const t = tanPi(r)
    if (name === 'tan') return t
    return t.kind === 'fn' ? undefined : makePow(t, MINUS_ONE)
  }
  if (r.n === 0n) {
    const inverse = { sin: 'asin', cos: 'acos', tan: 'atan', cot: 'acot' }[name]
    if (x.kind === 'fn' && x.name === inverse) return x.args[0]
    return undefined
  }
  // Shift by a multiple of π/2.
  const period = name === 'sin' || name === 'cos' ? rat(2n) : R1
  const m = mod(r, period)
  const half = rmul(m, rat(2n))
  if (half.d !== 1n) {
    if (rcmp(m, r) === 0) return undefined
    return makeFn(name, [makeAdd([rest, makeMul([numFromRational(m), PI])])])
  }
  const k = Number(half.n)
  const table: Record<string, [string, number][]> = {
    sin: [
      ['sin', 1],
      ['cos', 1],
      ['sin', -1],
      ['cos', -1],
    ],
    cos: [
      ['cos', 1],
      ['sin', -1],
      ['cos', -1],
      ['sin', 1],
    ],
    tan: [
      ['tan', 1],
      ['cot', -1],
    ],
    cot: [
      ['cot', 1],
      ['tan', -1],
    ],
  }
  const [f, s] = (table[name] as [string, number][])[k] as [string, number]
  const v = makeFn(f, [rest])
  return s < 0 ? neg(v) : v
}

function inverseTrig(name: string, x: Expr): Expr | undefined {
  const odd = name !== 'acos'
  if (odd && couldExtractMinus(x)) return neg(makeFn(name, [neg(x)]))
  const table: Record<string, [Expr, Expr][]> = {
    asin: [
      [ZERO, ZERO],
      [q(1, 2), makeMul([q(1, 6), PI])],
      [makeMul([q(1, 2), sqrtOf(2)]), makeMul([q(1, 4), PI])],
      [makeMul([q(1, 2), sqrtOf(3)]), makeMul([q(1, 3), PI])],
      [ONE, makeMul([q(1, 2), PI])],
    ],
    acos: [
      [ONE, ZERO],
      [makeMul([q(1, 2), sqrtOf(3)]), makeMul([q(1, 6), PI])],
      [makeMul([q(1, 2), sqrtOf(2)]), makeMul([q(1, 4), PI])],
      [q(1, 2), makeMul([q(1, 3), PI])],
      [ZERO, makeMul([q(1, 2), PI])],
      [q(-1, 2), makeMul([q(2, 3), PI])],
      [makeMul([q(-1, 2), sqrtOf(2)]), makeMul([q(3, 4), PI])],
      [makeMul([q(-1, 2), sqrtOf(3)]), makeMul([q(5, 6), PI])],
      [MINUS_ONE, PI],
    ],
    atan: [
      [ZERO, ZERO],
      [makeMul([q(1, 3), sqrtOf(3)]), makeMul([q(1, 6), PI])],
      [ONE, makeMul([q(1, 4), PI])],
      [sqrtOf(3), makeMul([q(1, 3), PI])],
      [OO, makeMul([q(1, 2), PI])],
    ],
    acot: [
      [ZERO, makeMul([q(1, 2), PI])],
      [makeMul([q(1, 3), sqrtOf(3)]), makeMul([q(1, 3), PI])],
      [ONE, makeMul([q(1, 4), PI])],
      [sqrtOf(3), makeMul([q(1, 6), PI])],
      [OO, ZERO],
    ],
  }
  for (const [a, v] of table[name] ?? []) if (a.key === x.key) return v
  return undefined
}

function hyperbolic(name: string, x: Expr): Expr | undefined {
  const odd = name !== 'cosh' && name !== 'acosh'
  if (name !== 'acosh' && couldExtractMinus(x)) {
    const v = makeFn(name, [neg(x)])
    return odd ? neg(v) : v
  }
  if (x.kind === 'num' && x.n === 0n) {
    return { sinh: ZERO, cosh: ONE, tanh: ZERO, coth: ZOO, asinh: ZERO, atanh: ZERO, acosh: undefined }[name]
  }
  if (name === 'acosh' && isRat(x, 1n)) return ZERO
  if (x.key === OO.key) return { sinh: OO, cosh: OO, tanh: ONE, coth: ONE, asinh: OO, acosh: OO }[name]
  const inverse: Record<string, string> = { sinh: 'asinh', cosh: 'acosh', tanh: 'atanh', asinh: 'sinh', atanh: 'tanh' }
  if (
    x.kind === 'fn' &&
    x.name === inverse[name] &&
    (name.startsWith('a') ? isReal(x.args[0] as Expr) === true : true)
  ) {
    return x.args[0]
  }
  return undefined
}

function expEval(x: Expr): Expr | undefined {
  if (x.kind === 'num' && x.n === 0n) return ONE
  if (isRat(x, 1n)) return E
  if (x.key === OO.key) return OO
  if (x.key === NEG_INF.key) return ZERO
  if (x.key === NAN.key) return NAN
  // exp(I*pi*r) for r a multiple of 1/2.
  if (x.kind === 'mul') {
    const [c, rest] = asCoeffMul(x)
    if (c.kind === 'num' && rest.kind === 'mul' && rest.args.length === 2 && rest.args[0]?.key === I.key) {
      if (rest.args[1]?.key === PI.key && (c.d === 1n || c.d === 2n)) {
        const k = Number(((((c.n * 2n) / c.d) % 4n) + 4n) % 4n)
        return [ONE, I, MINUS_ONE, neg(I)][k]
      }
    }
  }
  // exp(a + n*log(b)) = b^n * exp(a)
  const terms = x.kind === 'add' ? x.args : [x]
  const pulled: Expr[] = []
  const kept: Expr[] = []
  for (const t of terms) {
    const [c, rest] = asCoeffMul(t)
    if (rest.kind === 'fn' && rest.name === 'log' && isNumeric(c)) pulled.push(makePow(rest.args[0] as Expr, c))
    else kept.push(t)
  }
  if (pulled.length === 0) return undefined
  return makeMul([...pulled, kept.length === 0 ? ONE : makeFn('exp', [makeAdd(kept)])])
}

function logEval(x: Expr): Expr | undefined {
  if (isRat(x, 1n)) return ZERO
  if (x.kind === 'num' && x.n === 0n) return ZOO
  if (x.key === E.key) return ONE
  if (x.key === OO.key || x.key === NEG_INF.key) return OO
  if (x.key === I.key) return makeMul([q(1, 2), I, PI])
  if (x.kind === 'num') {
    if (x.n < 0n) {
      const pos = numFromRational(rabs(ratOf(x)))
      return makeAdd([makeFn('log', [pos]), makeMul([I, PI])])
    }
    if (x.n === 1n) return neg(makeFn('log', [numFromRational(rat(x.d))]))
    return undefined
  }
  if (x.kind === 'float') return x.value > 0 ? numFromFloat(Math.log(x.value)) : undefined
  if (x.kind === 'mul' && x.args.length === 2 && isRat(x.args[0] as Expr, -1n) && x.args[1]?.key === I.key) {
    return makeMul([q(-1, 2), I, PI])
  }
  if (x.kind === 'fn' && x.name === 'exp' && isReal(x.args[0] as Expr) === true) return x.args[0]
  return undefined
}

function absEval(x: Expr): Expr | undefined {
  if (x.kind === 'num') return numFromRational(rabs(ratOf(x)))
  if (x.kind === 'float') return numFromFloat(Math.abs(x.value))
  if (x.key === I.key) return ONE
  if (x.key === OO.key || x.key === NEG_INF.key || x.key === ZOO.key) return OO
  if (couldExtractMinus(x)) return makeFn('Abs', [neg(x)])
  if (isNonnegative(x) === true) return x
  if (isNegative(x) === true) return neg(x)
  if (x.kind === 'mul') {
    // As SymPy: factors with a known absolute value come out; the unknown ones stay together inside one Abs.
    const known: Expr[] = []
    const unknown: Expr[] = []
    for (const t of x.args) {
      if (t.kind === 'pow' && t.exp.kind === 'num' && t.exp.d === 1n && t.exp.n < 0n) {
        known.push(makePow(makeFn('Abs', [t.base]), t.exp))
        continue
      }
      const a = makeFn('Abs', [t])
      if (a.kind === 'fn' && a.name === 'Abs') unknown.push(t)
      else known.push(a)
    }
    if (known.length === 0) return undefined
    return makeMul([...known, unknown.length === 0 ? ONE : rawFn('Abs', [makeMul(unknown)])])
  }
  if (x.kind === 'fn' && x.name === 'Abs') return x
  const z = evalComplexConstant(x)
  if (z !== undefined && z[1] === 0) return z[0] >= 0 ? x : neg(x)
  return undefined
}

function signEval(x: Expr): Expr | undefined {
  if (x.kind === 'num') return q(x.n > 0n ? 1 : x.n < 0n ? -1 : 0)
  if (x.kind === 'float') return q(Math.sign(x.value))
  if (isPositive(x) === true) return ONE
  if (isNegative(x) === true) return MINUS_ONE
  if (couldExtractMinus(x)) return neg(makeFn('sign', [neg(x)]))
  if (x.kind === 'mul') {
    const [c, rest] = asCoeffMul(x)
    if (c.kind === 'num' && !isRat(c, 1n)) return makeMul([signEval(c) ?? ONE, makeFn('sign', [rest])])
  }
  const z = evalComplexConstant(x)
  if (z !== undefined && z[1] === 0) return q(Math.sign(z[0]))
  return undefined
}

function roundingEval(name: 'floor' | 'ceiling', x: Expr): Expr | undefined {
  if (x.kind === 'num') {
    const f = rfloor(ratOf(x))
    return numFromRational(rat(name === 'floor' || x.d === 1n ? f : f + 1n))
  }
  if (x.kind === 'float') return numFromFloat(name === 'floor' ? Math.floor(x.value) : Math.ceil(x.value))
  if (isInteger(x) === true) return x
  const z = evalComplexConstant(x)
  if (z !== undefined && z[1] === 0) {
    const v = name === 'floor' ? Math.floor(z[0]) : Math.ceil(z[0])
    // Only trust the double when the value is not within rounding distance of an integer.
    if (Math.abs(z[0] - Math.round(z[0])) > 1e-9) return q(v)
  }
  return undefined
}

function factorialBig(n: bigint): bigint {
  let f = 1n
  for (let k = 2n; k <= n; k++) f *= k
  return f
}

function gammaEval(x: Expr): Expr | undefined {
  if (x.kind !== 'num') return undefined
  if (x.d === 1n) {
    if (x.n <= 0n) return ZOO
    if (x.n <= 1000n) return numFromRational(rat(factorialBig(x.n - 1n)))
    return undefined
  }
  if (x.d === 2n && x.n > -200n && x.n < 400n) {
    // Γ(n + 1/2) = (2n)! / (4^n n!) √π, extended to negative n by the reflection Γ(1/2 - n) = (-4)^n n! / (2n)! √π.
    const n = (x.n - 1n) / 2n
    const sqrtPi = makePow(PI, HALF)
    if (n >= 0n) return makeMul([numFromRational(rat(factorialBig(2n * n), 4n ** n * factorialBig(n))), sqrtPi])
    const m = -n
    return makeMul([numFromRational(rat((-4n) ** m * factorialBig(m), factorialBig(2n * m))), sqrtPi])
  }
  return undefined
}

function evalRule(name: string, args: readonly Expr[]): Expr | undefined {
  const x = args[0]
  if (x === undefined) return undefined
  switch (name) {
    case 'exp':
      return expEval(x)
    case 'log':
      return logEval(x)
    case 'sin':
    case 'cos':
    case 'tan':
    case 'cot':
      return trig(name, x)
    case 'sec':
    case 'csc': {
      const c = trig(name === 'sec' ? 'cos' : 'sin', x)
      return c === undefined || c.kind === 'fn' ? undefined : makePow(c, MINUS_ONE)
    }
    case 'asin':
    case 'acos':
    case 'atan':
    case 'acot':
      return inverseTrig(name, x)
    case 'sinh':
    case 'cosh':
    case 'tanh':
    case 'coth':
    case 'asinh':
    case 'acosh':
    case 'atanh':
      return hyperbolic(name, x)
    case 'Abs':
      return absEval(x)
    case 'sign':
      return signEval(x)
    case 'floor':
    case 'ceiling':
      return roundingEval(name, x)
    case 'erf':
      if (x.kind === 'num' && x.n === 0n) return ZERO
      if (x.key === OO.key) return ONE
      if (x.key === NEG_INF.key) return MINUS_ONE
      if (couldExtractMinus(x)) return neg(makeFn('erf', [neg(x)]))
      return undefined
    case 'gamma':
      return gammaEval(x)
    case 'factorial':
      if (x.kind === 'num' && x.d === 1n && x.n >= 0n && x.n <= 1000n) return numFromRational(rat(factorialBig(x.n)))
      return undefined
    case 're':
      return isReal(x) === true ? x : undefined
    case 'im':
      return isReal(x) === true ? ZERO : undefined
    case 'conjugate':
      return isReal(x) === true ? x : undefined
    case 'arg':
      return isPositive(x) === true ? ZERO : isNegative(x) === true ? PI : undefined
    case 'atan2': {
      const b = args[1]
      if (b === undefined) return undefined
      if (isPositive(b) === true) return makeFn('atan', [makeMul([x, makePow(b, MINUS_ONE)])])
      if (x.kind === 'num' && x.n === 0n && isNegative(b) === true) return PI
      return undefined
    }
    default:
      return undefined
  }
}

/** Arity of each known function (others are undefined functions of any arity). @internal */
export const ARITY: Readonly<Record<string, number>> = {
  atan2: 2,
  polygamma: 2,
}

/**
 * Apply a function by name, evaluating it where a rule applies. Unknown names become undefined functions.
 * @internal
 */
export function makeFn(name: string, args: readonly Expr[]): Expr {
  if (args.some((a) => a.key === NAN.key)) return NAN
  return evalRule(name, args) ?? rawFn(name, args)
}
