// Arithmetic evaluation for `is/2` and the comparison built-ins, following SWI-Prolog's defaults (iso false,
// prefer_rationals false): integer operations stay integers, `/` of integers is an integer when exact and a float
// otherwise, and integers are exact up to 2^53 (beyond that an int_overflow evaluation error is raised).
import { evaluationError, instantiationError, typeError } from './_errors.ts'
import { compound, type Num, num, type Term } from './terms.ts'

const int = (v: number, where: string): Num => {
  if (!Number.isSafeInteger(v)) {
    if (Number.isFinite(v)) evaluationError('int_overflow', where)
    evaluationError('undefined', where)
  }
  return { kind: 'num', value: v === 0 ? 0 : v, float: false }
}
const flt = (v: number, where: string): Num => {
  if (Number.isNaN(v)) evaluationError('undefined', where)
  if (!Number.isFinite(v)) evaluationError('float_overflow', where)
  return { kind: 'num', value: v, float: true }
}

function requireInt(x: Num, where: string): number {
  if (x.float) typeError('integer', x, where)
  return x.value
}

const CONSTANTS: Readonly<Record<string, Num>> = {
  pi: num(Math.PI, true),
  e: num(Math.E, true),
  inf: num(Number.POSITIVE_INFINITY, true),
  infinite: num(Number.POSITIVE_INFINITY, true),
  nan: num(Number.NaN, true),
  epsilon: num(Number.EPSILON, true),
  max_tagged_integer: num(2 ** 53 - 1),
}

type Unary = (x: Num, w: string) => Num
type Binary = (x: Num, y: Num, w: string) => Num

const both = (x: Num, y: Num): boolean => !x.float && !y.float

const UNARY: Readonly<Record<string, Unary>> = {
  '-': (x, w) => (x.float ? flt(-x.value, w) : int(-x.value, w)),
  '+': (x) => x,
  abs: (x, w) => (x.float ? flt(Math.abs(x.value), w) : int(Math.abs(x.value), w)),
  sign: (x, w) => (x.float ? flt(Math.sign(x.value), w) : int(Math.sign(x.value), w)),
  float: (x, w) => flt(x.value, w),
  integer: (x, w) => int(x.float ? Math.round(x.value) : x.value, w),
  float_integer_part: (x, w) => flt(Math.trunc(x.value), w),
  float_fractional_part: (x, w) => flt(x.value - Math.trunc(x.value), w),
  truncate: (x, w) => int(Math.trunc(x.value), w),
  round: (x, w) => int(Math.sign(x.value) * Math.round(Math.abs(x.value)), w),
  ceiling: (x, w) => int(Math.ceil(x.value), w),
  floor: (x, w) => int(Math.floor(x.value), w),
  sqrt: (x, w) => (x.value < 0 ? evaluationError('undefined', w) : flt(Math.sqrt(x.value), w)),
  sin: (x, w) => flt(Math.sin(x.value), w),
  cos: (x, w) => flt(Math.cos(x.value), w),
  tan: (x, w) => flt(Math.tan(x.value), w),
  asin: (x, w) => flt(Math.asin(x.value), w),
  acos: (x, w) => flt(Math.acos(x.value), w),
  atan: (x, w) => flt(Math.atan(x.value), w),
  sinh: (x, w) => flt(Math.sinh(x.value), w),
  cosh: (x, w) => flt(Math.cosh(x.value), w),
  tanh: (x, w) => flt(Math.tanh(x.value), w),
  asinh: (x, w) => flt(Math.asinh(x.value), w),
  acosh: (x, w) => flt(Math.acosh(x.value), w),
  atanh: (x, w) => flt(Math.atanh(x.value), w),
  exp: (x, w) => flt(Math.exp(x.value), w),
  log: (x, w) => (x.value <= 0 ? evaluationError('undefined', w) : flt(Math.log(x.value), w)),
  log2: (x, w) => (x.value <= 0 ? evaluationError('undefined', w) : flt(Math.log2(x.value), w)),
  '\\': (x, w) => int(~requireInt(x, w), w),
  msb: (x, w) => int(31 - Math.clz32(requireInt(x, w)), w),
  succ: (x, w) => int(requireInt(x, w) + 1, w),
}

const intDiv = (x: Num, y: Num, w: string): [number, number] => {
  const a = requireInt(x, w)
  const b = requireInt(y, w)
  if (b === 0) evaluationError('zero_divisor', w)
  return [a, b]
}

function power(x: Num, y: Num, w: string, caret: boolean): Num {
  if (both(x, y)) {
    if (y.value < 0) {
      if (x.value === 1) return int(1, w)
      if (x.value === -1) return int(y.value % 2 === 0 ? 1 : -1, w)
      if (caret) {
        if (x.value === 0) evaluationError('zero_divisor', w)
        typeError('float', x, w)
      }
      return flt(x.value ** y.value, w)
    }
    return int(x.value ** y.value, w)
  }
  return flt(x.value ** y.value, w)
}

const BINARY: Readonly<Record<string, Binary>> = {
  '+': (x, y, w) => (both(x, y) ? int(x.value + y.value, w) : flt(x.value + y.value, w)),
  '-': (x, y, w) => (both(x, y) ? int(x.value - y.value, w) : flt(x.value - y.value, w)),
  '*': (x, y, w) => (both(x, y) ? int(x.value * y.value, w) : flt(x.value * y.value, w)),
  '/': (x, y, w) => {
    if (y.value === 0) evaluationError('zero_divisor', w)
    if (both(x, y) && x.value % y.value === 0) return int(x.value / y.value, w)
    return flt(x.value / y.value, w)
  },
  '//': (x, y, w) => {
    const [a, b] = intDiv(x, y, w)
    return int(Math.trunc(a / b), w)
  },
  div: (x, y, w) => {
    const [a, b] = intDiv(x, y, w)
    return int(Math.floor(a / b), w)
  },
  mod: (x, y, w) => {
    const [a, b] = intDiv(x, y, w)
    const m = a % b
    return int(m !== 0 && Math.sign(m) !== Math.sign(b) ? m + b : m, w)
  },
  rem: (x, y, w) => {
    const [a, b] = intDiv(x, y, w)
    return int(a % b, w)
  },
  min: (x, y) => (y.value < x.value ? y : x),
  max: (x, y) => (y.value > x.value ? y : x),
  '**': (x, y, w) => power(x, y, w, false),
  '^': (x, y, w) => power(x, y, w, true),
  atan2: (x, y, w) => flt(Math.atan2(x.value, y.value), w),
  atan: (x, y, w) => flt(Math.atan2(x.value, y.value), w),
  copysign: (x, y, w) => flt(Math.sign(y.value) * Math.abs(x.value) || (Object.is(y.value, -0) ? -0 : 0), w),
  log: (x, y, w) =>
    x.value <= 0 || y.value <= 0 ? evaluationError('undefined', w) : flt(Math.log(y.value) / Math.log(x.value), w),
  '>>': (x, y, w) => int(Math.floor(requireInt(x, w) / 2 ** requireInt(y, w)), w),
  '<<': (x, y, w) => int(requireInt(x, w) * 2 ** requireInt(y, w), w),
  '/\\': (x, y, w) => int(requireInt(x, w) & requireInt(y, w), w),
  '\\/': (x, y, w) => int(requireInt(x, w) | requireInt(y, w), w),
  xor: (x, y, w) => int(requireInt(x, w) ^ requireInt(y, w), w),
  gcd: (x, y, w) => {
    let a = Math.abs(requireInt(x, w))
    let b = Math.abs(requireInt(y, w))
    while (b !== 0) [a, b] = [b, a % b]
    return int(a, w)
  },
  truncate: (x, _y, w) => int(Math.trunc(x.value), w),
}

/** Evaluate an arithmetic expression (the right side of `is/2`), dereferencing variables with `deref`. */
export function evaluate(t: Term, deref: (t: Term) => Term, where: string): Num {
  const x = deref(t)
  if (x.kind === 'num') return x
  if (x.kind === 'var') return instantiationError(where)
  if (x.kind === 'atom') {
    const c = CONSTANTS[x.name]
    if (c !== undefined) return c
    if (x.name === '[]') return typeError('evaluable', x, where)
    return typeError('evaluable', compound('/', [x, num(0)]), where)
  }
  // A one-element list evaluates its element ("a" in code-list mode is the character code).
  if (x.functor === '.' && x.args.length === 2) {
    const tail = deref(x.args[1] as Term)
    if (tail.kind === 'atom' && tail.name === '[]') return evaluate(x.args[0] as Term, deref, where)
  }
  if (x.args.length === 1) {
    const f = UNARY[x.functor]
    if (f !== undefined) return f(evaluate(x.args[0] as Term, deref, where), where)
  } else if (x.args.length === 2) {
    const f = BINARY[x.functor]
    if (f !== undefined) {
      return f(evaluate(x.args[0] as Term, deref, where), evaluate(x.args[1] as Term, deref, where), where)
    }
  }
  return typeError('evaluable', compound('/', [{ kind: 'atom', name: x.functor }, num(x.args.length)]), where)
}

/** Compare two evaluated numbers by value: negative, zero or positive. */
export function compareNumbers(a: Num, b: Num): number {
  return a.value < b.value ? -1 : a.value > b.value ? 1 : 0
}
