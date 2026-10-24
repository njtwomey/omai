// Three-valued assumption queries (true, false or unknown), a small version of SymPy's `is_positive` and friends.
import type { Expr } from './expr.ts'

/** A truth value that may be unknown. @internal */
export type Fuzzy = boolean | undefined

function and3(xs: readonly Fuzzy[]): Fuzzy {
  let unknown = false
  for (const x of xs) {
    if (x === false) return false
    if (x === undefined) unknown = true
  }
  return unknown ? undefined : true
}

function sign(e: Expr): -1 | 0 | 1 | undefined {
  switch (e.kind) {
    case 'num':
      return e.n > 0n ? 1 : e.n < 0n ? -1 : 0
    case 'float':
      return e.value > 0 ? 1 : e.value < 0 ? -1 : e.value === 0 ? 0 : undefined
    case 'const':
      return e.name === 'pi' || e.name === 'E' || e.name === 'oo' ? 1 : e.name === '-oo' ? -1 : undefined
    case 'sym':
      if (e.assumptions.positive) return 1
      return undefined
    default:
      return undefined
  }
}

/** Whether `e` is known to be real. @internal */
export function isReal(e: Expr): Fuzzy {
  switch (e.kind) {
    case 'num':
    case 'float':
      return true
    case 'const':
      return e.name === 'pi' || e.name === 'E' ? true : e.name === 'I' ? false : undefined
    case 'sym':
      return e.assumptions.real ? true : undefined
    case 'add':
    case 'mul':
      return and3(e.args.map(isReal)) === true ? true : undefined
    case 'pow':
      if (isReal(e.exp) !== true) return undefined
      if (isPositive(e.base) === true) return true
      if (isReal(e.base) === true && e.exp.kind === 'num' && e.exp.d === 1n) return true
      return undefined
    case 'fn': {
      const a = e.args[0]
      if (a === undefined) return undefined
      if (
        ['exp', 'sin', 'cos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'erf', 'sign', 'floor', 'ceiling'].includes(
          e.name,
        )
      ) {
        return isReal(a) === true ? true : undefined
      }
      if (e.name === 'Abs') return true
      if (e.name === 'log') return isPositive(a) === true ? true : undefined
      return undefined
    }
    default:
      return undefined
  }
}

/** Whether `e` is known to be strictly positive. @internal */
export function isPositive(e: Expr): Fuzzy {
  const s = sign(e)
  if (s !== undefined) return s === 1
  switch (e.kind) {
    case 'add': {
      const ps = e.args.map(isNonnegative)
      if (and3(ps) === true && e.args.some((a) => isPositive(a) === true)) return true
      return undefined
    }
    case 'mul': {
      let negatives = 0
      for (const a of e.args) {
        if (isPositive(a) === true) continue
        if (isNegative(a) === true) negatives++
        else return undefined
      }
      return negatives % 2 === 0
    }
    case 'pow':
      if (isPositive(e.base) === true && isReal(e.exp) === true) return true
      if (isReal(e.base) === true && isNonzero(e.base) === true && e.exp.kind === 'num' && e.exp.d === 1n) {
        if (e.exp.n % 2n === 0n) return true
      }
      return undefined
    case 'fn': {
      const a = e.args[0]
      if (a === undefined) return undefined
      if (e.name === 'exp' || e.name === 'cosh') return isReal(a) === true ? true : undefined
      if (e.name === 'Abs') return isNonzero(a) === true ? true : undefined
      return undefined
    }
    default:
      return undefined
  }
}

/** Whether `e` is known to be strictly negative. @internal */
export function isNegative(e: Expr): Fuzzy {
  const s = sign(e)
  if (s !== undefined) return s === -1
  if (e.kind === 'mul') {
    let negatives = 0
    for (const a of e.args) {
      if (isPositive(a) === true) continue
      if (isNegative(a) === true) negatives++
      else return undefined
    }
    return negatives % 2 === 1
  }
  if (e.kind === 'add') {
    if (e.args.every((a) => isNegative(a) === true || isZeroExpr(a))) return true
    return undefined
  }
  return undefined
}

function isZeroExpr(e: Expr): boolean {
  return e.kind === 'num' && e.n === 0n
}

/** Whether `e` is known to be non-negative. @internal */
export function isNonnegative(e: Expr): Fuzzy {
  const s = sign(e)
  if (s !== undefined) return s >= 0
  if (e.kind === 'sym' && e.assumptions.nonnegative) return true
  if (isPositive(e) === true) return true
  if (e.kind === 'fn' && e.name === 'Abs') return true
  if (e.kind === 'pow' && e.exp.kind === 'num' && e.exp.d === 1n && e.exp.n % 2n === 0n && isReal(e.base) === true) {
    return true
  }
  if (e.kind === 'pow' && isNonnegative(e.base) === true && isReal(e.exp) === true) return true
  if (e.kind === 'add' && e.args.every((a) => isNonnegative(a) === true)) return true
  if (e.kind === 'mul' && e.args.every((a) => isNonnegative(a) === true)) return true
  return undefined
}

/** Whether `e` is known to be non-zero. @internal */
export function isNonzero(e: Expr): Fuzzy {
  const s = sign(e)
  if (s !== undefined) return s !== 0
  if (e.kind === 'sym') return e.assumptions.nonzero || e.assumptions.positive ? true : undefined
  if (e.kind === 'mul') return and3(e.args.map(isNonzero)) === true ? true : undefined
  if (e.kind === 'pow') return isNonzero(e.base) === true ? true : undefined
  if (e.kind === 'fn' && e.name === 'exp') return true
  if (isPositive(e) === true || isNegative(e) === true) return true
  return undefined
}

/** Whether `e` is known to be an integer. @internal */
export function isInteger(e: Expr): Fuzzy {
  switch (e.kind) {
    case 'num':
      return e.d === 1n
    case 'float':
      return Number.isInteger(e.value) ? undefined : false
    case 'sym':
      return e.assumptions.integer ? true : undefined
    case 'add':
    case 'mul':
      return and3(e.args.map(isInteger)) === true ? true : undefined
    case 'pow':
      if (isInteger(e.base) === true && isNonnegative(e.exp) === true && isInteger(e.exp) === true) return true
      return undefined
    default:
      return undefined
  }
}
