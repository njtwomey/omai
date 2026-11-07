// Shared plumbing for the systems domain: number views of values, complex lists and highest-first polynomials.
import { MdmlError, toRows } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { type ComplexArray, polyadd, polyFromRoots, polyroots, polytrim } from '../numerics/polynomial/index.ts'
import { binom, factorial as factorialOf } from '../numerics/special/index.ts'
import { item, primalOf, promoteFloat, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

// Polynomials here follow scipy.signal: coefficients highest degree first.

/** A value as floating point. */
export function float(x: TensorLike): Value {
  return (promoteFloat(x) as [Value])[0]
}

/** The primal numbers of a value, row-major. */
export function numbers(v: Value): number[] {
  return toFlat(primalOf(v))
}

/** The primal rows of a matrix value. */
export function rows(v: Value): number[][] {
  const [m = 0, n = 0] = v.shape
  return toRows(numbers(v), m, n)
}

/** A list of complex numbers from parallel parts. */
export function toList(z: ComplexArray): ComplexNumber[] {
  return z.re.map((re, i) => C.of(re, z.im[i] ?? 0))
}

/** Parallel parts from a list of complex numbers. */
export function fromList(zs: readonly ComplexNumber[]): ComplexArray {
  return { re: zs.map((z) => z.re), im: zs.map((z) => z.im) }
}

/** Real numbers or a complex array as a complex array. */
export function asComplexArray(z: readonly number[] | ComplexArray): ComplexArray {
  if (Array.isArray(z)) return { re: [...(z as readonly number[])], im: z.map(() => 0) }
  const c = z as ComplexArray
  return { re: [...c.re], im: [...c.im] }
}

/**
 * The roots of a highest-first polynomial (numpy.roots: leading zeros dropped, trailing zeros giving exact zero
 * roots), sorted by real then imaginary part.
 */
export function rootsOf(c: readonly number[]): ComplexArray {
  const p = [...c]
  while (p.length > 1 && p[0] === 0) p.shift()
  let atZero = 0
  while (p.length > 1 && p[p.length - 1] === 0) {
    p.pop()
    atZero++
  }
  const r = polyroots(p.reverse())
  if (atZero === 0) return r
  const pairs = [
    ...r.re.map((re, i) => [re, r.im[i] ?? 0] as const),
    ...Array.from({ length: atZero }, () => [0, 0] as const),
  ]
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return { re: pairs.map((q) => q[0]), im: pairs.map((q) => q[1]) }
}

/** The monic highest-first polynomial with the given roots (numpy.poly); real when the roots come in conjugate pairs. */
export function polyOf(roots: ComplexArray): number[] {
  return polyFromRoots(roots).reverse()
}

/** Evaluate a highest-first real polynomial at a complex point (Horner). */
export function evalAt(c: readonly number[], z: ComplexNumber): ComplexNumber {
  let acc = C.of(0)
  for (const v of c) acc = C.add(C.mul(acc, z), v)
  return acc
}

/** Strip leading zeros (keeping at least one coefficient): numpy.polynomial's polytrim on the reversed list. */
export function trimLeading(c: readonly number[], tol = 0): number[] {
  return polytrim([...c].reverse(), tol).reverse()
}

/** Add highest-first polynomials (right-aligned): polyadd on the reversed lists. */
export function polyAddHf(a: readonly number[], b: readonly number[]): number[] {
  return polyadd([...a].reverse(), [...b].reverse()).reverse()
}

/** The sample time shared by systems being combined: equal, or one of them continuous-agnostic is an error. */
export function commonDt(name: string, ...dts: readonly number[]): number {
  const first = dts[0] ?? 0
  for (const dt of dts) {
    if (dt !== first) throw new MdmlError('value', `${name}: systems have different sample times (${dts.join(', ')})`)
  }
  return first
}

/** k! as a number. */
export function factorial(k: number): number {
  return item(factorialOf(k))
}

/** The binomial coefficient C(n, k) as a number. */
export function choose(n: number, k: number): number {
  return item(binom(n, k))
}
