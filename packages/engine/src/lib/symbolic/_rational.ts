// Exact rational arithmetic on bigints: the number field of symbolic expressions. Every rational is normalised
// (positive denominator, numerator and denominator coprime), so equal values have equal representations.
import { OmaiError } from '../base/index.ts'
import { charge } from './_budget.ts'

/** A normalised rational number n / d with d > 0 and gcd(n, d) = 1. @internal */
export interface Rational {
  readonly n: bigint
  readonly d: bigint
}

/** Absolute value of a bigint. @internal */
export function babs(a: bigint): bigint {
  return a < 0n ? -a : a
}

/** Greatest common divisor of two bigints (non-negative). @internal */
export function bgcd(a: bigint, b: bigint): bigint {
  let x = babs(a)
  let y = babs(b)
  let steps = 1
  while (y !== 0n) {
    const t = x % y
    x = y
    y = t
    steps++
  }
  charge(steps)
  return x
}

/** Least common multiple of two bigints (non-negative). @internal */
export function blcm(a: bigint, b: bigint): bigint {
  if (a === 0n || b === 0n) return 0n
  return babs((a / bgcd(a, b)) * b)
}

/** Build a normalised rational. @internal */
export function rat(n: bigint, d: bigint = 1n): Rational {
  if (d === 0n) throw new OmaiError('domain', 'rational: zero denominator')
  if (d < 0n) {
    n = -n
    d = -d
  }
  const g = bgcd(n, d)
  return g === 1n || g === 0n ? { n, d } : { n: n / g, d: d / g }
}

/** Rational constants. @internal */
export const R0: Rational = { n: 0n, d: 1n }
/** @internal */
export const R1: Rational = { n: 1n, d: 1n }
/** @internal */
export const RM1: Rational = { n: -1n, d: 1n }
/** @internal */
export const RHALF: Rational = { n: 1n, d: 2n }

/** Sum. @internal */
export function radd(a: Rational, b: Rational): Rational {
  if (a.d === b.d) return rat(a.n + b.n, a.d)
  return rat(a.n * b.d + b.n * a.d, a.d * b.d)
}

/** Difference. @internal */
export function rsub(a: Rational, b: Rational): Rational {
  return radd(a, rneg(b))
}

/** Product. @internal */
export function rmul(a: Rational, b: Rational): Rational {
  return rat(a.n * b.n, a.d * b.d)
}

/** Quotient. @internal */
export function rdiv(a: Rational, b: Rational): Rational {
  if (b.n === 0n) throw new OmaiError('domain', 'rational: division by zero')
  return rat(a.n * b.d, a.d * b.n)
}

/** Negation. @internal */
export function rneg(a: Rational): Rational {
  return { n: -a.n, d: a.d }
}

/** Absolute value. @internal */
export function rabs(a: Rational): Rational {
  return a.n < 0n ? { n: -a.n, d: a.d } : a
}

/** Integer power (negative exponents invert). @internal */
export function rpow(a: Rational, k: number): Rational {
  if (k >= 0) return { n: a.n ** BigInt(k), d: a.d ** BigInt(k) }
  if (a.n === 0n) throw new OmaiError('domain', 'rational: zero to a negative power')
  return rat(a.d ** BigInt(-k), a.n ** BigInt(-k))
}

/** Three-way comparison. @internal */
export function rcmp(a: Rational, b: Rational): number {
  const l = a.n * b.d
  const r = b.n * a.d
  return l < r ? -1 : l > r ? 1 : 0
}

/** Equality. @internal */
export function req(a: Rational, b: Rational): boolean {
  return a.n === b.n && a.d === b.d
}

/** Whether the rational is an integer. @internal */
export function risInt(a: Rational): boolean {
  return a.d === 1n
}

/** Floor of a rational, as a bigint. @internal */
export function rfloor(a: Rational): bigint {
  const q = a.n / a.d
  return a.n < 0n && q * a.d !== a.n ? q - 1n : q
}

/** Nearest double to a rational (exact for small values, correctly scaled for large ones). @internal */
export function rnum(a: Rational): number {
  const n = Number(a.n)
  const d = Number(a.d)
  if (Number.isFinite(n) && Number.isFinite(d)) return n / d
  // Scale both down to keep the quotient representable.
  const shift = BigInt(Math.max(babs(a.n).toString(2).length, a.d.toString(2).length) - 1000)
  return Number(a.n >> (shift > 0n ? shift : 0n)) / Number(a.d >> (shift > 0n ? shift : 0n))
}

/**
 * The exact rational with the shortest decimal expansion that round-trips to `x` (so 0.1 becomes 1/10, as typed).
 * @internal
 */
export function rfromNumber(x: number): Rational {
  if (!Number.isFinite(x)) throw new OmaiError('value', `rational: ${x} is not finite`)
  if (Number.isInteger(x)) return { n: BigInt(x), d: 1n }
  return rfromDecimal(String(x))
}

/** Parse a decimal string such as `-1.25e-3` exactly. @internal */
export function rfromDecimal(text: string): Rational {
  const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(text.trim())
  if (m === null) throw new OmaiError('value', `rational: cannot parse '${text}'`)
  const [, sign = '', whole = '', frac = '', expo = '0'] = m
  const digits = BigInt(`${whole}${frac}` || '0')
  const e = Number(expo) - frac.length
  const n = sign === '-' ? -digits : digits
  return e >= 0 ? rat(n * 10n ** BigInt(e), 1n) : rat(n, 10n ** BigInt(-e))
}

/** Text form `n` or `n/d`. @internal */
export function rstr(a: Rational): string {
  return a.d === 1n ? a.n.toString() : `${a.n}/${a.d}`
}

/** Integer k-th root of a non-negative bigint, rounded down. @internal */
export function biroot(a: bigint, k: number): bigint {
  if (a < 0n) throw new OmaiError('domain', 'iroot: negative argument')
  if (a < 2n || k === 1) return a
  const K = BigInt(k)
  // Newton iteration from an over-estimate.
  let x = 1n << BigInt(Math.ceil(a.toString(2).length / k))
  for (;;) {
    const y = ((K - 1n) * x + a / x ** (K - 1n)) / K
    if (y >= x) break
    x = y
  }
  while (x ** K > a) x -= 1n
  while ((x + 1n) ** K <= a) x += 1n
  return x
}

/** Trial-division bound for {@link factorInteger}; a larger leftover cofactor is kept whole. */
const TRIAL_LIMIT = 100000n

/**
 * Factor a positive integer into prime powers by trial division up to 10^5. A leftover cofactor above the bound is
 * returned as one factor (it is prime or a product of large primes; perfect powers of it are still detected by
 * callers through {@link biroot}).
 * @internal
 */
export function factorInteger(a: bigint): [bigint, number][] {
  const out: [bigint, number][] = []
  let m = babs(a)
  if (m < 2n) return out
  for (const p of [2n, 3n]) {
    let e = 0
    while (m % p === 0n) {
      m /= p
      e++
    }
    if (e > 0) out.push([p, e])
  }
  for (let p = 5n; p <= TRIAL_LIMIT && p * p <= m; p += 6n) {
    for (const q of [p, p + 2n]) {
      let e = 0
      while (m % q === 0n) {
        m /= q
        e++
      }
      if (e > 0) out.push([q, e])
    }
  }
  if (m > 1n) {
    // A perfect power cofactor (for example a large prime squared) is split into its root.
    for (let k = 2; k <= 64 && 1n << BigInt(k) <= m; k++) {
      const r = biroot(m, k)
      if (r ** BigInt(k) === m) {
        out.push([r, k])
        m = 1n
        break
      }
    }
    if (m > 1n) out.push([m, 1])
  }
  return out
}
