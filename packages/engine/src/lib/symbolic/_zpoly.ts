// Dense univariate polynomials over the integers and modulo a prime (constant term first), shared by the
// factoriser and by the modular shortcuts of the rational gcds: an image modulo a prime whose gcd is constant proves
// that the gcd over the rationals is constant, at a fraction of the cost of Euclid's algorithm over the rationals.
import { charge } from './_budget.ts'
import { bgcd, type Rational, rat } from './_rational.ts'

/** Dense integer coefficients, constant term first. @internal */
export type Z = bigint[]

/** A prime for modular checks (2^31 - 1): large enough that unlucky reductions are rare. @internal */
export const CHECK_PRIME = 2147483647n

/** Trim trailing zeros. @internal */
export function ztrim(a: Z): Z {
  let n = a.length
  while (n > 0 && a[n - 1] === 0n) n--
  return a.slice(0, n)
}

/** The residue of a in [0, m). @internal */
export function mod(a: bigint, m: bigint): bigint {
  const r = a % m
  return r < 0n ? r + m : r
}

/** The residue of a in (-m/2, m/2]. @internal */
export function symmetric(a: bigint, m: bigint): bigint {
  const r = mod(a, m)
  return r > m / 2n ? r - m : r
}

/** Reduce modulo m. @internal */
export function zmod(a: Z, m: bigint): Z {
  charge(a.length)
  return ztrim(a.map((c) => mod(c, m)))
}

/** Sum modulo m. @internal */
export function zadd(a: Z, b: Z, m: bigint): Z {
  const n = Math.max(a.length, b.length)
  charge(n)
  const out: Z = []
  for (let i = 0; i < n; i++) out.push(mod((a[i] ?? 0n) + (b[i] ?? 0n), m))
  return ztrim(out)
}

/** Difference modulo m. @internal */
export function zsub(a: Z, b: Z, m: bigint): Z {
  return zadd(
    a,
    b.map((c) => -c),
    m,
  )
}

/** Product modulo m. @internal */
export function zmul(a: Z, b: Z, m: bigint): Z {
  if (a.length === 0 || b.length === 0) return []
  charge(a.length * b.length)
  const out: Z = new Array<bigint>(a.length + b.length - 1).fill(0n)
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as bigint
    if (x === 0n) continue
    for (let j = 0; j < b.length; j++) out[i + j] = (out[i + j] as bigint) + x * (b[j] as bigint)
  }
  return zmod(out, m)
}

/** Inverse of a modulo m. @internal */
export function zinv(a: bigint, m: bigint): bigint {
  let [r0, r1] = [mod(a, m), m]
  let [s0, s1] = [1n, 0n]
  while (r1 !== 0n) {
    const q = r0 / r1
    ;[r0, r1] = [r1, r0 - q * r1]
    ;[s0, s1] = [s1, s0 - q * s1]
  }
  if (r0 !== 1n) throw new Error('not invertible')
  return mod(s0, m)
}

/** Division by a polynomial whose leading coefficient is invertible modulo m. @internal */
export function zdivmod(a: Z, b: Z, m: bigint): [Z, Z] {
  const bt = ztrim(b)
  const db = bt.length - 1
  const inv = zinv(bt[db] as bigint, m)
  let r = zmod(a, m)
  const q: Z = new Array<bigint>(Math.max(0, r.length - db)).fill(0n)
  while (r.length - 1 >= db && r.length > 0) {
    const k = r.length - 1 - db
    const c = mod((r[r.length - 1] as bigint) * inv, m)
    q[k] = c
    const sub: Z = new Array<bigint>(k).fill(0n).concat(bt.map((x) => x * c))
    r = zsub(r, sub, m)
  }
  return [ztrim(q), r]
}

/** Monic version modulo m. @internal */
export function zmonic(a: Z, m: bigint): Z {
  const t = ztrim(a)
  if (t.length === 0) return t
  const inv = zinv(t[t.length - 1] as bigint, m)
  return zmod(
    t.map((c) => c * inv),
    m,
  )
}

/** Monic gcd modulo a prime. @internal */
export function zgcd(a: Z, b: Z, p: bigint): Z {
  let x = zmod(a, p)
  let y = zmod(b, p)
  while (y.length > 0) {
    const r = zdivmod(x, y, p)[1]
    x = y
    y = r
  }
  return zmonic(x, p)
}

/** Derivative. @internal */
export function zderiv(a: Z): Z {
  return ztrim(a.slice(1).map((c, k) => c * BigInt(k + 1)))
}

/** Content (gcd of the coefficients, non-negative). @internal */
export function zcontent(a: Z): bigint {
  let g = 0n
  for (const c of a) g = bgcd(g, c)
  return g
}

/** Primitive part with positive leading coefficient. @internal */
export function zprimitive(a: Z): Z {
  const g = zcontent(a)
  const t = ztrim(a)
  const sign = (t[t.length - 1] ?? 1n) < 0n ? -1n : 1n
  return g === 0n ? t : t.map((c) => (c / g) * sign)
}

/** Integer polynomial with the same roots (denominators cleared). @internal */
export function ratsToZ(a: readonly Rational[]): Z {
  let den = 1n
  for (const c of a) den = (den / bgcd(den, c.d)) * c.d
  return ztrim(a.map((c) => (c.n * den) / c.d))
}

/** Rational coefficients from integer ones. @internal */
export function zToRats(a: Z): Rational[] {
  return a.map((c) => rat(c))
}

/**
 * Whether two integer polynomials of positive degree are certainly coprime over the rationals: their images modulo
 * a prime dividing neither leading coefficient have a constant gcd (the gcd over Q, made primitive, divides both and
 * keeps its degree modulo such a prime). False means "not proven", not "not coprime".
 * @internal
 */
export function zcoprimeModular(a: Z, b: Z, p: bigint = CHECK_PRIME): boolean {
  const at = ztrim(a)
  const bt = ztrim(b)
  if (at.length < 2 || bt.length < 2) return false
  if ((at[at.length - 1] as bigint) % p === 0n || (bt[bt.length - 1] as bigint) % p === 0n) return false
  return zgcd(at, bt, p).length === 1
}

/** A non-zero constant multiple of the remainder of a by b over Z (a pseudo-remainder, scaled as little as needed). */
function zprem(a: Z, b: Z): Z {
  const bt = ztrim(b)
  const db = bt.length - 1
  const lb = bt[db] as bigint
  let r = ztrim(a)
  while (r.length - 1 >= db && r.length > 0) {
    charge(r.length)
    const k = r.length - 1 - db
    const lr = r[r.length - 1] as bigint
    const g = bgcd(lb, lr)
    const sr = lb / g
    const sb = lr / g
    const next = r.map((c) => c * sr)
    for (let i = 0; i <= db; i++) next[k + i] = (next[k + i] as bigint) - sb * (bt[i] as bigint)
    r = ztrim(next)
  }
  return r
}

/**
 * The gcd over Q of two non-zero integer polynomials as a primitive integer polynomial (up to sign), by the
 * primitive remainder sequence: integer arithmetic with the content removed at every step, which keeps the
 * coefficients far smaller than Euclid over the rationals does.
 * @internal
 */
export function zgcdPrimitive(a: Z, b: Z): Z {
  let x = zprimitive(a)
  let y = zprimitive(b)
  if (x.length < y.length) [x, y] = [y, x]
  while (y.length > 1) {
    const r = zprem(x, y)
    x = y
    y = zprimitive(r)
  }
  return y.length === 1 ? [1n] : x
}
