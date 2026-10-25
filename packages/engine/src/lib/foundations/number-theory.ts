/**
 * Elementary number theory with the steps shown: Euclid's algorithm and its extended form with Bezout coefficients, gcd
 * and lcm, modular arithmetic (residues, inverses, square-and-multiply exponentiation), the sieve of Eratosthenes,
 * Miller-Rabin primality testing, factorisation by trial division and Pollard's rho, Euler's totient and divisors, the
 * Chinese remainder theorem, continued fractions and convergents, and exact Fibonacci and Lucas numbers.
 */

import { WhiteboxError } from '../base/index.ts'

function integer(x: number, name: string): number {
  if (!Number.isSafeInteger(x)) throw new WhiteboxError('value', `${name}: expected a safe integer, got ${x}`)
  return x
}

function positive(x: number, name: string): number {
  if (!(Number.isSafeInteger(x) && x >= 1))
    throw new WhiteboxError('value', `${name}: expected a positive integer, got ${x}`)
  return x
}

/** One division step `a = q b + r` of Euclid's algorithm. */
export interface EuclidStep {
  readonly a: number
  readonly b: number
  readonly quotient: number
  readonly remainder: number
}

/**
 * Euclid's algorithm with its steps: replace `(a, b)` by `(b, a mod b)` until the remainder is zero; the last nonzero
 * remainder is the greatest common divisor.
 *
 * @param a - An integer.
 * @param b - An integer.
 * @returns The gcd (non-negative) and the division steps.
 * @concept euclidean-algorithm
 * @example
 * ```ts
 * import { euclid } from 'whitebox-engine/foundations'
 * const e = euclid(252, 105)
 * e.gcd // => 21
 * e.steps.map((s) => s.remainder) // => [42, 21, 0]
 * ```
 */
export function euclid(a: number, b: number): { readonly gcd: number; readonly steps: EuclidStep[] } {
  let x = Math.abs(integer(a, 'euclid'))
  let y = Math.abs(integer(b, 'euclid'))
  const steps: EuclidStep[] = []
  while (y !== 0) {
    const q = Math.floor(x / y)
    const r = x - q * y
    steps.push({ a: x, b: y, quotient: q, remainder: r })
    x = y
    y = r
  }
  return { gcd: x, steps }
}

/**
 * The greatest common divisor of integers (0 for none or all zero).
 *
 * @param values - Integers.
 * @returns The gcd.
 * @concept euclidean-algorithm
 * @example
 * ```ts
 * import { gcd } from 'whitebox-engine/foundations'
 * gcd(12, 18, 27) // => 3
 * ```
 */
export function gcd(...values: readonly number[]): number {
  let g = 0
  for (const v of values) g = euclid(g, v).gcd
  return g
}

/**
 * The least common multiple of integers (0 if any is zero).
 *
 * @param values - Integers.
 * @returns The lcm.
 * @concept euclidean-algorithm
 * @example
 * ```ts
 * import { lcm } from 'whitebox-engine/foundations'
 * lcm(4, 6, 10) // => 60
 * ```
 */
export function lcm(...values: readonly number[]): number {
  let l = 1
  for (const v of values) {
    if (v === 0) return 0
    l = Math.abs((l / gcd(l, v)) * v)
  }
  return l
}

/** One row of the extended Euclidean algorithm: `r = s a + t b`. */
export interface ExtendedEuclidStep {
  readonly quotient: number
  readonly r: number
  readonly s: number
  readonly t: number
}

/**
 * The extended Euclidean algorithm with its table: Bezout coefficients x, y with `a x + b y = gcd(a, b)`, carried
 * along the remainders (each row keeps `r_i = s_i a + t_i b`).
 *
 * @param a - An integer.
 * @param b - An integer.
 * @returns The gcd, the coefficients and the rows.
 * @concept extended-euclidean-algorithm
 * @example
 * ```ts
 * import { extendedEuclid } from 'whitebox-engine/foundations'
 * const e = extendedEuclid(240, 46)
 * [e.gcd, e.x, e.y] // => [2, -9, 47]
 * ```
 */
export function extendedEuclid(
  a: number,
  b: number,
): { readonly gcd: number; readonly x: number; readonly y: number; readonly steps: ExtendedEuclidStep[] } {
  integer(a, 'extendedEuclid')
  integer(b, 'extendedEuclid')
  let [r0, r1] = [a, b]
  let [s0, s1] = [1, 0]
  let [t0, t1] = [0, 1]
  const steps: ExtendedEuclidStep[] = [
    { quotient: 0, r: r0, s: s0, t: t0 },
    { quotient: 0, r: r1, s: s1, t: t1 },
  ]
  while (r1 !== 0) {
    const q = Math.floor(r0 / r1)
    ;[r0, r1] = [r1, r0 - q * r1]
    ;[s0, s1] = [s1, s0 - q * s1]
    ;[t0, t1] = [t1, t0 - q * t1]
    steps.push({ quotient: q, r: r1, s: s1, t: t1 })
  }
  // Make the gcd non-negative.
  const sign = r0 < 0 ? -1 : 1
  return { gcd: sign * r0, x: sign * s0 || 0, y: sign * t0 || 0, steps }
}

/**
 * The least non-negative residue `a mod m` (unlike JavaScript's `%`, never negative).
 *
 * @param a - An integer.
 * @param m - A positive modulus.
 * @returns `a mod m` in `[0, m)`.
 * @concept modular-arithmetic
 * @example
 * ```ts
 * import { modulo } from 'whitebox-engine/foundations'
 * modulo(-7, 3) // => 2
 * ```
 */
export function modulo(a: number, m: number): number {
  positive(m, 'modulo')
  return Number(((BigInt(integer(a, 'modulo')) % BigInt(m)) + BigInt(m)) % BigInt(m))
}

/**
 * The modular inverse: the x in `[0, m)` with `a x = 1 (mod m)`, from the extended Euclidean algorithm.
 *
 * @param a - An integer.
 * @param m - A modulus >= 2.
 * @returns The inverse.
 * @throws WhiteboxError with code `value` when `gcd(a, m) != 1`.
 * @concept modular-inverse
 * @example
 * ```ts
 * import { modInverse } from 'whitebox-engine/foundations'
 * modInverse(3, 11) // => 4
 * ```
 */
export function modInverse(a: number, m: number): number {
  positive(m, 'modInverse')
  const { gcd: g, x } = extendedEuclid(modulo(a, m), m)
  if (g !== 1) throw new WhiteboxError('value', `modInverse: ${a} has no inverse mod ${m} (gcd ${g})`)
  return modulo(x, m)
}

/** One step of square-and-multiply. */
export interface PowerStep {
  readonly bit: 0 | 1
  /** The running result after this bit. */
  readonly result: number
  /** The base squared so far, `b^(2^i) mod m`. */
  readonly square: number
}

/**
 * Modular exponentiation `b^e mod m` by binary (square-and-multiply) exponentiation, exact for moduli up to 2^53
 * (products are formed in BigInt), with one step per bit of the exponent from the lowest.
 *
 * @param base - The base.
 * @param exponent - A non-negative exponent.
 * @param m - The modulus (>= 1).
 * @returns The power and the steps.
 * @concept modular-exponentiation
 * @example
 * ```ts
 * import { modPow } from 'whitebox-engine/foundations'
 * modPow(4, 13, 497).value // => 445
 * ```
 */
export function modPow(
  base: number,
  exponent: number,
  m: number,
): { readonly value: number; readonly steps: PowerStep[] } {
  positive(m, 'modPow')
  if (!(Number.isSafeInteger(exponent) && exponent >= 0))
    throw new WhiteboxError('value', 'modPow: exponent must be a non-negative integer')
  const M = BigInt(m)
  let result = 1n % M
  let b = BigInt(modulo(base, m))
  let e = exponent
  const steps: PowerStep[] = []
  while (e > 0) {
    const bit = (e % 2) as 0 | 1
    if (bit === 1) result = (result * b) % M
    steps.push({ bit, result: Number(result), square: Number(b) })
    b = (b * b) % M
    e = Math.floor(e / 2)
  }
  return { value: Number(result), steps }
}

const powBig = (b: bigint, e: bigint, m: bigint): bigint => {
  let r = 1n % m
  let x = b % m
  let k = e
  while (k > 0n) {
    if (k & 1n) r = (r * x) % m
    x = (x * x) % m
    k >>= 1n
  }
  return r
}

/** Result of {@link sieve}. */
export interface Sieve {
  readonly primes: number[]
  /** `smallestFactor[k]` for k <= n: the prime that crossed k out (k itself for primes; 0 for 0 and 1). */
  readonly smallestFactor: number[]
}

/**
 * The sieve of Eratosthenes up to n: cross out the multiples of each prime p from p^2, recording which prime crossed
 * each number out (its smallest prime factor, which also factorises every k <= n quickly).
 *
 * @param n - The bound.
 * @returns The primes up to n and the smallest prime factors.
 * @concept sieve-of-eratosthenes
 * @example
 * ```ts
 * import { sieve } from 'whitebox-engine/foundations'
 * sieve(20).primes // => [2, 3, 5, 7, 11, 13, 17, 19]
 * sieve(20).smallestFactor[15] // => 3
 * ```
 */
export function sieve(n: number): Sieve {
  if (!(Number.isSafeInteger(n) && n >= 0)) throw new WhiteboxError('value', 'sieve: expected a non-negative integer')
  if (n > 1e8) throw new WhiteboxError('value', 'sieve: bound above 1e8 is too large to tabulate')
  const spf = new Int32Array(n + 1)
  const primes: number[] = []
  for (let k = 2; k <= n; k++) {
    if (spf[k] === 0) {
      spf[k] = k
      primes.push(k)
      for (let j = k * k; j <= n; j += k) if (spf[j] === 0) spf[j] = k
    }
  }
  return { primes, smallestFactor: Array.from(spf) }
}

/** The evidence of one Miller-Rabin round. */
export interface MillerRabinRound {
  readonly base: number
  /** The sequence `a^d, a^(2d), ..., a^(2^s d)` mod n. */
  readonly sequence: number[]
  /** Whether this base proves n composite. */
  readonly witness: boolean
}

/**
 * The Miller-Rabin test with its rounds: write `n - 1 = 2^s d` with d odd; a base a is a witness of compositeness unless
 * `a^d = 1` or some `a^(2^r d) = -1 (mod n)`. With the first twelve primes as bases the answer is exact for every
 * n < 3.3 * 10^24, so for all safe integers.
 *
 * @param n - The number to test.
 * @param bases - The bases (default the primes up to 37).
 * @returns Whether n is (probably) prime and each round's sequence.
 * @concept miller-rabin
 * @example
 * ```ts
 * import { millerRabin } from 'whitebox-engine/foundations'
 * millerRabin(561).prime // => false
 * millerRabin(2147483647).prime // => true
 * ```
 */
export function millerRabin(
  n: number,
  bases: readonly number[] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
): { readonly prime: boolean; readonly rounds: MillerRabinRound[] } {
  integer(n, 'millerRabin')
  if (n < 2) return { prime: false, rounds: [] }
  if (n % 2 === 0) return { prime: n === 2, rounds: [] }
  for (const p of bases) {
    if (n === p) return { prime: true, rounds: [] }
    if (n % p === 0) return { prime: false, rounds: [] }
  }
  const N = BigInt(n)
  let d = N - 1n
  let s = 0
  while ((d & 1n) === 0n) {
    d >>= 1n
    s++
  }
  const rounds: MillerRabinRound[] = []
  for (const a of bases) {
    const A = BigInt(a) % N
    if (A === 0n) continue
    let x = powBig(A, d, N)
    const sequence = [Number(x)]
    let passes = x === 1n || x === N - 1n
    for (let r = 1; r < s && !passes; r++) {
      x = (x * x) % N
      sequence.push(Number(x))
      if (x === N - 1n) passes = true
    }
    rounds.push({ base: a, sequence, witness: !passes })
    if (!passes) return { prime: false, rounds }
  }
  return { prime: true, rounds }
}

/**
 * Whether an integer is prime (deterministic Miller-Rabin for every safe integer).
 *
 * @param n - The integer.
 * @returns `true` for primes.
 * @concept primes
 * @example
 * ```ts
 * import { isPrime } from 'whitebox-engine/foundations'
 * [1, 2, 9, 97].map(isPrime) // => [false, true, false, true]
 * ```
 */
export function isPrime(n: number): boolean {
  return millerRabin(n).prime
}

/**
 * Factor an integer by trial division by 2 and the odd numbers up to its square root, recording each division that
 * succeeds.
 *
 * @param n - A positive integer (practical up to about 10^14).
 * @returns The prime factors in non-decreasing order, and the trial divisors that divided.
 * @concept factorisation
 * @example
 * ```ts
 * import { trialDivision } from 'whitebox-engine/foundations'
 * trialDivision(360).factors // => [2, 2, 2, 3, 3, 5]
 * ```
 */
export function trialDivision(n: number): { readonly factors: number[]; readonly trials: number } {
  let m = positive(n, 'trialDivision')
  const factors: number[] = []
  let trials = 0
  for (let d = 2; d * d <= m; d += d === 2 ? 1 : 2) {
    trials++
    while (m % d === 0) {
      factors.push(d)
      m /= d
    }
  }
  if (m > 1) factors.push(m)
  return { factors, trials }
}

/** One iteration of Pollard's rho: the tortoise, the hare and the gcd of their difference with n. */
export interface RhoStep {
  readonly x: number
  readonly y: number
  readonly d: number
}

/**
 * Pollard's rho with Floyd's cycle detection: iterate `x -> x^2 + c (mod n)` with a tortoise and a hare until
 * `gcd(|x - y|, n)` exposes a factor (the sequence mod an unknown prime factor p repeats after about sqrt(p) steps).
 * Retries with the next c when the cycle closes without one.
 *
 * @param n - A composite number (odd, not a prime power is the easy case).
 * @param options - Starting point (default 2), first constant (default 1) and step cap (default 100000).
 * @returns A nontrivial factor, the c that found it and the steps of the successful attempt.
 * @throws WhiteboxError with code `value` for a prime or n < 4.
 * @concept pollard-rho
 * @example
 * ```ts
 * import { pollardRho } from 'whitebox-engine/foundations'
 * pollardRho(8051).factor // => 97
 * ```
 */
export function pollardRho(
  n: number,
  options: { readonly start?: number; readonly c?: number; readonly maxSteps?: number } = {},
): { readonly factor: number; readonly c: number; readonly steps: RhoStep[] } {
  positive(n, 'pollardRho')
  if (n < 4 || isPrime(n)) throw new WhiteboxError('value', `pollardRho: ${n} has no nontrivial factor`)
  if (n % 2 === 0) return { factor: 2, c: 0, steps: [] }
  const N = BigInt(n)
  for (let c = options.c ?? 1; c < (options.c ?? 1) + 20; c++) {
    const C = BigInt(c)
    const f = (v: bigint): bigint => (v * v + C) % N
    let x = BigInt(options.start ?? 2)
    let y = x
    const steps: RhoStep[] = []
    for (let i = 0; i < (options.maxSteps ?? 100000); i++) {
      x = f(x)
      y = f(f(y))
      const diff = x > y ? x - y : y - x
      const d = euclid(Number(diff), n).gcd
      steps.push({ x: Number(x), y: Number(y), d })
      if (d === n) break
      if (d > 1) return { factor: d, c, steps }
    }
  }
  throw new WhiteboxError('convergence', `pollardRho: no factor of ${n} found`)
}

/**
 * The prime factorisation `n = prod p_i^e_i` (trial division by small primes, then Pollard's rho with Miller-Rabin
 * for what remains), as sympy's `factorint`.
 *
 * @param n - A positive integer.
 * @returns `[prime, exponent]` pairs in increasing order of prime (empty for 1).
 * @concept factorisation
 * @example
 * ```ts
 * import { factorise } from 'whitebox-engine/foundations'
 * factorise(1001) // => [[7, 1], [11, 1], [13, 1]]
 * factorise(2 ** 10 * 3 ** 4) // => [[2, 10], [3, 4]]
 * ```
 */
export function factorise(n: number): [number, number][] {
  let m = positive(n, 'factorise')
  const primes: number[] = []
  for (const p of sieve(1000).primes) {
    while (m % p === 0) {
      primes.push(p)
      m /= p
    }
  }
  const stack = m > 1 ? [m] : []
  while (stack.length > 0) {
    const k = stack.pop() as number
    if (isPrime(k)) primes.push(k)
    else {
      const { factor } = pollardRho(k)
      stack.push(factor, k / factor)
    }
  }
  primes.sort((a, b) => a - b)
  const out: [number, number][] = []
  for (const p of primes) {
    const last = out[out.length - 1]
    if (last !== undefined && last[0] === p) last[1]++
    else out.push([p, 1])
  }
  return out
}

/**
 * Euler's totient `phi(n)`: how many of 1..n are coprime to n, `n prod_{p | n} (1 - 1/p)`.
 *
 * @param n - A positive integer.
 * @returns phi(n).
 * @concept eulers-totient
 * @example
 * ```ts
 * import { eulerPhi } from 'whitebox-engine/foundations'
 * eulerPhi(36) // => 12
 * ```
 */
export function eulerPhi(n: number): number {
  let phi = positive(n, 'eulerPhi')
  for (const [p] of factorise(n)) phi = (phi / p) * (p - 1)
  return phi
}

/**
 * The positive divisors of n in increasing order, from its factorisation.
 *
 * @param n - A positive integer.
 * @returns The divisors.
 * @concept divisors
 * @example
 * ```ts
 * import { divisors } from 'whitebox-engine/foundations'
 * divisors(28) // => [1, 2, 4, 7, 14, 28]
 * ```
 */
export function divisors(n: number): number[] {
  let ds = [1]
  for (const [p, e] of factorise(n)) {
    const next: number[] = []
    for (const d of ds) for (let k = 0, q = 1; k <= e; k++, q *= p) next.push(d * q)
    ds = next
  }
  return ds.sort((a, b) => a - b)
}

/**
 * The Chinese remainder theorem: the x with `x = r_i (mod m_i)` for every i, unique modulo the lcm of the moduli,
 * built congruence by congruence (moduli need not be coprime; null when the congruences conflict), as sympy's `crt`.
 *
 * @param residues - The residues r_i.
 * @param moduli - The positive moduli m_i.
 * @returns `{ x, modulus }` with x in `[0, modulus)`, or null when there is no solution.
 * @concept chinese-remainder-theorem
 * @example
 * ```ts
 * import { chineseRemainder } from 'whitebox-engine/foundations'
 * chineseRemainder([2, 3, 2], [3, 5, 7]) // => { x: 23, modulus: 105 }
 * chineseRemainder([1, 2], [4, 6]) // => null
 * ```
 */
export function chineseRemainder(
  residues: readonly number[],
  moduli: readonly number[],
): { readonly x: number; readonly modulus: number } | null {
  if (residues.length !== moduli.length) throw new WhiteboxError('value', 'chineseRemainder: one residue per modulus')
  let x = 0n
  let m = 1n
  for (let i = 0; i < moduli.length; i++) {
    const mi = BigInt(positive(moduli[i] ?? 0, 'chineseRemainder'))
    const ri = BigInt(modulo(residues[i] ?? 0, Number(mi)))
    // Solve x + m t = ri (mod mi): m t = ri - x (mod mi).
    const g = BigInt(gcd(Number(m % mi), Number(mi)))
    const diff = (((ri - x) % mi) + mi) % mi
    if (diff % g !== 0n) return null
    const mg = mi / g
    const t = ((diff / g) * BigInt(modInverse(Number((m / g) % mg), Number(mg)))) % mg
    x = x + m * t
    m = m * mg
    x = ((x % m) + m) % m
  }
  if (m > BigInt(Number.MAX_SAFE_INTEGER))
    throw new WhiteboxError('value', 'chineseRemainder: the combined modulus exceeds 2^53')
  return { x: Number(x), modulus: Number(m) }
}

/**
 * The simple continued fraction `[a0; a1, a2, ...]` of a number: `a0 = floor(x)`, then repeat on `1 / (x - a0)`. For a
 * rational p/q give both integers for the exact (Euclid) expansion; for a real number the expansion is cut after
 * `terms` partial quotients or when the remainder vanishes.
 *
 * @param x - A real number, or the numerator p of a fraction.
 * @param q - The denominator (for an exact expansion of p/q).
 * @param terms - The most partial quotients for a real number (default 20).
 * @returns The partial quotients.
 * @concept continued-fractions
 * @example
 * ```ts
 * import { continuedFractionExpansion } from 'whitebox-engine/foundations'
 * continuedFractionExpansion(415, 93) // => [4, 2, 6, 7]
 * continuedFractionExpansion(Math.SQRT2, undefined, 5) // => [1, 2, 2, 2, 2]
 * ```
 */
export function continuedFractionExpansion(x: number, q?: number, terms = 20): number[] {
  if (q !== undefined) {
    integer(x, 'continuedFractionExpansion')
    if (integer(q, 'continuedFractionExpansion') === 0)
      throw new WhiteboxError('value', 'continuedFractionExpansion: zero denominator')
    const sign = q < 0 ? -1 : 1
    let p = sign * x
    let d = sign * q
    const out: number[] = []
    while (d !== 0) {
      const a = Math.floor(p / d)
      out.push(a)
      ;[p, d] = [d, p - a * d]
    }
    return out
  }
  const out: number[] = []
  let v = x
  for (let i = 0; i < terms; i++) {
    const a = Math.floor(v)
    out.push(a)
    const frac = v - a
    if (frac < 1e-12) break
    v = 1 / frac
  }
  return out
}

/**
 * The convergents `p_k / q_k` of a continued fraction, from `p_k = a_k p_{k-1} + p_{k-2}` and the same for q: the
 * best rational approximations, alternately below and above the value.
 *
 * @param partialQuotients - `[a0, a1, ...]`.
 * @returns The convergents as `[p, q]` pairs.
 * @concept continued-fractions
 * @example
 * ```ts
 * import { convergents } from 'whitebox-engine/foundations'
 * convergents([3, 7, 15, 1]) // => [[3, 1], [22, 7], [333, 106], [355, 113]]
 * ```
 */
export function convergents(partialQuotients: readonly number[]): [number, number][] {
  const out: [number, number][] = []
  let [p0, q0, p1, q1] = [1, 0, 0, 1]
  for (const a of partialQuotients) {
    const p = a * p0 + p1
    const q = a * q0 + q1
    out.push([p, q])
    ;[p1, q1, p0, q0] = [p0, q0, p, q]
  }
  return out
}

function fibLucas(n: number): [bigint, bigint] {
  // Fast doubling: F(2k) = F(k) (2 F(k+1) - F(k)), F(2k+1) = F(k)^2 + F(k+1)^2.
  let a = 0n
  let b = 1n
  for (let bit = 31 - Math.clz32(n); bit >= 0 && n > 0; bit--) {
    const c = a * (2n * b - a)
    const d = a * a + b * b
    if ((n >> bit) & 1) {
      a = d
      b = c + d
    } else {
      a = c
      b = d
    }
  }
  return [a, b]
}

/**
 * The Fibonacci number F(n) (`F(0) = 0`, `F(1) = 1`, `F(n) = F(n-1) + F(n-2)`; `F(-n) = (-1)^(n+1) F(n)`) by fast
 * doubling, exactly as a BigInt.
 *
 * @param n - The index (|n| < 2^31).
 * @returns F(n).
 * @concept fibonacci-numbers
 * @example
 * ```ts
 * import { fibonacci } from 'whitebox-engine/foundations'
 * fibonacci(10) // => 55n
 * fibonacci(100) // => 354224848179261915075n
 * ```
 */
export function fibonacci(n: number): bigint {
  integer(n, 'fibonacci')
  const k = Math.abs(n)
  const [f] = fibLucas(k)
  return n < 0 && k % 2 === 0 ? -f : f
}

/**
 * The Lucas number L(n) (`L(0) = 2`, `L(1) = 1`, the same recurrence; `L(n) = F(n-1) + F(n+1)`), exactly.
 *
 * @param n - The index (|n| < 2^31).
 * @returns L(n).
 * @concept lucas-numbers
 * @example
 * ```ts
 * import { lucas } from 'whitebox-engine/foundations'
 * lucas(10) // => 123n
 * ```
 */
export function lucas(n: number): bigint {
  integer(n, 'lucas')
  const k = Math.abs(n)
  const [f, f1] = fibLucas(k)
  const l = 2n * f1 - f
  return n < 0 && k % 2 === 1 ? -l : l
}
