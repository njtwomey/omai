import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import * as F from './index.ts'

const int = fc.integer({ min: -1_000_000, max: 1_000_000 })
const pos = fc.integer({ min: 1, max: 1_000_000 })

describe('Euclid and modular arithmetic', () => {
  it('Bezout coefficients satisfy a x + b y = gcd, which divides both', () => {
    fc.assert(
      fc.property(int, int, (a, b) => {
        const { gcd, x, y, steps } = F.extendedEuclid(a, b)
        expect(a * x + b * y).toBe(gcd)
        expect(gcd).toBe(F.gcd(a, b))
        if (gcd !== 0) {
          expect(a % gcd).toBe(0 * Math.sign(a % gcd))
          expect(Math.abs(b % gcd)).toBe(0)
        }
        for (const s of steps) expect(s.r).toBe(a * s.s + b * s.t)
      }),
    )
    for (const s of F.euclid(1071, 462).steps) expect(s.a).toBe(s.quotient * s.b + s.remainder)
    expect(F.gcd()).toBe(0)
    expect(F.lcm(3, 0)).toBe(0)
    expect(() => F.euclid(0.5, 2)).toThrow()
  })

  it('inverses and powers agree with BigInt arithmetic', () => {
    fc.assert(
      fc.property(pos, fc.integer({ min: 0, max: 10_000 }), fc.integer({ min: 2, max: 1_000_003 }), (b, e, m) => {
        let want = 1n
        let base = BigInt(b) % BigInt(m)
        for (let k = e; k > 0; k >>= 1) {
          if (k & 1) want = (want * base) % BigInt(m)
          base = (base * base) % BigInt(m)
        }
        expect(F.modPow(b, e, m).value).toBe(Number(want))
        if (F.gcd(b, m) === 1) expect(F.modulo(F.modInverse(b, m) * b, m)).toBe(1)
      }),
    )
    expect(() => F.modInverse(6, 9)).toThrow(/no inverse/)
    expect(F.modPow(5, 0, 1).value).toBe(0)
    expect(F.modPow(3, 5, 7).steps.map((s) => s.bit)).toEqual([1, 0, 1])
    expect(() => F.modPow(2, -1, 7)).toThrow()
    expect(() => F.modulo(3, 0)).toThrow()
  })
})

describe('primes and factorisation', () => {
  it('the sieve agrees with Miller-Rabin and records smallest factors', () => {
    const { primes, smallestFactor } = F.sieve(2000)
    const set = new Set(primes)
    for (let n = 0; n <= 2000; n++) {
      expect(F.isPrime(n)).toBe(set.has(n))
      if (n >= 2) expect(n % (smallestFactor[n] ?? 0)).toBe(0)
    }
    expect(() => F.sieve(-1)).toThrow()
    expect(() => F.sieve(2e8)).toThrow()
  })

  it('Miller-Rabin records the witness against a composite', () => {
    const r = F.millerRabin(1763, [2])
    expect(r.prime).toBe(false)
    expect(r.rounds[0]?.witness).toBe(true)
    // 2047 = 23 * 89 is a strong pseudoprime to base 2: base 2 alone is fooled.
    expect(F.millerRabin(2047, [2]).prime).toBe(true)
    expect(F.millerRabin(2047).prime).toBe(false)
  })

  it('factorisations multiply back and determine phi and the divisors', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1e12 }), (n) => {
        const f = F.factorise(n)
        expect(f.reduce((p, [q, e]) => p * q ** e, 1)).toBe(n)
        for (const [q] of f) expect(F.isPrime(q)).toBe(true)
        const phi = F.eulerPhi(n)
        expect(phi).toBe(f.reduce((p, [q, e]) => p * (q - 1) * q ** (e - 1), 1))
      }),
      { numRuns: 50 },
    )
    const ds = F.divisors(720)
    expect(ds.length).toBe(30)
    expect(ds.reduce((s, d) => s + F.eulerPhi(d), 0)).toBe(720)
    expect(F.trialDivision(97).factors).toEqual([97])
  })

  it("Pollard's rho finds a factor and records its walk", () => {
    const r = F.pollardRho(10403)
    expect(10403 % r.factor).toBe(0)
    expect(r.factor).not.toBe(1)
    expect(r.steps.length).toBeGreaterThan(0)
    expect(F.pollardRho(1000).factor).toBe(2)
    expect(() => F.pollardRho(97)).toThrow()
  })
})

describe('Chinese remainder theorem and continued fractions', () => {
  it('the solution satisfies every congruence', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: -50, max: 50 }), fc.integer({ min: 1, max: 40 })), {
          minLength: 1,
          maxLength: 4,
        }),
        (pairs) => {
          const s = F.chineseRemainder(
            pairs.map((p) => p[0]),
            pairs.map((p) => p[1]),
          )
          if (s === null) return
          expect(s.modulus).toBe(F.lcm(...pairs.map((p) => p[1])))
          for (const [r, m] of pairs) expect(F.modulo(s.x - r, m)).toBe(0)
        },
      ),
    )
    expect(() => F.chineseRemainder([1], [2, 3])).toThrow()
  })

  it('convergents alternate around the value and approach it', () => {
    const cf = F.continuedFractionExpansion(Math.PI, undefined, 6)
    expect(cf).toEqual([3, 7, 15, 1, 292, 1])
    const conv = F.convergents(cf)
    for (const [k, [p, q]] of conv.entries()) {
      const err = p / q - Math.PI
      if (k > 0) expect(Math.sign(err)).toBe(k % 2 === 0 ? -1 : 1)
      expect(Math.abs(err)).toBeLessThan(1 / q ** 2)
    }
    expect(F.continuedFractionExpansion(3)).toEqual([3])
    expect(F.continuedFractionExpansion(7, -3)).toEqual([-3, 1, 2])
    expect(() => F.continuedFractionExpansion(1, 0)).toThrow()
  })
})

describe('Fibonacci and Lucas numbers', () => {
  it("satisfy Cassini's identity and the doubling formulas", () => {
    fc.assert(
      fc.property(fc.integer({ min: -200, max: 200 }), (n) => {
        const sign = n % 2 === 0 ? 1n : -1n
        expect(F.fibonacci(n + 1) * F.fibonacci(n - 1) - F.fibonacci(n) ** 2n).toBe(sign)
        expect(F.lucas(n)).toBe(F.fibonacci(n - 1) + F.fibonacci(n + 1))
        expect(F.fibonacci(2 * n)).toBe(F.fibonacci(n) * F.lucas(n))
      }),
    )
    expect(F.fibonacci(0)).toBe(0n)
    expect(F.lucas(0)).toBe(2n)
  })
})
