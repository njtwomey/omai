// Factorisation over the rationals. Univariate: square-free decomposition (Yun), then Berlekamp-Zassenhaus
// (Berlekamp's deterministic factorisation modulo a small prime, quadratic Hensel lifting to beyond the
// Landau-Mignotte bound, and recombination of lifted factors by trial division). Multivariate: contents and
// square-free parts recursively, then Kronecker's substitution x_i -> t^(D^i) and recombination of the univariate
// factors, which finds every factor (the images of true factors are products of univariate factors).

import { charge } from './_budget.ts'
import {
  fromTerms,
  type P,
  pconst,
  pcontentIn,
  pdeg,
  pdiff,
  pdivExact,
  pgcd,
  pIsConst,
  pIsZero,
  pimage,
  pmul,
  pnumericContent,
  psub,
  pvars,
  type Term,
  type U,
  udeg,
  usqf,
  utrim,
} from './_poly.ts'
import { babs, bgcd, biroot, R1, type Rational, rat, rdiv, rmul } from './_rational.ts'
import {
  mod,
  ratsToZ,
  symmetric,
  type Z,
  zadd,
  zderiv,
  zdivmod,
  zgcd,
  zinv,
  zmod,
  zmonic,
  zmul,
  zprimitive,
  zsub,
  zToRats,
  ztrim,
} from './_zpoly.ts'

/** s, t with s*a + t*b = 1 modulo p (a, b coprime). */
function zxgcd(a: Z, b: Z, p: bigint): [Z, Z] {
  let [r0, r1] = [zmod(a, p), zmod(b, p)]
  let [s0, s1] = [[1n] as Z, [] as Z]
  let [t0, t1] = [[] as Z, [1n] as Z]
  while (r1.length > 0) {
    const [q, r] = zdivmod(r0, r1, p)
    ;[r0, r1] = [r1, r]
    ;[s0, s1] = [s1, zsub(s0, zmul(q, s1, p), p)]
    ;[t0, t1] = [t1, zsub(t0, zmul(q, t1, p), p)]
  }
  const inv = zinv(r0[0] as bigint, p)
  return [
    zmod(
      s0.map((c) => c * inv),
      p,
    ),
    zmod(
      t0.map((c) => c * inv),
      p,
    ),
  ]
}

/** x^e mod (f, p). */
function zpowmodX(e: bigint, f: Z, p: bigint): Z {
  let result: Z = [1n]
  let base: Z = zdivmod([0n, 1n], f, p)[1]
  let k = e
  while (k > 0n) {
    if (k & 1n) result = zdivmod(zmul(result, base, p), f, p)[1]
    base = zdivmod(zmul(base, base, p), f, p)[1]
    k >>= 1n
  }
  return result
}

/** Basis of the nullspace of (Q - I) for Berlekamp's algorithm, modulo p. */
function nullspace(rows: Z[], n: number, p: bigint): Z[] {
  // Matrix M (n x n) with M[i][j] = Q[j][i] - delta; we solve M v = 0 by Gauss-Jordan elimination.
  const m: bigint[][] = []
  for (let i = 0; i < n; i++) {
    const row: bigint[] = []
    for (let j = 0; j < n; j++) row.push(mod(((rows[j] as Z)[i] ?? 0n) - (i === j ? 1n : 0n), p))
    m.push(row)
  }
  const pivots: number[] = []
  let r = 0
  for (let c = 0; c < n && r < n; c++) {
    let piv = -1
    for (let i = r; i < n; i++) if ((m[i] as bigint[])[c] !== 0n) piv = piv < 0 ? i : piv
    if (piv < 0) continue
    ;[m[r], m[piv]] = [m[piv] as bigint[], m[r] as bigint[]]
    charge(n * n)
    const row = m[r] as bigint[]
    const inv = zinv(row[c] as bigint, p)
    for (let j = 0; j < n; j++) row[j] = mod((row[j] as bigint) * inv, p)
    for (let i = 0; i < n; i++) {
      if (i === r) continue
      const other = m[i] as bigint[]
      const f = other[c] as bigint
      if (f === 0n) continue
      for (let j = 0; j < n; j++) other[j] = mod((other[j] as bigint) - f * (row[j] as bigint), p)
    }
    pivots.push(c)
    r++
  }
  const free: number[] = []
  for (let c = 0; c < n; c++) if (!pivots.includes(c)) free.push(c)
  return free.map((fc) => {
    const v: bigint[] = new Array<bigint>(n).fill(0n)
    v[fc] = 1n
    pivots.forEach((pc, i) => {
      v[pc] = mod(-((m[i] as bigint[])[fc] as bigint), p)
    })
    return ztrim(v)
  })
}

/** Factor a square-free polynomial modulo p into monic irreducibles (Berlekamp). */
function berlekamp(f: Z, p: bigint): Z[] {
  const fm = zmonic(f, p)
  const n = fm.length - 1
  if (n <= 1) return [fm]
  const rows: Z[] = []
  const xp = zpowmodX(p, fm, p)
  let cur: Z = [1n]
  for (let i = 0; i < n; i++) {
    rows.push(cur)
    cur = zdivmod(zmul(cur, xp, p), fm, p)[1]
  }
  const basis = nullspace(rows, n, p)
  const r = basis.length
  let factors: Z[] = [fm]
  for (const v of basis) {
    if (factors.length >= r) break
    if (v.length <= 1) continue
    const next: Z[] = []
    for (const h of factors) {
      let pieces: Z[] = [h]
      if (h.length - 1 > 1) {
        for (let s = 0n; s < p && pieces.length < r; s++) {
          const split: Z[] = []
          for (const q of pieces) {
            const g = zgcd(q, zsub(v, [s], p), p)
            if (g.length > 1 && g.length < q.length) split.push(g, zdivmod(q, g, p)[0])
            else split.push(q)
          }
          pieces = split.map((q) => zmonic(q, p))
        }
      }
      next.push(...pieces)
    }
    factors = next
  }
  return factors
}

/** One quadratic Hensel step: f = g*h mod m to modulo mm (h monic, s*g + t*h = 1). */
function henselStep(f: Z, g: Z, h: Z, s: Z, t: Z, mm: bigint): [Z, Z, Z, Z] {
  const e = zsub(zmod(f, mm), zmul(g, h, mm), mm)
  const [q, r] = zdivmod(zmul(s, e, mm), h, mm)
  const g2 = zadd(zadd(g, zmul(t, e, mm), mm), zmul(q, g, mm), mm)
  const h2 = zadd(h, r, mm)
  const b = zsub(zadd(zmul(s, g2, mm), zmul(t, h2, mm), mm), [1n], mm)
  const [c, d] = zdivmod(zmul(s, b, mm), h2, mm)
  const s2 = zsub(s, d, mm)
  const t2 = zsub(zsub(t, zmul(t, b, mm), mm), zmul(c, g2, mm), mm)
  return [g2, h2, s2, t2]
}

/** Lift f = lc * prod(factors) mod p to mod p^k; returns monic lifted factors. */
function henselLift(f: Z, factors: Z[], p: bigint, pk: bigint): Z[] {
  if (factors.length === 1) return [zmonic(zmod(f, pk), pk)]
  const lc = f[f.length - 1] as bigint
  const first = factors[0] as Z
  const rest = factors.slice(1)
  let h: Z = [1n]
  for (const q of rest) h = zmul(h, q, p)
  let g = zmod(
    first.map((c) => c * lc),
    p,
  )
  let [s, t] = zxgcd(g, h, p)
  let m = p
  while (m < pk) {
    const mm = m * m < pk ? m * m : pk
    ;[g, h, s, t] = henselStep(f, g, h, s, t, mm)
    m = mm
  }
  const liftedFirst = zmonic(g, pk)
  return [liftedFirst, ...henselLift(h, rest, p, pk)]
}

function* combinations(n: number, k: number, start = 0): Generator<number[]> {
  if (k === 0) {
    yield []
    return
  }
  for (let i = start; i <= n - k; i++) for (const rest of combinations(n, k - 1, i + 1)) yield [i, ...rest]
}

/** Exact division over Z; undefined if b does not divide a. */
function zexactDiv(a: Z, b: Z): Z | undefined {
  let r = ztrim(a)
  const bt = ztrim(b)
  const db = bt.length - 1
  const lb = bt[db] as bigint
  const q: Z = new Array<bigint>(Math.max(0, r.length - db)).fill(0n)
  while (r.length - 1 >= db && r.length > 0) {
    charge(r.length)
    const top = r[r.length - 1] as bigint
    if (top % lb !== 0n) return undefined
    const c = top / lb
    const k = r.length - 1 - db
    q[k] = c
    const next = [...r]
    for (let i = 0; i <= db; i++) next[k + i] = (next[k + i] as bigint) - c * (bt[i] as bigint)
    r = ztrim(next)
  }
  return r.length === 0 ? ztrim(q) : undefined
}

const PRIMES: bigint[] = (() => {
  const out: bigint[] = []
  for (let n = 3; out.length < 60; n += 2) {
    let prime = true
    for (let d = 3; d * d <= n; d += 2) if (n % d === 0) prime = false
    if (prime) out.push(BigInt(n))
  }
  return out
})()

/** Irreducible factors over Z of a square-free primitive integer polynomial with positive leading coefficient. */
function zassenhaus(f: Z): Z[] {
  const n = f.length - 1
  if (n <= 1) return [f]
  const lc = f[n] as bigint
  // Choose the prime (among the first few suitable ones) giving the fewest modular factors.
  let best: { p: bigint; factors: Z[] } | undefined
  let tried = 0
  for (const p of PRIMES) {
    if (lc % p === 0n) continue
    const fp = zmod(f, p)
    if (zgcd(fp, zderiv(fp), p).length > 1) continue
    const factors = berlekamp(fp, p)
    if (best === undefined || factors.length < best.factors.length) best = { p, factors }
    if (factors.length === 1 || ++tried >= 5) break
  }
  if (best === undefined) throw new Error('no suitable prime')
  if (best.factors.length === 1) return [f]
  const { p } = best
  let maxCoeff = 0n
  for (const c of f) maxCoeff = babs(c) > maxCoeff ? babs(c) : maxCoeff
  const bound = 2n * babs(lc) * (biroot(BigInt(n + 1), 2) + 1n) * (1n << BigInt(n)) * maxCoeff
  let pk = p
  while (pk <= bound) pk *= p
  let lifted = henselLift(f, best.factors, p, pk)
  const out: Z[] = []
  let g = f
  let size = 1
  while (2 * size <= lifted.length) {
    let found = false
    for (const subset of combinations(lifted.length, size)) {
      charge(size)
      const lcg = g[g.length - 1] as bigint
      let cand: Z = [lcg]
      for (const i of subset) cand = zmul(cand, lifted[i] as Z, pk)
      cand = zprimitive(cand.map((c) => symmetric(c, pk)))
      const q = zexactDiv(g, cand)
      if (q !== undefined) {
        out.push(cand)
        g = q
        lifted = lifted.filter((_, i) => !subset.includes(i))
        found = true
        break
      }
    }
    if (!found) size++
  }
  if (g.length > 1) out.push(zprimitive(g))
  return out
}

/** Factor a univariate polynomial over Q: constant and irreducible primitive integer factors with multiplicities. @internal */
export function factorUnivariate(f: U): [Rational, [U, number][]] {
  const t = utrim(f)
  if (udeg(t) <= 0) return [t[0] ?? rat(0n), []]
  const factors: [U, number][] = []
  for (const [part, mult] of usqf(t)) {
    for (const z of zassenhaus(zprimitive(ratsToZ(part)))) factors.push([zToRats(z), mult])
  }
  // The constant is what remains after dividing by the primitive factors.
  let lcProd: Rational = R1
  for (const [g, k] of factors) for (let i = 0; i < k; i++) lcProd = rmul(lcProd, g[g.length - 1] as Rational)
  return [rdiv(t[t.length - 1] as Rational, lcProd), factors]
}

/** Factor a multivariate polynomial over Q into a constant and irreducible factors with multiplicities. @internal */
export function factorMultivariate(f: P): [Rational, [P, number][]] {
  if (pIsZero(f)) return [rat(0n), []]
  const [c, prim] = pnumericContent(f)
  if (pIsConst(prim)) return [c, []]
  const out: [P, number][] = []
  // Monomial content.
  const mins = new Array<number>(f.n).fill(Number.POSITIVE_INFINITY)
  for (const t of prim.terms)
    t.e.forEach((k, i) => {
      mins[i] = Math.min(mins[i] as number, k)
    })
  let rest = prim
  mins.forEach((m, i) => {
    if (m > 0 && Number.isFinite(m)) {
      const e = new Array<number>(f.n).fill(0)
      e[i] = 1
      out.push([fromTerms(f.n, [{ e, c: R1 }]), m])
      rest = fromTerms(
        f.n,
        rest.terms.map((t) => ({ e: t.e.map((k, j) => (j === i ? k - m : k)), c: t.c })),
      )
    }
  })
  for (const [g, k] of factorPrimitive(rest)) out.push([g, k])
  // Normalise: each factor primitive with positive leading coefficient; fold constants into c.
  let constant = c
  const normal: [P, number][] = []
  for (const [g, k] of out) {
    const [gc, gp] = pnumericContent(g)
    for (let i = 0; i < k; i++) constant = rmul(constant, gc)
    if (!pIsConst(gp)) normal.push([gp, k])
  }
  // Fix the constant so that the product reproduces f exactly.
  let prod = pconst(f.n, constant)
  for (const [g, k] of normal) for (let i = 0; i < k; i++) prod = pmul(prod, g)
  const ratio = rdiv((f.terms[0] as Term).c, (prod.terms[0] as Term).c)
  return [rmul(constant, ratio), normal]
}

function factorPrimitive(f: P): [P, number][] {
  if (pIsConst(f)) return []
  const vs = pvars(f)
  if (vs.length === 1) {
    const v = vs[0] as number
    const dense: U = new Array<Rational>(pdeg(f, v) + 1).fill(rat(0n))
    for (const t of f.terms) dense[t.e[v] as number] = t.c
    const [, facs] = factorUnivariate(dense)
    return facs.map(([g, k]) => [
      fromTerms(
        f.n,
        g.map((cf, d) => {
          const e = new Array<number>(f.n).fill(0)
          e[v] = d
          return { e, c: cf }
        }),
      ),
      k,
    ])
  }
  const v = vs[0] as number
  const out: [P, number][] = []
  const cont = pcontentIn(f, v)
  let pp = f
  if (!pIsConst(cont)) {
    out.push(...factorPrimitive(cont))
    pp = pdivExact(f, cont) ?? f
  }
  // Square-free decomposition in v (Yun) with multivariate gcds.
  const d = pdiff(pp, v)
  const a0 = pgcd(pp, d)
  let b = pdivExact(pp, a0) ?? pp
  let cc = pdivExact(d, a0) ?? d
  let dd = psub(cc, pdiff(b, v))
  let i = 1
  while (!pIsConst(b)) {
    const ai = pgcd(b, dd)
    b = pdivExact(b, ai) ?? b
    cc = pdivExact(dd, ai) ?? dd
    dd = psub(cc, pdiff(b, v))
    if (!pIsConst(ai)) for (const g of kronecker(ai)) out.push([g, i])
    i++
  }
  return out
}

const MAX_TEST_DEGREE = 12

/**
 * Whether f (in several variables) is certainly irreducible: for a variable w in which f is primitive, f of degree 1
 * in w, or an image f(w; others at integer points) of the same degree that is irreducible over Q, proves it (a
 * factorisation of f would map to one of the image with the same degrees). False means "not proven".
 */
function provablyIrreducible(f: P, vs: readonly number[]): boolean {
  const order = [...vs].sort((a, b) => pdeg(f, a) - pdeg(f, b))
  for (const w of order) {
    const d = pdeg(f, w)
    if (d > MAX_TEST_DEGREE) break
    if (!pIsConst(pcontentIn(f, w))) continue
    if (d === 1) return true
    for (let attempt = 0; attempt < 2; attempt++) {
      const image = pimage(f, w, attempt)
      if (image[d] === 0n) continue
      const [, facs] = factorUnivariate(zToRats(image))
      if (facs.length === 1 && (facs[0] as [U, number])[1] === 1) return true
    }
  }
  return false
}

/** Irreducible factors of a square-free polynomial in several variables, by Kronecker's substitution. */
function kronecker(f: P): P[] {
  const vs = pvars(f)
  if (vs.length <= 1) return factorPrimitive(f).map(([g]) => g)
  // Most polynomials are irreducible, and an image in one variable usually shows it at a fraction of the cost.
  if (provablyIrreducible(f, vs)) return [f]
  let D = 1
  for (const v of vs) D = Math.max(D, pdeg(f, v) + 1)
  const degree = f.terms.reduce(
    (m, t) =>
      Math.max(
        m,
        vs.reduce((s, v, j) => s + (t.e[v] as number) * D ** j, 0),
      ),
    0,
  )
  if (degree > 400) return [f]
  const image: Z = new Array<bigint>(degree + 1).fill(0n)
  let den = 1n
  for (const t of f.terms) den = (den / bgcd(den, t.c.d)) * t.c.d
  for (const t of f.terms) {
    const k = vs.reduce((s, v, j) => s + (t.e[v] as number) * D ** j, 0)
    image[k] = (image[k] as bigint) + (t.c.n * den) / t.c.d
  }
  const [, ufacs] = factorUnivariate(zToRats(zprimitive(image)))
  const pieces: Z[] = []
  for (const [g, k] of ufacs) for (let i = 0; i < k; i++) pieces.push(ratsToZ(g))
  // Powers of t are part of the multiset too (they come from monomials in the images).
  const decode = (z: Z): P | undefined => {
    const terms: Term[] = []
    for (let k = 0; k < z.length; k++) {
      const c = z[k] as bigint
      if (c === 0n) continue
      const e = new Array<number>(f.n).fill(0)
      let rem = k
      for (const v of vs) {
        e[v] = rem % D
        rem = Math.floor(rem / D)
      }
      if (rem !== 0) return undefined
      terms.push({ e, c: rat(c) })
    }
    return fromTerms(f.n, terms)
  }
  const out: P[] = []
  let g = f
  let remaining = pieces
  let size = 1
  let budget = 1 << 14
  while (2 * size <= remaining.length && budget > 0) {
    let found = false
    for (const subset of combinations(remaining.length, size)) {
      charge(size)
      if (--budget <= 0) break
      let prod: Z = [1n]
      for (const i of subset) prod = zmulInt(prod, remaining[i] as Z)
      const cand = decode(prod)
      if (cand === undefined || pIsConst(cand)) continue
      const q = pdivExact(g, cand)
      if (q !== undefined) {
        out.push(cand)
        g = q
        remaining = remaining.filter((_, i) => !subset.includes(i))
        found = true
        break
      }
    }
    if (!found) size++
  }
  if (!pIsConst(g)) out.push(g)
  return out
}

function zmulInt(a: Z, b: Z): Z {
  charge(a.length * b.length)
  const out: Z = new Array<bigint>(a.length + b.length - 1).fill(0n)
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) out[i + j] = (out[i + j] as bigint) + (a[i] as bigint) * (b[j] as bigint)
  return ztrim(out)
}
