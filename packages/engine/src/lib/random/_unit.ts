import { type Key, randomBits } from './key.ts'

const TWO_POW_53 = 9007199254740992

/** `n` uniform doubles in [0, 1) with 53 random bits each (two 32-bit words per value, as numpy does). */
export function unitDoubles(k: Key, n: number): Float64Array {
  const bits = randomBits(k, 2 * n)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const a = (bits[2 * i] ?? 0) >>> 5
    const b = (bits[2 * i + 1] ?? 0) >>> 6
    out[i] = (a * 67108864 + b) / TWO_POW_53
  }
  return out
}

/** Standard normal from a uniform stream (Marsaglia polar method). */
export function normalFrom(u: () => number): number {
  for (;;) {
    const x = 2 * u() - 1
    const y = 2 * u() - 1
    const s = x * x + y * y
    if (s > 0 && s < 1) return x * Math.sqrt((-2 * Math.log(s)) / s)
  }
}

/** log Gamma for positive arguments (Lanczos, g = 7, n = 9; about 15 digits). Private to the samplers. */
export function logGammaSmall(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGammaSmall(1 - x)
  const z = x - 1
  let a = c[0] ?? 0
  const t = z + 7.5
  for (let i = 1; i < 9; i++) a += (c[i] ?? 0) / (z + i)
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a)
}
