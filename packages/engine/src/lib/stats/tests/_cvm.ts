// The Cramer-von Mises null distribution as scipy.stats computes it: the limiting CDF (Anderson and Darling's series in
// Bessel K functions) and Csorgo and Faraway's finite-n correction.
import { gamma as gammaFn, kv, lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'

const K = (v: number, x: number): number => item(kv(v, x))

/** The limiting CDF of the Cramer-von Mises statistic. */
export function cvmCdfInf(x: number): number {
  let tot = 0
  for (let k = 0; k < 1000; k++) {
    const u = Math.exp(item(lgamma(k + 0.5)) - item(lgamma(k + 1))) / (Math.PI ** 1.5 * Math.sqrt(x))
    const y = 4 * k + 1
    const q = (y * y) / (16 * x)
    const z = u * Math.sqrt(y) * Math.exp(-q) * K(0.25, q)
    tot += z
    if (!(Math.abs(z) >= 1e-7)) break
  }
  return tot
}

function psi1Mod(x: number): number {
  const ed2 = (y: number): number => {
    const z = (y * y) / 4
    return (Math.exp(-z) * (y / 2) ** 1.5 * (K(0.25, z) + K(0.75, z))) / Math.sqrt(Math.PI)
  }
  const ed3 = (y: number): number => {
    const z = (y * y) / 4
    return (Math.exp(-z) / Math.sqrt(Math.PI)) * (y / 2) ** 2.5 * (2 * K(0.25, z) + 3 * K(0.75, z) - K(1.25, z))
  }
  const ak = (k: number): number => {
    const m = 2 * k + 1
    const sx = 2 * Math.sqrt(x)
    const y1 = x ** 0.75
    const y2 = x ** 1.25
    const g12 = item(gammaFn(k + 0.5))
    const g32 = item(gammaFn(k + 1.5))
    const e1 = (m * g12 * ed2((4 * k + 3) / sx)) / (9 * y1)
    const e2 = (g12 * ed3((4 * k + 1) / sx)) / (72 * y2)
    const e3 = (2 * (m + 2) * g32 * ed3((4 * k + 5) / sx)) / (12 * y2)
    const e4 = (7 * m * g12 * ed2((4 * k + 1) / sx)) / (144 * y1)
    const e5 = (7 * m * g12 * ed2((4 * k + 5) / sx)) / (144 * y1)
    return e1 + e2 + e3 + e4 + e5
  }
  let tot = 0
  for (let k = 0; k < 1000; k++) {
    const z = -ak(k) / (Math.PI * item(gammaFn(k + 1)))
    tot += z
    if (!(Math.abs(z) >= 1e-7)) break
  }
  return tot
}

/** The CDF of the one-sample Cramer-von Mises statistic for sample size n (Csorgo and Faraway 1996). */
export function cvmCdf(x: number, n: number): number {
  if (x >= n / 3) return 1
  if (!(x > 1 / (12 * n))) return 0
  return cvmCdfInf(x) * (1 + 1 / (12 * n)) + psi1Mod(x) / n
}
