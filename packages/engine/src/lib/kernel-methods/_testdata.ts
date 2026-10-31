// Shared fixtures of the tests of this folder (not exported from the folder): small planted problems from a key.
import { key as makeKey, normal, uniform } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'

/** `n` rows of `d` uniform features in `[lo, hi]` from a seed. */
export function uniformRows(n: number, d: number, seed: number, lo = -2, hi = 2): number[][] {
  return toArray(uniform(makeKey(seed), [n, d], { low: lo, high: hi })) as number[][]
}

/** `n` standard normal draws from a seed. */
export function normals(n: number, seed: number): number[] {
  return Array.from(toFlat(normal(makeKey(seed), [n])))
}

/** Two Gaussian blobs in the plane, `n` per class, centred at `(-gap/2, 0)` and `(gap/2, 0)`, labels 0 and 1. */
export function blobs(n: number, gap: number, seed: number): { x: number[][]; y: number[] } {
  const z = normals(4 * n, seed)
  const x: number[][] = []
  const y: number[] = []
  for (let i = 0; i < 2 * n; i++) {
    const c = i < n ? 0 : 1
    x.push([(c === 0 ? -gap / 2 : gap / 2) + (z[2 * i] ?? 0), z[2 * i + 1] ?? 0])
    y.push(c)
  }
  return { x, y }
}
