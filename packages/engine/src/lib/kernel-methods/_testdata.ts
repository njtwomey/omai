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
