import { Generator, type Key } from '../random/index.ts'

/** Scalar draws from one key for the stochastic searches, buffered from `Generator` blocks (a pure function of the key). */
export interface Draw {
  /** A uniform in [0, 1). */
  uniform(): number
  /** A uniform integer in [0, n). */
  int(n: number): number
  /** A standard normal. */
  normal(): number
  /** A fresh key, for a callee that takes keys. */
  key(): Key
}

const BLOCK = 64

/** Draws from a key. */
export function draws(key: Key): Draw {
  const rng = new Generator(key)
  let u: Float64Array = new Float64Array(0)
  let ui = 0
  let z: Float64Array = new Float64Array(0)
  let zi = 0
  const uniform = (): number => {
    if (ui >= u.length) {
      u = rng.uniform([BLOCK]).data
      ui = 0
    }
    return u[ui++] ?? 0
  }
  return {
    uniform,
    int: (n) => Math.min(n - 1, Math.floor(uniform() * n)),
    normal: () => {
      if (zi >= z.length) {
        z = rng.normal([BLOCK]).data
        zi = 0
      }
      return z[zi++] ?? 0
    },
    key: () => rng.nextKey(),
  }
}
