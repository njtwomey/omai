// Joint action profiles of a normal-form game: row-major indices over the players' action counts (player 0 slowest).
import { stridesOf } from '../../tensor/index.ts'

/** The number of joint profiles. */
export function profileCount(shape: readonly number[]): number {
  let n = 1
  for (const k of shape) n *= k
  return n
}

/** The profile (one action per player) of a flat index. */
export function decodeProfile(index: number, shape: readonly number[]): number[] {
  const out = new Array<number>(shape.length).fill(0)
  let r = index
  for (let i = shape.length - 1; i >= 0; i--) {
    const k = shape[i] ?? 1
    out[i] = r % k
    r = Math.floor(r / k)
  }
  return out
}

/** The flat index of a profile. */
export function encodeProfile(profile: readonly number[], strides: readonly number[]): number {
  let idx = 0
  for (let i = 0; i < profile.length; i++) idx += (profile[i] ?? 0) * (strides[i] ?? 0)
  return idx
}

/** The payoffs `u` of each of `player`'s actions against every joint profile of the others: `rows[a][s]`. */
export function actionRows(shape: readonly number[], u: ArrayLike<number>, player: number): number[][] {
  const k = shape[player] ?? 0
  const stride = stridesOf(shape)[player] ?? 1
  const rows: number[][] = Array.from({ length: k }, () => [])
  const total = profileCount(shape)
  for (let s = 0; s < total; s++) rows[Math.floor(s / stride) % k]?.push(u[s] ?? 0)
  return rows
}

/**
 * The probability of every joint profile under independent mixed strategies, skipping `skip` (whose factor is 1), as a
 * flat array in profile order.
 */
export function profileWeights(
  shape: readonly number[],
  strategies: readonly ArrayLike<number>[],
  skip = -1,
): Float64Array {
  const total = profileCount(shape)
  const out = new Float64Array(total)
  const profile = new Array<number>(shape.length).fill(0)
  for (let s = 0; s < total; s++) {
    let w = 1
    for (let q = 0; q < shape.length && w !== 0; q++) if (q !== skip) w *= strategies[q]?.[profile[q] ?? 0] ?? 0
    out[s] = w
    for (let q = shape.length - 1; q >= 0; q--) {
      profile[q] = (profile[q] ?? 0) + 1
      if ((profile[q] ?? 0) < (shape[q] ?? 1)) break
      profile[q] = 0
    }
  }
  return out
}
