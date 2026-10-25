/**
 * The Philox4x32-10 block function (Salmon et al., 2011): a keyed bijection from a 128-bit counter to 128 random bits,
 * computed exactly in float64 arithmetic. It is the counter-based generator underneath every key in whitebox-js.
 */

const M0 = 0xd2511f53
const M1 = 0xcd9e8d57
const W0 = 0x9e3779b9
const W1 = 0xbb67ae85

/** High 32 bits of the 64-bit product of two unsigned 32-bit integers, exact in float64 arithmetic. */
function mulhi(a: number, b: number): number {
  const aLo = a & 0xffff
  const aHi = a >>> 16
  const bLo = b & 0xffff
  const bHi = b >>> 16
  const ll = aLo * bLo
  const lh = aLo * bHi
  const hl = aHi * bLo
  const hh = aHi * bHi
  const mid = (ll >>> 16) + (lh & 0xffff) + (hl & 0xffff)
  return (hh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0
}

/**
 * The Philox4x32-10 block function (Salmon et al., 2011): a keyed bijection from a 128-bit counter to 128 random bits.
 *
 * Counter-based generators need no mutable state: the same key and counter always give the same bits, and any block
 * can be computed directly. This makes every random draw in whitebox-js reproducible and order independent.
 *
 * @param counter - Four unsigned 32-bit words.
 * @param key - Two unsigned 32-bit words.
 * @returns Four unsigned 32-bit words.
 * @concept counter-based-rng
 * @see https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 * @example
 * ```ts
 * import { philox4x32 } from 'whitebox-engine/random'
 * philox4x32([0, 0, 0, 0], [0, 0]) // => [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
 * ```
 */
export function philox4x32(
  counter: readonly [number, number, number, number],
  key: readonly [number, number],
): [number, number, number, number] {
  let [c0, c1, c2, c3] = counter
  let [k0, k1] = key
  for (let round = 0; round < 10; round++) {
    const hi0 = mulhi(M0, c0)
    const lo0 = Math.imul(M0, c0) >>> 0
    const hi1 = mulhi(M1, c2)
    const lo1 = Math.imul(M1, c2) >>> 0
    c0 = (hi1 ^ c1 ^ k0) >>> 0
    c1 = lo1
    c2 = (hi0 ^ c3 ^ k1) >>> 0
    c3 = lo0
    k0 = (k0 + W0) >>> 0
    k1 = (k1 + W1) >>> 0
  }
  return [c0, c1, c2, c3]
}
