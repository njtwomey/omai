/**
 * Seed rerolling without hidden randomness: `rerollSeed` derives the next seed from the current one by a key, so the
 * same seed always rerolls to the same next seed, within `MAX_SEED`.
 */

import { foldIn, key, randomBits } from 'whitebox-engine/random'

/** The largest seed {@link SeedInput} offers. */
export const MAX_SEED = 999_999

/**
 * A fresh seed, derived from the current one by a key (no hidden randomness: the same seed always rerolls to the same
 * next seed, so a page's sequence of rerolls is reproducible).
 *
 * @param seed - The current seed.
 * @returns A different seed in `[0, MAX_SEED]`.
 */
export function rerollSeed(seed: number): number {
  const bits = randomBits(foldIn(key(seed), 0x5eed), 1)[0] ?? 0
  const next = bits % (MAX_SEED + 1)
  return next === seed ? (next + 1) % (MAX_SEED + 1) : next
}
