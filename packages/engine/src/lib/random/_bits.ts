// Block `block` of the bit stream of a key: the one place that fixes the counter layout of `randomBits`, so that code
// needing only a few words (the lazy blocks of `uniformStream`) reads the same words as the whole-array generator.
import { philox4x32 } from './philox.ts'

// The domain tag in the fourth counter word keeps bit generation and key derivation from ever sharing a block.
const BITS_TAG = 0

/** The four words of Philox block `block` (a non-negative integer) of the bit stream of the key with these words. */
export function bitsBlock(keyWords: readonly [number, number], block: number): [number, number, number, number] {
  return philox4x32([block >>> 0, Math.floor(block / 2 ** 32) >>> 0, 0, BITS_TAG], keyWords)
}
