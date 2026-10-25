/**
 * A numpy-style stateful random generator (`Generator`, created with `defaultRng`): a thin wrapper that holds a key and
 * splits it on every call, for code that prefers `rng.normal(...)` to passing keys by hand.
 */

import type { Shape, Tensor } from '../tensor/index.ts'
import { normal, uniform } from './continuous.ts'
import { randint } from './discrete.ts'
import { isKey, type Key, key as makeKey, split } from './key.ts'
import { permutation } from './permutation.ts'

/**
 * A numpy-style random generator: a thin stateful wrapper over the functional key API.
 *
 * Each call splits the held key and draws from the fresh half, so a generator created from the same seed produces the
 * same sequence. Library code takes {@link Key} arguments; use this class in scripts and notebooks where carrying keys
 * around is noise.
 *
 * @param seed - An integer seed or an existing key.
 * @returns The generator.
 * @concept counter-based-rng
 * @example
 * ```ts
 * import { Generator } from 'whitebox-engine/random'
 * const a = new Generator(7).normal([3])
 * const b = new Generator(7).normal([3])
 * a.data[2] === b.data[2] // => true
 * ```
 */
export class Generator {
  #key: Key

  constructor(seed: number | Key) {
    this.#key = isKey(seed) ? seed : makeKey(seed)
  }

  /** Take a fresh key and advance the generator. */
  nextKey(): Key {
    const [next, use] = split(this.#key)
    this.#key = next as Key
    return use as Key
  }

  /** Uniform samples in `[low, high)`; see {@link uniform}. */
  uniform(shape: Shape, low = 0, high = 1): Tensor<'float64'> {
    return uniform(this.nextKey(), shape, { low, high })
  }

  /** Normal samples; see {@link normal}. */
  normal(shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
    return normal(this.nextKey(), shape, { loc, scale })
  }

  /** Integers in `[low, high)`; see {@link randint}. */
  integers(low: number, high: number, shape: Shape): Tensor<'int32'> {
    return randint(this.nextKey(), shape, low, high)
  }

  /** A random permutation of `0..n-1`; see {@link permutation}. */
  permutation(n: number): Tensor<'int32'> {
    return permutation(this.nextKey(), n)
  }
}

/**
 * Create a {@link Generator} (the `numpy.random.default_rng` spelling).
 *
 * @param seed - An integer seed.
 * @returns The generator.
 * @example
 * ```ts
 * import { defaultRng } from 'whitebox-engine/random'
 * defaultRng(0).uniform([2]).shape // => [2]
 * ```
 */
export function defaultRng(seed: number): Generator {
  return new Generator(seed)
}
