// Small helpers shared by the automata and agent-based models: a state codec for checkpoints, modular indices on a
// torus, a hash of a board for period detection, and the bits of a keyed uniform stream. Not exported.
import { type Json, MdmlError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import { uniform } from '../../random/index.ts'
import { stateFromJson, stateToJson } from '../../trace/index.ts'

/** The `serialise` and `deserialise` of an algorithm whose state is plain objects, arrays and typed arrays. */
export function codec<S>(): { serialise: (s: S) => Json; deserialise: (j: Json) => S } {
  return { serialise: (s) => stateToJson(s), deserialise: (j) => stateFromJson(j) as S }
}

/** `n` uniform draws in [0, 1) from a key. */
export function draws(key: Key, n: number): Float64Array {
  return uniform(key, [Math.max(n, 1)]).data
}

/** i modulo n, in [0, n). */
export function mod(i: number, n: number): number {
  return ((i % n) + n) % n
}

/** Check that a size is a positive integer. */
export function positiveInt(name: string, what: string, x: number): void {
  if (!(Number.isInteger(x) && x >= 1))
    throw new MdmlError('value', `${name}: ${what} must be a positive integer, got ${x}`)
}

/** Check that a number lies in [0, 1]. */
export function unit(name: string, what: string, x: number): void {
  if (!(x >= 0 && x <= 1)) throw new MdmlError('value', `${name}: ${what} must be in [0, 1], got ${x}`)
}

/** A 53-bit hash of the bytes of a board (two FNV-1a passes), for spotting a state that has come round again. */
export function hashBoard(cells: ArrayLike<number>): number {
  let a = 0x811c9dc5
  let b = 0x9747b28c
  for (let i = 0; i < cells.length; i++) {
    const c = (cells[i] ?? 0) & 0xff
    a = Math.imul(a ^ c, 0x01000193)
    b = Math.imul(b ^ (c + 0x9e), 0x85ebca6b) ^ (b >>> 13)
  }
  return (a >>> 0) * 0x200000 + ((b >>> 0) >>> 11)
}
