/**
 * What a mapping is: one entry of an ordered table (`_table.ts`) that recognises a pattern of the model and emits its part of the
 * graph. An entry has a name, a line saying what it maps, the factor operators it needs, a pure test of whether the node is its
 * pattern (`matches`) and the emission (`map`). An entry that can only tell by trying (it evaluates the node's expressions and
 * finds they are not its form) answers `NOT_MINE`, and the runner then undoes whatever the attempt made (`attempt`), so a failed
 * recognition leaves nothing in the graph and the next entry sees the graph as it was. The first entry that takes a node maps it:
 * the order of a table is part of its meaning, and is written down in the entries.
 */

import { OmaiError } from '../../../base/index.ts'
import type { OperatorName } from '../../../message-passing/index.ts'
import type { BuilderCheckpoint } from './_builder.ts'
import type { Context } from './_context.ts'
import { Refusal } from './_refusal.ts'

/** The answer of an entry that tried a node and found it is not its pattern. */
export const NOT_MINE: unique symbol = Symbol('not-mine')

/** The type of `NOT_MINE`. */
export type NotMine = typeof NOT_MINE

/** One entry of a mapping table, for the arguments `A` a table passes (the node, and what the table has made of it) and the result `R`. */
export interface Mapping<A extends readonly unknown[], R> {
  /** The name of the mapping: stable, one per entry, what a refusal or a doc refers to. */
  readonly name: string
  /** One line: what the entry maps, in words (the generated list of what the compiler maps is made of these). */
  readonly maps: string
  /** Why the entry sits where it does, when its neighbours matter (an earlier entry would take the node, or a later one is its fallback). */
  readonly order?: string
  /** The distribution families the entry serves, when the table is keyed by family (absent: any). */
  readonly families?: readonly string[]
  /** The factor operators of the engine the entry can emit (a test checks they exist, and that every operator is claimed). */
  readonly requires: readonly OperatorName[]
  /** Whether the node is this entry's pattern: pure, no emission. Absent: the entry takes every node that reaches it. */
  readonly matches?: (cx: Context, ...args: A) => boolean
  /** Map the node. `NOT_MINE` (only from an entry that recognises by trying) hands the node to the next entry, with the graph restored. */
  readonly map: (cx: Context, ...args: A) => R | NotMine
}

/** A table: the entries in priority order. */
export type MappingTable<A extends readonly unknown[], R> = readonly Mapping<A, R>[]

/** The state a failed attempt restores: the graph under construction, and the evaluator's caches that refer to it. */
interface Checkpoint {
  readonly graph: BuilderCheckpoint
  readonly applied: number
  readonly cells: number
}

function checkpointOf(cx: Context): Checkpoint {
  return { graph: cx.g.checkpoint(), applied: cx.applied.size, cells: cx.cells.size }
}

/** Delete the entries of a map that were added after it had `size` entries (maps keep insertion order; nothing is overwritten by an attempt). */
function truncateMap(m: Map<string, unknown>, size: number): void {
  if (m.size <= size) return
  let k = 0
  for (const key of [...m.keys()]) if (k++ >= size) m.delete(key)
}

/**
 * Run `f`, which recognises by trying. If it answers `NOT_MINE` after making variables or factors, they are taken out again (with the
 * auxiliary names, the atoms, the apply chains and the cached cells that refer to them), so the graph is as it was before. An attempt that
 * made nothing leaves the caches as they are (a recognition that only evaluated keeps its work for the next entry). A `Refusal` thrown from
 * `f` is a refusal of the node and goes up unchanged.
 */
export function attempt<T>(cx: Context, f: () => T | NotMine): T | NotMine {
  const before = checkpointOf(cx)
  const result = f()
  if (result === NOT_MINE && cx.g.changedSince(before.graph)) {
    cx.g.restore(before.graph)
    truncateMap(cx.applied, before.applied)
    truncateMap(cx.cells, before.cells)
  }
  return result
}

/**
 * Evaluate something that may fail because the node is not of the form the caller looks for: `NOT_MINE` when the evaluation refuses
 * (the refusal is dropped: it was only a "no"), else its value. Whatever the evaluation made in the graph before it refused is taken
 * out again.
 */
export function probe<T>(cx: Context, f: () => T): T | NotMine {
  const before = checkpointOf(cx)
  try {
    return f()
  } catch (e) {
    if (!(e instanceof Refusal)) throw e
    if (cx.g.changedSince(before.graph)) {
      cx.g.restore(before.graph)
      truncateMap(cx.applied, before.applied)
      truncateMap(cx.cells, before.cells)
    }
    return NOT_MINE
  }
}

/** Map with the first entry of `table` whose pattern the arguments are; the entries after the one that took the node are not asked. */
export function mapFirst<A extends readonly unknown[], R>(
  table: MappingTable<A, R>,
  cx: Context,
  what: string,
  ...args: A
): R {
  for (const entry of table) {
    if (entry.matches !== undefined && !entry.matches(cx, ...args)) continue
    const result = attempt(cx, () => entry.map(cx, ...args))
    if (result !== NOT_MINE) return result
  }
  throw new OmaiError(
    'value',
    `factor graph: no mapping of the ${what} took the node (the last entry of its table must take every node)`,
  )
}
