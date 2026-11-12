// The warm starts of `ep-grid`: a node keeps the final messages of its EP run when it settled, and the next node starts from those
// of the settled node nearest to it on the lattice. What is kept and read back is checked to be finite: a message that is not (an
// infinite entry, or the `null` that JSON makes of one in a serialised state) is never an initial message, so the node starts cold.

import type { CompiledGraph, InitialMessage } from '../../message-passing/index.ts'

/** The part of a node the warm start reads. */
export interface WarmNode {
  readonly index: readonly number[]
  readonly converged: boolean
  readonly logEvidence: number
  readonly messages?: readonly (readonly number[])[]
}

/** Whether every entry of every message is a finite number (a `null` from a serialised infinite entry is not). */
function finiteMessages(
  messages: readonly (readonly number[])[] | undefined,
): messages is readonly (readonly number[])[] {
  return (
    messages?.every((m) => Array.isArray(m) && m.every((v) => typeof v === 'number' && Number.isFinite(v))) === true
  )
}

/**
 * What a node keeps of its run for the warm start of its neighbours: the final messages, only when the run settled with a finite
 * evidence and every message is finite (otherwise nothing: such a node is never a start, and the state stays small).
 *
 * @param converged - Whether the EP run converged.
 * @param logEvidence - Its evidence (possibly not finite).
 * @param messages - The message on every edge of its last state.
 * @returns `{ messages }` to spread into the node, or nothing.
 */
export function settledMessages(
  converged: boolean,
  logEvidence: number | undefined,
  messages: readonly (readonly number[])[],
): { readonly messages?: readonly (readonly number[])[] } {
  return converged && logEvidence !== undefined && Number.isFinite(logEvidence) && finiteMessages(messages)
    ? { messages }
    : {}
}

/**
 * The start messages of a node from the final messages of another run of the same factor graph structure. None when the messages
 * are missing, differ in structure (the edge count or a message's length), or have an entry that is not a finite number: the node
 * then starts cold.
 *
 * @param g - The graph of the node about to run.
 * @param messages - The final messages of the neighbour's run, by edge.
 * @returns The initial messages, or an empty list.
 */
export function warmStart(g: CompiledGraph, messages: readonly (readonly number[])[] | undefined): InitialMessage[] {
  if (!finiteMessages(messages) || messages.length !== g.edges.length) return []
  const out: InitialMessage[] = []
  for (const [factor, f] of g.factors.entries())
    for (const [port, p] of f.ports.entries()) {
      const eta = p.edge >= 0 ? messages[p.edge] : undefined
      const edge = p.edge >= 0 ? g.edges[p.edge] : undefined
      const flat = edge === undefined ? undefined : g.variables[edge.variable]?.family.flat
      if (eta === undefined || flat === undefined || eta.length !== flat.length) return []
      out.push({ factor, port, eta: [...eta] })
    }
  return out
}

/**
 * The settled node nearest to `index` (in steps of the lattice, summed over the axes) whose messages are usable, or none.
 *
 * @param nodes - The nodes run so far.
 * @param index - The lattice position about to run.
 * @returns The node, or undefined.
 */
export function nearestSettled<N extends WarmNode>(nodes: readonly N[], index: readonly number[]): N | undefined {
  let best: N | undefined
  let bestDistance = Number.POSITIVE_INFINITY
  for (const n of nodes) {
    if (!n.converged || !Number.isFinite(n.logEvidence) || !finiteMessages(n.messages)) continue
    const dist = n.index.reduce((s, v, k) => s + Math.abs(v - (index[k] ?? 0)), 0)
    if (dist < bestDistance) {
      best = n
      bestDistance = dist
    }
  }
  return best
}
