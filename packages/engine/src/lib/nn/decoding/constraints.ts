/**
 * Constrained decoding: rules restricting which tokens may come next, applied by masking the forbidden logits to minus
 * infinity. A mask processor, constraints from a deterministic token automaton (grammar-constrained decoding) and from
 * a trie of allowed sequences (a fixed list of choices or labels).
 */

import type { LogitProcessor } from './processors.ts'

const NEG = Number.NEGATIVE_INFINITY

/** Which tokens may come next, given the tokens generated so far (prompt excluded). */
export type TokenConstraint = (generated: readonly number[]) => readonly boolean[] | ReadonlySet<number>

/**
 * A processor that sets the logits of tokens a constraint forbids to -infinity (constrained decoding).
 *
 * @param constraint - The allowed-token function of the generated tokens.
 * @param promptLength - How much of the prefix is the prompt (not passed to the constraint).
 * @returns The processor.
 * @example
 * ```ts
 * import { maskProcessor } from 'whitebox-engine/nn'
 * maskProcessor(() => new Set([1]), 0)([5, 1, 3], []) // => [-Infinity, 1, -Infinity]
 * ```
 */
export function maskProcessor(constraint: TokenConstraint, promptLength: number): LogitProcessor {
  return (logits, prefix) => {
    const allowed = constraint(prefix.slice(promptLength))
    const ok = (i: number): boolean =>
      allowed instanceof Set ? allowed.has(i) : (allowed as readonly boolean[])[i] === true
    return logits.map((z, i) => (ok(i) ? z : NEG))
  }
}

/** A deterministic automaton over token ids: decoding may only follow defined transitions. */
export interface TokenAutomaton<S> {
  readonly start: S
  /** The state after a token, or undefined when the token is not allowed. */
  next(state: S, token: number): S | undefined
  /** Whether generation may end in this state (allows the end-of-sequence token). */
  accepting?(state: S): boolean
}

/**
 * A constraint from an automaton (grammar-constrained decoding): the allowed tokens are those with a transition from
 * the state reached by the generated tokens, plus the end-of-sequence token in accepting states.
 *
 * @param automaton - The automaton.
 * @param vocabSize - The number of token ids to consider.
 * @param eosId - The end-of-sequence token, allowed in accepting states.
 * @returns The constraint.
 * @concept constrained-decoding
 * @example
 * ```ts
 * import { automatonConstraint } from 'whitebox-engine/nn'
 * // Alternate tokens 0 and 1, starting with 0.
 * const c = automatonConstraint({ start: 0, next: (s, t) => (t === s ? 1 - s : undefined) }, 2)
 * [...c([0, 1, 0])] // => [1]
 * ```
 */
export function automatonConstraint<S>(
  automaton: TokenAutomaton<S>,
  vocabSize: number,
  eosId?: number,
): TokenConstraint {
  return (generated) => {
    let s: S | undefined = automaton.start
    for (const t of generated) {
      if (s === undefined) break
      if (t === eosId) return new Set<number>()
      s = automaton.next(s, t)
    }
    const allowed = new Set<number>()
    if (s === undefined) return allowed
    for (let t = 0; t < vocabSize; t++) if (t !== eosId && automaton.next(s, t) !== undefined) allowed.add(t)
    if (eosId !== undefined && automaton.accepting?.(s) === true) allowed.add(eosId)
    return allowed
  }
}

interface TrieNode {
  readonly children: Map<number, TrieNode>
  end: boolean
}

/**
 * A constraint that only allows the given token sequences (a trie; choices from a fixed list, a JSON enum, a set of
 * labels): each step allows the children of the node reached, and the end-of-sequence token once a whole sequence is
 * complete.
 *
 * @param sequences - The allowed sequences of token ids.
 * @param vocabSize - The number of token ids.
 * @param eosId - The end-of-sequence token.
 * @returns The constraint.
 * @concept constrained-decoding
 * @example
 * ```ts
 * import { trieConstraint } from 'whitebox-engine/nn'
 * [...trieConstraint([[0, 1], [0, 2], [3]], 5, 4)([0])] // => [1, 2]
 * ```
 */
export function trieConstraint(
  sequences: readonly (readonly number[])[],
  vocabSize: number,
  eosId?: number,
): TokenConstraint {
  const root: TrieNode = { children: new Map(), end: false }
  for (const s of sequences) {
    let node = root
    for (const t of s) {
      let child = node.children.get(t)
      if (child === undefined) {
        child = { children: new Map(), end: false }
        node.children.set(t, child)
      }
      node = child
    }
    node.end = true
  }
  return automatonConstraint<TrieNode>(
    { start: root, next: (n, t) => n.children.get(t), accepting: (n) => n.end },
    vocabSize,
    eosId,
  )
}
