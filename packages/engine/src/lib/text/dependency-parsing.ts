import { WhiteboxError } from '../base/index.ts'

/**
 * A dependency tree over tokens 1..n with an artificial root 0: `heads[i]` is the head of token i (`heads[0]` is -1).
 */
export type Heads = readonly number[]

/**
 * Eisner's algorithm (Eisner 1996): the highest-scoring projective dependency tree for arc scores
 * `scores[h][d]` (head h, dependent d; index 0 is the root), by dynamic programming over complete and incomplete spans
 * in O(n^3). Ties prefer the earlier split.
 *
 * @param scores - The (n + 1) x (n + 1) arc scores.
 * @param options - Allow only one child of the root (default false).
 * @returns The heads and the tree's total score.
 * @throws WhiteboxError with code `shape` for a non-square score matrix.
 * @concept dependency-parsing
 * @example
 * ```ts
 * import { eisner } from 'whitebox-engine/text'
 * const s = [[0, 9, 2, 1], [0, 0, 8, 1], [0, 3, 0, 7], [0, 1, 2, 0]]
 * eisner(s) // => { heads: [-1, 0, 1, 2], score: 24 }
 * ```
 */
export function eisner(
  scores: readonly (readonly number[])[],
  options: { readonly singleRoot?: boolean } = {},
): { readonly heads: number[]; readonly score: number } {
  const n1 = scores.length
  if (scores.some((r) => r.length !== n1)) throw new WhiteboxError('shape', 'eisner: scores must be square')
  const n = n1 - 1
  if (n <= 0) return { heads: [-1], score: 0 }
  const NEG = Number.NEGATIVE_INFINITY
  const sc = (h: number, d: number): number => scores[h]?.[d] ?? NEG
  // C[s][t][dir], I[s][t][dir]: dir 0 = head on the right (t), dir 1 = head on the left (s).
  const size = n1 * n1 * 2
  const C = new Float64Array(size).fill(NEG)
  const I = new Float64Array(size).fill(NEG)
  const Cb = new Int32Array(size).fill(-1)
  const Ib = new Int32Array(size).fill(-1)
  const k3 = (s: number, t: number, d: number): number => (s * n1 + t) * 2 + d
  for (let s = 0; s < n1; s++) {
    C[k3(s, s, 0)] = 0
    C[k3(s, s, 1)] = 0
  }
  for (let len = 1; len < n1; len++) {
    for (let s = 0; s + len < n1; s++) {
      const t = s + len
      // Incomplete spans: an arc between s and t over two facing complete spans.
      let best = NEG
      let arg = -1
      for (let r = s; r < t; r++) {
        const v = (C[k3(s, r, 1)] ?? NEG) + (C[k3(r + 1, t, 0)] ?? NEG)
        if (v > best) {
          best = v
          arg = r
        }
      }
      // Arc t -> s (head right); the root never has a head.
      if (s > 0) {
        I[k3(s, t, 0)] = best + sc(t, s)
        Ib[k3(s, t, 0)] = arg
      }
      if (options.singleRoot === true && s === 0) {
        // The root's only child: nothing else may hang from the root inside the arc.
        I[k3(0, t, 1)] = (C[k3(0, 0, 1)] ?? NEG) + (C[k3(1, t, 0)] ?? NEG) + sc(0, t)
        Ib[k3(0, t, 1)] = 0
      } else {
        I[k3(s, t, 1)] = best + sc(s, t)
        Ib[k3(s, t, 1)] = arg
      }
      // Complete spans.
      best = NEG
      arg = -1
      for (let r = s; r < t; r++) {
        const v = (C[k3(s, r, 0)] ?? NEG) + (I[k3(r, t, 0)] ?? NEG)
        if (v > best) {
          best = v
          arg = r
        }
      }
      C[k3(s, t, 0)] = best
      Cb[k3(s, t, 0)] = arg
      best = NEG
      arg = -1
      for (let r = s + 1; r <= t; r++) {
        const v = (I[k3(s, r, 1)] ?? NEG) + (C[k3(r, t, 1)] ?? NEG)
        if (v > best) {
          best = v
          arg = r
        }
      }
      C[k3(s, t, 1)] = best
      Cb[k3(s, t, 1)] = arg
    }
  }
  const heads = new Array<number>(n1).fill(-1)
  const complete = (s: number, t: number, d: number): void => {
    if (s === t) return
    const r = Cb[k3(s, t, d)] ?? -1
    if (d === 0) {
      complete(s, r, 0)
      incomplete(r, t, 0)
    } else {
      incomplete(s, r, 1)
      complete(r, t, 1)
    }
  }
  const incomplete = (s: number, t: number, d: number): void => {
    if (d === 0) heads[s] = t
    else heads[t] = s
    const r = Ib[k3(s, t, d)] ?? -1
    complete(s, r, 1)
    complete(r + 1, t, 0)
  }
  complete(0, n, 1)
  return { heads, score: C[k3(0, n, 1)] ?? NEG }
}

/**
 * Whether a dependency tree is projective: no two arcs cross (arcs from the root included).
 *
 * @param heads - The heads (`heads[0]` = -1).
 * @returns True for a projective tree.
 * @concept dependency-parsing
 * @example
 * ```ts
 * import { isProjective } from 'whitebox-engine/text'
 * [isProjective([-1, 2, 0, 2]), isProjective([-1, 3, 4, 0, 3])] // => [true, false]
 * ```
 */
export function isProjective(heads: Heads): boolean {
  const arcs: [number, number][] = []
  for (let d = 1; d < heads.length; d++) {
    const h = heads[d] ?? 0
    arcs.push([Math.min(h, d), Math.max(h, d)])
  }
  for (const [a, b] of arcs) for (const [c, d] of arcs) if (a < c && c < b && b < d) return false
  return true
}

/** An arc-standard transition. */
export type ArcStandardTransition = 'shift' | 'left' | 'right'

/**
 * Apply arc-standard transitions (Nivre 2004) to tokens 1..n, starting with the root on the stack: `shift` moves the
 * next token to the stack, `left` makes the top the head of the item below it (which is popped), `right` makes the
 * item below the top the head of the top (which is popped).
 *
 * @param n - The number of tokens.
 * @param transitions - The transitions.
 * @returns The heads (unattached tokens keep -1).
 * @throws WhiteboxError with code `value` for an illegal transition.
 * @concept dependency-parsing
 * @example
 * ```ts
 * import { applyArcStandard } from 'whitebox-engine/text'
 * applyArcStandard(3, ['shift', 'shift', 'left', 'shift', 'right', 'right']) // => [-1, 2, 0, 2]
 * ```
 */
export function applyArcStandard(n: number, transitions: readonly ArcStandardTransition[]): number[] {
  const heads = new Array<number>(n + 1).fill(-1)
  const stack = [0]
  let next = 1
  for (const t of transitions) {
    if (t === 'shift') {
      if (next > n) throw new WhiteboxError('value', 'arc-standard: shift with an empty buffer')
      stack.push(next++)
      continue
    }
    if (stack.length < 2) throw new WhiteboxError('value', `arc-standard: ${t} needs two items on the stack`)
    const top = stack.pop() as number
    const below = stack.pop() as number
    if (t === 'left') {
      if (below === 0) throw new WhiteboxError('value', 'arc-standard: the root cannot be a dependent')
      heads[below] = top
      stack.push(top)
    } else {
      heads[top] = below
      stack.push(below)
    }
  }
  return heads
}

/**
 * The static oracle of the arc-standard system: the transition sequence that builds a given projective tree (left
 * when the item below the top is the top's dependent, right when the top is the dependent of the item below and has
 * all its own dependents, otherwise shift).
 *
 * @param heads - The tree (projective).
 * @returns The transitions.
 * @throws WhiteboxError with code `value` for a non-projective tree.
 * @concept dependency-parsing
 * @example
 * ```ts
 * import { arcStandardOracle } from 'whitebox-engine/text'
 * arcStandardOracle([-1, 2, 0, 2]) // => ['shift', 'shift', 'left', 'shift', 'right', 'right']
 * ```
 */
export function arcStandardOracle(heads: Heads): ArcStandardTransition[] {
  if (!isProjective(heads)) throw new WhiteboxError('value', 'arcStandardOracle: the tree is not projective')
  const n = heads.length - 1
  const remaining = new Array<number>(n + 1).fill(0)
  for (let d = 1; d <= n; d++) remaining[heads[d] ?? 0] = (remaining[heads[d] ?? 0] ?? 0) + 1
  const stack = [0]
  let next = 1
  const out: ArcStandardTransition[] = []
  for (let guard = 0; guard < 2 * n + 1; guard++) {
    if (stack.length >= 2) {
      const top = stack[stack.length - 1] ?? 0
      const below = stack[stack.length - 2] ?? 0
      if (below !== 0 && heads[below] === top) {
        out.push('left')
        stack.splice(stack.length - 2, 1)
        remaining[top] = (remaining[top] ?? 0) - 1
        continue
      }
      if (heads[top] === below && remaining[top] === 0) {
        out.push('right')
        stack.pop()
        remaining[below] = (remaining[below] ?? 0) - 1
        continue
      }
    }
    if (next > n) break
    out.push('shift')
    stack.push(next++)
  }
  return out
}

/**
 * Greedy transition-based parsing (Nivre 2004; Chen and Manning 2014's decoder): at each configuration, apply the
 * legal arc-standard transition with the highest score from a scoring function of the stack and buffer.
 *
 * @param n - The number of tokens.
 * @param score - The score of a transition in a configuration (stack and next buffer index).
 * @returns The heads and the transitions taken.
 * @concept dependency-parsing
 * @example
 * ```ts
 * import { greedyArcStandard } from 'whitebox-engine/text'
 * const gold = [-1, 2, 0, 2]
 * const score = (t: string, stack: readonly number[]) => {
 *   const top = stack[stack.length - 1] ?? 0, below = stack[stack.length - 2] ?? -1
 *   return t === 'left' ? Number(gold[below] === top) : t === 'right' ? Number(gold[top] === below) : 0.5
 * }
 * greedyArcStandard(3, score).heads // => [-1, 2, 0, 2]
 * ```
 */
export function greedyArcStandard(
  n: number,
  score: (transition: ArcStandardTransition, stack: readonly number[], next: number) => number,
): { readonly heads: number[]; readonly transitions: ArcStandardTransition[] } {
  const stack = [0]
  let next = 1
  const transitions: ArcStandardTransition[] = []
  for (let step = 0; step < 2 * n; step++) {
    const legal: ArcStandardTransition[] = []
    if (next <= n) legal.push('shift')
    if (stack.length >= 3) legal.push('left')
    if (stack.length >= 2 && (stack.length > 2 || next > n)) legal.push('right')
    if (legal.length === 0) break
    let best = legal[0] as ArcStandardTransition
    let bestScore = Number.NEGATIVE_INFINITY
    for (const t of legal) {
      const v = score(t, stack, next)
      if (v > bestScore) {
        bestScore = v
        best = t
      }
    }
    transitions.push(best)
    if (best === 'shift') stack.push(next++)
    else if (best === 'left') stack.splice(stack.length - 2, 1)
    else stack.pop()
  }
  return { heads: applyArcStandard(n, transitions), transitions }
}
