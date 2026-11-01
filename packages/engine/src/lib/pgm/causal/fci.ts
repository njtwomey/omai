/**
 * Fast Causal Inference (FCI; Spirtes, Meek and Richardson 1995, with Zhang's orientation rules), constraint-based
 * causal discovery that allows hidden confounders and selection bias. It learns a partial ancestral graph (PAG), whose
 * edge marks say which causal directions every compatible model shares.
 */

import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  type TracePolicy,
  type Tracer,
  tensor,
} from '../../trace/index.ts'
import { type DataRow, dataVariables } from '../learning/_data.ts'
import { type CiTest, type IndependenceOptions, independenceOracle } from '../learning/independence.ts'
import { pcAlgorithm } from '../learning/structure.ts'

/** An edge mark of a partial ancestral graph. */
export type Mark = 'tail' | 'arrow' | 'circle'

/** An edge of a PAG between a and b (a before b): the mark at a's end and the mark at b's end. */
export interface PagEdge {
  readonly a: string
  readonly b: string
  readonly markA: Mark
  readonly markB: Mark
}

/**
 * A partial ancestral graph (PAG): the Markov equivalence class of maximal ancestral graphs that FCI learns. A tail
 * means "is an ancestor of", an arrowhead "is not an ancestor of", and a circle that both occur in the class; a <-> b
 * signals a latent common cause.
 *
 * @concept causal-discovery
 */
export interface Pag {
  readonly kind: 'pag'
  readonly nodes: readonly string[]
  readonly edges: readonly PagEdge[]
}

/** Options of {@link fci}. */
export interface FciOptions extends IndependenceOptions {
  /** The conditional independence test (default `fisherZ`, as causal-learn). */
  readonly test?: CiTest
  /** The largest conditioning set in the adjacency search (default unlimited). */
  readonly maxConditioning?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** Mutable marks: mark(a, b) is the mark at b on the edge between a and b. */
class Marks {
  readonly m = new Map<string, Mark>()
  readonly nodes: readonly string[]
  constructor(nodes: readonly string[]) {
    this.nodes = nodes
  }
  key(a: string, b: string): string {
    return `${a}\u0000${b}`
  }
  adj(a: string, b: string): boolean {
    return this.m.has(this.key(a, b))
  }
  get(a: string, b: string): Mark | undefined {
    return this.m.get(this.key(a, b))
  }
  set(a: string, b: string, mark: Mark): boolean {
    if (this.get(a, b) === mark) return false
    this.m.set(this.key(a, b), mark)
    return true
  }
  add(a: string, b: string): void {
    this.m.set(this.key(a, b), 'circle')
    this.m.set(this.key(b, a), 'circle')
  }
  remove(a: string, b: string): void {
    this.m.delete(this.key(a, b))
    this.m.delete(this.key(b, a))
  }
  neighbours(a: string): string[] {
    return this.nodes.filter((b) => b !== a && this.adj(a, b))
  }
  reset(): void {
    for (const k of this.m.keys()) this.m.set(k, 'circle')
  }
}

const sepKey = (a: string, b: string): string => (a < b ? `${a}|${b}` : `${b}|${a}`)

/** Rule 0: orient unshielded triples a *-* c *-* b with c outside sepset(a, b) as colliders. */
function rule0(g: Marks, sepsets: ReadonlyMap<string, readonly string[]>): void {
  for (const c of g.nodes) {
    const ns = g.neighbours(c)
    for (let i = 0; i < ns.length; i++) {
      for (let j = i + 1; j < ns.length; j++) {
        const a = ns[i] as string
        const b = ns[j] as string
        if (g.adj(a, b)) continue
        if ((sepsets.get(sepKey(a, b)) ?? []).includes(c)) continue
        g.set(a, c, 'arrow')
        g.set(b, c, 'arrow')
      }
    }
  }
}

/** Possible-D-Sep(x): nodes reachable from x along paths whose every interior node is a collider or in a triangle. */
function possibleDsep(g: Marks, x: string, y: string): string[] {
  const out = new Set<string>()
  const seen = new Set<string>()
  const queue: [string, string][] = []
  for (const n of g.neighbours(x)) {
    out.add(n)
    queue.push([x, n])
    seen.add(`${x}>${n}`)
  }
  while (queue.length > 0) {
    const [a, b] = queue.shift() as [string, string]
    for (const c of g.neighbours(b)) {
      if (c === a || c === x) continue
      const collider = g.get(a, b) === 'arrow' && g.get(c, b) === 'arrow'
      if (!collider && !g.adj(a, c)) continue
      const key = `${b}>${c}`
      if (seen.has(key)) continue
      seen.add(key)
      out.add(c)
      queue.push([b, c])
    }
  }
  out.delete(y)
  out.delete(x)
  return [...out].sort()
}

function* subsetsBySize(items: readonly string[], minSize: number): Generator<string[]> {
  for (let k = minSize; k <= items.length; k++) {
    const rec = function* (start: number, acc: string[]): Generator<string[]> {
      if (acc.length === k) {
        yield [...acc]
        return
      }
      for (let i = start; i < items.length; i++) yield* rec(i + 1, [...acc, items[i] as string])
    }
    yield* rec(0, [])
  }
}

/** Whether a -> b (tail at a, arrowhead at b). */
const directed = (g: Marks, a: string, b: string): boolean => g.get(a, b) === 'arrow' && g.get(b, a) === 'tail'

/** Uncovered paths from a to `target` whose every edge satisfies `ok(u, v)`, with at least `minNodes` nodes. */
function uncoveredPaths(
  g: Marks,
  a: string,
  target: string,
  ok: (u: string, v: string) => boolean,
  minNodes: number,
  limit = 2000,
): string[][] {
  const out: string[][] = []
  const walk = (path: string[]): void => {
    if (out.length >= limit) return
    const last = path[path.length - 1] as string
    for (const v of g.neighbours(last)) {
      if (path.includes(v) || !ok(last, v)) continue
      const prev = path[path.length - 2]
      if (prev !== undefined && g.adj(prev, v)) continue // uncovered
      const next = [...path, v]
      if (v === target) {
        if (next.length >= minNodes) out.push(next)
        continue
      }
      walk(next)
    }
  }
  walk([a])
  return out
}

/** A potentially directed edge u -> v on a path: no arrowhead at u and no tail at v. */
const potentiallyDirected =
  (g: Marks) =>
  (u: string, v: string): boolean =>
    g.get(v, u) !== 'arrow' && g.get(u, v) !== 'tail'

type FciRule = (g: Marks, sepsets: ReadonlyMap<string, readonly string[]>) => boolean

/** The number logged for each sweep of {@link fciRules}: R6 and R7 are one sweep, logged as 6. */
const FCI_RULE_NUMBERS: readonly number[] = [1, 2, 3, 4, 5, 6, 8, 9, 10]

/** Zhang's orientation rules R1 to R10, each one sweep over the graph that reports whether it changed a mark. */
const fciRules: readonly FciRule[] = [
  // R1: a *-> b o-* c, a and c not adjacent: b -> c.
  (g) => {
    let changed = false
    for (const b of g.nodes)
      for (const a of g.neighbours(b))
        if (g.get(a, b) === 'arrow')
          for (const c of g.neighbours(b))
            if (c !== a && !g.adj(a, c) && g.get(c, b) === 'circle') {
              changed = g.set(c, b, 'tail') || changed
              changed = g.set(b, c, 'arrow') || changed
            }
    return changed
  },
  // R2: a -> b *-> c or a *-> b -> c, and a *-o c: a *-> c.
  (g) => {
    let changed = false
    for (const a of g.nodes)
      for (const c of g.neighbours(a))
        if (g.get(a, c) === 'circle')
          for (const b of g.neighbours(a))
            if (b !== c && g.adj(b, c)) {
              const first = directed(g, a, b) && g.get(b, c) === 'arrow'
              const second = g.get(a, b) === 'arrow' && directed(g, b, c)
              if (first || second) changed = g.set(a, c, 'arrow') || changed
            }
    return changed
  },
  // R3: a *-> b <-* c, a *-o d o-* c, a and c not adjacent, d *-o b: d *-> b.
  (g) => {
    let changed = false
    for (const b of g.nodes) {
      const into = g.neighbours(b).filter((v) => g.get(v, b) === 'arrow')
      for (let i = 0; i < into.length; i++)
        for (let j = i + 1; j < into.length; j++) {
          const a = into[i] as string
          const c = into[j] as string
          if (g.adj(a, c)) continue
          for (const d of g.neighbours(b))
            if (
              d !== a &&
              d !== c &&
              g.adj(a, d) &&
              g.adj(c, d) &&
              g.get(a, d) === 'circle' &&
              g.get(c, d) === 'circle' &&
              g.get(d, b) === 'circle'
            )
              changed = g.set(d, b, 'arrow') || changed
        }
    }
    return changed
  },
  // R4: discriminating paths <theta, ..., a, b, c> for b with b o-* c.
  (g, sepsets) => {
    let changed = false
    for (const b of g.nodes)
      for (const c of g.neighbours(b)) {
        if (g.get(c, b) !== 'circle') continue
        for (const a of g.neighbours(b)) {
          if (a === c || !(g.get(b, a) === 'arrow' && directed(g, a, c))) continue
          const theta = discriminate(g, a, b, c)
          if (theta === undefined) continue
          if ((sepsets.get(sepKey(theta, c)) ?? []).includes(b)) {
            changed = g.set(b, c, 'arrow') || changed
            changed = g.set(c, b, 'tail') || changed
          } else {
            changed = g.set(a, b, 'arrow') || changed
            changed = g.set(b, a, 'arrow') || changed
            changed = g.set(c, b, 'arrow') || changed
            changed = g.set(b, c, 'arrow') || changed
          }
        }
      }
    return changed
  },
  // R5: a o-o b with an uncovered circle path <a, c, ..., d, b>, a and d, b and c not adjacent: all tails.
  (g) => {
    let changed = false
    for (const a of g.nodes)
      for (const b of g.neighbours(a)) {
        if (!(a < b && g.get(a, b) === 'circle' && g.get(b, a) === 'circle')) continue
        const circlePath = (u: string, v: string): boolean => g.get(u, v) === 'circle' && g.get(v, u) === 'circle'
        for (const p of uncoveredPaths(g, a, b, circlePath, 4)) {
          const c = p[1] as string
          const d = p[p.length - 2] as string
          if (g.adj(a, d) || g.adj(b, c)) continue
          changed = g.set(a, b, 'tail') || changed
          changed = g.set(b, a, 'tail') || changed
          for (let k = 0; k + 1 < p.length; k++) {
            changed = g.set(p[k] as string, p[k + 1] as string, 'tail') || changed
            changed = g.set(p[k + 1] as string, p[k] as string, 'tail') || changed
          }
          break
        }
      }
    return changed
  },
  // R6: a - b and b o-* c: b -* c.  R7: a -o b, b o-* c, a and c not adjacent: b -* c.
  (g) => {
    let changed = false
    for (const b of g.nodes)
      for (const a of g.neighbours(b))
        for (const c of g.neighbours(b)) {
          if (c === a || g.get(c, b) !== 'circle') continue
          if (g.get(a, b) === 'tail' && g.get(b, a) === 'tail') changed = g.set(c, b, 'tail') || changed
          else if (g.get(b, a) === 'tail' && g.get(a, b) === 'circle' && !g.adj(a, c))
            changed = g.set(c, b, 'tail') || changed
        }
    return changed
  },
  // R8: a -> b -> c or a -o b -> c, and a o-> c: a -> c.
  (g) => {
    let changed = false
    for (const a of g.nodes)
      for (const c of g.neighbours(a)) {
        if (!(g.get(a, c) === 'arrow' && g.get(c, a) === 'circle')) continue
        for (const b of g.neighbours(a))
          if (
            b !== c &&
            g.adj(b, c) &&
            directed(g, b, c) &&
            (directed(g, a, b) || (g.get(b, a) === 'tail' && g.get(a, b) === 'circle'))
          )
            changed = g.set(c, a, 'tail') || changed
      }
    return changed
  },
  // R9: a o-> c with an uncovered potentially directed path <a, b, ..., c>, b and c not adjacent: a -> c.
  (g) => {
    let changed = false
    for (const a of g.nodes)
      for (const c of g.neighbours(a)) {
        if (!(g.get(a, c) === 'arrow' && g.get(c, a) === 'circle')) continue
        const paths = uncoveredPaths(g, a, c, potentiallyDirected(g), 4)
        if (paths.some((p) => !g.adj(p[1] as string, c))) changed = g.set(c, a, 'tail') || changed
      }
    return changed
  },
  // R10: a o-> c, b -> c <- t, uncovered p.d. paths from a to b and to t whose second nodes differ and are not
  // adjacent: a -> c.
  (g) => {
    let changed = false
    for (const a of g.nodes)
      for (const c of g.neighbours(a)) {
        if (!(g.get(a, c) === 'arrow' && g.get(c, a) === 'circle')) continue
        const parents = g.neighbours(c).filter((v) => v !== a && directed(g, v, c))
        let done = false
        for (let i = 0; i < parents.length && !done; i++)
          for (let j = i + 1; j < parents.length && !done; j++) {
            const b = parents[i] as string
            const t = parents[j] as string
            const firsts = (target: string): string[] => [
              ...new Set(uncoveredPaths(g, a, target, potentiallyDirected(g), 2).map((p) => p[1] as string)),
            ]
            const mus = firsts(b)
            const omegas = firsts(t)
            if (mus.some((mu) => omegas.some((om) => mu !== om && !g.adj(mu, om)))) {
              changed = g.set(c, a, 'tail') || changed
              done = true
            }
          }
      }
    return changed
  },
]

/** The start theta of a discriminating path <theta, ..., a, b, c> for b, if any. */
function discriminate(g: Marks, a: string, b: string, c: string): string | undefined {
  // Walk back from a through colliders that are parents of c until a node not adjacent to c is reached.
  const queue: { cur: string; path: string[] }[] = [{ cur: a, path: [c, b, a] }]
  while (queue.length > 0) {
    const { cur, path } = queue.shift() as { cur: string; path: string[] }
    for (const v of g.neighbours(cur)) {
      if (path.includes(v) || g.get(v, cur) !== 'arrow') continue
      if (!g.adj(v, c)) return v
      if (directed(g, v, c) && g.get(cur, v) === 'arrow') queue.push({ cur: v, path: [...path, v] })
    }
  }
  return undefined
}

/** The paths FCI logs, as a type. */
export type FciPath = 'fci/phase' | 'fci/rule' | 'fci/oriented' | 'fci/removed' | 'fci/sepset' | 'fci/circles'

/**
 * What FCI logs, by path: the phase of the step (0 colliders from the skeleton, 1 Possible-D-Sep pruning, 2 colliders
 * again then, 3 Zhang's rules), the rule swept (0 for colliders, 1 to 10 with R6 and R7 swept together as 6, -1 while pruning), the number of edge marks the
 * step changed, the edge a pruning step removed and its separating set (as node indices, in the order of the data's
 * variables) and the circle marks left.
 */
export const fciContract: Contract<unknown, FciPath> = defineContract({
  'fci/phase': scalar({ description: '0 colliders, 1 Possible-D-Sep pruning, 2 colliders again, 3 orientation rules' }),
  'fci/rule': scalar({
    description:
      'the rule swept: 0 for the collider rule, 1 to 10 for Zhang (R6 and R7 together as 6), -1 while pruning',
  }),
  'fci/oriented': scalar({ description: 'edge marks the step changed' }),
  'fci/removed': tensor({
    role: 'state',
    shape: [2],
    description: 'the edge a pruning step removed, as node indices (empty: none)',
  }),
  'fci/sepset': tensor({ role: 'state', shape: ['sepset'], description: 'its separating set, as node indices' }),
  'fci/circles': scalar({ description: 'circle marks left in the graph' }),
})

/** Options of {@link fciAlgorithm}. */
export interface FciAlgorithmOptions extends FciOptions {
  /** A precomputed adjacency search (the PC skeleton and its separating sets); default: run it. */
  readonly skeleton?: {
    readonly adjacency: ReadonlyMap<string, Iterable<string>>
    readonly sepsets: ReadonlyMap<string, readonly string[]>
  }
}

/** The state of {@link fciAlgorithm}. */
export interface FciState {
  /** Where the run is: 0 colliders, 1 pruning, 3 rules, 4 finished (2 happens inside the step that ends pruning). */
  readonly phase: 0 | 1 | 3 | 4
  /** The marks: `[a + NUL + b, mark]`, the mark at b of the edge between a and b. */
  readonly marks: readonly (readonly [string, Mark])[]
  /** Separating sets by sorted pair key. */
  readonly sepsets: readonly (readonly [string, readonly string[]])[]
  /** The position among the node pairs in the pruning phase. */
  readonly cursor: number
  /** The next sweep, 0 to 8 (R1 to R5, R6 with R7, R8, R9, R10). */
  readonly rule: number
  /** Whether a rule changed a mark in the current cycle. */
  readonly changed: boolean
  /** Sweeps and steps made. */
  readonly steps: number
}

const marksToState = (g: Marks): [string, Mark][] => [...g.m.entries()]
const marksFromState = (names: readonly string[], marks: readonly (readonly [string, Mark])[]): Marks => {
  const g = new Marks(names)
  for (const [k, v] of marks) g.m.set(k, v)
  return g
}

/**
 * Fast Causal Inference as a steppable algorithm: the PC adjacency search gives a skeleton and separating sets; then one
 * step orients the unshielded colliders, each following step tests one remaining edge against subsets of its
 * Possible-D-Sep (removing it and recording the separating set when they separate), one step re-orients the colliders
 * from scratch, and then each step sweeps one of Zhang's rules R1 to R10 (R6 and R7 together) until a whole cycle changes nothing.
 *
 * @param data - The rows (continuous for Fisher z; discrete for chi-square or G tests).
 * @param options - Test, significance, largest conditioning set and optionally a precomputed `skeleton`.
 * @returns The algorithm; its final state's marks are the PAG (see {@link fci}).
 * @concept causal-discovery
 * @example
 * ```ts
 * import { fciAlgorithm } from 'mdml-engine/pgm/causal'
 * fciAlgorithm([{ a: 0, b: 1 }, { a: 1, b: 0 }, { a: 2, b: 3 }, { a: 3, b: 2 }]).name // => 'pgm/fci'
 * ```
 */
export function fciAlgorithm(
  data: readonly DataRow[],
  options: FciAlgorithmOptions = {},
): Algorithm<FciState, FciPath> {
  const names = dataVariables(data)
  const test = options.test ?? 'fisherZ'
  const maxCond = options.maxConditioning ?? names.length
  const index = new Map(names.map((v, i) => [v, i]))
  const idx = (vs: readonly string[]): Int32Array => Int32Array.from(vs, (v) => index.get(v) ?? -1)
  const pairs: [string, string][] = []
  for (const x of names) for (const y of names) if (x < y) pairs.push([x, y])
  const oracle = independenceOracle(data, test, options)
  const circles = (g: Marks): number => {
    let c = 0
    for (const v of g.m.values()) if (v === 'circle') c++
    return c
  }
  return {
    name: 'pgm/fci',
    config: { variables: names.length, test: typeof test === 'string' ? test : 'custom', maxConditioning: maxCond },
    init: () => {
      const skeleton =
        options.skeleton ??
        runAlgorithm(pcAlgorithm(data, { ...options, test, maxConditioning: maxCond, sepsetRule: 'first' }), {
          maxSteps: maxCond + 2,
        }).state
      const g = new Marks(names)
      for (const [v, ns] of skeleton.adjacency) for (const w of ns) if (v < w) g.add(v, w)
      return {
        phase: 0,
        marks: marksToState(g),
        sepsets: [...skeleton.sepsets].map(([k, v]) => [k, [...v]] as const),
        cursor: 0,
        rule: 0,
        changed: false,
        steps: 0,
      }
    },
    step: (s, ctx) => {
      if (s.phase === 4) return s
      const g = marksFromState(names, s.marks)
      const before = new Map(g.m)
      const sepsets = new Map(s.sepsets.map(([k, v]) => [k, v] as const))
      let phase: FciState['phase'] = s.phase
      let cursor = s.cursor
      let rule = s.rule
      let changed = s.changed
      let ruleNumber = -1
      let removed: [string, string] | undefined
      let separating: readonly string[] = []
      if (phase === 0) {
        rule0(g, sepsets)
        ruleNumber = 0
        phase = 1
      } else if (phase === 1) {
        // Possible-D-Sep pruning: the next remaining edge, or, once there is none, the re-orientation of the colliders.
        while (cursor < pairs.length && !g.adj(...(pairs[cursor] as [string, string]))) cursor++
        const pair = pairs[cursor]
        if (pair === undefined) {
          g.reset()
          rule0(g, sepsets)
          ruleNumber = 0
          phase = 3
          rule = 0
          changed = false
        } else {
          cursor++
          const [x, y] = pair
          let gone = false
          for (const [from, to] of [
            [x, y],
            [y, x],
          ] as const) {
            if (gone) break
            for (const sub of subsetsBySize(possibleDsep(g, from, to), 1)) {
              if (oracle(x, y, sub)) {
                g.remove(x, y)
                sepsets.set(sepKey(x, y), sub)
                removed = [x, y]
                separating = sub
                gone = true
                break
              }
            }
          }
        }
      } else {
        ruleNumber = FCI_RULE_NUMBERS[rule] ?? -1
        changed = (fciRules[rule] as FciRule)(g, sepsets) || changed
        rule++
        if (rule === fciRules.length) {
          if (!changed) phase = 4
          rule = 0
          changed = false
        }
      }
      let marksChanged = 0
      for (const [k, v] of g.m) if (before.get(k) !== v) marksChanged++
      for (const k of before.keys()) if (!g.m.has(k)) marksChanged++
      if (ctx.wants('fci/phase')) ctx.log('fci/phase', s.phase)
      if (ctx.wants('fci/rule')) ctx.log('fci/rule', ruleNumber)
      if (ctx.wants('fci/oriented')) ctx.log('fci/oriented', marksChanged)
      if (ctx.wants('fci/removed')) ctx.log('fci/removed', removed === undefined ? new Int32Array(0) : idx(removed))
      if (ctx.wants('fci/sepset')) ctx.log('fci/sepset', idx(separating))
      if (ctx.wants('fci/circles')) ctx.log('fci/circles', circles(g))
      return {
        phase,
        marks: marksToState(g),
        sepsets: [...sepsets].map(([k, v]) => [k, [...v]] as const),
        cursor,
        rule,
        changed,
        steps: s.steps + 1,
      }
    },
    done: (s) => (s.phase === 4 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FciState,
  }
}

/** The PAG and the separating sets a finished {@link fciAlgorithm} state holds. */
function pagOf(
  names: readonly string[],
  s: FciState,
): { readonly pag: Pag; readonly separatingSets: Readonly<Record<string, readonly string[]>> } {
  const g = marksFromState(names, s.marks)
  const edges: PagEdge[] = []
  for (const a of names)
    for (const b of names)
      if (a < b && g.adj(a, b)) edges.push({ a, b, markA: g.get(b, a) ?? 'circle', markB: g.get(a, b) ?? 'circle' })
  return { pag: { kind: 'pag', nodes: names, edges }, separatingSets: Object.fromEntries(s.sepsets) }
}

/**
 * Fast Causal Inference (Spirtes, Meek and Richardson 1995; Zhang 2008): causal discovery that allows latent
 * confounders and selection. The PC adjacency search gives a skeleton and separating sets; unshielded colliders are
 * oriented; edges are pruned by testing subsets of Possible-D-Sep; the colliders are re-oriented and Zhang's complete
 * rules R1-R10 (R4 on discriminating paths with the separating sets) are applied until nothing changes (causal-learn's
 * `fci`). This is {@link fciAlgorithm} run to the end.
 *
 * @param data - The rows (continuous for Fisher z; discrete for chi-square or G tests).
 * @param options - Test, significance, largest conditioning set and tracer (for the adjacency search).
 * @returns The PAG and the separating sets of removed edges.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { fci } from 'mdml-engine/pgm/causal'
 * import { key, normal } from 'mdml-engine/random'
 * // a -> c <- b: the collider is oriented, the other marks stay circles.
 * const z = normal(key(3), [500, 3]).data
 * const rows = Array.from({ length: 500 }, (_, i) => { const a = z[3 * i] ?? 0, b = z[3 * i + 1] ?? 0; return { a, b, c: a + b + 0.5 * (z[3 * i + 2] ?? 0) } })
 * fci(rows).pag.edges.map((e) => `${e.a} ${e.markA}-${e.markB} ${e.b}`) // => ['a circle-arrow c', 'b circle-arrow c']
 * ```
 */
export function fci(
  data: readonly DataRow[],
  options: FciOptions = {},
): { readonly pag: Pag; readonly separatingSets: Readonly<Record<string, readonly string[]>> } {
  const names = dataVariables(data)
  const test = options.test ?? 'fisherZ'
  const maxCond = options.maxConditioning ?? names.length
  const skeleton = runAlgorithm(
    pcAlgorithm(data, { ...options, test, maxConditioning: maxCond, sepsetRule: 'first' }),
    {
      maxSteps: maxCond + 2,
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
    },
  ).state
  const { state } = runAlgorithm(fciAlgorithm(data, { ...options, skeleton }), {
    maxSteps: Number.MAX_SAFE_INTEGER,
  })
  return pagOf(names, state)
}
