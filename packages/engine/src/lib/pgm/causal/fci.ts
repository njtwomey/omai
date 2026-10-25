/**
 * Fast Causal Inference (FCI; Spirtes, Meek and Richardson 1995, with Zhang's orientation rules), constraint-based
 * causal discovery that allows hidden confounders and selection bias. It learns a partial ancestral graph (PAG), whose
 * edge marks say which causal directions every compatible model shares.
 */

import { runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
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

function applyRules(g: Marks, sepsets: ReadonlyMap<string, readonly string[]>): void {
  const N = g.nodes
  let changed = true
  while (changed) {
    changed = false
    // R1: a *-> b o-* c, a and c not adjacent: b -> c.
    for (const b of N)
      for (const a of g.neighbours(b))
        if (g.get(a, b) === 'arrow')
          for (const c of g.neighbours(b))
            if (c !== a && !g.adj(a, c) && g.get(c, b) === 'circle') {
              changed = g.set(c, b, 'tail') || changed
              changed = g.set(b, c, 'arrow') || changed
            }
    // R2: a -> b *-> c or a *-> b -> c, and a *-o c: a *-> c.
    for (const a of N)
      for (const c of g.neighbours(a))
        if (g.get(a, c) === 'circle')
          for (const b of g.neighbours(a))
            if (b !== c && g.adj(b, c)) {
              const first = directed(g, a, b) && g.get(b, c) === 'arrow'
              const second = g.get(a, b) === 'arrow' && directed(g, b, c)
              if (first || second) changed = g.set(a, c, 'arrow') || changed
            }
    // R3: a *-> b <-* c, a *-o d o-* c, a and c not adjacent, d *-o b: d *-> b.
    for (const b of N) {
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
    // R4: discriminating paths <theta, ..., a, b, c> for b with b o-* c.
    for (const b of N)
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
    // R5: a o-o b with an uncovered circle path <a, c, ..., d, b>, a and d, b and c not adjacent: all tails.
    for (const a of N)
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
    // R6: a - b and b o-* c: b -* c.  R7: a -o b, b o-* c, a and c not adjacent: b -* c.
    for (const b of N)
      for (const a of g.neighbours(b))
        for (const c of g.neighbours(b)) {
          if (c === a || g.get(c, b) !== 'circle') continue
          if (g.get(a, b) === 'tail' && g.get(b, a) === 'tail') changed = g.set(c, b, 'tail') || changed
          else if (g.get(b, a) === 'tail' && g.get(a, b) === 'circle' && !g.adj(a, c))
            changed = g.set(c, b, 'tail') || changed
        }
    // R8: a -> b -> c or a -o b -> c, and a o-> c: a -> c.
    for (const a of N)
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
    // R9: a o-> c with an uncovered potentially directed path <a, b, ..., c>, b and c not adjacent: a -> c.
    for (const a of N)
      for (const c of g.neighbours(a)) {
        if (!(g.get(a, c) === 'arrow' && g.get(c, a) === 'circle')) continue
        const paths = uncoveredPaths(g, a, c, potentiallyDirected(g), 4)
        if (paths.some((p) => !g.adj(p[1] as string, c))) changed = g.set(c, a, 'tail') || changed
      }
    // R10: a o-> c, b -> c <- t, uncovered p.d. paths from a to b and to t whose second nodes differ and are not
    // adjacent: a -> c.
    for (const a of N)
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
  }
}

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

/**
 * Fast Causal Inference (Spirtes, Meek and Richardson 1995; Zhang 2008): causal discovery that allows latent
 * confounders and selection. The PC adjacency search gives a skeleton and separating sets; unshielded colliders are
 * oriented; edges are pruned by testing subsets of Possible-D-Sep; the colliders are re-oriented and Zhang's complete
 * rules R1-R10 (R4 on discriminating paths with the separating sets) are applied until nothing changes (causal-learn's
 * `fci`).
 *
 * @param data - The rows (continuous for Fisher z; discrete for chi-square or G tests).
 * @param options - Test, significance, largest conditioning set and tracer (for the adjacency search).
 * @returns The PAG and the separating sets of removed edges.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { fci } from 'whitebox-engine/pgm/causal'
 * import { key, normal } from 'whitebox-engine/random'
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
  const g = new Marks(names)
  for (const [v, ns] of skeleton.adjacency) for (const w of ns) if (v < w) g.add(v, w)
  const sepsets = new Map(skeleton.sepsets)
  rule0(g, sepsets)
  // Possible-D-Sep pruning.
  const oracle = independenceOracle(data, test, options)
  for (const x of names)
    for (const y of names) {
      if (!(x < y) || !g.adj(x, y)) continue
      let removed = false
      for (const [from, to] of [
        [x, y],
        [y, x],
      ] as const) {
        if (removed) break
        for (const s of subsetsBySize(possibleDsep(g, from, to), 1)) {
          if (oracle(x, y, s)) {
            g.remove(x, y)
            sepsets.set(sepKey(x, y), s)
            removed = true
            break
          }
        }
      }
    }
  g.reset()
  rule0(g, sepsets)
  applyRules(g, sepsets)
  const edges: PagEdge[] = []
  for (const a of names)
    for (const b of names)
      if (a < b && g.adj(a, b)) edges.push({ a, b, markA: g.get(b, a) ?? 'circle', markB: g.get(a, b) ?? 'circle' })
  return { pag: { kind: 'pag', nodes: names, edges }, separatingSets: Object.fromEntries(sepsets) }
}
