/**
 * Score-based and functional causal discovery from observational data: greedy equivalence search (GES, Chickering
 * 2002), which searches Markov equivalence classes by score, DirectLiNGAM, which exploits non-Gaussian noise to orient
 * a linear model, and NOTEARS, which learns a DAG by continuous optimisation with a smooth acyclicity constraint. GES
 * and NOTEARS are traced algorithms.
 */

import { argmaxOf, type Json, WhiteboxError } from '../../base/index.ts'
import { expm, lstsq } from '../../numerics/linalg/index.ts'
import { applyUpdates, type GradientTransformation } from '../../optim/index.ts'
import { fromStorage, toArray, toFlat, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { type Dag, dag } from '../graphs.ts'
import { type DataRow, dataVariables, numericColumns } from '../learning/_data.ts'
import { cpdag, type Pdag, pdag, pdagToDag } from '../learning/equivalence.ts'
import {
  type DiscreteScore,
  discreteScorer,
  type GaussianScore,
  gaussianScorer,
  type ScoreOptions,
} from '../learning/scores.ts'

export { pc } from '../learning/structure.ts'

/** Options of {@link gesAlgorithm} and {@link ges}. */
export interface GesOptions extends ScoreOptions {
  /** The decomposable score (default `bicGauss`, the linear-Gaussian BIC of causal-learn's GES). */
  readonly score?: DiscreteScore | GaussianScore
  /** The largest number of parents allowed (default unlimited). */
  readonly maxParents?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a GES run: the current equivalence class, its score and the phase. */
export interface GesState {
  readonly cpdag: Pdag
  readonly score: number
  readonly phase: 'forward' | 'backward' | 'done'
}

interface Adjacency {
  readonly directed: Set<string>
  readonly undirected: Set<string>
}

const dkey = (a: string, b: string): string => `${a}\u0000${b}`
const ukey = (a: string, b: string): string => (a < b ? dkey(a, b) : dkey(b, a))

function adjacencyOf(p: Pdag): Adjacency {
  return {
    directed: new Set(p.directed.map(([a, b]) => dkey(a, b))),
    undirected: new Set(p.undirected.map(([a, b]) => ukey(a, b))),
  }
}

function* subsets(items: readonly string[]): Generator<string[]> {
  const n = items.length
  for (let mask = 0; mask < 1 << n; mask++) yield items.filter((_, i) => mask & (1 << i))
}

/**
 * Greedy equivalence search (Chickering 2002) as an {@link Algorithm} over Markov equivalence classes: the forward
 * phase applies the valid Insert(X, Y, T) of largest score gain (T a set of neighbours of Y not adjacent to X, with
 * NA(Y, X) and T a clique and every semi-directed path from Y to X blocked by them) until none gains; the backward
 * phase then applies Delete(X, Y, H) moves (NA(Y, X) minus H a clique) the same way. After each move the class is
 * recomputed as the CPDAG of a consistent extension. Each step is one move, recorded as a `move` message.
 *
 * @param data - The rows (continuous for the Gaussian scores).
 * @param options - Score, parent limit and score options.
 * @returns The algorithm.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { gesAlgorithm } from 'whitebox-engine/pgm/causal'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const data = Array.from({ length: 200 }, (_, i) => ({ a: Math.sin(i), b: 2 * Math.sin(i) + Math.cos(3 * i) }))
 * runAlgorithm(gesAlgorithm(data), { maxSteps: 20 }).state.cpdag.undirected // => [['a', 'b']]
 * ```
 */
export function gesAlgorithm(data: readonly DataRow[], options: GesOptions = {}): Algorithm<GesState> {
  const names = dataVariables(data)
  const score = options.score ?? 'bicGauss'
  const local =
    score === 'bicGauss' || score === 'aicGauss' || score === 'logLikelihoodGauss'
      ? gaussianScorer(data, names, score)
      : discreteScorer(data, names, score, options)
  const maxParents = options.maxParents ?? Number.POSITIVE_INFINITY
  const sorted = (xs: Iterable<string>): string[] => [...xs].sort()
  const scoreOf = (v: string, parents: Iterable<string>): number => local(v, sorted(parents))
  const total = (p: Pdag): number => {
    const d = pdagToDag(p)
    return d.nodes.reduce((s, n) => s + scoreOf(n.name, n.parents), 0)
  }
  const adjacent = (g: Adjacency, a: string, b: string): boolean =>
    g.undirected.has(ukey(a, b)) || g.directed.has(dkey(a, b)) || g.directed.has(dkey(b, a))
  const neighbours = (g: Adjacency, y: string): string[] => names.filter((v) => v !== y && g.undirected.has(ukey(v, y)))
  const parents = (g: Adjacency, y: string): string[] => names.filter((v) => g.directed.has(dkey(v, y)))
  const isClique = (g: Adjacency, s: readonly string[]): boolean =>
    s.every((a, i) => s.slice(i + 1).every((b) => adjacent(g, a, b)))
  /** Whether every semi-directed path from y to x meets the blocking set. */
  const blocked = (g: Adjacency, y: string, x: string, block: Set<string>): boolean => {
    const seen = new Set([y])
    const stack = [y]
    while (stack.length > 0) {
      const v = stack.pop() as string
      for (const w of names) {
        if (seen.has(w) || block.has(w)) continue
        if (g.directed.has(dkey(v, w)) || g.undirected.has(ukey(v, w))) {
          if (w === x) return false
          seen.add(w)
          stack.push(w)
        }
      }
    }
    return true
  }
  const rebuild = (directed: readonly [string, string][], undirected: readonly [string, string][]): Pdag =>
    cpdag(pdagToDag(pdag(names, directed, undirected)))
  const edgesOf = (g: Adjacency): { d: [string, string][]; u: [string, string][] } => ({
    d: [...g.directed].map((k) => k.split('\u0000') as [string, string]),
    u: [...g.undirected].map((k) => k.split('\u0000') as [string, string]),
  })
  return {
    name: 'ges',
    config: { score, maxParents: Number.isFinite(maxParents) ? maxParents : -1 } as Json,
    schema: channels({
      metrics: { score: { description: 'score of the current equivalence class', direction: 'higher' } },
      messages: { move: { description: 'an Insert or Delete operation' } },
    }),
    init: () => {
      const empty = pdag(names, [], [])
      return { cpdag: empty, score: total(empty), phase: 'forward' }
    },
    step: (s, ctx) => {
      const g = adjacencyOf(s.cpdag)
      let best: { delta: number; apply: () => Pdag; op: Json } | undefined
      if (s.phase === 'forward') {
        for (const x of names) {
          for (const y of names) {
            if (x === y || adjacent(g, x, y)) continue
            const ny = neighbours(g, y)
            const na = ny.filter((t) => adjacent(g, t, x))
            const t0 = ny.filter((t) => !adjacent(g, t, x))
            const pa = parents(g, y)
            for (const t of subsets(t0)) {
              const nat = [...na, ...t]
              if (pa.length + nat.length + 1 > maxParents || !isClique(g, nat) || !blocked(g, y, x, new Set(nat)))
                continue
              const delta = scoreOf(y, [...pa, ...nat, x]) - scoreOf(y, [...pa, ...nat])
              if (best === undefined || delta > best.delta) {
                best = {
                  delta,
                  op: { operation: 'insert', x, y, t },
                  apply: () => {
                    const { d, u } = edgesOf(g)
                    return rebuild(
                      [...d, [x, y], ...t.map((v): [string, string] => [v, y])],
                      u.filter(([a, b]) => !t.some((v) => ukey(v, y) === ukey(a, b))),
                    )
                  },
                }
              }
            }
          }
        }
      } else if (s.phase === 'backward') {
        for (const x of names) {
          for (const y of names) {
            if (x === y || !(g.undirected.has(ukey(x, y)) || g.directed.has(dkey(x, y)))) continue
            const na = neighbours(g, y).filter((h) => h !== x && adjacent(g, h, x))
            const pa = parents(g, y).filter((v) => v !== x)
            for (const h of subsets(na)) {
              const rest = na.filter((v) => !h.includes(v))
              if (!isClique(g, rest)) continue
              const delta = scoreOf(y, [...pa, ...rest]) - scoreOf(y, [...pa, ...rest, x])
              if (best === undefined || delta > best.delta) {
                best = {
                  delta,
                  op: { operation: 'delete', x, y, h },
                  apply: () => {
                    const { d, u } = edgesOf(g)
                    const dir = d.filter(([a, b]) => !(a === x && b === y))
                    const und = u.filter(([a, b]) => ukey(a, b) !== ukey(x, y))
                    for (const v of h) {
                      if (g.undirected.has(ukey(y, v))) dir.push([y, v])
                      if (g.undirected.has(ukey(x, v))) dir.push([x, v])
                    }
                    const oriented = new Set(dir.map(([a, b]) => ukey(a, b)))
                    return rebuild(
                      dir,
                      und.filter(([a, b]) => !oriented.has(ukey(a, b))),
                    )
                  },
                }
              }
            }
          }
        }
      }
      const chosen = best as { delta: number; apply: () => Pdag; op: Json } | undefined
      if (chosen === undefined || !(chosen.delta > 1e-10)) {
        return { ...s, phase: s.phase === 'forward' ? 'backward' : 'done' }
      }
      ctx.run.message(ctx.t, 'move', chosen.op)
      const next = chosen.apply()
      return { cpdag: next, score: total(next), phase: s.phase }
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
    observe: (s) => ({ metrics: { score: s.score } }),
  }
}

/**
 * Learn a Markov equivalence class by greedy equivalence search ({@link gesAlgorithm}); with the Gaussian BIC this is
 * causal-learn's `ges` with `local_score_BIC`.
 *
 * @param data - The rows.
 * @param options - Score, parent limit, score options and tracer.
 * @returns The learned CPDAG and its score.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { ges } from 'whitebox-engine/pgm/causal'
 * const data = Array.from({ length: 300 }, (_, i) => {
 *   const a = Math.sin(1.3 * i), b = Math.cos(2.1 * i)
 *   return { a, b, c: a + b + 0.3 * Math.sin(5.7 * i) }
 * })
 * ges(data).cpdag.directed // => [['a', 'c'], ['b', 'c']]
 * ```
 */
export function ges(
  data: readonly DataRow[],
  options: GesOptions = {},
): { readonly cpdag: Pdag; readonly score: number; readonly status: RunStatus } {
  const r = runAlgorithm(gesAlgorithm(data, options), {
    maxSteps: 10000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return { cpdag: r.state.cpdag, score: r.state.score, status: r.status }
}

/** Standardise a column (population standard deviation, as numpy). */
function standardise(x: Float64Array): Float64Array {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / n)
  return x.map((v) => (v - m) / sd)
}

/** The residual of x regressed on y the way causal-learn's DirectLiNGAM writes it: cov(x, y; ddof 1) / var(y; ddof 0). */
function residual(x: Float64Array, y: Float64Array): Float64Array {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let cov = 0
  let vy = 0
  for (let i = 0; i < n; i++) {
    cov += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my)
    vy += ((y[i] ?? 0) - my) ** 2
  }
  const beta = cov / (n - 1) / (vy / n)
  return x.map((v, i) => v - beta * (y[i] ?? 0))
}

/** Hyvarinen's maximum-entropy approximation of the differential entropy of a standardised variable. */
function entropy(u: Float64Array): number {
  const n = u.length
  let a = 0
  let b = 0
  for (const v of u) {
    a += Math.log(Math.cosh(v))
    b += v * Math.exp((-v * v) / 2)
  }
  const k1 = 79.047
  const k2 = 7.4129
  const gamma = 0.37457
  return (1 + Math.log(2 * Math.PI)) / 2 - k1 * (a / n - gamma) ** 2 - k2 * (b / n) ** 2
}

function scaled(x: Float64Array): Float64Array {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / n)
  return x.map((v) => v / sd)
}

/** Result of {@link directLingam}. */
export interface LingamResult {
  /** The estimated causal order (variable names, causes first). */
  readonly order: readonly string[]
  /** B[i][j]: the direct effect of variable j on variable i (in data-column order). */
  readonly adjacency: readonly (readonly number[])[]
  readonly variables: readonly string[]
  /** The DAG of the non-zero effects. */
  readonly dag: Dag
}

/**
 * DirectLiNGAM (Shimizu et al. 2011) for linear non-Gaussian acyclic models: repeatedly pick the most exogenous
 * remaining variable by the pairwise likelihood-ratio measure of Hyvarinen and Smith (2013, entropies by maximum-entropy
 * approximation), regress it out of the others, and finally estimate each variable's direct effects by least squares
 * on the variables before it in the order (causal-learn's DirectLiNGAM with `measure="pwling"`; its adaptive-lasso
 * pruning is replaced by an optional threshold on the effects).
 *
 * @param data - Rows of numbers.
 * @param options - `threshold`: effects at most this in magnitude are set to zero (default 0).
 * @returns The causal order, the effect matrix and the DAG.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { directLingam } from 'whitebox-engine/pgm/causal'
 * import { key, uniform } from 'whitebox-engine/random'
 * const u = uniform(key(0), [400, 2]).data
 * const rows = Array.from({ length: 400 }, (_, i) => { const x = (u[2 * i] ?? 0) - 0.5; return { y: 2 * x + ((u[2 * i + 1] ?? 0) - 0.5), x } })
 * directLingam(rows).order // => ['x', 'y']
 * ```
 */
export function directLingam(data: readonly DataRow[], options: { readonly threshold?: number } = {}): LingamResult {
  const names = dataVariables(data)
  const X = numericColumns(data, names)
  const d = names.length
  const work: Float64Array[] = X.map((c) => Float64Array.from(c))
  let remaining = names.map((_, i) => i)
  const order: number[] = []
  while (remaining.length > 0) {
    let m = remaining[0] ?? 0
    if (remaining.length > 1) {
      const scores = remaining.map((i) => {
        let M = 0
        for (const j of remaining) {
          if (i === j) continue
          const xi = standardise(work[i] as Float64Array)
          const xj = standardise(work[j] as Float64Array)
          const rij = residual(xi, xj)
          const rji = residual(xj, xi)
          const diff = entropy(xj) + entropy(scaled(rij)) - (entropy(xi) + entropy(scaled(rji)))
          M += Math.min(0, diff) ** 2
        }
        return -M
      })
      m = remaining[argmaxOf(scores)] ?? m
    }
    for (const i of remaining) if (i !== m) work[i] = residual(work[i] as Float64Array, work[m] as Float64Array)
    order.push(m)
    remaining = remaining.filter((i) => i !== m)
  }
  const threshold = options.threshold ?? 0
  const n = data.length
  const B = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (const [k, target] of order.entries()) {
    if (k === 0) continue
    const preds = order.slice(0, k)
    const design = Array.from({ length: n }, (_, r) => [1, ...preds.map((p) => X[p]?.[r] ?? 0)])
    const beta = toArray(lstsq(design, Array.from(X[target] ?? []))) as number[]
    for (const [q, p] of preds.entries()) {
      const b = beta[q + 1] ?? 0
      ;(B[target] as number[])[p] = Math.abs(b) > threshold ? b : 0
    }
  }
  const edges: [string, string][] = []
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) if ((B[i]?.[j] ?? 0) !== 0) edges.push([names[j] ?? '', names[i] ?? ''])
  return { order: order.map((i) => names[i] ?? ''), adjacency: B, variables: names, dag: dag(edges, names) }
}

/** Options of {@link notearsAlgorithm} and {@link notears}. */
export interface NotearsOptions {
  /** L1 penalty on the weights (default 0.1). */
  readonly lambda1?: number
  /** Weights at most this in magnitude are dropped from the final graph (default 0.3). */
  readonly threshold?: number
  /** Stop when the acyclicity constraint h(W) falls below this (default 1e-8). */
  readonly hTol?: number
  /** The largest penalty coefficient (default 1e16). */
  readonly rhoMax?: number
  /** Inner optimiser steps per augmented-Lagrangian iteration (default 300). */
  readonly innerSteps?: number
  /** An inner first-order optimiser from optim (default: projected gradient descent with a backtracking line search). */
  readonly method?: GradientTransformation
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a NOTEARS run. */
export interface NotearsState {
  /** The weighted adjacency, row-major W[i][j] = weight of i -> j. */
  readonly W: Float64Array
  readonly h: number
  readonly rho: number
  readonly alpha: number
  readonly loss: number
}

/**
 * NOTEARS (Zheng et al. 2018) as an {@link Algorithm}: continuous optimisation of a linear SEM's weighted adjacency W
 * under the smooth acyclicity constraint h(W) = tr(exp(W o W)) - d = 0, by the augmented Lagrangian
 * (1/2n) ||X - XW||^2 + lambda1 ||W||_1 + rho/2 h^2 + alpha h. Each step solves the inner problem approximately (W
 * split into non-negative positive and negative parts; projected gradient steps with a backtracking line search, or a
 * given `optim` transformation, where the original uses L-BFGS-B), then raises
 * rho until h has fallen by a factor of four and updates alpha. Records h, the loss and rho.
 *
 * @param data - Rows of numbers.
 * @param options - Penalty, tolerances and the inner optimiser.
 * @returns The algorithm.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { notearsAlgorithm } from 'whitebox-engine/pgm/causal'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const rows = Array.from({ length: 100 }, (_, i) => ({ x: Math.sin(i), y: 2 * Math.sin(i) + 0.5 * Math.cos(3 * i) }))
 * runAlgorithm(notearsAlgorithm(rows, { innerSteps: 50 }), { maxSteps: 3 }).steps // => 3
 * ```
 */
export function notearsAlgorithm(data: readonly DataRow[], options: NotearsOptions = {}): Algorithm<NotearsState> {
  const names = dataVariables(data)
  const d = names.length
  const n = data.length
  const cols = numericColumns(data, names).map((c) => {
    const m = c.reduce((a, b) => a + b, 0) / n
    return c.map((v) => v - m)
  })
  const lambda1 = options.lambda1 ?? 0.1
  const hTol = options.hTol ?? 1e-8
  const rhoMax = options.rhoMax ?? 1e16
  const inner = options.innerSteps ?? 300
  const lossAndGrad = (W: Float64Array): { loss: number; grad: Float64Array } => {
    // R = X - XW; loss = ||R||^2 / 2n; grad = -X^T R / n.
    const grad = new Float64Array(d * d)
    let loss = 0
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < d; j++) {
        let pred = 0
        for (let i = 0; i < d; i++) pred += (cols[i]?.[r] ?? 0) * (W[i * d + j] ?? 0)
        const res = (cols[j]?.[r] ?? 0) - pred
        loss += res * res
        for (let i = 0; i < d; i++) grad[i * d + j] = (grad[i * d + j] ?? 0) - ((cols[i]?.[r] ?? 0) * res) / n
      }
    }
    return { loss: loss / (2 * n), grad }
  }
  const acyclicity = (W: Float64Array): { h: number; grad: Float64Array } => {
    const sq = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (W[i * d + j] ?? 0) ** 2))
    const E = toArray(expm(sq) as Value) as number[][]
    let h = -d
    for (let i = 0; i < d; i++) h += E[i]?.[i] ?? 0
    const grad = new Float64Array(d * d)
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) grad[i * d + j] = (E[j]?.[i] ?? 0) * 2 * (W[i * d + j] ?? 0)
    return { h, grad }
  }
  const project = (x: Float64Array): Float64Array =>
    x.map((v, k) => {
      const i = Math.floor((k % (d * d)) / d)
      return i === k % d ? 0 : Math.max(v, 0)
    })
  /** The augmented Lagrangian and its gradient in [P, N] (W = P - N). */
  const objective = (x: Float64Array, rho: number, alpha: number): { value: number; grad: Float64Array } => {
    const W = Float64Array.from({ length: d * d }, (_, k) => (x[k] ?? 0) - (x[d * d + k] ?? 0))
    const { loss, grad: gl } = lossAndGrad(W)
    const { h, grad: gh } = acyclicity(W)
    const g = new Float64Array(2 * d * d)
    let l1 = 0
    for (let k = 0; k < d * d; k++) {
      const gw = (gl[k] ?? 0) + (rho * h + alpha) * (gh[k] ?? 0)
      g[k] = gw + lambda1
      g[d * d + k] = -gw + lambda1
      l1 += (x[k] ?? 0) + (x[d * d + k] ?? 0)
    }
    return { value: loss + (rho / 2) * h * h + alpha * h + lambda1 * l1, grad: g }
  }
  const solveInner = (start: Float64Array, rho: number, alpha: number): Float64Array => {
    let x: Float64Array = new Float64Array(2 * d * d)
    for (let k = 0; k < d * d; k++) {
      const w = start[k] ?? 0
      x[k] = Math.max(w, 0)
      x[d * d + k] = Math.max(-w, 0)
    }
    x = project(x)
    if (options.method !== undefined) {
      // A first-order transformation from optim, projected after each step.
      const method = options.method
      const asTensor = (v: Float64Array): Value => fromStorage(Float64Array.from(v), [v.length], 'float64')
      let state = method.init(asTensor(x))
      for (let it = 0; it < inner; it++) {
        const step = method.update(asTensor(objective(x, rho, alpha).grad), state, asTensor(x))
        state = step.state
        x = project(Float64Array.from(toFlat(applyUpdates(asTensor(x), step.updates) as Value)))
      }
    } else {
      // Projected gradient with a backtracking (Armijo) step: unlike Adam it keeps the gradient's scale across
      // coordinates, which the acyclicity penalty needs once rho is large.
      let t = 1
      let cur = objective(x, rho, alpha)
      for (let it = 0; it < inner; it++) {
        let accepted = false
        for (let tries = 0; tries < 60; tries++) {
          const next = project(x.map((v, k) => v - t * (cur.grad[k] ?? 0)))
          const f = objective(next, rho, alpha)
          let decrease = 0
          let dist = 0
          for (let k = 0; k < x.length; k++) {
            const dk = (next[k] ?? 0) - (x[k] ?? 0)
            decrease += (cur.grad[k] ?? 0) * dk
            dist += dk * dk
          }
          if (f.value <= cur.value + decrease + dist / (2 * t)) {
            const moved = Math.sqrt(dist)
            x = next
            cur = f
            accepted = true
            t *= 2
            if (moved < 1e-12) it = inner
            break
          }
          t /= 2
        }
        if (!accepted) break
      }
    }
    return Float64Array.from({ length: d * d }, (_, k) => (x[k] ?? 0) - (x[d * d + k] ?? 0))
  }
  return {
    name: 'notears',
    config: { lambda1, hTol, rhoMax, innerSteps: inner } as Json,
    schema: channels({
      metrics: {
        h: { description: 'acyclicity constraint tr(exp(W o W)) - d', direction: 'lower' },
        loss: { description: 'least-squares loss', direction: 'lower' },
        rho: { description: 'penalty coefficient' },
      },
    }),
    init: () => ({
      W: new Float64Array(d * d),
      h: Number.POSITIVE_INFINITY,
      rho: 1,
      alpha: 0,
      loss: lossAndGrad(new Float64Array(d * d)).loss,
    }),
    step: (s) => {
      let rho = s.rho
      let W = s.W
      let h = Number.POSITIVE_INFINITY
      while (rho < rhoMax) {
        W = solveInner(s.W, rho, s.alpha)
        h = acyclicity(W).h
        if (!Number.isFinite(s.h) || h <= 0.25 * s.h) break
        rho *= 10
      }
      return { W, h, rho, alpha: s.alpha + rho * h, loss: lossAndGrad(W).loss }
    },
    done: (s, t) => (t > 0 && (s.h <= hTol || s.rho >= rhoMax) ? 'converged' : undefined),
    observe: (s) => ({ metrics: { h: Number.isFinite(s.h) ? s.h : -1, loss: s.loss, rho: s.rho } }),
  }
}

/**
 * Learn a DAG by NOTEARS ({@link notearsAlgorithm}) and threshold the weights.
 *
 * @param data - Rows of numbers.
 * @param options - Penalty, threshold, tolerances, inner optimiser and tracer.
 * @returns The weighted adjacency (row-major, W[i][j] = weight of i -> j), the DAG and convergence information.
 * @concept causal-discovery
 * @example
 * ```ts
 * import { notears } from 'whitebox-engine/pgm/causal'
 * import { dagEdges } from 'whitebox-engine/pgm'
 * import { key, normal } from 'whitebox-engine/random'
 * const z = normal(key(1), [300, 2]).data
 * const rows = Array.from({ length: 300 }, (_, i) => { const x = z[2 * i] ?? 0; return { x, y: 1.5 * x + 0.5 * (z[2 * i + 1] ?? 0) } })
 * dagEdges(notears(rows).dag).length // => 1
 * ```
 */
export function notears(
  data: readonly DataRow[],
  options: NotearsOptions & { readonly maxIter?: number } = {},
): { readonly weights: number[][]; readonly dag: Dag; readonly h: number; readonly status: RunStatus } {
  const names = dataVariables(data)
  const d = names.length
  const r = runAlgorithm(notearsAlgorithm(data, options), {
    maxSteps: options.maxIter ?? 100,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const threshold = options.threshold ?? 0.3
  const weights = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => {
      const w = r.state.W[i * d + j] ?? 0
      return Math.abs(w) < threshold ? 0 : w
    }),
  )
  const edges: [string, string][] = []
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) if ((weights[i]?.[j] ?? 0) !== 0) edges.push([names[i] ?? '', names[j] ?? ''])
  let result: Dag
  try {
    result = dag(edges, names)
  } catch {
    throw new WhiteboxError(
      'value',
      'notears: the thresholded graph has a cycle (raise the threshold or the iterations)',
    )
  }
  return { weights, dag: result, h: r.state.h, status: r.status }
}
