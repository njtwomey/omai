// Minimising a binary pairwise energy by a minimum s-t cut: the Kolmogorov-Zabih construction for submodular terms
// and the doubled graph of roof duality (QPBO) for arbitrary terms. Shared by the graph-cut MAP and the move-making
// methods (not re-exported).
import { digraph, minCut } from '../../graph/index.ts'

/** A binary pairwise term over variables i and j: the energies E(0,0), E(0,1), E(1,0) and E(1,1). */
export interface BinaryTerm {
  readonly i: number
  readonly j: number
  readonly a: number
  readonly b: number
  readonly c: number
  readonly d: number
}

/** B + C - A - D: non-negative exactly when the term is submodular. */
export function submodularMargin(t: BinaryTerm): number {
  return t.b + t.c - t.a - t.d
}

/** Capacities by ordered node pair, summed. */
class Capacities {
  private readonly map = new Map<number, number>()
  constructor(private readonly nodes: number) {}
  add(from: number, to: number, value: number): void {
    if (!(value > 0)) return
    const key = from * this.nodes + to
    this.map.set(key, (this.map.get(key) ?? 0) + value)
  }
  edges(): [number, number, number][] {
    return [...this.map.entries()].map(([key, v]) => [Math.floor(key / this.nodes), key % this.nodes, v])
  }
}

/**
 * Minimise sum_i u[i][x_i] + sum of submodular pair terms over x in {0, 1}^n by a minimum cut (Kolmogorov and Zabih
 * 2004): each term A + (C - A) x_i + (D - C) x_j + (B + C - A - D) [x_i = 0, x_j = 1] moves its unary parts into the
 * data costs and leaves one non-negative edge i -> j; a node on the source side takes label 0.
 */
export function minimiseSubmodular(
  n: number,
  u0: ArrayLike<number>,
  u1: ArrayLike<number>,
  terms: readonly BinaryTerm[],
): { readonly labels: Uint8Array; readonly value: number } {
  const diff = new Float64Array(n)
  let constant = 0
  const s = n
  const t = n + 1
  const cap = new Capacities(n + 2)
  for (let i = 0; i < n; i++) {
    diff[i] = (u1[i] ?? 0) - (u0[i] ?? 0)
    constant += u0[i] ?? 0
  }
  for (const term of terms) {
    constant += term.a
    diff[term.i] = (diff[term.i] ?? 0) + term.c - term.a
    diff[term.j] = (diff[term.j] ?? 0) + term.d - term.c
    cap.add(term.i, term.j, submodularMargin(term))
  }
  for (let i = 0; i < n; i++) {
    const d = diff[i] ?? 0
    if (d > 0) cap.add(s, i, d)
    else {
      cap.add(i, t, -d)
      constant += d
    }
  }
  const labels = new Uint8Array(n).fill(1)
  const edges = cap.edges()
  if (edges.length === 0) return { labels: Uint8Array.from(diff, (d) => (d > 0 ? 0 : d < 0 ? 1 : 0)), value: constant }
  const cut = minCut(digraph(edges, { nodes: n + 2 }), s, t)
  for (const v of cut.partition[0]) if (v < n) labels[v] = 0
  return { labels, value: constant + cut.value }
}

/** The result of {@link roofDuality}: a partial labelling (-1 where undecided) and the roof-dual lower bound. */
export interface RoofDual {
  /** 0 or 1 for a persistent label, -1 for a variable the cut leaves undecided. */
  readonly labels: Int8Array
  /** A lower bound on the minimum energy: the value of the minimum cut of the doubled graph. */
  readonly lowerBound: number
}

/**
 * Roof duality (Hammer, Hansen and Simeone 1984; QPBO by Boros and Hammer 2002, Kolmogorov and Rother 2007): double
 * the variables with their complements (node i and node n + i), give every term to both copies at half capacity, and
 * cut the resulting graph, which is submodular whatever the terms. A variable whose copy and complement disagree is
 * labelled and the label is weakly persistent: some minimiser of the energy agrees with every such label.
 */
export function roofDuality(
  n: number,
  u0: ArrayLike<number>,
  u1: ArrayLike<number>,
  terms: readonly BinaryTerm[],
): RoofDual {
  const nodes = 2 * n + 2
  const s = 2 * n
  const t = 2 * n + 1
  const diff = new Float64Array(n)
  let constant = 0
  const cap = new Capacities(nodes)
  for (let i = 0; i < n; i++) {
    diff[i] = (u1[i] ?? 0) - (u0[i] ?? 0)
    constant += u0[i] ?? 0
  }
  for (const term of terms) {
    const { i, j } = term
    constant += term.a
    const w = submodularMargin(term)
    if (w >= 0) {
      // w [x_i = 0, x_j = 1] = w [~x_j = 0, ~x_i = 1]: the edge i -> j and its mirror ~j -> ~i.
      diff[i] = (diff[i] ?? 0) + term.c - term.a
      diff[j] = (diff[j] ?? 0) + term.d - term.c
      cap.add(i, j, w / 2)
      cap.add(n + j, n + i, w / 2)
    } else {
      // -w [x_i = 1, x_j = 1] = -w [x_i = 1, ~x_j = 0]: the edge ~j -> i and its mirror ~i -> j.
      diff[i] = (diff[i] ?? 0) + term.c - term.a
      diff[j] = (diff[j] ?? 0) + term.b - term.a
      cap.add(n + j, i, -w / 2)
      cap.add(n + i, j, -w / 2)
    }
  }
  for (let i = 0; i < n; i++) {
    const d = diff[i] ?? 0
    if (d > 0) {
      cap.add(s, i, d / 2)
      cap.add(n + i, t, d / 2)
    } else {
      cap.add(i, t, -d / 2)
      cap.add(s, n + i, -d / 2)
      constant += d
    }
  }
  const labels = new Int8Array(n).fill(-1)
  const edges = cap.edges()
  if (edges.length === 0)
    return { labels: Int8Array.from(diff, (d) => (d > 0 ? 0 : d < 0 ? 1 : -1)), lowerBound: constant }
  const cut = minCut(digraph(edges, { nodes }), s, t)
  const source = new Uint8Array(nodes)
  for (const v of cut.partition[0]) source[v] = 1
  for (let i = 0; i < n; i++) {
    const x = source[i] === 1 ? 0 : 1
    const xbar = source[n + i] === 1 ? 0 : 1
    if (x + xbar === 1) labels[i] = x
  }
  return { labels, lowerBound: constant + cut.value }
}
