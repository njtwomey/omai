// Graphs from oracle fixtures (oracle/whitebox_oracle/graphs.py `spec`) and fast-check arbitraries for graph
// property tests. Not published.
import fc from 'fast-check'
import { type Graph, graph } from '../lib/graph/index.ts'

/** The graph a fixture case describes with `n`, `directed` and `edges` ([source, target, weight] triples). */
export function graphFromCase(params: Readonly<Record<string, unknown>>, suffix = ''): Graph {
  const n = params[`n${suffix}`] as number
  const directed = params[`directed${suffix}`] as boolean
  const edges = params[`edges${suffix}`] as [number, number, number][]
  return graph(edges, { nodes: n, directed })
}

/** Options of {@link arbitraryGraph}. */
export interface ArbitraryGraphOptions {
  readonly minNodes?: number
  readonly maxNodes?: number
  readonly directed?: boolean
  /** Integer weights in [1, maxWeight] (default 1: unweighted). */
  readonly maxWeight?: number
  /** Edge probability (default 0.3). */
  readonly density?: number
}

/** A random simple graph (no self-loops) as a fast-check arbitrary. */
export function arbitraryGraph(options: ArbitraryGraphOptions = {}): fc.Arbitrary<Graph> {
  const { minNodes = 2, maxNodes = 9, directed = false, maxWeight = 1, density = 0.3 } = options
  return fc.integer({ min: minNodes, max: maxNodes }).chain((n) => {
    const pairs: [number, number][] = []
    for (let i = 0; i < n; i++) for (let j = directed ? 0 : i + 1; j < n; j++) if (i !== j) pairs.push([i, j])
    return fc
      .tuple(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: pairs.length, maxLength: pairs.length }),
        fc.array(fc.integer({ min: 1, max: maxWeight }), { minLength: pairs.length, maxLength: pairs.length }),
      )
      .map(([keep, weights]) =>
        graph(
          pairs
            .map(([u, v], k) => [u, v, weights[k] ?? 1] as [number, number, number])
            .filter((_, k) => (keep[k] ?? 1) < density),
          { nodes: n, directed },
        ),
      )
  })
}
