// Random pairwise discrete models for the tests of the MAP methods: binary or multi-label, on chains, grids and random
// graphs, with arbitrary, submodular, Potts or truncated-linear pair terms, written as energies. Not published.
import { logFactor, type MarkovNetwork, markovNetwork } from '../lib/pgm/index.ts'
import { key, normal } from '../lib/random/index.ts'
import { toFlat } from '../lib/tensor/index.ts'

/** How the pair tables are drawn. */
export type PairKind = 'any' | 'submodular' | 'potts' | 'truncatedLinear'

/** Options of {@link randomPairwise}. */
export interface RandomPairwiseOptions {
  readonly seed: number
  readonly shape: 'chain' | 'grid' | 'random'
  /** Variables of a chain or a random graph; for a grid the rows (and `cols`). */
  readonly size: number
  readonly cols?: number
  readonly labels?: number
  readonly kind?: PairKind
  /** Scale of the unary energies (default 1). */
  readonly unaryScale?: number
}

/** The edges of the topology as pairs of variable indices. */
export function topologyEdges(shape: RandomPairwiseOptions['shape'], size: number, cols: number, u: Float64Array): [number, number][] {
  const edges: [number, number][] = []
  if (shape === 'chain') for (let i = 0; i + 1 < size; i++) edges.push([i, i + 1])
  else if (shape === 'grid') {
    for (let r = 0; r < size; r++)
      for (let c = 0; c < cols; c++) {
        if (c + 1 < cols) edges.push([r * cols + c, r * cols + c + 1])
        if (r + 1 < size) edges.push([r * cols + c, (r + 1) * cols + c])
      }
  } else {
    let at = 0
    for (let i = 0; i < size; i++) for (let j = i + 1; j < size; j++) if (Math.abs(u[at++] ?? 0) < 0.6) edges.push([i, j])
  }
  return edges
}

/** A random pairwise Markov network whose log values are minus the drawn energies. */
export function randomPairwise(options: RandomPairwiseOptions): MarkovNetwork {
  const K = options.labels ?? 2
  const cols = options.cols ?? options.size
  const n = options.shape === 'grid' ? options.size * cols : options.size
  const noise = Float64Array.from(toFlat(normal(key(options.seed), [n * n + n * K + n * n * K * K + 8])))
  const edges = topologyEdges(options.shape, options.size, cols, noise.subarray(0, n * n))
  const draw = ((): (() => number) => {
    let at = n * n
    return () => noise[at++] ?? 0
  })()
  const variables = Array.from({ length: n }, (_, i) => ({
    name: `v${i}`,
    states: Array.from({ length: K }, (_, k) => String(k)),
  }))
  const potentials = variables.map((v) =>
    logFactor([v.name], [K], Array.from({ length: K }, () => -(options.unaryScale ?? 1) * (options.kind === 'potts' || options.kind === 'truncatedLinear' ? Math.abs(draw()) : draw()))),
  )
  const kind = options.kind ?? 'any'
  for (const [i, j] of edges) {
    const w = Math.abs(draw()) + 0.1
    const energy = new Float64Array(K * K)
    for (let a = 0; a < K; a++)
      for (let b = 0; b < K; b++) {
        if (kind === 'any' || kind === 'submodular') energy[a * K + b] = draw()
        else if (kind === 'potts') energy[a * K + b] = a === b ? 0 : w
        else energy[a * K + b] = w * Math.min(Math.abs(a - b), 2)
      }
    if (kind === 'submodular') energy[3] = Math.min(energy[3] ?? 0, (energy[1] ?? 0) + (energy[2] ?? 0) - (energy[0] ?? 0))
    potentials.push(logFactor([`v${i}`, `v${j}`], [K, K], Array.from(energy, (e) => -e)))
  }
  return markovNetwork(variables, potentials)
}
