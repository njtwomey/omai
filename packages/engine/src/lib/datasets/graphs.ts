/**
 * Graph datasets: the karate club and Les Misérables networks, contextual stochastic block models for node
 * classification, a graph-classification set of random graphs labelled by family (Erdős-Rényi, Barabási-Albert,
 * Watts-Strogatz, grids, trees and more), small molecules read from SMILES, and kinship knowledge-graph triples with
 * planted rules.
 */

import { OmaiError } from '../base/index.ts'
import {
  type DatasetGenerator,
  type Generated,
  type GraphData,
  type GraphRecord,
  graphData,
  graphRecord,
  type TripleData,
  triples,
} from '../data/index.ts'
import { foldIn, type Key, normal, permutation, poisson, randint, uniform } from '../random/index.ts'
import { structure } from '../scene/index.ts'
import { dict, int, real } from '../space/index.ts'
import {
  KARATE_CLUBS,
  KARATE_EDGES,
  KARATE_WEIGHTS,
  LES_MISERABLES_EDGES,
  LES_MISERABLES_NAMES,
  LES_MISERABLES_WEIGHTS,
} from './_data/graphs.data.ts'
import { ATOMIC_WEIGHTS, formulaOf, MOLECULES, parseSmiles } from './_molecules.ts'
import { labels, matrix, stamp } from './_util.ts'

function pairs(flat: readonly number[]): number[][] {
  return Array.from({ length: flat.length / 2 }, (_, i) => [flat[2 * i] ?? 0, flat[2 * i + 1] ?? 0])
}

/**
 * Zachary's karate club (1977): the friendships of 34 members of a club that split in two, with interaction counts as
 * edge weights and each member's eventual club (`Mr. Hi` or `Officer`) as node labels, the classic community
 * detection example.
 *
 * @returns The graph (one graph with node labels).
 * @example
 * ```ts
 * import { karateClub } from 'omai-engine/datasets'
 * karateClub().graphs[0]?.edges.shape // => [78, 2]
 * ```
 */
export function karateClub(): GraphData {
  const clubs = KARATE_CLUBS.map((c) => (c === 'Mr. Hi' ? 0 : 1))
  return graphData(
    [graphRecord(pairs(KARATE_EDGES), { nNodes: 34, edgeWeights: KARATE_WEIGHTS as number[], nodeLabels: clubs })],
    {
      classes: ['Mr. Hi', 'Officer'],
      meta: {
        name: 'karateClub',
        description: "Zachary's karate club: 34 members, 78 friendships, and the two clubs it split into.",
        source: 'NetworkX karate_club_graph',
        citation:
          'Zachary (1977), An information flow model for conflict and fission in small groups, J. Anthropological Research 33(4).',
        licence: 'Public domain data (NetworkX, BSD-3-Clause)',
      },
    },
  )
}

/**
 * The co-appearance network of the characters of Les Misérables (Knuth, 1993): 77 characters, edges weighted by the
 * number of chapters in which two characters appear together.
 *
 * @returns The weighted graph with character names.
 * @example
 * ```ts
 * import { lesMiserables } from 'omai-engine/datasets'
 * lesMiserables().graphs[0]?.nNodes // => 77
 * ```
 */
export function lesMiserables(): GraphData {
  return graphData(
    [
      graphRecord(pairs(LES_MISERABLES_EDGES), {
        nNodes: LES_MISERABLES_NAMES.length,
        edgeWeights: LES_MISERABLES_WEIGHTS as number[],
        nodeNames: LES_MISERABLES_NAMES,
      }),
    ],
    {
      meta: {
        name: 'lesMiserables',
        description: 'Co-appearances of the characters of Les Miserables (Knuth, The Stanford GraphBase).',
        source: 'NetworkX les_miserables_graph',
        citation: 'Knuth (1993), The Stanford GraphBase, Addison-Wesley.',
        licence: 'Public domain data (NetworkX, BSD-3-Clause)',
      },
    },
  )
}

/** Sample the edges of a stochastic block model with block-pair probabilities `p`. */
function sbmEdges(k: Key, blocks: readonly number[], p: (a: number, b: number) => number): number[][] {
  const n = blocks.length
  const u = uniform(k, [(n * (n - 1)) / 2]).data
  const edges: number[][] = []
  let c = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if ((u[c++] ?? 1) < p(blocks[i] ?? 0, blocks[j] ?? 0)) edges.push([i, j])
  return edges
}

/**
 * A contextual stochastic block model (Deshpande et al., 2018): `blocks` equal communities, edges with probability
 * `p_in` within and `p_out` between communities (set by the average degree and the signal `lambda`), and node
 * features `mu m_c + N(0, I)` around a random unit mean per community. Train, validation and test masks split the
 * nodes 60/20/20. The truth has the communities, the edge probabilities and the feature means, so graph and feature
 * information can be traded off exactly.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { contextualSbm } from 'omai-engine/datasets'
 * generate(contextualSbm, { knobs: { nodes: 60, features: 4 } }).data.graphs[0]?.nodeFeatures?.shape // => [60, 4]
 * ```
 */
export const contextualSbm: DatasetGenerator<GraphData> = {
  name: 'contextualSbm',
  kind: 'graph',
  description: 'A stochastic block model with Gaussian node features that depend on the community.',
  knobs: dict({
    nodes: int(2, 20_000, { label: 'nodes', description: 'The number of nodes.', default: 300 }),
    blocks: int(2, 50, { label: 'communities', description: 'The number of communities.', default: 2 }),
    degree: real(0.1, 1000, {
      description: 'The expected number of edges per node.',
      default: 10,
      label: 'average degree',
    }),
    lambda: real(0, 50, {
      description: 'How much likelier edges within a community are than across (0: no signal).',
      default: 2,
      label: 'graph signal',
    }),
    features: int(0, 1000, {
      label: 'feature dimensions',
      description: 'The number of Gaussian node features.',
      default: 8,
    }),
    mu: real(0, 50, {
      description: 'How far apart the communities’ feature means are (0: no signal).',
      default: 1,
      label: 'feature signal',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['nodes'] as number
    const kb = knobs['blocks'] as number
    const d = knobs['degree'] as number
    const lam = knobs['lambda'] as number
    const f = knobs['features'] as number
    const mu = knobs['mu'] as number
    const blocks = Array.from(permutation(foldIn(k, 0), n).data, (i) => i % kb)
    // Average degree d with c_in - c_out = lambda sqrt(d) (scaled by the number of blocks).
    const cIn = Math.min(n, d + (lam * Math.sqrt(d) * (kb - 1)) / kb)
    const cOut = Math.max(0, d - (lam * Math.sqrt(d)) / kb)
    const pIn = Math.min(1, cIn / n)
    const pOut = Math.min(1, cOut / n)
    const edges = sbmEdges(foldIn(k, 1), blocks, (a, b) => (a === b ? pIn : pOut))
    const means = Array.from({ length: kb }, (_, c) => {
      const v = Array.from(normal(foldIn(foldIn(k, 2), c), [f]).data)
      const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1
      return v.map((x) => x / norm)
    })
    const z = normal(foldIn(k, 3), [n, f]).data
    const x = matrix(
      Float64Array.from(
        { length: n * f },
        (_, i) => mu * (means[blocks[Math.floor(i / Math.max(1, f))] ?? 0]?.[i % Math.max(1, f)] ?? 0) + (z[i] ?? 0),
      ),
      n,
      f,
    )
    const split = uniform(foldIn(k, 4), [n]).data
    const g = graphRecord(edges.length === 0 ? [] : edges, {
      nNodes: n,
      nodeLabels: blocks,
      ...(f > 0 ? { nodeFeatures: x } : {}),
    })
    return {
      data: graphData([g], {
        masks: {
          train: Array.from(split, (u) => u < 0.6),
          val: Array.from(split, (u) => u >= 0.6 && u < 0.8),
          test: Array.from(split, (u) => u >= 0.8),
        },
        classes: Array.from({ length: kb }, (_, c) => `block${c}`),
        meta: stamp('contextualSbm', contextualSbm.description, knobs),
      }),
      truth: { clusters: labels(blocks), extra: { pIn, pOut, featureMeans: means, mu } },
    }
  },
}

/** The graph families of {@link familyGraph}. */
export type GraphFamily =
  | 'erdosRenyi'
  | 'barabasiAlbert'
  | 'wattsStrogatz'
  | 'grid'
  | 'cycle'
  | 'path'
  | 'star'
  | 'complete'
  | 'tree'
  | 'geometric'

/** All graph families, in label order. */
export const GRAPH_FAMILIES: readonly GraphFamily[] = [
  'erdosRenyi',
  'barabasiAlbert',
  'wattsStrogatz',
  'grid',
  'cycle',
  'path',
  'star',
  'complete',
  'tree',
  'geometric',
]

/**
 * One random graph from a family: Erdős-Rényi (edge probability `p`), Barabási-Albert (`m` edges per new node),
 * Watts-Strogatz (ring of degree `2m` rewired with probability `p`), a grid, cycle, path, star or complete graph, a
 * uniform random labelled tree (from a Prüfer sequence) or a random geometric graph in the unit square (radius `r`).
 *
 * @param key - The random key.
 * @param family - The family.
 * @param n - The number of nodes.
 * @param options - `p`, `m` and `r` (defaults 0.2, 2 and 0.3).
 * @returns The graph.
 * @throws OmaiError with code `value` for an unknown family.
 * @example
 * ```ts
 * import { familyGraph } from 'omai-engine/datasets'
 * import { key } from 'omai-engine/random'
 * familyGraph(key(0), 'tree', 10).edges.shape // => [9, 2]
 * ```
 */
export function familyGraph(
  key: Key,
  family: GraphFamily,
  n: number,
  options: { readonly p?: number; readonly m?: number; readonly r?: number } = {},
): GraphRecord {
  const p = options.p ?? 0.2
  const m = Math.max(1, Math.min(options.m ?? 2, n - 1))
  const edges: number[][] = []
  const seen = new Set<string>()
  const add = (a: number, b: number) => {
    if (a === b) return
    const [i, j] = a < b ? [a, b] : [b, a]
    const id = `${i},${j}`
    if (seen.has(id)) return
    seen.add(id)
    edges.push([i, j])
  }
  switch (family) {
    case 'erdosRenyi': {
      const u = uniform(key, [(n * (n - 1)) / 2]).data
      let c = 0
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if ((u[c++] ?? 1) < p) add(i, j)
      break
    }
    case 'barabasiAlbert': {
      const ends: number[] = []
      for (let i = 0; i <= m && i < n; i++)
        for (let j = 0; j < i; j++) {
          add(j, i)
          ends.push(i, j)
        }
      for (let v = m + 1; v < n; v++) {
        const targets = new Set<number>()
        for (let t = 0; targets.size < m && t < 100 * m; t++) {
          const r = Math.floor((uniform(foldIn(foldIn(key, v), t), [1]).data[0] ?? 0) * ends.length)
          targets.add(ends[r] ?? 0)
        }
        for (const t of targets) {
          add(v, t)
          ends.push(v, t)
        }
      }
      break
    }
    case 'wattsStrogatz': {
      const u = uniform(foldIn(key, 0), [n * m]).data
      const r = randint(foldIn(key, 1), [n * m], 0, n).data
      for (let i = 0; i < n; i++) {
        for (let s = 1; s <= m; s++) {
          const c = i * m + s - 1
          add(i, (u[c] ?? 1) < p ? (r[c] ?? 0) : (i + s) % n)
        }
      }
      break
    }
    case 'grid': {
      const w = Math.max(1, Math.round(Math.sqrt(n)))
      for (let i = 0; i < n; i++) {
        if ((i + 1) % w !== 0 && i + 1 < n) add(i, i + 1)
        if (i + w < n) add(i, i + w)
      }
      break
    }
    case 'cycle':
      for (let i = 0; i < n; i++) if (n > 2 || i < n - 1) add(i, (i + 1) % n)
      break
    case 'path':
      for (let i = 0; i + 1 < n; i++) add(i, i + 1)
      break
    case 'star':
      for (let i = 1; i < n; i++) add(0, i)
      break
    case 'complete':
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) add(i, j)
      break
    case 'tree': {
      if (n === 2) add(0, 1)
      if (n > 2) {
        const prufer = Array.from(randint(key, [n - 2], 0, n).data)
        const degree = new Array<number>(n).fill(1)
        for (const v of prufer) degree[v] = (degree[v] ?? 1) + 1
        for (const v of prufer) {
          const leaf = degree.indexOf(1)
          add(leaf, v)
          degree[leaf] = 0
          degree[v] = (degree[v] ?? 1) - 1
        }
        const last = degree.flatMap((d, i) => (d === 1 ? [i] : []))
        add(last[0] ?? 0, last[1] ?? 0)
      }
      break
    }
    case 'geometric': {
      const r = options.r ?? 0.3
      const pts = uniform(key, [n, 2]).data
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (Math.hypot((pts[2 * i] ?? 0) - (pts[2 * j] ?? 0), (pts[2 * i + 1] ?? 0) - (pts[2 * j + 1] ?? 0)) <= r)
            add(i, j)
      return graphRecord(edges, { nNodes: n, nodeFeatures: matrix(pts, n, 2) })
    }
    default:
      throw new OmaiError('value', `familyGraph: unknown family ${String(family)}`)
  }
  return graphRecord(edges, { nNodes: n })
}

/**
 * A graph-classification dataset of random graphs labelled by their family (Erdős-Rényi, Barabási-Albert,
 * Watts-Strogatz, grids, cycles, paths, stars, complete graphs, trees and geometric graphs), with sizes uniform between
 * `minNodes` and `maxNodes`, for testing what graph neural networks can tell apart.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { graphFamilies } from 'omai-engine/datasets'
 * generate(graphFamilies, { knobs: { graphs: 20 } }).data.classes?.length // => 10
 * ```
 */
export const graphFamilies: DatasetGenerator<GraphData> = {
  name: 'graphFamilies',
  kind: 'graph',
  description: 'Random graphs from ten families, labelled by family.',
  knobs: dict({
    graphs: int(1, 100_000, { label: 'graphs', description: 'The number of graphs.', default: 200 }),
    minNodes: int(3, 10_000, { label: 'fewest nodes', description: 'The smallest graph size.', default: 10 }),
    maxNodes: int(3, 10_000, { label: 'most nodes', description: 'The largest graph size.', default: 30 }),
  }),
  generate: (k, knobs) => {
    const g = knobs['graphs'] as number
    const lo = knobs['minNodes'] as number
    const hi = Math.max(lo, knobs['maxNodes'] as number)
    const fam = randint(foldIn(k, 0), [g], 0, GRAPH_FAMILIES.length).data
    const sizes = randint(foldIn(k, 1), [g], lo, hi + 1).data
    const graphs = Array.from(fam, (f, i) =>
      familyGraph(foldIn(foldIn(k, 2), i), GRAPH_FAMILIES[f] as GraphFamily, sizes[i] ?? lo),
    )
    return {
      data: graphData(graphs, {
        graphLabels: Array.from(fam),
        classes: GRAPH_FAMILIES,
        meta: stamp('graphFamilies', graphFamilies.description, knobs),
      }),
      truth: { clusters: labels(Array.from(fam)) },
    }
  },
}

/**
 * Small molecules as heavy-atom graphs, read from SMILES: node labels are elements, edge labels bond types (single,
 * double, triple, aromatic), and each molecule is labelled aromatic or not. The truth gives every molecule's name,
 * formula (with implicit hydrogens), molecular weight and ring count (the cyclomatic number of its graph).
 *
 * @returns The graphs and their exact properties.
 * @example
 * ```ts
 * import { smallMolecules } from 'omai-engine/datasets'
 * const m = smallMolecules()
 * (m.truth.extra?.['formulas'] as string[])[m.data.graphs.length - 2] // => 'C8H10N4O2'
 * ```
 */
export function smallMolecules(): Generated<GraphData> {
  const elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B']
  const bondTypes = [1, 2, 3, 1.5]
  const parsed = MOLECULES.map((m) => parseSmiles(m.smiles))
  const graphs = parsed.map((p) =>
    graphRecord(
      p.bonds.map(([a, b]) => [a, b]),
      {
        nNodes: p.atoms.length,
        nodeLabels: p.atoms.map((a) => elements.indexOf(a)),
        edgeLabels: p.bonds.map(([, , o]) => bondTypes.indexOf(o)),
        nodeFeatures: matrix(
          p.atoms.flatMap((a, i) => [
            ...elements.map((e) => (e === a ? 1 : 0)),
            p.hydrogens[i] ?? 0,
            p.aromatic[i] === true ? 1 : 0,
          ]),
          p.atoms.length,
          elements.length + 2,
        ),
      },
    ),
  )
  const formulas = parsed.map(formulaOf)
  const weights = parsed.map(
    (p) =>
      Math.round(
        (p.atoms.reduce((s, a) => s + (ATOMIC_WEIGHTS[a] ?? 0), 0) +
          (ATOMIC_WEIGHTS['H'] ?? 0) * p.hydrogens.reduce((s, h) => s + h, 0)) *
          1000,
      ) / 1000,
  )
  const ringCounts = parsed.map((p) => p.bonds.length - p.atoms.length + 1)
  const aromaticLabel = parsed.map((p) => (p.aromatic.some(Boolean) ? 1 : 0))
  return {
    data: graphData(graphs, {
      graphLabels: aromaticLabel,
      classes: ['aliphatic', 'aromatic'],
      meta: {
        name: 'smallMolecules',
        description: `${MOLECULES.length} common small molecules as heavy-atom graphs, labelled aromatic or not.`,
        source: 'omai (SMILES of common molecules)',
        licence: 'Public domain (chemical facts)',
      },
    }),
    truth: {
      extra: {
        names: MOLECULES.map((m) => m.name),
        smiles: MOLECULES.map((m) => m.smiles),
        formulas,
        molecularWeights: weights,
        ringCounts,
        elements,
        bondTypes: ['single', 'double', 'triple', 'aromatic'],
      },
    },
  }
}

const RELATIONS = ['parentOf', 'childOf', 'spouseOf', 'siblingOf', 'grandparentOf', 'auntOrUncleOf'] as const

/**
 * Knowledge-graph triples from random family trees with planted rules (kinship-style link prediction): couples have
 * Poisson numbers of children over several generations, and the relations `parentOf`, `childOf`, `spouseOf`,
 * `siblingOf`, `grandparentOf` and `auntOrUncleOf` are all derived from `parentOf` and `spouseOf` by fixed rules.
 * A fraction of the true triples is observed; the truth holds every true triple, the held-out ones and the rules.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { kinshipTriples } from 'omai-engine/datasets'
 * generate(kinshipTriples).data.relationNames.length // => 6
 * ```
 */
export const kinshipTriples: DatasetGenerator<TripleData> = {
  name: 'kinshipTriples',
  kind: 'triples',
  description: 'Family-tree triples whose relations follow fixed rules, partly observed.',
  knobs: dict({
    families: int(1, 1000, { label: 'families', description: 'The number of family trees.', default: 5 }),
    generations: int(2, 8, { label: 'generations', description: 'The depth of each family tree.', default: 3 }),
    children: real(0.5, 6, {
      description: 'The mean number of children per couple.',
      default: 2,
      label: 'mean children per couple',
    }),
    observed: real(0.01, 1, {
      description: 'The fraction of true triples observed (the rest are to be inferred).',
      default: 0.8,
      label: 'fraction of true triples observed',
    }),
  }),
  generate: (k, knobs) => {
    const parents = new Map<number, number[]>()
    const spouses: [number, number][] = []
    let next = 0
    let salt = 0
    for (let f = 0; f < (knobs['families'] as number); f++) {
      let couples: [number, number][] = [[next, next + 1]]
      spouses.push([next, next + 1])
      next += 2
      for (let g = 1; g < (knobs['generations'] as number); g++) {
        const nextCouples: [number, number][] = []
        for (const [a, b] of couples) {
          const kids = poisson(foldIn(k, salt++), knobs['children'] as number, [1]).data[0] ?? 0
          for (let c = 0; c < kids; c++) {
            const child = next++
            parents.set(child, [a, b])
            const spouse = next++
            spouses.push([child, spouse])
            nextCouples.push([child, spouse])
          }
        }
        couples = nextCouples
      }
    }
    const facts = new Set<string>()
    const fact = (h: number, r: number, t: number) => facts.add(`${h},${r},${t}`)
    const childrenOf = new Map<number, number[]>()
    for (const [c, ps] of parents) for (const p of ps) childrenOf.set(p, [...(childrenOf.get(p) ?? []), c])
    for (const [c, ps] of parents)
      for (const p of ps) {
        fact(p, 0, c)
        fact(c, 1, p)
        for (const gp of parents.get(p) ?? []) fact(gp, 4, c)
        for (const sib of childrenOf.get(p) ?? []) if (sib !== c) fact(c, 3, sib)
      }
    for (const [a, b] of spouses) {
      fact(a, 2, b)
      fact(b, 2, a)
    }
    // Aunts and uncles: siblings of a parent (and their spouses).
    for (const [c, ps] of parents)
      for (const p of ps)
        for (const gp of parents.get(p) ?? [])
          for (const au of childrenOf.get(gp) ?? []) {
            if (au === p) continue
            fact(au, 5, c)
            for (const [a, b] of spouses) if (a === au) fact(b, 5, c)
          }
    const all = [...facts]
      .map((s) => s.split(',').map(Number) as [number, number, number])
      .sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2])
    const u = uniform(foldIn(k, 999_999), [all.length]).data
    const seen = all.filter((_, i) => (u[i] ?? 1) < (knobs['observed'] as number))
    const held = all.filter((_, i) => (u[i] ?? 1) >= (knobs['observed'] as number))
    const names = Array.from({ length: next }, (_, i) => `person${i}`)
    return {
      data: triples(
        seen.map((t) => t[0]),
        seen.map((t) => t[1]),
        seen.map((t) => t[2]),
        {
          entityNames: names,
          relationNames: RELATIONS,
          meta: stamp('kinshipTriples', kinshipTriples.description, knobs),
        },
      ),
      truth: {
        structure: structure(
          'knowledge-graph-rules',
          RELATIONS.map((r) => ({ id: r, label: r, role: 'other' as const })),
          [
            { from: 'parentOf', to: 'childOf' },
            { from: 'parentOf', to: 'grandparentOf' },
            { from: 'parentOf', to: 'siblingOf' },
            { from: 'siblingOf', to: 'auntOrUncleOf' },
            { from: 'spouseOf', to: 'auntOrUncleOf' },
          ],
        ),
        extra: {
          allTriples: all,
          heldOut: held,
          rules: [
            'childOf(x, y) <- parentOf(y, x)',
            'grandparentOf(x, z) <- parentOf(x, y), parentOf(y, z)',
            'siblingOf(x, y) <- parentOf(z, x), parentOf(z, y), x != y',
            'auntOrUncleOf(x, y) <- siblingOf(x, z), parentOf(z, y)',
            'auntOrUncleOf(x, y) <- spouseOf(x, w), siblingOf(w, z), parentOf(z, y)',
            'spouseOf(x, y) <- spouseOf(y, x)',
          ],
        },
      },
    }
  },
}
