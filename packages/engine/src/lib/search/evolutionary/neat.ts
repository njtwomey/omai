/**
 * NEAT, NeuroEvolution of Augmenting Topologies (Stanley and Miikkulainen 2002), at a compact scale: genomes of nodes
 * and connection genes that carry innovation numbers, so that two genomes can be aligned gene by gene whatever their
 * history; structural mutations that add a connection or split one with a new node, starting from a minimal network;
 * speciation by a compatibility distance (excess and disjoint genes and the mean weight difference), so that new
 * structures compete within their niche until their weights are tuned; explicit fitness sharing and the crossover of
 * aligned genes. Networks are feedforward. XOR, which no network without a hidden unit solves, is the built-in truth.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { type Key, uniformStream } from '../../random/index.ts'
import { type Algorithm, type Contract, defineContract, event, scalar, tensor } from '../../trace/index.ts'

/** A node gene. */
export interface NeatNode {
  readonly id: number
  readonly kind: 'input' | 'bias' | 'output' | 'hidden'
}

/** A connection gene: its innovation number identifies the structural change that created it. */
export interface NeatConnection {
  readonly innovation: number
  readonly from: number
  readonly to: number
  readonly weight: number
  readonly enabled: boolean
}

/** A genome with the fitness it was last given and the species it belongs to. */
export interface NeatGenome {
  readonly nodes: readonly NeatNode[]
  readonly connections: readonly NeatConnection[]
  readonly fitness: number
  readonly species: number
}

/** The paths the NEAT algorithm logs. */
export type NeatPath =
  | 'generation/best'
  | 'generation/mean'
  | 'generation/species'
  | 'generation/nodes'
  | 'generation/connections'
  | 'structure/added'

/** The contract fragment of {@link neatAlgorithm}. */
export const neatContract: Contract<NeatState, NeatPath> = defineContract({
  'generation/best': scalar({ description: 'the best fitness in the generation', direction: 'higher' }),
  'generation/mean': scalar({ description: 'the mean fitness', direction: 'higher' }),
  'generation/species': tensor({ role: 'state', description: 'the size of each species', shape: ['species'] }),
  'generation/nodes': scalar({ description: 'the mean number of nodes per genome' }),
  'generation/connections': scalar({ description: 'the mean number of enabled connections per genome' }),
  'structure/added': event({ context: { kind: "'node' or 'connection'", innovation: 'its innovation number' } }),
})

/** The state of a NEAT run: the population, the innovation record and the species. */
export interface NeatState {
  readonly generation: number
  readonly population: readonly NeatGenome[]
  /** Innovations seen so far as `[from, to, innovation]`, so that the same change gets the same number. */
  readonly innovations: readonly (readonly [number, number, number])[]
  /** The node each split connection created, as `[innovation, node id]`. */
  readonly splits: readonly (readonly [number, number])[]
  readonly nextInnovation: number
  readonly nextNode: number
  /** One representative genome per species, with the species id. */
  readonly representatives: readonly { readonly id: number; readonly genome: NeatGenome }[]
  readonly nextSpecies: number
  readonly best: NeatGenome
  readonly bestFitness: number
  readonly solved: boolean
}

/** A feedforward network built from a genome: inputs in, outputs out. */
export type NeatNetwork = (inputs: readonly number[]) => number[]

/**
 * The network a genome describes: nodes in topological order of the enabled connections, each hidden and output node
 * computing `sigmoid(4.9 * sum of weighted inputs)`; the bias node is 1. A node with no enabled input gives 0.5.
 *
 * @param genome - The genome (its enabled connections must be acyclic, as mutation keeps them).
 * @returns The network.
 * @example
 * ```ts
 * import { neatNetwork } from 'mdml-engine/search'
 * const g = { nodes: [{ id: 0, kind: 'input' }, { id: 1, kind: 'bias' }, { id: 2, kind: 'output' }], connections: [{ innovation: 0, from: 0, to: 2, weight: 1, enabled: true }], fitness: 0, species: 0 } as const
 * neatNetwork(g)([0])[0] // => 0.5
 * ```
 */
export function neatNetwork(genome: NeatGenome): NeatNetwork {
  const inputs = genome.nodes.filter((n) => n.kind === 'input')
  const bias = genome.nodes.find((n) => n.kind === 'bias')
  const outputs = genome.nodes.filter((n) => n.kind === 'output')
  const incoming = new Map<number, NeatConnection[]>()
  for (const c of genome.connections) if (c.enabled) incoming.set(c.to, [...(incoming.get(c.to) ?? []), c])
  return (x) => {
    const value = new Map<number, number>()
    for (const [i, n] of inputs.entries()) value.set(n.id, x[i] ?? 0)
    if (bias !== undefined) value.set(bias.id, 1)
    const evaluate = (id: number): number => {
      const known = value.get(id)
      if (known !== undefined) return known
      let sum = 0
      for (const c of incoming.get(id) ?? []) sum += c.weight * evaluate(c.from)
      const out = 1 / (1 + Math.exp(-4.9 * sum))
      value.set(id, out)
      return out
    }
    return outputs.map((n) => evaluate(n.id))
  }
}

/** The four XOR cases: inputs and the target output. */
export const XOR_CASES: readonly (readonly [readonly [number, number], number])[] = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
]

/**
 * The XOR fitness of the original paper: `(4 - sum of absolute errors)^2` over the four cases, at most 16.
 *
 * @param network - The network to score.
 * @returns The fitness.
 * @example
 * ```ts
 * import { xorFitness } from 'mdml-engine/search'
 * xorFitness(([a, b]) => [Number((a ?? 0) !== (b ?? 0))]) // => 16
 * ```
 */
export function xorFitness(network: NeatNetwork): number {
  let error = 0
  for (const [x, target] of XOR_CASES) error += Math.abs((network(x)[0] ?? 0) - target)
  return (4 - error) ** 2
}

/**
 * Whether a network computes XOR: every output on the right side of one half.
 *
 * @param network - The network.
 * @returns Whether all four cases are classified correctly.
 * @example
 * ```ts
 * import { solvesXor } from 'mdml-engine/search'
 * solvesXor(([a, b]) => [Number((a ?? 0) !== (b ?? 0))]) // => true
 * ```
 */
export function solvesXor(network: NeatNetwork): boolean {
  return XOR_CASES.every(([x, target]) => (network(x)[0] ?? 0) > 0.5 === (target === 1))
}

/** Options of {@link neatAlgorithm}. */
export interface NeatOptions {
  /** Inputs excluding the bias (default 2). */
  readonly inputs?: number
  /** Outputs (default 1). */
  readonly outputs?: number
  /** The fitness of a network (default {@link xorFitness}). */
  readonly fitness?: (network: NeatNetwork) => number
  /** The fitness at which the run counts as solved (default 15.5 for XOR; use `Infinity` to never stop). */
  readonly target?: number
  /** Genomes per generation (default 150). */
  readonly populationSize?: number
  /** Compatibility coefficients for excess genes, disjoint genes and mean weight difference (defaults 1, 1, 0.4). */
  readonly coefficients?: readonly [number, number, number]
  /** Genomes closer than this to a species' representative join it (default 3). */
  readonly compatibilityThreshold?: number
  /** Probabilities of mutating weights, adding a connection and adding a node (defaults 0.8, 0.1, 0.05). */
  readonly mutation?: { readonly weights?: number; readonly connection?: number; readonly node?: number }
  /** The fraction of each species that breeds (default 0.3). */
  readonly survival?: number
  /** The probability that an offspring comes from crossover rather than mutation alone (default 0.75). */
  readonly crossover?: number
}

interface Rng {
  next(): number
  normal(): number
  int(n: number): number
}

function rngOf(key: Key): Rng {
  const u = uniformStream(key)
  return {
    next: u,
    normal: () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u()),
    int: (n) => Math.min(n - 1, Math.floor(u() * n)),
  }
}

/** Whether `start` reaches `target` along the connection genes, disabled ones included (crossover can re-enable them). */
function reaches(g: NeatGenome, start: number, target: number): boolean {
  const seen = new Set<number>()
  const stack = [start]
  while (stack.length > 0) {
    const at = stack.pop() as number
    if (at === target) return true
    if (seen.has(at)) continue
    seen.add(at)
    for (const c of g.connections) if (c.from === at) stack.push(c.to)
  }
  return false
}

/**
 * NEAT as an algorithm, one generation per step: score every genome, speciate the population by the compatibility
 * distance `c1 E / N + c2 D / N + c3 W` (excess genes, disjoint genes, mean weight difference of matching genes), share
 * fitness within species, keep the champion of each large species, and breed the next generation by crossover of aligned
 * genes (the fitter parent's excess and disjoint genes) and mutation: perturbed weights, new connections between nodes
 * that keep the graph acyclic, and new nodes that split a connection (the old one disabled, the incoming new one weighted 1,
 * the outgoing one with the old weight). The same structural change gets the same innovation number everywhere.
 *
 * @param options - The problem (default XOR), population, speciation and mutation settings.
 * @returns The algorithm; the state holds the population, the best genome and whether it solved the task.
 * @throws MdmlError with code `value` for a population below 4.
 * @concept neuroevolution
 * @example
 * ```ts
 * import { neatAlgorithm } from 'mdml-engine/search'
 * import { key } from 'mdml-engine/random'
 * const alg = neatAlgorithm({ populationSize: 20 })
 * alg.init(key(0)).population.length // => 20
 * ```
 */
export function neatAlgorithm(options: NeatOptions = {}): Algorithm<NeatState, NeatPath> {
  const nIn = options.inputs ?? 2
  const nOut = options.outputs ?? 1
  const fitnessOf = options.fitness ?? xorFitness
  const target = options.target ?? 15.5
  const size = options.populationSize ?? 150
  const [c1, c2, c3] = options.coefficients ?? [1, 1, 0.4]
  const threshold = options.compatibilityThreshold ?? 3
  const pWeights = options.mutation?.weights ?? 0.8
  const pConnection = options.mutation?.connection ?? 0.1
  const pNode = options.mutation?.node ?? 0.05
  const survival = options.survival ?? 0.3
  const pCrossover = options.crossover ?? 0.75
  if (size < 4) throw new MdmlError('value', 'neat: the population needs at least 4 genomes')
  const biasId = nIn
  const nodes: NeatNode[] = [
    ...Array.from({ length: nIn }, (_, i) => ({ id: i, kind: 'input' as const })),
    { id: biasId, kind: 'bias' as const },
    ...Array.from({ length: nOut }, (_, i) => ({ id: nIn + 1 + i, kind: 'output' as const })),
  ]
  const innovationOf = (
    s: { innovations: [number, number, number][]; next: number },
    from: number,
    to: number,
  ): number => {
    const known = s.innovations.find((x) => x[0] === from && x[1] === to)
    if (known !== undefined) return known[2]
    s.innovations.push([from, to, s.next])
    return s.next++
  }
  const distance = (a: NeatGenome, b: NeatGenome): number => {
    const mapA = new Map(a.connections.map((c) => [c.innovation, c]))
    const mapB = new Map(b.connections.map((c) => [c.innovation, c]))
    const maxA = Math.max(-1, maxOf([...mapA.keys()]))
    const maxB = Math.max(-1, maxOf([...mapB.keys()]))
    let excess = 0
    let disjoint = 0
    let matching = 0
    let diff = 0
    for (const [i, c] of mapA) {
      const o = mapB.get(i)
      if (o !== undefined) {
        matching += 1
        diff += Math.abs(c.weight - o.weight)
      } else if (i > maxB) excess += 1
      else disjoint += 1
    }
    for (const i of mapB.keys()) {
      if (mapA.has(i)) continue
      if (i > maxA) excess += 1
      else disjoint += 1
    }
    const n =
      Math.max(a.connections.length, b.connections.length) < 20
        ? 1
        : Math.max(a.connections.length, b.connections.length)
    return (c1 * excess) / n + (c2 * disjoint) / n + c3 * (matching === 0 ? 0 : diff / matching)
  }
  return {
    name: 'neat',
    config: { inputs: nIn, outputs: nOut, populationSize: size, threshold, survival, crossover: pCrossover },
    init: (key) => {
      const rng = rngOf(key)
      const book = { innovations: [] as [number, number, number][], next: 0 }
      const population: NeatGenome[] = Array.from({ length: size }, () => ({
        nodes,
        connections: nodes
          .filter((n) => n.kind !== 'output')
          .flatMap((from) =>
            nodes
              .filter((n) => n.kind === 'output')
              .map((to) => ({
                innovation: innovationOf(book, from.id, to.id),
                from: from.id,
                to: to.id,
                weight: 2 * rng.normal(),
                enabled: true,
              })),
          ),
        fitness: 0,
        species: -1,
      }))
      return {
        generation: 0,
        population,
        innovations: book.innovations,
        splits: [],
        nextInnovation: book.next,
        nextNode: nIn + 1 + nOut,
        representatives: [],
        nextSpecies: 0,
        best: population[0] as NeatGenome,
        bestFitness: Number.NEGATIVE_INFINITY,
        solved: false,
      }
    },
    step: (s, ctx) => {
      const rng = rngOf(ctx.key)
      // 1. Score, and keep the best.
      const scored = s.population.map((g) => ({ ...g, fitness: fitnessOf(neatNetwork(g)) }))
      let best = scored[0] as NeatGenome
      for (const g of scored) if (g.fitness > best.fitness) best = g
      // 2. Speciate against the representatives of the last generation, in order.
      const reps = s.representatives.map((r) => ({ ...r }))
      let nextSpecies = s.nextSpecies
      const species = new Map<number, NeatGenome[]>()
      for (const g of scored) {
        let home = reps.find((r) => distance(g, r.genome) < threshold)
        if (home === undefined) {
          home = { id: nextSpecies++, genome: g }
          reps.push(home)
        }
        species.set(home.id, [...(species.get(home.id) ?? []), { ...g, species: home.id }])
      }
      // A species with no members this generation is dropped; the others get a new random representative.
      const alive = [...species.entries()]
      const representatives = alive.map(([id, gs]) => ({ id, genome: gs[rng.int(gs.length)] as NeatGenome }))
      // 3. Offspring per species in proportion to its mean shared fitness.
      const shared = alive.map(([, gs]) => gs.reduce((a, g) => a + g.fitness, 0) / gs.length)
      const total = shared.reduce((a, b) => a + b, 0)
      const quotas = shared.map((f) => (total > 0 ? (f / total) * size : size / alive.length))
      const counts = quotas.map(Math.floor)
      let left = size - counts.reduce((a, b) => a + b, 0)
      const order = quotas.map((q, i) => [q - Math.floor(q), i] as const).sort((a, b) => b[0] - a[0])
      for (let i = 0; left > 0; i = (i + 1) % order.length, left--)
        counts[order[i]?.[1] ?? 0] = (counts[order[i]?.[1] ?? 0] ?? 0) + 1
      // 4. Breed.
      const book = {
        innovations: [...s.innovations.map((x) => [...x] as [number, number, number])],
        next: s.nextInnovation,
      }
      const splits = s.splits.map((x) => [...x] as [number, number])
      let nextNode = s.nextNode
      const added: { kind: string; innovation: number }[] = []
      const mutate = (g: NeatGenome): NeatGenome => {
        let connections = [...g.connections]
        let nodesOf = [...g.nodes]
        if (rng.next() < pWeights)
          connections = connections.map((c) =>
            rng.next() < 0.9 ? { ...c, weight: c.weight + 0.5 * rng.normal() } : { ...c, weight: 2 * rng.normal() },
          )
        if (rng.next() < pConnection) {
          const from = nodesOf[rng.int(nodesOf.length)] as NeatNode
          const to = nodesOf[rng.int(nodesOf.length)] as NeatNode
          const present = connections.some((c) => c.from === from.id && c.to === to.id)
          const legal = to.kind !== 'input' && to.kind !== 'bias' && from.kind !== 'output' && !present
          if (legal && !reaches({ ...g, connections }, to.id, from.id)) {
            const innovation = innovationOf(book, from.id, to.id)
            connections.push({ innovation, from: from.id, to: to.id, weight: 2 * rng.normal(), enabled: true })
            added.push({ kind: 'connection', innovation })
          }
        }
        if (rng.next() < pNode) {
          const open = connections.filter((c) => c.enabled)
          const split = open[rng.int(open.length)]
          if (split !== undefined) {
            let id = splits.find((x) => x[0] === split.innovation)?.[1]
            if (id === undefined) {
              id = nextNode++
              splits.push([split.innovation, id])
            }
            if (!nodesOf.some((n) => n.id === id)) {
              nodesOf = [...nodesOf, { id, kind: 'hidden' }]
              const into = innovationOf(book, split.from, id)
              const out = innovationOf(book, id, split.to)
              connections = connections.map((c) => (c.innovation === split.innovation ? { ...c, enabled: false } : c))
              connections.push({ innovation: into, from: split.from, to: id, weight: 1, enabled: true })
              connections.push({ innovation: out, from: id, to: split.to, weight: split.weight, enabled: true })
              added.push({ kind: 'node', innovation: into })
            }
          }
        }
        return { ...g, nodes: nodesOf, connections }
      }
      const cross = (a: NeatGenome, b: NeatGenome): NeatGenome => {
        // a is the fitter parent: its excess and disjoint genes are inherited; matching genes come from either.
        const other = new Map(b.connections.map((c) => [c.innovation, c]))
        const connections = a.connections.map((c) => {
          const o = other.get(c.innovation)
          const pick = o !== undefined && rng.next() < 0.5 ? o : c
          const off = o !== undefined && (!c.enabled || !o.enabled)
          return { ...pick, enabled: off ? rng.next() < 0.25 : pick.enabled }
        })
        return { ...a, connections }
      }
      const next: NeatGenome[] = []
      for (const [k, [, gs]] of alive.entries()) {
        const ranked = [...gs].sort((x, y) => y.fitness - x.fitness)
        const parents = ranked.slice(0, Math.max(1, Math.ceil(ranked.length * survival)))
        let quota = counts[k] ?? 0
        if (quota > 0 && ranked.length >= 5) {
          next.push({ ...(ranked[0] as NeatGenome), fitness: 0 })
          quota -= 1
        }
        for (; quota > 0; quota--) {
          const a = parents[rng.int(parents.length)] as NeatGenome
          let child = a
          if (rng.next() < pCrossover && parents.length > 1) {
            const b = parents[rng.int(parents.length)] as NeatGenome
            child = a.fitness >= b.fitness ? cross(a, b) : cross(b, a)
          }
          next.push({ ...mutate(child), fitness: 0, species: -1 })
        }
      }
      while (next.length < size) next.push({ ...mutate(best), fitness: 0, species: -1 })
      const sizes = alive.map(([, gs]) => gs.length)
      ctx.log('generation/best', best.fitness)
      ctx.log('generation/mean', scored.reduce((a, g) => a + g.fitness, 0) / scored.length)
      ctx.log('generation/species', sizes)
      ctx.log('generation/nodes', scored.reduce((a, g) => a + g.nodes.length, 0) / scored.length)
      ctx.log(
        'generation/connections',
        scored.reduce((a, g) => a + g.connections.filter((c) => c.enabled).length, 0) / scored.length,
      )
      for (const a of added) ctx.emit('structure/added', a)
      return {
        generation: s.generation + 1,
        population: next.slice(0, size),
        innovations: book.innovations,
        splits,
        nextInnovation: book.next,
        nextNode,
        representatives,
        nextSpecies,
        best: best.fitness >= s.bestFitness ? best : s.best,
        bestFitness: Math.max(best.fitness, s.bestFitness),
        solved: best.fitness >= target,
      }
    },
    done: (s) => (s.solved ? 'completed' : undefined),
  }
}
