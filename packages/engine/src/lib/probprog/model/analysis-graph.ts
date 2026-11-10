/**
 * The graph side of a spec's structure: the Markov blanket of each latent (its parents, children and the other parents of
 * its children, so what a single-site update must read), the connected components of the interaction graph (independent
 * sub-problems: the posterior factorises over them), the discrete part (the finite discrete latents with the continuous
 * ones held fixed: its interaction graph, an elimination order from `omai-engine/pgm`, the induced width and what summing
 * the sites out costs) and the plate axes along which the observations are exchangeable and can be taken in minibatches.
 * Everything is read from the spec, at the level of nodes: a node over a plate is one vertex, with the dimensions that
 * decide whether its elements interact.
 *
 * @index function graphStructure: Markov blankets, independent components, discrete-part width and minibatch axes of a spec.
 */

import {
  connectedComponents,
  eliminationOrder,
  inducedWidth,
  type UndirectedGraph,
  undirectedGraph,
} from '../../pgm/index.ts'
import { type LatentMemo, latentsIn, type NodesByName, nodeExprs } from './analysis-exprs.ts'
import { exprShape } from './check.ts'
import { distributionFamilies } from './dists/families.ts'
import { canonicalDims } from './ids.ts'
import { defaultOperators, type NamedShape } from './operators.ts'
import type { ExprSpec, ModelNode, ModelSpec } from './spec.ts'

/** What a single-site update of a latent has to read. */
export interface SiteBlanket {
  readonly site: string
  /** The latents its distribution depends on. */
  readonly parents: readonly string[]
  /** The nodes (latent, observed or factor) whose density depends on it. */
  readonly children: readonly string[]
  /** The other latents its children depend on. */
  readonly coParents: readonly string[]
  /** Parents, children and co-parents together. */
  readonly blanket: readonly string[]
  /**
   * The dimensions of the site along which the blanket factorises: each element sees only its own element of every child
   * (no reduction over the dimension, no lookup by index), so a plate of independent updates.
   */
  readonly perElement: readonly string[]
}

/** A set of latents that interact with no other: the posterior factorises over components. */
export interface IndependentComponent {
  readonly latents: readonly string[]
  readonly observed: readonly string[]
}

/** The interaction of finite discrete latents once the continuous ones are held fixed, for one connected part. */
export interface DiscreteComponent {
  readonly sites: readonly string[]
  /** The induced width of the elimination order (the largest elimination clique minus one). */
  readonly width: number
  /** An elimination order (the better of min-fill and min-degree). */
  readonly order: readonly string[]
  /** The number of joint states of the largest clique of one element, when every cardinality is known. */
  readonly cliqueStates?: number
  /** The number of states of each site, where known. */
  readonly cardinality: Readonly<Record<string, number>>
  /** Whether the elements along a site's plate are independent (no reduction over the plate), so a sum costs per element. */
  readonly elementwise: boolean
  /** The dimensions the elements interact along when they do not factorise. */
  readonly coupledAlong: readonly string[]
  /** The number of elements of the largest site's plate, when known. */
  readonly elements?: number
}

/** The finite discrete latents and how coupled they are. */
export interface DiscretePart {
  readonly sites: readonly string[]
  readonly components: readonly DiscreteComponent[]
  /** The largest induced width among the components (0 when none interact), or `-1` when there are no discrete latents. */
  readonly width: number
  /** The concatenated elimination orders. */
  readonly orderHint: readonly string[]
}

/** A plate dimension along which the observations are exchangeable: a minibatch of it is an unbiased estimate. */
export interface MinibatchAxis {
  readonly dim: string
  readonly size?: number
  /** The observed nodes over it. */
  readonly observed: readonly string[]
  /** The latents over it (local to each element: a minibatch updates only the sampled ones). */
  readonly local: readonly string[]
  /** The latents not over it (shared by every element). */
  readonly shared: readonly string[]
}

/** Everything {@link graphStructure} reads of a spec. */
export interface GraphEnv {
  readonly spec: ModelSpec
  readonly byName: NodesByName
  readonly memo: LatentMemo
  readonly sizeOf: (dim: string) => number | undefined
}

/** What it finds. */
export interface GraphStructure {
  readonly blankets: readonly SiteBlanket[]
  readonly components: readonly IndependentComponent[]
  readonly discretePart: DiscretePart
  readonly minibatchAxes: readonly MinibatchAxis[]
}

/** The dimensions an expression reduces or looks up along, for the subexpressions that satisfy `pick`. */
function reductions(
  e: ExprSpec,
  scope: ReadonlyMap<string, NamedShape>,
  table: ReturnType<typeof defaultOperators>,
  pick: (x: ExprSpec) => boolean,
  out: { reduced: Set<string>; indexed: Set<string> },
): void {
  if (e.kind === 'op') {
    for (const a of e.args) reductions(a, scope, table, pick, out)
    if (!pick(e)) return
    try {
      const args = e.args.map((a) => exprShape(a, scope, table))
      if (e.op === 'at') {
        const first = args[0]?.dims[0]
        if (first !== undefined) out.indexed.add(first)
        return
      }
      const result = exprShape(e, scope, table)
      for (const s of args) for (const d of s.dims) if (!result.dims.includes(d)) out.reduced.add(d)
    } catch {
      // A shape the registry cannot infer here is not a reduction we can name.
    }
  } else if (e.kind === 'custom') for (const a of e.args) reductions(a, scope, table, pick, out)
}

function cardinalityOf(n: ModelNode): number | undefined {
  if (n.kind !== 'latent') return undefined
  const { family, params } = n.dist
  if (family === 'bernoulli') return 2
  const constant = (name: string): ExprSpec | undefined => params[name]
  if (family === 'binomial') {
    const p = constant('n')
    return p?.kind === 'const' && typeof p.value === 'number' ? p.value + 1 : undefined
  }
  if (family === 'categorical') {
    for (const k of ['probs', 'logits']) {
      let v = constant(k)?.kind === 'const' ? (constant(k) as { value: unknown }).value : undefined
      while (Array.isArray(v) && Array.isArray(v[0])) v = v[0]
      if (Array.isArray(v)) return v.length
    }
  }
  return undefined
}

/**
 * Read the graph structure of a spec: blankets, independent components, the discrete part and the minibatch axes.
 *
 * @param env - The spec, its value nodes by name, the memo of derived nodes' latents and the sizes of dimensions.
 * @returns The structure.
 */
export function graphStructure(env: GraphEnv): GraphStructure {
  const { spec, byName, memo } = env
  const families = distributionFamilies()
  const table = defaultOperators()
  const scope = new Map<string, NamedShape>()
  for (const n of spec.nodes)
    if (n.kind !== 'observed' && n.kind !== 'factor') scope.set(n.name, { dims: n.dims, eventRank: n.eventRank })
  const dimsOf = (n: ModelNode): string[] => canonicalDims(spec, n.dims)
  const latents = spec.nodes.filter((n) => n.kind === 'latent')
  const isDensity = (n: ModelNode): boolean => n.kind === 'latent' || n.kind === 'observed' || n.kind === 'factor'
  const parentsOf = new Map<string, string[]>()
  for (const n of spec.nodes) {
    if (!isDensity(n)) continue
    const ps = new Set<string>()
    for (const e of nodeExprs(n)) for (const l of latentsIn(e, byName, memo)) if (l !== n.name) ps.add(l)
    parentsOf.set(`${n.kind}:${n.name}`, [...ps])
  }
  const key = (n: ModelNode): string => `${n.kind}:${n.name}`
  const childrenOf = new Map<string, ModelNode[]>()
  for (const n of spec.nodes) {
    if (!isDensity(n)) continue
    for (const p of parentsOf.get(key(n)) ?? []) childrenOf.set(p, [...(childrenOf.get(p) ?? []), n])
  }

  // Blankets.
  const blankets: SiteBlanket[] = latents.map((x) => {
    const parents = parentsOf.get(key(x)) ?? []
    const kids = childrenOf.get(x.name) ?? []
    const co = new Set<string>()
    for (const c of kids) for (const p of parentsOf.get(key(c)) ?? []) if (p !== x.name) co.add(p)
    for (const p of parents) co.delete(p)
    const own = dimsOf(x)
    const perElement = own.filter((d) => {
      if (!kids.every((c) => dimsOf(c).includes(d))) return false
      const out = { reduced: new Set<string>(), indexed: new Set<string>() }
      for (const c of kids)
        for (const e of nodeExprs(c))
          reductions(e, scope, table, (sub) => latentsIn(sub, byName, memo).has(x.name), out)
      return !out.reduced.has(d) && !out.indexed.has(d)
    })
    return {
      site: x.name,
      parents,
      children: kids.map((c) => c.name),
      coParents: [...co],
      blanket: [...new Set([...parents, ...kids.map((c) => c.name), ...co])],
      perElement,
    }
  })

  // Components of the interaction graph over the latents.
  const scopes: string[][] = []
  for (const n of spec.nodes) {
    if (!isDensity(n)) continue
    const s = [...(n.kind === 'latent' ? [n.name] : []), ...(parentsOf.get(key(n)) ?? [])]
    scopes.push(s)
  }
  const link = (g: Map<string, Set<string>>, a: string, b: string): void => {
    g.get(a)?.add(b)
    g.get(b)?.add(a)
  }
  const graph = undirectedGraph(
    [],
    latents.map((l) => l.name),
  )
  for (const s of scopes)
    for (let i = 0; i < s.length; i++)
      for (let j = i + 1; j < s.length; j++) link(graph, s[i] as string, s[j] as string)
  const comps = connectedComponents(graph)
  const components: IndependentComponent[] = comps.map((c) => {
    const set = new Set(c)
    const observed = spec.nodes
      .filter((n) => n.kind === 'observed' && (parentsOf.get(key(n)) ?? []).some((p) => set.has(p)))
      .map((n) => n.name)
    return { latents: c, observed }
  })

  // The discrete part: finite discrete latents, the continuous ones held fixed.
  const discrete = latents.filter((n) => n.kind === 'latent' && families[n.dist.family]?.finite === true)
  const dnames = new Set(discrete.map((n) => n.name))
  const dgraph: UndirectedGraph = (() => {
    const g = undirectedGraph(
      [],
      discrete.map((n) => n.name),
    )
    for (const s of scopes) {
      const t = s.filter((v) => dnames.has(v))
      for (let i = 0; i < t.length; i++) for (let j = i + 1; j < t.length; j++) link(g, t[i] as string, t[j] as string)
    }
    return g
  })()
  const card: Record<string, number> = {}
  for (const n of discrete) {
    const k = cardinalityOf(n)
    if (k !== undefined) card[n.name] = k
  }
  const dcomponents: DiscreteComponent[] = connectedComponents(dgraph).map((sites) => {
    const sub = undirectedGraph([], sites)
    for (const v of sites) for (const w of dgraph.get(v) ?? []) link(sub, v, w)
    const candidates = (['minFill', 'minDegree'] as const).map((h) => {
      const order = eliminationOrder(sub, sites, h, card)
      return { order, width: inducedWidth(sub, order) }
    })
    const best = candidates.reduce((a, b) => (b.width < a.width ? b : a))
    // The elements of a site interact when a reduction over its plate depends on it.
    const coupled = new Set<string>()
    for (const v of sites) {
      const x = byName.get(v)
      if (x === undefined) continue
      const out = { reduced: new Set<string>(), indexed: new Set<string>() }
      for (const c of childrenOf.get(v) ?? [])
        for (const e of nodeExprs(c)) reductions(e, scope, table, (sub2) => latentsIn(sub2, byName, memo).has(v), out)
      for (const d of dimsOf(x)) if (out.reduced.has(d)) coupled.add(d)
    }
    const known = sites.every((s) => card[s] !== undefined)
    let cliqueStates: number | undefined
    if (known) {
      // The product of the cardinalities of the largest clique along the order.
      let work = new Map([...sub].map(([v, ns]) => [v, new Set(ns)]))
      cliqueStates = 1
      for (const v of best.order) {
        const nb = [...(work.get(v) ?? [])]
        cliqueStates = Math.max(
          cliqueStates,
          [v, ...nb].reduce((p, w) => p * (card[w] ?? 1), 1),
        )
        for (const a of nb) {
          for (const b of nb) if (a !== b) work.get(a)?.add(b)
          work.get(a)?.delete(v)
        }
        work.delete(v)
      }
      work = new Map()
    }
    let elements: number | undefined = 1
    for (const v of sites) {
      const x = byName.get(v)
      let k = 1
      for (const d of x === undefined ? [] : dimsOf(x)) {
        const size = env.sizeOf(d)
        if (size === undefined) {
          k = Number.NaN
          break
        }
        k *= size
      }
      if (Number.isNaN(k)) {
        elements = undefined
        break
      }
      elements = Math.max(elements ?? 1, k)
    }
    return {
      sites,
      width: best.width,
      order: best.order,
      ...(cliqueStates === undefined ? {} : { cliqueStates }),
      cardinality: Object.fromEntries(sites.filter((s) => card[s] !== undefined).map((s) => [s, card[s] as number])),
      elementwise: coupled.size === 0,
      coupledAlong: [...coupled],
      ...(elements === undefined ? {} : { elements }),
    }
  })
  const discretePart: DiscretePart = {
    sites: discrete.map((n) => n.name),
    components: dcomponents,
    width: dcomponents.reduce((m, c) => Math.max(m, c.width), discrete.length === 0 ? -1 : 0),
    orderHint: dcomponents.flatMap((c) => c.order),
  }

  // Minibatch axes: dimensions of observed nodes that nothing reduces or looks up along.
  const all = { reduced: new Set<string>(), indexed: new Set<string>() }
  for (const n of spec.nodes)
    if (n.kind !== 'observed' && n.kind !== 'data')
      for (const e of nodeExprs(n)) reductions(e, scope, table, () => true, all)
  for (const n of spec.nodes)
    if (n.kind === 'observed') for (const e of nodeExprs(n)) reductions(e, scope, table, () => true, all)
  for (const n of spec.nodes) if (n.kind === 'data' && n.indexes !== undefined) all.indexed.add(n.indexes)
  const axes: MinibatchAxis[] = []
  const seen = new Set<string>()
  for (const n of spec.nodes) {
    if (n.kind !== 'observed') continue
    for (const d of dimsOf(n)) {
      if (seen.has(d) || all.reduced.has(d) || all.indexed.has(d)) continue
      seen.add(d)
      const size = env.sizeOf(d)
      axes.push({
        dim: d,
        ...(size === undefined ? {} : { size }),
        observed: spec.nodes.filter((o) => o.kind === 'observed' && dimsOf(o).includes(d)).map((o) => o.name),
        local: latents.filter((l) => dimsOf(l).includes(d)).map((l) => l.name),
        shared: latents.filter((l) => !dimsOf(l).includes(d)).map((l) => l.name),
      })
    }
  }
  return { blankets, components, discretePart, minibatchAxes: axes }
}
