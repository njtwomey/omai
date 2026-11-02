/**
 * A layered (Sugiyama-style) layout for directed graphs, as dagre draws them: cycles broken by reversing edges, ranks
 * by longest path, long edges routed through virtual points, crossings reduced by repeated barycentre ordering with
 * groups kept together, and coordinates that keep the spacing. The default layout of structure diagrams.
 */

import {
  type Box,
  boxBoundary,
  type EdgeRoute,
  type GraphLayout,
  groupBoxes,
  type LayoutGraph,
  normaliseLayout,
} from './graph.ts'

/** Options of {@link layered}. */
export interface LayeredOptions {
  /** Ranks run top to bottom (`down`, the default) or left to right (`right`). */
  readonly direction?: 'down' | 'right'
  /** Space between ranks (default 56). */
  readonly rankSep?: number
  /** Space between neighbours in a rank (default 28). */
  readonly nodeSep?: number
  /** Ordering sweeps that reduce crossings (default 12). */
  readonly sweeps?: number
  /** Space around the drawing (default 16). */
  readonly margin?: number
}

interface Item {
  readonly id: string
  readonly width: number
  readonly height: number
  readonly group?: string
  readonly dummy: boolean
}

/** Remove cycles by reversing the edges that close them (depth first, in node order); returns the acyclic edges. */
function acyclic(ids: readonly string[], edges: readonly (readonly [string, string])[]): [string, string][] {
  const out = new Map<string, string[]>()
  for (const [a, b] of edges) out.set(a, [...(out.get(a) ?? []), b])
  const state = new Map<string, 1 | 2>()
  const back = new Set<string>()
  const visit = (v: string): void => {
    state.set(v, 1)
    for (const w of out.get(v) ?? []) {
      const s = state.get(w)
      if (s === 1) back.add(`${v}\u0000${w}`)
      else if (s === undefined) visit(w)
    }
    state.set(v, 2)
  }
  for (const v of ids) if (!state.has(v)) visit(v)
  return edges.map(([a, b]) => (back.has(`${a}\u0000${b}`) ? [b, a] : [a, b]))
}

/** Ranks by longest path from the sources. */
function ranks(ids: readonly string[], edges: readonly (readonly [string, string])[]): Map<string, number> {
  const preds = new Map<string, string[]>()
  for (const [a, b] of edges) if (a !== b) preds.set(b, [...(preds.get(b) ?? []), a])
  const rank = new Map<string, number>()
  const visiting = new Set<string>()
  const of = (v: string): number => {
    const known = rank.get(v)
    if (known !== undefined) return known
    if (visiting.has(v)) return 0
    visiting.add(v)
    const r = Math.max(-1, ...(preds.get(v) ?? []).map(of)) + 1
    visiting.delete(v)
    rank.set(v, r)
    return r
  }
  for (const v of ids) of(v)
  return rank
}

function crossings(layers: readonly string[][], down: ReadonlyMap<string, readonly string[]>): number {
  let total = 0
  for (let r = 0; r + 1 < layers.length; r++) {
    const pos = new Map((layers[r + 1] as string[]).map((v, i) => [v, i]))
    const pairs: [number, number][] = []
    ;(layers[r] as string[]).forEach((v, i) => {
      for (const w of down.get(v) ?? []) pairs.push([i, pos.get(w) ?? 0])
    })
    for (let i = 0; i < pairs.length; i++)
      for (let j = i + 1; j < pairs.length; j++) {
        const [a1, b1] = pairs[i] as [number, number]
        const [a2, b2] = pairs[j] as [number, number]
        if ((a1 - a2) * (b1 - b2) < 0) total++
      }
  }
  return total
}

/**
 * A layered (Sugiyama-style) layout for directed graphs, as dagre draws them: cycles broken by reversing edges, ranks
 * by longest path, long edges routed through virtual points, ranks ordered by repeated barycentre sweeps to reduce
 * crossings (members of a group kept together, for plates and blocks), and positions pulled towards neighbours while
 * keeping each rank's spacing. Deterministic: the same graph always gives the same layout.
 *
 * @param graph - Nodes with sizes (and groups), and edges.
 * @param options - Direction, spacing, sweeps and margin.
 * @returns Node boxes, edge routes (polylines through the ranks they cross), group boxes and the size.
 * @index function Lay out a directed graph in layers: `layered({ nodes, edges })`.
 * @example
 * ```ts
 * import { layered } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c'].map((id) => ({ id, width: 40, height: 20 })), edges: [{ from: 'a', to: 'b' }, { from: 'a', to: 'c' }] }
 * const l = layered(g)
 * l.nodes['a'].y < l.nodes['b'].y // => true
 * l.nodes['b'].y === l.nodes['c'].y // => true
 * ```
 */
export function layered(graph: LayoutGraph, options: LayeredOptions = {}): GraphLayout {
  const { direction = 'down', rankSep = 56, nodeSep = 28, sweeps = 12, margin = 16 } = options
  const right = direction === 'right'
  // Work as if ranks run downwards; swap the axes of the sizes for left-to-right.
  const items = new Map<string, Item>()
  for (const n of graph.nodes)
    items.set(n.id, {
      id: n.id,
      width: right ? n.height : n.width,
      height: right ? n.width : n.height,
      ...(n.group === undefined ? {} : { group: n.group }),
      dummy: false,
    })
  const ids = [...items.keys()]
  const real = graph.edges.filter((e) => items.has(e.from) && items.has(e.to) && e.from !== e.to)
  const dag = acyclic(
    ids,
    real.map((e) => [e.from, e.to] as const),
  )
  const rank = ranks(ids, dag)
  // Split long edges into chains of unit edges through dummy items.
  const down = new Map<string, string[]>()
  const up = new Map<string, string[]>()
  const link = (a: string, b: string): void => {
    down.set(a, [...(down.get(a) ?? []), b])
    up.set(b, [...(up.get(b) ?? []), a])
  }
  const chains: string[][] = []
  // The ends of each dummy's edge: a dummy sits inside the groups of either end (an edge into a block stays in it).
  const dummyEnds = new Map<string, readonly [string, string]>()
  dag.forEach(([a, b], k) => {
    const ra = rank.get(a) ?? 0
    const rb = rank.get(b) ?? 0
    const chain = [a]
    for (let r = ra + 1; r < rb; r++) {
      const d = `\u0000${k}:${r}`
      items.set(d, { id: d, width: 0, height: 0, dummy: true })
      dummyEnds.set(d, [a, b])
      rank.set(d, r)
      chain.push(d)
    }
    chain.push(b)
    for (let i = 0; i + 1 < chain.length; i++) link(chain[i] as string, chain[i + 1] as string)
    chains.push(chain)
  })
  const depth = Math.max(0, ...rank.values()) + 1
  let layers: string[][] = Array.from({ length: depth }, () => [])
  // Initial order: depth first from the sources, in node order.
  const seen = new Set<string>()
  const dfs = (v: string): void => {
    if (seen.has(v)) return
    seen.add(v)
    layers[rank.get(v) ?? 0]?.push(v)
    for (const w of down.get(v) ?? []) dfs(w)
  }
  for (const v of ids) if ((up.get(v) ?? []).length === 0) dfs(v)
  for (const v of items.keys()) dfs(v)

  // A node's groups from its own outwards; ordering keeps the members of each outermost group together.
  const parentOf = new Map((graph.groups ?? []).flatMap((g) => (g.parent === undefined ? [] : [[g.id, g.parent]])))
  const chainOf = (g: string | undefined): string[] => {
    const out: string[] = []
    while (g !== undefined && !out.includes(g) && out.length < 32) {
      out.push(g)
      g = parentOf.get(g)
    }
    return out
  }
  const groupsOf = new Map([...items.values()].map((it) => [it.id, chainOf(it.group)]))
  const groupOf = (v: string): string | undefined => groupsOf.get(v)?.at(-1)
  const reorder = (layer: string[], neighbours: ReadonlyMap<string, readonly string[]>, ref: string[]): string[] => {
    const pos = new Map(ref.map((v, i) => [v, i]))
    const bary = new Map<string, number>()
    layer.forEach((v, i) => {
      const ns = (neighbours.get(v) ?? []).map((w) => pos.get(w)).filter((p): p is number => p !== undefined)
      bary.set(v, ns.length === 0 ? i : ns.reduce((a, b) => a + b, 0) / ns.length)
    })
    // Keep a group's members together: order by the group's mean barycentre, then by the member's own.
    const groupMean = new Map<string, number>()
    for (const v of layer) {
      const g = groupOf(v)
      if (g === undefined) continue
      const members = layer.filter((w) => groupOf(w) === g)
      groupMean.set(g, members.reduce((a, w) => a + (bary.get(w) ?? 0), 0) / members.length)
    }
    const key = (v: string): number => {
      const g = groupOf(v)
      return g === undefined ? (bary.get(v) ?? 0) : (groupMean.get(g) ?? 0)
    }
    return [...layer].sort((a, b) => key(a) - key(b) || (bary.get(a) ?? 0) - (bary.get(b) ?? 0))
  }
  // Members of a group sit together in each rank, gathered where the group's first member is.
  const gather = (layer: readonly string[]): string[] => {
    const out: string[] = []
    const done = new Set<string>()
    for (const v of layer) {
      const g = groupOf(v)
      if (g === undefined) out.push(v)
      else if (!done.has(g)) {
        done.add(g)
        out.push(...layer.filter((w) => groupOf(w) === g))
      }
    }
    return out
  }
  layers = layers.map(gather)
  let best = layers.map((l) => [...l])
  let bestCross = crossings(best, down)
  for (let s = 0; s < sweeps && bestCross > 0; s++) {
    if (s % 2 === 0) for (let r = 1; r < depth; r++) layers[r] = reorder(layers[r] ?? [], up, layers[r - 1] ?? [])
    else for (let r = depth - 2; r >= 0; r--) layers[r] = reorder(layers[r] ?? [], down, layers[r + 1] ?? [])
    const c = crossings(layers, down)
    if (c < bestCross) {
      bestCross = c
      best = layers.map((l) => [...l])
    }
  }
  layers = best

  // Coordinates: pack each rank, then pull nodes towards the mean of their neighbours, keeping the spacing.
  const x = new Map<string, number>()
  const halfW = (v: string): number => (items.get(v)?.width ?? 0) / 2
  const gap = (a: string, b: string): number =>
    halfW(a) + halfW(b) + (items.get(a)?.dummy || items.get(b)?.dummy ? nodeSep / 2 : nodeSep)
  for (const layer of layers) {
    let at = 0
    layer.forEach((v, i) => {
      if (i > 0) at += gap(layer[i - 1] as string, v)
      x.set(v, at)
    })
    const mid = at / 2
    for (const v of layer) x.set(v, (x.get(v) ?? 0) - mid)
  }
  const settle = (layer: string[], want: Map<string, number>): void => {
    // Place left to right at the wanted positions, never closer than the gap; then right to left the same way, and
    // average, so the rank keeps its order and spacing while following its neighbours.
    const l2r: number[] = []
    layer.forEach((v, i) => {
      const w = want.get(v) ?? x.get(v) ?? 0
      l2r.push(i === 0 ? w : Math.max(w, (l2r[i - 1] as number) + gap(layer[i - 1] as string, v)))
    })
    const r2l: number[] = new Array(layer.length).fill(0)
    for (let i = layer.length - 1; i >= 0; i--) {
      const v = layer[i] as string
      const w = want.get(v) ?? x.get(v) ?? 0
      r2l[i] = i === layer.length - 1 ? w : Math.min(w, (r2l[i + 1] as number) - gap(v, layer[i + 1] as string))
    }
    layer.forEach((v, i) => {
      x.set(v, ((l2r[i] as number) + (r2l[i] as number)) / 2)
    })
    // The average can break spacing where the two passes disagree; one last left-to-right pass restores it.
    for (let i = 1; i < layer.length; i++) {
      const prev = x.get(layer[i - 1] as string) ?? 0
      const v = layer[i] as string
      x.set(v, Math.max(x.get(v) ?? 0, prev + gap(layer[i - 1] as string, v)))
    }
  }
  for (let pass = 0; pass < 8; pass++) {
    const towards = pass % 2 === 0 ? up : down
    const order = pass % 2 === 0 ? layers : [...layers].reverse()
    for (const layer of order) {
      const want = new Map<string, number>()
      for (const v of layer) {
        const ns = towards.get(v) ?? []
        if (ns.length > 0) want.set(v, ns.reduce((a, w) => a + (x.get(w) ?? 0), 0) / ns.length)
      }
      settle(layer, want)
    }
  }
  // Groups are drawn as boxes around their members across ranks, so a node outside a group (or in another group) that
  // sits in a rank the group spans must clear the group's extent: push it, and its neighbours beyond it, out of the way.
  const boxesOf = (v: string): string[] => {
    const ends = dummyEnds.get(v)
    if (ends === undefined) return groupsOf.get(v) ?? []
    return [...new Set([...(groupsOf.get(ends[0]) ?? []), ...(groupsOf.get(ends[1]) ?? [])])]
  }
  const members = new Map<string, string[]>()
  for (const v of items.keys())
    if (!items.get(v)?.dummy) for (const g of groupsOf.get(v) ?? []) members.set(g, [...(members.get(g) ?? []), v])
  for (let round = 0; round < 6; round++) {
    let moved = false
    for (const [g, ms] of members) {
      const lo = Math.min(...ms.map((v) => (x.get(v) ?? 0) - halfW(v)))
      const hi = Math.max(...ms.map((v) => (x.get(v) ?? 0) + halfW(v)))
      const ranks = new Set(ms.map((v) => rank.get(v) ?? 0))
      // The group's box reaches 12 px past its members per level of nesting inside it.
      const depth = Math.max(...ms.map((v) => (groupsOf.get(v) ?? []).indexOf(g)))
      const ancestors = new Set(chainOf(g))
      for (const r of ranks) {
        const layer = layers[r] ?? []
        const own = layer.map((v, i) => (boxesOf(v).includes(g) ? i : -1)).filter((i) => i >= 0)
        layer.forEach((v, i) => {
          if (boxesOf(v).includes(g)) return
          const ownBoxes = boxesOf(v).filter((h) => !ancestors.has(h)).length
          const pad = 12 * (depth + 1) + 12 * ownBoxes + (items.get(v)?.dummy ? 4 : 10)
          const xv = x.get(v) ?? 0
          if (xv + halfW(v) <= lo - pad || xv - halfW(v) >= hi + pad) return
          // Out on the side it is ordered on (or, beside no member, the nearer side), with everything beyond it.
          const after = own.length > 0 ? i > (own[0] ?? 0) : xv >= (lo + hi) / 2
          const delta = after ? hi + pad + halfW(v) - xv : lo - pad - halfW(v) - xv
          const shifted = after ? layer.slice(i) : layer.slice(0, i + 1)
          for (const w of shifted) x.set(w, (x.get(w) ?? 0) + delta)
          moved = true
        })
      }
    }
    if (!moved) break
  }

  // Rank depths: each rank as tall as its tallest node.
  const rankY: number[] = []
  let y = 0
  layers.forEach((layer, r) => {
    const h = Math.max(0, ...layer.map((v) => items.get(v)?.height ?? 0))
    if (r > 0) y += rankSep
    rankY.push(y + h / 2)
    y += h
  })
  const place = (v: string): Box => {
    const it = items.get(v) as Item
    const bx = x.get(v) ?? 0
    const by = rankY[rank.get(v) ?? 0] ?? 0
    return right
      ? { x: by, y: bx, width: it.height, height: it.width }
      : { x: bx, y: by, width: it.width, height: it.height }
  }
  const nodes: Record<string, Box> = {}
  for (const v of ids) nodes[v] = place(v)
  // Routes: through the dummies of each chain, clipped to the end boxes; reversed edges are turned back.
  const routes: EdgeRoute[] = []
  let k = 0
  for (const e of real) {
    const chain = chains[k] as string[]
    k++
    const pts = chain.map((v): [number, number] => {
      const b = place(v)
      return [b.x, b.y]
    })
    const forward = chain[0] === e.from
    const ordered = forward ? pts : [...pts].reverse()
    const a = nodes[e.from] as Box
    const b = nodes[e.to] as Box
    const first = ordered[1] ?? [b.x, b.y]
    const last = ordered[ordered.length - 2] ?? [a.x, a.y]
    routes.push({
      from: e.from,
      to: e.to,
      points: [boxBoundary(a, first), ...ordered.slice(1, -1), boxBoundary(b, last)],
    })
  }
  return normaliseLayout(nodes, routes, groupBoxes(graph, nodes), margin)
}
