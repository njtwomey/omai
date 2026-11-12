// The walk of `ep-grid`'s `order: 'walk'`: which lattice node to run next. It reads the lattice only through the axes' ranges and the
// nodes' position, evidence and prior, so it is tested on synthetic evidence (a peak with failing tails) without any model.

/** How many lattice steps from a node within the window a failed node may lie and still be expanded. */
const FAILED_REACH = 2

/** The part of an axis the walk reads: the range of point indices run so far. */
export interface WalkAxis {
  readonly from: number
  readonly to: number
}

/** The part of a node the walk reads. */
export interface WalkNode {
  readonly index: readonly number[]
  /** The EP evidence of the node; `-Infinity` (or not finite) when its run failed or has none. */
  readonly logEvidence: number
  readonly logPrior: number
}

/** The part of the grid's state the walk reads. */
export interface WalkState {
  readonly axes: readonly WalkAxis[]
  readonly nodes: readonly WalkNode[]
}

/**
 * The next node of a walk over the grid: the middle of the first range, then the unvisited lattice neighbour (one step along one
 * axis) of the node that weighs most (log evidence plus log prior) among those within `window` nats of the best. A node with no
 * evidence (its EP run failed) has nothing to judge by: while no node has any evidence it is expanded first (the walk must find the
 * mass somewhere), but once the walk has found mass a failed node is expanded only when it lies within `FAILED_REACH` lattice steps
 * of a node within the window of the best, and then after every node that has evidence: a tail where EP fails is the edge of the
 * region, not a place to flood, and a failed node inside the mass (a hole) is still crossed. With one hyperparameter this steps
 * outwards on the side whose edge node weighs more; with several it is a best-first flood fill of the region where the posterior
 * has mass, so its cost follows the number of nodes within the window and not the product of the axes. None when the region is
 * closed, the range is used up or the budget is.
 *
 * @param state - The axes and the nodes run so far.
 * @param window - How many nats below the best a node may weigh and still be expanded.
 * @param maxNodes - The budget of nodes: none is returned at it.
 * @returns The lattice index to run next, as a list of one, or an empty list.
 */
export function walkNext(state: WalkState, window: number, maxNodes: number): readonly (readonly number[])[] {
  const axes = state.axes
  if (axes.length === 0 || state.nodes.length >= maxNodes) return []
  if (state.nodes.length === 0) return [axes.map((a) => Math.floor((a.from + a.to) / 2))]
  const score = (n: WalkNode): number => n.logEvidence + n.logPrior
  let top = Number.NEGATIVE_INFINITY
  for (const n of state.nodes) if (Number.isFinite(n.logEvidence) && score(n) > top) top = score(n)
  // a lattice point as one number (the axes' ranges make a mixed-radix code), so a step costs a few set lookups per node
  const span = axes.map((a) => a.to - a.from + 1)
  const code = (index: readonly number[]): number =>
    index.reduce((c, v, k) => c * (span[k] as number) + (v - (axes[k] as WalkAxis).from), 0)
  const visited = new Set(state.nodes.map((n) => code(n.index)))
  const live = new Set<number>()
  for (const n of state.nodes) if (Number.isFinite(n.logEvidence) && score(n) > top - window) live.add(code(n.index))
  const reach = (n: WalkNode): boolean => {
    // a lattice point within FAILED_REACH steps (summed over the axes) of a live node, found by stepping out from `n`
    let frontier: readonly (readonly number[])[] = [n.index]
    for (let step = 0; step < FAILED_REACH; step++) {
      const out: number[][] = []
      for (const index of frontier)
        for (const [k, a] of axes.entries())
          for (const side of [-1, 1]) {
            const at = (index[k] ?? 0) + side
            if (at < a.from || at > a.to) continue
            const next = index.slice()
            next[k] = at
            if (live.has(code(next))) return true
            out.push(next)
          }
      frontier = out
    }
    return false
  }
  let best: number[] | undefined
  let bestRank = Number.NEGATIVE_INFINITY
  for (const n of state.nodes) {
    const finite = Number.isFinite(n.logEvidence)
    if (finite && score(n) <= top - window) continue
    // nothing has evidence yet: a failed node is all there is to expand; otherwise it ranks below every node that has evidence
    if (!finite && Number.isFinite(top) && !reach(n)) continue
    const rank = finite ? score(n) : Number.isFinite(top) ? -Number.MAX_VALUE : Number.POSITIVE_INFINITY
    for (const [k, a] of axes.entries()) {
      for (const side of [-1, 1]) {
        const at = (n.index[k] ?? 0) + side
        if (at < a.from || at > a.to) continue
        const next = n.index.slice()
        next[k] = at
        if (visited.has(code(next))) continue
        // the heavier parent first; on a tie the smaller index (the left edge of an interval)
        if (rank > bestRank || (rank === bestRank && best !== undefined && lexLess(next, best))) {
          best = next
          bestRank = rank
        }
      }
    }
  }
  return best === undefined ? [] : [best]
}

/** Whether `a` comes before `b` in lexicographic order. */
function lexLess(a: readonly number[], b: readonly number[]): boolean {
  for (const [i, v] of a.entries()) if (v !== (b[i] ?? 0)) return v < (b[i] ?? 0)
  return false
}
