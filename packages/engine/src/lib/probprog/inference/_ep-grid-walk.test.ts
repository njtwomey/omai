// The walk of the EP grid on synthetic evidence: a peak whose tails fail (the EP run has no evidence there) must not make the walk
// flood the failing lattice before the peak is covered.
import { describe, expect, it } from 'vitest'
import { type WalkNode, type WalkState, walkNext } from './_ep-grid-walk.ts'

const SIDE = 41
const MIDDLE = 20

/** A peak at the middle of a `SIDE` by `SIDE` lattice: `-r^2 / 4` within a radius of 6.5 lattice steps, a failed run (no evidence) outside it. */
function evidence(index: readonly number[]): number {
  const r2 = (index[0] as number) - MIDDLE
  const c2 = (index[1] as number) - MIDDLE
  const d2 = r2 * r2 + c2 * c2
  return d2 <= 6.5 * 6.5 ? -d2 / 4 : Number.NEGATIVE_INFINITY
}

/** Walk until the walk stops or the budget is used: the nodes it ran. */
function walk(window: number, maxNodes: number): readonly WalkNode[] {
  const nodes: WalkNode[] = []
  const axes = [
    { from: 0, to: SIDE - 1 },
    { from: 0, to: SIDE - 1 },
  ]
  for (;;) {
    const state: WalkState = { axes, nodes }
    const [next] = walkNext(state, window, maxNodes)
    if (next === undefined) return nodes
    nodes.push({ index: next, logEvidence: evidence(next), logPrior: 0 })
  }
}

describe('walkNext with a peak whose tails fail', () => {
  it('covers every node with mass within a budget that is far below the lattice, and stops there', () => {
    const budget = 400
    const nodes = walk(9, budget)
    const ran = new Set(nodes.map((n) => n.index.join(',')))
    const mass: string[] = []
    for (let i = 0; i < SIDE; i++)
      for (let j = 0; j < SIDE; j++) if (Number.isFinite(evidence([i, j]))) mass.push(`${i},${j}`)
    // the peak is covered: every node that has evidence was run
    expect(mass.filter((m) => !ran.has(m))).toEqual([])
    // the failed nodes run are the shell around the peak, not the lattice
    expect(nodes.length).toBeLessThan(budget)
    expect(nodes.length).toBeLessThan(mass.length + 4 * 2 * 14 + 40)
  })

  it('still expands failed nodes when no node has any evidence yet (it must find the mass somewhere)', () => {
    const failed: WalkNode = { index: [20, 20], logEvidence: Number.NEGATIVE_INFINITY, logPrior: 0 }
    const axes = [
      { from: 0, to: 40 },
      { from: 0, to: 40 },
    ]
    expect(walkNext({ axes, nodes: [failed] }, 9, 100)).toEqual([[19, 20]])
  })

  it('expands a failed node next to the mass after the nodes that have evidence (a hole in the peak is crossed)', () => {
    // a one-dimensional peak with a failed node at its middle: the nodes beyond it are reached through it
    const axes = [{ from: 0, to: 40 }]
    const at = (i: number, logEvidence: number): WalkNode => ({ index: [i], logEvidence, logPrior: 0 })
    const nodes = [at(20, -1), at(19, -0.5), at(21, Number.NEGATIVE_INFINITY)]
    // the free neighbours: 18 (of 19, with evidence) and 22 (of the failed 21, near the mass): the one with evidence first
    expect(walkNext({ axes, nodes }, 9, 100)).toEqual([[18]])
    // with the left side used up the failed node's neighbour is run
    const left = [...nodes, ...[18, 17, 16, 15].map((i) => at(i, -(20 - i) - 2))]
    expect(walkNext({ axes, nodes: left }, 3, 100)).toEqual([[22]])
  })
})
