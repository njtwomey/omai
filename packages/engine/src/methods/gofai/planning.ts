/**
 * Classical planning played twice in the blocks world: the forward state-space search that finds the plan, expansion by
 * expansion with path costs and heuristic values, and then the plan carried out action by action, both drawn as blocks
 * on a table with the gripper above.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import {
  applyAction,
  aStarSearchAlgorithm,
  bestFirstSearchAlgorithm,
  breadthFirstSearchAlgorithm,
  forwardProblem,
  type PlanningHeuristic,
  type PlanState,
  type SearchNode,
  type SearchState,
  type StripsAction,
} from '../../lib/search/index.ts'
import { iterate } from '../../lib/trace/index.ts'
import { blocksWorld, type Towers, towersOf } from './planning-domains.ts'

/** Options of {@link planBlocks}. */
export interface BlocksPlanOptions {
  /** The search over states (default A*). */
  readonly search?: 'a-star' | 'greedy' | 'weighted-a-star' | 'breadth-first'
  /** The heuristic (default h-ff). */
  readonly heuristic?: PlanningHeuristic
  /** The weight of weighted A* (default 2). */
  readonly weight?: number
  /** Stop after this many expansions (default 20000). */
  readonly maxExpansions?: number
  /** Record at most this many search frames (default 200). */
  readonly maxFrames?: number
}

/** A plan for a blocks-world problem and what finding it took. */
export interface BlocksPlan {
  readonly status: 'solved' | 'failure' | 'cutoff' | 'budget'
  readonly plan: readonly string[]
  readonly cost: number
  readonly expanded: number
  readonly generated: number
}

function sample<T>(frames: readonly T[], cap: number): T[] {
  if (frames.length <= cap) return [...frames]
  return Array.from({ length: cap }, (_, i) => frames[Math.round((i * (frames.length - 1)) / (cap - 1))] as T)
}

/**
 * Plan in the blocks world by forward state-space search, recording two plays drawn as blocks on a table with the
 * gripper above: every state the search expands (with its path cost and heuristic, and the frontier's size), then the
 * plan carried out action by action (with the goal facts achieved).
 *
 * @param initial - The towers at the start, bottom to top.
 * @param goal - The towers wanted.
 * @param options - The search, heuristic, weight, budget and frames.
 * @returns The plan and the effort, with two `boardSteps` records: the search, then the plan.
 */
export function planBlocks(initial: Towers, goal: Towers, options: BlocksPlanOptions = {}): Explained<BlocksPlan> {
  const problem = blocksWorld(initial, goal)
  const blocks = [...initial.flat()].sort()
  const n = blocks.length
  if (n === 0) throw new MdmlError('value', 'planBlocks: no blocks')
  const p = forwardProblem(problem, options.heuristic ?? 'h-ff')
  const maxExpansions = options.maxExpansions ?? 20_000
  const o = { maxExpansions }
  const h = (n: SearchNode<PlanState, StripsAction>): number => p.heuristic?.(n.state) ?? 0
  const search = options.search ?? 'a-star'
  const algorithm =
    search === 'breadth-first'
      ? breadthFirstSearchAlgorithm(p, o)
      : search === 'greedy'
        ? bestFirstSearchAlgorithm(p, h, o)
        : search === 'weighted-a-star'
          ? bestFirstSearchAlgorithm(p, (x) => x.pathCost + (options.weight ?? 2) * h(x), o)
          : aStarSearchAlgorithm(p, o)
  // Rows: the gripper, n levels of towers (top first), the table. Columns: one per possible tower.
  const rows = n + 2
  const columns = n
  const categories = ['empty', 'hand', 'table', ...blocks]
  const draw = (state: PlanState): { cells: number[]; labels: string[] } => {
    const cells = new Array<number>(rows * columns).fill(0)
    const labels = new Array<string>(rows * columns).fill('')
    for (let c = 0; c < columns; c++) {
      cells[c] = 1
      cells[(rows - 1) * columns + c] = 2
    }
    const { towers, holding } = towersOf(state)
    if (holding !== null) {
      cells[0] = 3 + blocks.indexOf(holding)
      labels[0] = holding
    }
    for (const [c, t] of towers.entries())
      for (const [level, b] of t.entries()) {
        const i = (rows - 2 - level) * columns + c
        cells[i] = 3 + blocks.indexOf(b)
        labels[i] = b
      }
    return { cells, labels }
  }
  type Frame = { cells: number[]; labels: string[]; caption: string; expanded: number; frontier: number; h: number }
  const frames: Frame[] = []
  let last: SearchState<PlanState, StripsAction> | undefined
  for (const { t, state: s } of iterate(algorithm, { maxSteps: 2 * maxExpansions + 16 })) {
    const node = s.current
    const state = node?.state ?? (t === 0 ? p.initial : undefined)
    if (state !== undefined) {
      const hv = p.heuristic?.(state) ?? 0
      const caption =
        node === null
          ? `start: h = ${hv}`
          : `expand ${node.action === undefined ? 'the start' : `after ${node.action.name}`}: g = ${node.pathCost}, h = ${hv}`
      frames.push({ ...draw(state), caption, expanded: s.stats.expanded, frontier: s.frontierSize, h: hv })
    }
    last = s
  }
  if (last === undefined) throw new MdmlError('value', 'planBlocks: the search did not start')
  const status = last.status === 'running' ? 'budget' : last.status
  const solution = status === 'solved' ? last.solution : null
  const actions: StripsAction[] = []
  for (let x = solution; x !== null && x.action !== undefined; x = x.parent) actions.unshift(x.action)
  const kept = sample(frames, options.maxFrames ?? 200)
  const layout = { type: 'grid' as const, rows, columns }
  const searchRecord: BoardSteps = boardSteps({
    method: `${search} (${options.heuristic ?? 'h-ff'})`,
    layout,
    categories,
    steps: kept.map((f) => ({ cells: f.cells, labels: f.labels, caption: f.caption })),
    metrics: [
      { name: 'expanded', values: kept.map((f) => f.expanded) },
      { name: 'heuristic', values: kept.map((f) => f.h) },
    ],
    stepName: 'expansion',
    summary: `${status} after ${last.stats.expanded} expansions`,
  })
  const shown = (towers: Towers): string => towers.map((t) => `[${t.join(' ')}]`).join(' ')
  const goals = (s: PlanState): number => problem.goal.filter((g) => s.includes(g)).length
  const states: PlanState[] = [problem.initial]
  for (const a of actions) states.push(applyAction(states[states.length - 1] as PlanState, a))
  const planRecord: BoardSteps = boardSteps({
    method: 'plan',
    layout,
    categories,
    steps: states.map((s, i) => ({
      ...draw(s),
      caption: i === 0 ? `start; goal ${shown(goal)}` : `${i}. ${actions[i - 1]?.name ?? ''}`,
    })),
    metrics: [{ name: 'goal facts achieved', values: states.map(goals) }],
    stepName: 'action',
    summary: `${actions.length} actions`,
  })
  return {
    status,
    plan: actions.map((a) => a.name),
    cost: solution?.pathCost ?? Number.POSITIVE_INFINITY,
    expanded: last.stats.expanded,
    generated: last.stats.generated,
    exposition: [searchRecord, planRecord],
  }
}
