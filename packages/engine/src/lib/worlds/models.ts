import { WhiteboxError } from '../base/index.ts'
import { type Key, uniform } from '../random/index.ts'
import {
  ACTIONS,
  type ActionSet,
  type AgentState,
  applyRules,
  type ModelOptions,
  type Rule,
  slotActions,
  trackedItems,
} from './rules.ts'
import { cellCoords, passable, type World } from './world.ts'

/**
 * One outcome of an action in a tabular model: gymnasium's `P[s][a]` entries (probability, next state, reward,
 * terminated), plus the path cost the rules assigned. Structurally an `Outcome<number>` of `whitebox-engine/rl`.
 */
export interface WorldOutcome {
  readonly probability: number
  readonly state: number
  readonly reward: number
  readonly terminated: boolean
  readonly cost: number
}

/**
 * A finite transition model of a world under rules: states are numbered `slot + slots * (heading + headings *
 * items)` (a cell's slot, then the heading if actions turn, then the held items), and every action of every state
 * has its outcomes. It feeds tabular MDPs (`whitebox-engine/rl`'s `tabularEnv` takes `outcomes` as its transitions),
 * planners and exact solvers.
 *
 * @concept markov-decision-processes
 */
export interface TransitionModel {
  readonly world: World
  readonly rules: readonly Rule[]
  readonly actions: ActionSet
  readonly nStates: number
  readonly nActions: number
  /** The cell of each slot (every cell, or only the passable ones). */
  readonly cells: readonly number[]
  /** The items tracked in states: `tracked[j]` is the index in `world.items` of bit `j`. */
  readonly tracked: readonly number[]
  /** The initial state distribution. */
  readonly initial: Float64Array
  stateIndex(state: AgentState): number
  state(index: number): AgentState
  outcomes(state: number, action: number): readonly WorldOutcome[]
  /** Whether a state is terminal (absorbing when the model absorbs). */
  terminal(state: number): boolean
}

const turns = (a: ActionSet): boolean => a.actions.some((x) => x.kind === 'turn' || x.kind === 'forward')

/**
 * The stochastic transition model of a world under rules: every outcome of every action, as gymnasium's `P` tables.
 * Terminal states (by the rules' `terminal` hooks) and impassable cells loop on themselves with reward 0 and
 * `terminated` set, unless `absorbing` is false.
 *
 * @param w - The world.
 * @param ruleSet - The rules, in order.
 * @param options - Actions, which cells are states, merging, absorption, edge clamping, the initial distribution and
 *   move options.
 * @returns The model.
 * @throws WhiteboxError with code `value` with more than 16 tracked items, or no start when `initial` is `'starts'`.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { fromAscii, rules, stepReward, terminateOn, transitionModel } from 'whitebox-engine/worlds'
 * const m = transitionModel(fromAscii(['S.G']), rules(stepReward(-1), terminateOn('goals')))
 * [m.nStates, m.nActions] // => [3, 4]
 * m.outcomes(1, 1) // => [{ probability: 1, state: 2, reward: -1, terminated: true, cost: 1 }]
 * m.outcomes(2, 0) // => [{ probability: 1, state: 2, reward: 0, terminated: true, cost: 0 }]
 * ```
 */
export function transitionModel(w: World, ruleSet: readonly Rule[], options: ModelOptions = {}): TransitionModel {
  const actions =
    options.actions ??
    (w.topology.kind === 'square' ? ACTIONS.urdl : w.topology.kind === 'hex' ? ACTIONS.hex : slotActions(w))
  const opts: ModelOptions = { ...options, actions }
  const tracked = trackedItems(w)
  if (tracked.length > 16) throw new WhiteboxError('value', 'transitionModel: at most 16 tracked keys and pickups')
  const cells: number[] = []
  for (let c = 0; c < w.n; c++) if (options.cells !== 'passable' || passable(w, c)) cells.push(c)
  const slotOf = new Int32Array(w.n).fill(-1)
  for (const [i, c] of cells.entries()) slotOf[c] = i
  const headings = turns(actions) ? (w.topology.kind === 'hex' ? 6 : 4) : 1
  const slots = cells.length
  const nStates = slots * headings * 2 ** tracked.length
  const stateIndex = (s: AgentState): number => {
    const slot = slotOf[s.cell] ?? -1
    if (slot < 0) throw new WhiteboxError('value', `transitionModel: cell ${s.cell} is not a state`)
    return slot + slots * ((headings > 1 ? s.heading : 0) + headings * s.items)
  }
  const state = (i: number): AgentState => {
    if (!Number.isInteger(i) || i < 0 || i >= nStates) throw new WhiteboxError('value', `transitionModel: state ${i}`)
    const slot = i % slots
    const rest = Math.floor(i / slots)
    return { cell: cells[slot] ?? 0, items: Math.floor(rest / headings), heading: headings > 1 ? rest % headings : -1 }
  }
  const isTerminal = (s: AgentState): boolean => !passable(w, s.cell) || ruleSet.some((r) => r.terminal?.(w, s))
  const outcomes = (i: number, a: number): WorldOutcome[] => {
    const s = state(i)
    if (options.absorbing !== false && isTerminal(s))
      return [{ probability: 1, state: i, reward: 0, terminated: true, cost: 0 }]
    const raw = applyRules(w, ruleSet, s, a, opts).map((e) => ({
      probability: e.probability,
      state: stateIndex({ cell: e.cell, items: e.items, heading: e.heading }),
      reward: e.reward,
      terminated: e.terminated,
      cost: e.cost,
    }))
    if (options.merge !== true) return raw
    const merged = new Map<string, WorldOutcome>()
    for (const o of raw) {
      const k = `${o.state}|${o.reward}|${o.terminated}|${o.cost}`
      const prev = merged.get(k)
      merged.set(k, prev === undefined ? o : { ...prev, probability: prev.probability + o.probability })
    }
    return [...merged.values()]
  }
  const heading0 = headings > 1 ? (options.heading ?? (w.topology.kind === 'hex' ? 0 : 1)) : -1
  const initial = new Float64Array(nStates)
  const starts =
    options.initial === 'uniform'
      ? cells.filter((c) => passable(w, c) && !isTerminal({ cell: c, items: 0, heading: heading0 }))
      : w.starts
  if (starts.length === 0) throw new WhiteboxError('value', 'transitionModel: the world has no start')
  for (const c of starts) {
    const i = stateIndex({ cell: c, items: 0, heading: heading0 })
    initial[i] = (initial[i] ?? 0) + 1 / starts.length
  }
  return {
    world: w,
    rules: ruleSet,
    actions,
    nStates,
    nActions: actions.actions.length,
    cells,
    tracked,
    initial,
    stateIndex,
    state,
    outcomes,
    terminal: (i) => isTerminal(state(i)),
  }
}

/**
 * Draw one outcome of an action (a simulator step): the outcomes' probabilities against one uniform draw.
 *
 * @param model - The transition model.
 * @param state - The state index.
 * @param action - The action.
 * @param key - The key.
 * @returns The outcome drawn.
 * @example
 * ```ts
 * import { fromAscii, rules, slip, transitionModel, sampleOutcome } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * const m = transitionModel(fromAscii(['S..']), rules(slip([[-1, 0.5], [1, 0.5]])))
 * [0, 2].includes(sampleOutcome(m, 1, 0, key(3)).state) // => true
 * ```
 */
export function sampleOutcome(model: TransitionModel, state: number, action: number, key: Key): WorldOutcome {
  const os = model.outcomes(state, action)
  const u = uniform(key, []).data[0] ?? 0
  let acc = 0
  for (const o of os) {
    acc += o.probability
    if (u < acc) return o
  }
  const last = os[os.length - 1]
  if (last === undefined) throw new WhiteboxError('value', 'sampleOutcome: the action has no outcomes')
  return last
}

/**
 * Draw a start state from the model's initial distribution.
 *
 * @param model - The transition model.
 * @param key - The key.
 * @returns The state index.
 * @example
 * ```ts
 * import { fromAscii, sampleInitial, transitionModel } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * sampleInitial(transitionModel(fromAscii(['.S.']), []), key(0)) // => 1
 * ```
 */
export function sampleInitial(model: TransitionModel, key: Key): number {
  const u = uniform(key, []).data[0] ?? 0
  let acc = 0
  let last = 0
  for (let i = 0; i < model.nStates; i++) {
    const p = model.initial[i] ?? 0
    if (p <= 0) continue
    acc += p
    last = i
    if (u < acc) return i
  }
  return last
}

/**
 * A deterministic model of a world under rules, shaped as a search problem (`whitebox-engine/search`'s
 * `SearchProblem<AgentState, number>`): the start, the actions that change the state, their results and costs, the
 * goal test (a goal cell), a state key and a label. Use it for breadth-first search, A* and planning.
 *
 * @concept state-space-search
 */
export interface DeterministicModel {
  readonly initial: AgentState
  readonly starts: readonly AgentState[]
  actions(state: AgentState): readonly number[]
  result(state: AgentState, action: number): AgentState
  stepCost(state: AgentState, action: number, next: AgentState): number
  reward(state: AgentState, action: number): number
  isGoal(state: AgentState): boolean
  isTerminal(state: AgentState): boolean
  key(state: AgentState): number
  label(state: AgentState): string
}

/**
 * The deterministic model of a world under rules (no slips, no stochastic wind): each action has exactly one effect.
 * `actions` lists the actions that change the state (bumps and no-ops are left out) and none in terminal states.
 *
 * @param w - The world.
 * @param ruleSet - The rules (deterministic).
 * @param options - Actions and move options, as {@link transitionModel}.
 * @returns The model.
 * @throws WhiteboxError with code `value` when the world has no start, or (when queried) an action branches.
 * @concept state-space-search
 * @example
 * ```ts
 * import { deterministicModel, fromAscii } from 'whitebox-engine/worlds'
 * const m = deterministicModel(fromAscii(['S.', '.G']), [])
 * m.actions(m.initial) // => [1, 2]
 * m.isGoal(m.result(m.result(m.initial, 1), 2)) // => true
 * ```
 */
export function deterministicModel(w: World, ruleSet: readonly Rule[], options: ModelOptions = {}): DeterministicModel {
  const tm = transitionModel(w, ruleSet, { ...options, absorbing: true })
  const one = (s: AgentState, a: number): WorldOutcome => {
    const os = tm.outcomes(tm.stateIndex(s), a)
    const o = os[0]
    if (o === undefined || os.length !== 1 || o.probability !== 1) {
      throw new WhiteboxError('value', `deterministicModel: action ${a} branches in cell ${s.cell}`)
    }
    return o
  }
  const starts: AgentState[] = []
  for (let i = 0; i < tm.nStates; i++) if ((tm.initial[i] ?? 0) > 0) starts.push(tm.state(i))
  const first = starts[0]
  if (first === undefined) throw new WhiteboxError('value', 'deterministicModel: the world has no start')
  const label = (s: AgentState): string => {
    const where = w.topology.kind === 'graph' ? String(s.cell) : `(${cellCoords(w, s.cell).join(', ')})`
    const extra = [s.heading >= 0 ? `heading ${s.heading}` : '', s.items > 0 ? `items ${s.items}` : '']
    return [where, ...extra.filter((x) => x !== '')].join(' ')
  }
  return {
    initial: first,
    starts,
    actions: (s) => {
      const i = tm.stateIndex(s)
      if (tm.terminal(i)) return []
      const out: number[] = []
      for (let a = 0; a < tm.nActions; a++) if (one(s, a).state !== i) out.push(a)
      return out
    },
    result: (s, a) => tm.state(one(s, a).state),
    stepCost: (s, a) => one(s, a).cost,
    reward: (s, a) => one(s, a).reward,
    isGoal: (s) => w.goals.includes(s.cell),
    isTerminal: (s) => tm.terminal(tm.stateIndex(s)),
    key: (s) => tm.stateIndex(s),
    label,
  }
}
