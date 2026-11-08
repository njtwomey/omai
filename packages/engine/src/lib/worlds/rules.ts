/**
 * Rules that turn a world into a task: agent states (cell, held items, heading), the standard action sets, and
 * composable rules for rewards, terrain costs, termination, teleports (as CliffWalking's cliff), slippery and noisy
 * moves, Sutton and Barto's wind, one-way cells, pickups and locked doors, resolved for one action by `applyRules`.
 *
 * @index function Composable rules turning a world into a task.
 */

import { OmaiError } from '../base/index.ts'
import { offset, SQUARE_OFFSETS } from './_topology.ts'
import { cellCoords, cellIndex, destination, type MoveOptions, passable, terrainOf, type World } from './world.ts'

/**
 * Where an agent is under a set of rules: its cell, the bitmask of tracked items it holds (bit `j` for the `j`-th key
 * or pickup of the world, in item order; doors are never tracked) and its heading (a side index, or -1 when the
 * actions do not turn).
 *
 * @concept grid-worlds
 */
export interface AgentState {
  readonly cell: number
  readonly items: number
  readonly heading: number
}

/** One action: a move in a fixed direction, staying put, turning by some sides, or moving in the heading. */
export type WorldAction =
  | { readonly kind: 'move'; readonly direction: number }
  | { readonly kind: 'stay' }
  | { readonly kind: 'turn'; readonly by: number }
  | { readonly kind: 'forward' }

/** A named, ordered list of actions (the action space of an environment). */
export interface ActionSet {
  readonly names: readonly string[]
  readonly actions: readonly WorldAction[]
}

const moves = (names: readonly string[], directions: readonly number[]): ActionSet => ({
  names,
  actions: directions.map((d): WorldAction => (d < 0 ? { kind: 'stay' } : { kind: 'move', direction: d })),
})

/**
 * The usual action sets: `urdl` (up, right, down, left: CliffWalking, the AIMA grid, four rooms), `ldru` (left, down,
 * right, up: FrozenLake), `king` (`urdl` then the diagonals NE, SE, SW, NW), `kingStay` (`king` and stay), `hex`
 * (the six hexagonal sides E, SE, SW, W, NW, NE) and `turning` (MiniGrid's turn left, turn right, forward).
 */
export const ACTIONS: {
  readonly urdl: ActionSet
  readonly ldru: ActionSet
  readonly king: ActionSet
  readonly kingStay: ActionSet
  readonly hex: ActionSet
  readonly turning: ActionSet
} = {
  urdl: moves(['up', 'right', 'down', 'left'], [0, 1, 2, 3]),
  ldru: moves(['left', 'down', 'right', 'up'], [3, 2, 1, 0]),
  king: moves(
    ['up', 'right', 'down', 'left', 'up-right', 'down-right', 'down-left', 'up-left'],
    [0, 1, 2, 3, 4, 5, 6, 7],
  ),
  kingStay: moves(
    ['up', 'right', 'down', 'left', 'up-right', 'down-right', 'down-left', 'up-left', 'stay'],
    [0, 1, 2, 3, 4, 5, 6, 7, -1],
  ),
  hex: moves(['E', 'SE', 'SW', 'W', 'NW', 'NE'], [0, 1, 2, 3, 4, 5]),
  turning: {
    names: ['left', 'right', 'forward'],
    actions: [{ kind: 'turn', by: -1 }, { kind: 'turn', by: 1 }, { kind: 'forward' }],
  },
}

/**
 * The actions of a graph world: one move per neighbour slot, up to the largest degree (a slot a node lacks bumps).
 *
 * @param w - The world.
 * @returns The action set (`slot 0`, `slot 1`, ...).
 * @example
 * ```ts
 * import { slotActions, world } from 'omai-engine/worlds'
 * slotActions(world({ topology: { kind: 'graph', n: 3, neighbours: [[1, 2], [0], [0]] } })).names // => ['slot 0', 'slot 1']
 * ```
 */
export function slotActions(w: World): ActionSet {
  const t = w.topology
  let k = 0
  if (t.kind === 'graph') for (const list of t.neighbours) k = Math.max(k, list.length)
  else k = t.kind === 'square' ? 4 : 6
  return moves(
    Array.from({ length: k }, (_, i) => `slot ${i}`),
    Array.from({ length: k }, (_, i) => i),
  )
}

/** One way an action can play out: the action actually taken (after slips) and an extra push in rows and columns. */
export interface Move {
  readonly probability: number
  readonly action: number
  /** Extra displacement `[rows, cols]` added to the step (wind); square grids only. */
  readonly push: readonly [number, number]
}

/** What a move does: the probability of this branch, the resulting state, and its reward, cost and termination. */
export interface Effect {
  readonly probability: number
  readonly cell: number
  readonly items: number
  readonly heading: number
  readonly reward: number
  readonly cost: number
  readonly terminated: boolean
}

/** What a rule sees when the agent enters a cell. */
export interface EntryContext {
  readonly world: World
  /** The state before the move. */
  readonly state: AgentState
  /** The action actually taken (after slips). */
  readonly action: number
  /** The direction moved (-1 for staying and turning). */
  readonly direction: number
  readonly from: number
  /** The cell entered (the starting cell when the move bumped). */
  readonly to: number
  /** Whether the move was blocked. */
  readonly bumped: boolean
  /** The model's tracked items: `tracked[j]` is the index in `world.items` of bit `j`. */
  readonly tracked: readonly number[]
}

/**
 * A rule: a named piece of what happens in a world. A rule may rewrite how an action plays out (`moves`: slips,
 * wind), forbid entering cells (`allows`: one-way cells, locked doors), rewrite what entering does (`enter`: costs,
 * rewards, termination, teleports, pickups; it may branch into weighted effects) and mark states terminal
 * (`terminal`: absorbing states). Rules apply in order, so a rule set reads like the rules of the game.
 *
 * @concept grid-worlds
 */
export interface Rule {
  readonly name: string
  moves?(moves: readonly Move[], world: World, state: AgentState, actions: ActionSet): readonly Move[]
  allows?(
    world: World,
    state: AgentState,
    from: number,
    to: number,
    direction: number,
    tracked: readonly number[],
  ): boolean
  enter?(effect: Effect, context: EntryContext): Effect | readonly Effect[]
  terminal?(world: World, state: AgentState): boolean
}

/** Which cells a rule applies to: listed cells, the world's goals, starts or hazards, a terrain, or a predicate. */
export type CellSelector =
  | readonly number[]
  | 'goals'
  | 'starts'
  | 'hazards'
  | { readonly terrain: string }
  | ((cell: number, world: World) => boolean)

function selector(select: CellSelector): (cell: number, world: World) => boolean {
  if (typeof select === 'function') return select
  if (select === 'goals') return (c, w) => w.goals.includes(c)
  if (select === 'starts') return (c, w) => w.starts.includes(c)
  if (select === 'hazards') return (c, w) => c >= 0 && c < w.n && terrainOf(w, c).hazard
  if (Array.isArray(select)) {
    const set = new Set(select)
    return (c) => set.has(c)
  }
  const name = (select as { readonly terrain: string }).terrain
  return (c, w) => c >= 0 && c < w.n && terrainOf(w, c).name === name
}

/**
 * Compose rules and rule sets into one ordered rule set (nested sets are flattened).
 *
 * @param parts - Rules or rule sets.
 * @returns The rule set.
 * @example
 * ```ts
 * import { rules, stepReward, terminateOn } from 'omai-engine/worlds'
 * rules(stepReward(-1), [terminateOn('goals')]).map((r) => r.name) // => ['stepReward', 'terminateOn']
 * ```
 */
export function rules(...parts: readonly (Rule | readonly Rule[])[]): Rule[] {
  return parts.flatMap((p) => (Array.isArray(p) ? [...(p as readonly Rule[])] : [p as Rule]))
}

/**
 * Add a reward to every action (a living cost when negative).
 *
 * @param reward - The reward.
 * @returns The rule.
 * @example
 * ```ts
 * import { stepReward } from 'omai-engine/worlds'
 * stepReward(-1).name // => 'stepReward'
 * ```
 */
export function stepReward(reward: number): Rule {
  return { name: 'stepReward', enter: (e) => ({ ...e, reward: e.reward + reward }) }
}

/**
 * A reward for entering selected cells, added to the reward so far or replacing it.
 *
 * @param select - The cells.
 * @param reward - The reward.
 * @param options - `replace: true` to replace the reward so far (CliffWalking's -100 instead of the step's -1).
 * @returns The rule.
 * @example
 * ```ts
 * import { rewardOn } from 'omai-engine/worlds'
 * rewardOn('goals', 1).name // => 'rewardOn'
 * ```
 */
export function rewardOn(select: CellSelector, reward: number, options: { readonly replace?: boolean } = {}): Rule {
  const hit = selector(select)
  return {
    name: 'rewardOn',
    enter: (e, ctx) =>
      hit(e.cell, ctx.world) ? { ...e, reward: options.replace === true ? reward : e.reward + reward } : e,
  }
}

/**
 * Scale the cost of each move by the terrain cost of the cell entered (moves cost 1, diagonals `sqrt 2`, turns 1 and
 * staying 0 before this rule).
 *
 * @returns The rule.
 * @example
 * ```ts
 * import { terrainCost } from 'omai-engine/worlds'
 * terrainCost().name // => 'terrainCost'
 * ```
 */
export function terrainCost(): Rule {
  return {
    name: 'terrainCost',
    enter: (e, ctx) => (ctx.direction < 0 ? e : { ...e, cost: e.cost * terrainOf(ctx.world, e.cell).cost }),
  }
}

/**
 * End the episode on entering selected cells; states in them are terminal (absorbing in transition models).
 *
 * @param select - The cells (for example `'goals'` or `'hazards'`).
 * @returns The rule.
 * @example
 * ```ts
 * import { terminateOn } from 'omai-engine/worlds'
 * terminateOn('hazards').name // => 'terminateOn'
 * ```
 */
export function terminateOn(select: CellSelector): Rule {
  const hit = selector(select)
  return {
    name: 'terminateOn',
    enter: (e, ctx) => (hit(e.cell, ctx.world) ? { ...e, terminated: true } : e),
    terminal: (w, s) => hit(s.cell, w),
  }
}

/**
 * Teleport on entering selected cells: to a fixed cell, the first start, or a cell computed from the one entered
 * (CliffWalking's cliff sends the agent back to the start).
 *
 * @param select - The cells that teleport.
 * @param to - The destination cell, `'start'`, or a function of the cell entered.
 * @returns The rule.
 * @example
 * ```ts
 * import { teleport } from 'omai-engine/worlds'
 * teleport({ terrain: 'cliff' }, 'start').name // => 'teleport'
 * ```
 */
export function teleport(select: CellSelector, to: number | 'start' | ((cell: number, world: World) => number)): Rule {
  const hit = selector(select)
  return {
    name: 'teleport',
    enter: (e, ctx) => {
      if (!hit(e.cell, ctx.world)) return e
      const cell =
        to === 'start' ? (ctx.world.starts[0] ?? e.cell) : typeof to === 'number' ? to : to(e.cell, ctx.world)
      return { ...e, cell }
    },
  }
}

/**
 * Slippery moves: the action actually taken is the intended one shifted around the first `ring` actions of the set
 * (default 4) by each offset, with its probability, in the order given. FrozenLake is
 * `slip([[-1, 1/3], [0, 1/3], [1, 1/3]])`; the AIMA grid `slip([[0, 0.8], [1, 0.1], [-1, 0.1]])`. Branches with zero
 * probability are dropped; actions beyond the ring are unaffected.
 *
 * @param branches - `[offset, probability]` pairs (probabilities summing to 1).
 * @param ring - How many actions form the ring.
 * @returns The rule.
 * @throws OmaiError with code `value` when the probabilities do not sum to 1.
 * @example
 * ```ts
 * import { slip } from 'omai-engine/worlds'
 * slip([[-1, 0.25], [0, 0.5], [1, 0.25]]).name // => 'slip'
 * ```
 */
export function slip(branches: readonly (readonly [number, number])[], ring = 4): Rule {
  const total = branches.reduce((s, [, p]) => s + p, 0)
  if (Math.abs(total - 1) > 1e-12) throw new OmaiError('value', `slip: probabilities sum to ${total}, not 1`)
  return {
    name: 'slip',
    moves: (ms) =>
      ms.flatMap((m) =>
        m.action >= ring
          ? [m]
          : branches
              .filter(([, p]) => p > 0)
              .map(([k, p]) => ({
                ...m,
                action: (((m.action + k) % ring) + ring) % ring,
                probability: m.probability * p,
              })),
      ),
  }
}

/**
 * Noisy moves: the intended action with probability `success`, each other action of the ring with an equal share of
 * the rest, listed in action order (the four-rooms domain: 2/3 and 1/9 each).
 *
 * @param success - The probability of the intended action.
 * @param ring - How many actions form the ring (default 4).
 * @returns The rule.
 * @example
 * ```ts
 * import { slipUniform } from 'omai-engine/worlds'
 * slipUniform(2 / 3).name // => 'slipUniform'
 * ```
 */
export function slipUniform(success: number, ring = 4): Rule {
  const other = (1 - success) / (ring - 1)
  return {
    name: 'slipUniform',
    moves: (ms) =>
      ms.flatMap((m) =>
        m.action >= ring
          ? [m]
          : Array.from({ length: ring }, (_, a) => ({
              ...m,
              action: a,
              probability: m.probability * (a === m.action ? success : other),
            })).filter((x) => x.probability > 0),
      ),
  }
}

/**
 * Wind (Sutton and Barto's windy grid world): each move is pushed up by the strength of the column the agent stands
 * in; with `stochastic`, a blowing wind is one stronger, as given, or one weaker with probability 1/3 each. Pushed
 * moves are clamped to the grid.
 *
 * @param strengths - Upward push per column.
 * @param options - Stochastic gusts.
 * @returns The rule.
 * @example
 * ```ts
 * import { wind } from 'omai-engine/worlds'
 * wind([0, 1, 2]).name // => 'wind'
 * ```
 */
export function wind(strengths: readonly number[], options: { readonly stochastic?: boolean } = {}): Rule {
  return {
    name: 'wind',
    moves: (ms, w, s) => {
      const c = cellCoords(w, s.cell)[1]
      const strength = strengths[c] ?? 0
      const gusts = options.stochastic === true && strength > 0 ? [-1, 0, 1] : [0]
      return ms.flatMap((m) =>
        gusts.map((g) => ({
          ...m,
          probability: m.probability / gusts.length,
          push: [m.push[0] - (strength + g), m.push[1]] as const,
        })),
      )
    },
  }
}

/**
 * One-way cells: each listed cell may only be entered moving in its direction (in addition to one-way terrain, which
 * every move respects).
 *
 * @param cells - Cell to the only direction it may be entered in.
 * @returns The rule.
 * @example
 * ```ts
 * import { oneWay } from 'omai-engine/worlds'
 * oneWay({ 4: 1 }).name // => 'oneWay'
 * ```
 */
export function oneWay(cells: Readonly<Record<number, number>>): Rule {
  return {
    name: 'oneWay',
    allows: (_w, _s, _from, to, direction) => {
      const d = cells[to]
      return d === undefined || d === direction
    },
  }
}

const holds = (w: World, s: AgentState, tracked: readonly number[], kind: 'key', colour: string): boolean =>
  tracked.some((i, j) => ((s.items >>> j) & 1) === 1 && w.items[i]?.kind === kind && w.items[i]?.colour === colour)

/**
 * Pickups: entering a cell collects its uncollected keys and pickups (setting their bits in the state), adding each
 * pickup's reward (or `options.reward` for every item collected).
 *
 * @param options - A reward per item collected, replacing the items' own rewards.
 * @returns The rule.
 * @example
 * ```ts
 * import { pickups } from 'omai-engine/worlds'
 * pickups().name // => 'pickups'
 * ```
 */
export function pickups(options: { readonly reward?: number } = {}): Rule {
  return {
    name: 'pickups',
    enter: (e, ctx) => {
      let items = e.items
      let reward = e.reward
      for (const [j, i] of ctx.tracked.entries()) {
        const it = ctx.world.items[i]
        if (it === undefined || it.cell !== e.cell || ((items >>> j) & 1) === 1) continue
        items |= 1 << j
        reward += options.reward ?? it.reward
      }
      return items === e.items ? e : { ...e, items, reward }
    },
  }
}

/**
 * Locked doors: a cell holding a door may be entered only while holding a key of the door's colour.
 *
 * @returns The rule.
 * @example
 * ```ts
 * import { lockedDoors } from 'omai-engine/worlds'
 * lockedDoors().name // => 'lockedDoors'
 * ```
 */
export function lockedDoors(): Rule {
  return {
    name: 'lockedDoors',
    allows: (w, s, _from, to, _direction, tracked) =>
      w.items.every((it) => it.kind !== 'door' || it.cell !== to || holds(w, s, tracked, 'key', it.colour)),
  }
}

/**
 * The items a transition model tracks in its states: the indices in `world.items` of the keys and pickups.
 *
 * @param w - The world.
 * @returns Item indices; bit `j` of a state's `items` is item `trackedItems(w)[j]`.
 * @example
 * ```ts
 * import { fromAscii, trackedItems } from 'omai-engine/worlds'
 * trackedItems(fromAscii(['SKD$G'])) // => [0, 2]
 * ```
 */
export function trackedItems(w: World): number[] {
  return w.items.flatMap((it, i) => (it.kind === 'door' ? [] : [i]))
}

// -------------------------------------------------------------------------------------------- applying the rules

/** How a world, its rules and an action set become a transition model. */
export interface ModelOptions extends MoveOptions {
  /** The actions (default `ACTIONS.urdl` on squares, `ACTIONS.hex` on hexagons, one per neighbour slot on graphs). */
  readonly actions?: ActionSet
  /** Which cells are states: every cell (default; impassable ones absorb) or only passable ones, renumbered. */
  readonly cells?: 'all' | 'passable'
  /** Merge outcomes that reach the same state with the same reward, cost and termination (default false). */
  readonly merge?: boolean
  /** Terminal states loop on themselves with reward 0 (default true), as gymnasium's tables do. */
  readonly absorbing?: boolean
  /** Moves off the edge of a square grid: blocked (default) or clamped per axis (windy grid worlds). */
  readonly edges?: 'block' | 'clamp'
  /** The initial distribution: uniform over the starts (default) or over every non-terminal passable cell. */
  readonly initial?: 'starts' | 'uniform'
  /** The starting heading for turning actions (default 1, east, on squares; 0 on hexagons). */
  readonly heading?: number
}

const defaultActions = (w: World): ActionSet =>
  w.topology.kind === 'square' ? ACTIONS.urdl : w.topology.kind === 'hex' ? ACTIONS.hex : slotActions(w)

/**
 * Where a clamped or pushed move on a square grid lands: the step plus the push, clamped per axis. A landing one step
 * away obeys walls and corners like any move; a longer jump only needs a passable cell. Otherwise the agent stays.
 */
function landing(
  w: World,
  from: number,
  direction: number,
  push: readonly [number, number],
  options: MoveOptions,
  allowed: (from: number, to: number, direction: number) => boolean,
): number {
  const t = w.topology
  if (t.kind !== 'square') return from
  const [dr, dc] = direction >= 0 ? offset(t, from, direction) : [0, 0]
  const [r, c] = cellCoords(w, from)
  const r1 = Math.min(Math.max(r + dr + push[0], 0), t.rows - 1)
  const c1 = Math.min(Math.max(c + dc + push[1], 0), t.cols - 1)
  const to = cellIndex(w, r1, c1)
  if (to === from) return from
  const d = SQUARE_OFFSETS.findIndex(([a, b]) => a === r1 - r && b === c1 - c)
  const ok = d >= 0 ? destination(w, from, d, { ...options, connectivity: 8 }) === to : passable(w, to)
  return ok && allowed(from, to, d) ? to : from
}

/**
 * Everything one action can do from one state: each branch of the rules' moves, resolved against the world and the
 * rules' constraints, then rewritten by their entry effects, in order.
 *
 * @param w - The world.
 * @param ruleSet - The rules.
 * @param state - The state.
 * @param action - The action index.
 * @param options - The action set, move options and edge rule.
 * @returns The effects (unmerged, possibly with zero-probability branches removed).
 * @throws OmaiError with code `value` for an action outside the set.
 * @example
 * ```ts
 * import { applyRules, fromAscii, rules, stepReward, terminateOn } from 'omai-engine/worlds'
 * const w = fromAscii(['S.G'])
 * applyRules(w, rules(stepReward(-1), terminateOn('goals')), { cell: 1, items: 0, heading: -1 }, 1).map((e) => [e.cell, e.reward, e.terminated]) // => [[2, -1, true]]
 * ```
 */
export function applyRules(
  w: World,
  ruleSet: readonly Rule[],
  state: AgentState,
  action: number,
  options: ModelOptions = {},
): Effect[] {
  const actions = options.actions ?? defaultActions(w)
  if (!Number.isInteger(action) || action < 0 || action >= actions.actions.length) {
    throw new OmaiError('value', `applyRules: action ${action} is not in the action set`)
  }
  const tracked = trackedItems(w)
  let ms: readonly Move[] = [{ probability: 1, action, push: [0, 0] }]
  for (const r of ruleSet) if (r.moves !== undefined) ms = r.moves(ms, w, state, actions)
  const sides = w.topology.kind === 'hex' ? 6 : 4
  const allowed = (from: number, to: number, d: number): boolean =>
    ruleSet.every((r) => r.allows === undefined || r.allows(w, state, from, to, d, tracked))
  const out: Effect[] = []
  for (const m of ms) {
    if (m.probability <= 0) continue
    const act = actions.actions[m.action]
    if (act === undefined) throw new OmaiError('value', `applyRules: action ${m.action} is not in the action set`)
    let heading = state.heading
    let direction = -1
    let cost = 1
    if (act.kind === 'turn') heading = (((Math.max(state.heading, 0) + act.by) % sides) + sides) % sides
    else if (act.kind === 'forward') direction = Math.max(state.heading, 0)
    else if (act.kind === 'move') direction = act.direction
    else cost = 0
    let to = state.cell
    const pushed = m.push[0] !== 0 || m.push[1] !== 0
    if (direction >= 0 && !pushed) {
      const plain = destination(w, state.cell, direction, { ...options, connectivity: 8 })
      if (plain >= 0) to = allowed(state.cell, plain, direction) ? plain : state.cell
      else if (options.edges === 'clamp') to = landing(w, state.cell, direction, [0, 0], options, allowed)
    } else if (pushed) to = landing(w, state.cell, direction, m.push, options, allowed)
    if (direction >= 4 && w.topology.kind === 'square') cost = Math.SQRT2
    let effects: Effect[] = [
      { probability: m.probability, cell: to, items: state.items, heading, reward: 0, cost, terminated: false },
    ]
    for (const r of ruleSet) {
      const enter = r.enter
      if (enter === undefined) continue
      effects = effects.flatMap((e) => {
        const ctx: EntryContext = {
          world: w,
          state,
          action: m.action,
          direction,
          from: state.cell,
          to,
          bumped: direction >= 0 && to === state.cell,
          tracked,
        }
        const next = enter(e, ctx)
        // A single effect carries its own probability; branches carry probabilities conditional on the effect.
        if (!Array.isArray(next)) return [next as Effect]
        return (next as readonly Effect[]).map((x) => ({ ...x, probability: x.probability * e.probability }))
      })
    }
    out.push(...effects.filter((e) => e.probability > 0))
  }
  return out
}
