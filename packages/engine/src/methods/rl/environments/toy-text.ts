import { foldIn, type Key, uniform } from '../../../lib/random/index.ts'
import { type Env, type EnvDefinition, type Outcome, tabularEnv } from '../../../lib/rl/index.ts'
import { circle, line, rect, type Scene, type SceneNode, scene, text } from '../../../lib/scene/index.ts'
import { bool, dict, discrete, int, real, tuple } from '../../../lib/space/index.ts'
import { renderGrid } from './gridworlds.ts'

// ---------------------------------------------------------------------------------------------------------- Blackjack

/** The cards of an infinite deck (aces count 1, picture cards 10), as gymnasium's Blackjack draws them. */
export const BLACKJACK_DECK: readonly number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

/** The state of Blackjack: the player's and the dealer's hands (the dealer's first card is showing). */
export interface BlackjackState {
  readonly player: readonly number[]
  readonly dealer: readonly number[]
}

/** The observation of Blackjack: the player's sum, the dealer's showing card and whether the player has a usable ace. */
export type BlackjackObservation = readonly [number, number, number]

const usableAce = (hand: readonly number[]): boolean => hand.includes(1) && hand.reduce((a, b) => a + b, 0) + 10 <= 21

/**
 * The value of a Blackjack hand, counting one ace as 11 when that does not bust (gymnasium's `sum_hand`).
 *
 * @param hand - The cards.
 * @returns The total.
 * @example
 * ```ts
 * import { blackjackSum } from 'whitebox-engine/methods/rl'
 * blackjackSum([1, 6]) // => 17
 * ```
 */
export function blackjackSum(hand: readonly number[]): number {
  const s = hand.reduce((a, b) => a + b, 0)
  return usableAce(hand) ? s + 10 : s
}

const score = (hand: readonly number[]): number => (blackjackSum(hand) > 21 ? 0 : blackjackSum(hand))
const isNatural = (hand: readonly number[]): boolean =>
  hand.length === 2 && [...hand].sort((a, b) => a - b).join(',') === '1,10'
const observeBlackjack = (s: BlackjackState): BlackjackObservation => [
  blackjackSum(s.player),
  s.dealer[0] ?? 0,
  usableAce(s.player) ? 1 : 0,
]

/** Options of Blackjack (gymnasium's `natural` and `sab` flags). */
export interface BlackjackOptions {
  /** Pay 1.5 for winning with a natural blackjack (ignored with `sab`). */
  readonly natural?: boolean
  /** Follow Sutton and Barto exactly: a natural beats a non-natural dealer for +1. */
  readonly sab?: boolean
}

/**
 * One step of Blackjack with the cards supplied by `draw` (gymnasium's BlackjackEnv.step): hit (1) adds a card and
 * busts above 21; stick (0) plays out the dealer, who draws to 17, and scores win +1, draw 0, loss -1.
 *
 * @param state - The hands.
 * @param action - 0 stick, 1 hit.
 * @param draw - Returns the next card of the deck.
 * @param options - The natural and Sutton-Barto rules.
 * @returns The next hands, the reward and whether the hand is over.
 * @example
 * ```ts
 * import { blackjackStep } from 'whitebox-engine/methods/rl'
 * blackjackStep({ player: [10, 9], dealer: [10, 7] }, 0, () => 2).reward // => 1
 * ```
 */
export function blackjackStep(
  state: BlackjackState,
  action: number,
  draw: () => number,
  options: BlackjackOptions = {},
): { readonly state: BlackjackState; readonly reward: number; readonly terminated: boolean } {
  if (action === 1) {
    const player = [...state.player, draw()]
    const bust = blackjackSum(player) > 21
    return { state: { ...state, player }, reward: bust ? -1 : 0, terminated: bust }
  }
  const dealer = [...state.dealer]
  while (blackjackSum(dealer) < 17) dealer.push(draw())
  const p = score(state.player)
  const d = score(dealer)
  let reward = Number(p > d) - Number(p < d)
  if (options.sab === true && isNatural(state.player) && !isNatural(dealer)) reward = 1
  else if (options.sab !== true && options.natural === true && isNatural(state.player) && reward === 1) reward = 1.5
  return { state: { player: state.player, dealer }, reward, terminated: true }
}

function cardStream(k: Key): () => number {
  let i = 0
  let block = 0
  let u = uniform(foldIn(k, 0), [32]).data
  return () => {
    if (i >= u.length) {
      block += 1
      u = uniform(foldIn(k, block), [32]).data
      i = 0
    }
    return BLACKJACK_DECK[Math.floor((u[i++] ?? 0) * BLACKJACK_DECK.length)] ?? 10
  }
}

function renderBlackjack(s: BlackjackState): Scene {
  const card = (x: number, y: number, v: number, hidden: boolean): SceneNode[] => [
    rect(x, y, 60, 84, { fill: hidden ? 'tab0' : 'bg', stroke: 'fg', strokeWidth: 1 }),
    ...(hidden
      ? []
      : [text(x + 30, y + 42, v === 1 ? 'A' : String(v), { anchor: 'middle', baseline: 'middle', size: 24 })]),
  ]
  const dealerDone = s.dealer.length > 2 || blackjackSum(s.dealer) >= 17
  return scene(
    600,
    500,
    [
      rect(0, 0, 600, 500, { fill: '#076324' }),
      text(20, 30, `Dealer: ${dealerDone ? blackjackSum(s.dealer) : (s.dealer[0] ?? 0)}`, {
        fill: '#ffffff',
        size: 20,
      }),
      ...s.dealer.flatMap((v, i) => card(20 + 70 * i, 50, v, i === 1 && !dealerDone)),
      text(20, 280, `Player: ${blackjackSum(s.player)}${usableAce(s.player) ? ' (usable ace)' : ''}`, {
        fill: '#ffffff',
        size: 20,
      }),
      ...s.player.flatMap((v, i) => card(20 + 70 * i, 300, v, false)),
    ],
    { title: 'Blackjack' },
  )
}

/**
 * Blackjack-v1 (gymnasium; Sutton and Barto, Example 5.1): reach a hand closer to 21 than the dealer's without going
 * over, drawing from an infinite deck. Observations are `[player sum, dealer card, usable ace]`; actions stick (0) or
 * hit (1); rewards +1, 0 and -1 (options `natural` and `sab`).
 *
 * @concept environments
 * @example
 * ```ts
 * import { blackjack } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * blackjack.create().reset(key(0)).observation.length // => 3
 * ```
 */
export const blackjack: EnvDefinition<BlackjackState, BlackjackObservation, number> = {
  id: 'Blackjack-v1',
  description: 'Beat the dealer at blackjack drawing from an infinite deck.',
  options: dict({ natural: bool({ default: false }), sab: bool({ default: false }) }),
  create: (options): Env<BlackjackState, BlackjackObservation, number> => {
    const rules: BlackjackOptions = { natural: options?.['natural'] === true, sab: options?.['sab'] === true }
    return {
      spec: { id: 'Blackjack-v1', description: 'Beat the dealer at blackjack drawing from an infinite deck.' },
      observationSpace: tuple([discrete(32), discrete(11), discrete(2)], {
        label: 'player sum, dealer card, usable ace',
      }),
      actionSpace: discrete(2, { label: 'stick (0) or hit (1)' }),
      reset: (k) => {
        const draw = cardStream(k)
        const dealer = [draw(), draw()]
        const player = [draw(), draw()]
        const s = { player, dealer }
        return { state: s, observation: observeBlackjack(s), info: {} }
      },
      step: (state, action, k) => {
        const r = blackjackStep(state, action, cardStream(k), rules)
        return {
          state: r.state,
          observation: observeBlackjack(r.state),
          reward: r.reward,
          terminated: r.terminated,
          truncated: false,
          info: {},
        }
      },
      render: renderBlackjack,
      observe: observeBlackjack,
    }
  },
}

// --------------------------------------------------------------------------------------------------------------- Taxi

const TAXI_MAP = [
  '+---------+',
  '|R: | : :G|',
  '| : | : : |',
  '| : : : : |',
  '| | : | : |',
  '|Y| : |B: |',
  '+---------+',
]
/** The four pick-up and drop-off locations R, G, Y, B as (row, column). */
export const TAXI_LOCATIONS: readonly (readonly [number, number])[] = [
  [0, 0],
  [0, 4],
  [4, 0],
  [4, 3],
]

/**
 * Encode a Taxi state (gymnasium's `TaxiEnv.encode`): `((row * 5 + col) * 5 + passenger) * 4 + destination`, where the
 * passenger is at location 0 to 3 or in the taxi (4).
 *
 * @param row - Taxi row.
 * @param col - Taxi column.
 * @param passenger - Passenger location (4 in the taxi).
 * @param destination - Destination index.
 * @returns The state index in 0..499.
 * @example
 * ```ts
 * import { taxiEncode } from 'whitebox-engine/methods/rl'
 * taxiEncode(3, 1, 2, 0) // => 328
 * ```
 */
export function taxiEncode(row: number, col: number, passenger: number, destination: number): number {
  return ((row * 5 + col) * 5 + passenger) * 4 + destination
}

/**
 * Decode a Taxi state index into `[row, col, passenger, destination]`.
 *
 * @param s - The state index (0..499).
 * @returns The parts.
 * @example
 * ```ts
 * import { taxiDecode } from 'whitebox-engine/methods/rl'
 * taxiDecode(328) // => [3, 1, 2, 0]
 * ```
 */
export function taxiDecode(s: number): [number, number, number, number] {
  const dest = s % 4
  const pass = Math.floor(s / 4) % 5
  const col = Math.floor(s / 20) % 5
  const row = Math.floor(s / 100)
  return [row, col, pass, dest]
}

const wallChar = (row: number, charCol: number): string => TAXI_MAP[1 + row]?.[charCol] ?? '|'
const sameLoc = (a: readonly [number, number], b: readonly [number, number]): boolean => a[0] === b[0] && a[1] === b[1]
const locIndex = (loc: readonly [number, number]): number => TAXI_LOCATIONS.findIndex((l) => sameLoc(l, loc))

/**
 * The legal-move mask of a Taxi state (gymnasium's `action_mask`): moves that are not blocked, pick-up where the
 * passenger waits, and drop-off at a location with the passenger aboard.
 *
 * @param s - The state index.
 * @returns Six flags (south, north, east, west, pick-up, drop-off).
 * @example
 * ```ts
 * import { taxiActionMask } from 'whitebox-engine/methods/rl'
 * taxiActionMask(0) // => [1, 0, 0, 0, 1, 0]
 * ```
 */
export function taxiActionMask(s: number): number[] {
  const [row, col, pass, dest] = taxiDecode(s)
  const loc: [number, number] = [row, col]
  return [
    row < 4 ? 1 : 0,
    row > 0 ? 1 : 0,
    col < 4 && wallChar(row, 2 * col + 2) === ':' ? 1 : 0,
    col > 0 && wallChar(row, 2 * col) === ':' ? 1 : 0,
    pass < 4 && sameLoc(loc, TAXI_LOCATIONS[pass] ?? [-1, -1]) ? 1 : 0,
    pass === 4 && (sameLoc(loc, TAXI_LOCATIONS[dest] ?? [-1, -1]) || locIndex(loc) >= 0) ? 1 : 0,
  ]
}

function pickup(loc: readonly [number, number], pass: number): [number, number] {
  return pass < 4 && sameLoc(loc, TAXI_LOCATIONS[pass] ?? [-1, -1]) ? [4, -1] : [pass, -10]
}

function dropoff(loc: readonly [number, number], pass: number, dest: number): [number, number, boolean] {
  if (sameLoc(loc, TAXI_LOCATIONS[dest] ?? [-1, -1]) && pass === 4) return [dest, 20, true]
  if (locIndex(loc) >= 0 && pass === 4) return [locIndex(loc), -1, false]
  return [pass, -10, false]
}

function calcNewPosition(row: number, col: number, dr: number, dc: number): [number, number] {
  const r = Math.max(0, Math.min(row + dr, 4))
  const c = Math.max(0, Math.min(col + dc, 4))
  if (dc === 1 && wallChar(r, 2 * c) !== ':') return [row, col]
  if (dc === -1 && wallChar(r, 2 * c + 2) !== ':') return [row, col]
  return [r, c]
}

/**
 * The outcomes of a Taxi action (gymnasium's `P` table, dry or rainy): south, north, east, west, pick-up, drop-off.
 * Every step costs -1, illegal pick-ups and drop-offs -10, a delivery pays +20 and ends the episode. In the rain a
 * move goes in the intended direction with `rainyProbability` and slips to either side otherwise.
 *
 * @param s - The state index.
 * @param action - The action.
 * @param options - Rain.
 * @returns The outcomes.
 * @example
 * ```ts
 * import { taxiTransitions, taxiEncode } from 'whitebox-engine/methods/rl'
 * taxiTransitions(taxiEncode(0, 0, 4, 0), 5)[0]?.reward // => 20
 * ```
 */
export function taxiTransitions(
  s: number,
  action: number,
  options: { readonly isRainy?: boolean; readonly rainyProbability?: number } = {},
): Outcome<number>[] {
  const [row, col, pass, dest] = taxiDecode(s)
  const loc: [number, number] = [row, col]
  let newRow = row
  let newCol = col
  let newPass = pass
  let reward = -1
  let terminated = false
  if (options.isRainy !== true) {
    if (action === 0) newRow = Math.min(row + 1, 4)
    else if (action === 1) newRow = Math.max(row - 1, 0)
    if (action === 2 && wallChar(row, 2 * col + 2) === ':') newCol = Math.min(col + 1, 4)
    else if (action === 3 && wallChar(row, 2 * col) === ':') newCol = Math.max(col - 1, 0)
    else if (action === 4) [newPass, reward] = pickup(loc, newPass)
    else if (action === 5) [newPass, reward, terminated] = dropoff(loc, newPass, dest)
    return [{ probability: 1, state: taxiEncode(newRow, newCol, newPass, dest), reward, terminated }]
  }
  const p = options.rainyProbability ?? 0.8
  const lateral = (1 - p) / 2
  const moves: Readonly<Record<number, readonly (readonly [number, number])[]>> = {
    0: [
      [1, 0],
      [0, 1],
      [0, -1],
    ],
    1: [
      [-1, 0],
      [0, -1],
      [0, 1],
    ],
    2: [
      [0, 1],
      [-1, 0],
      [1, 0],
    ],
    3: [
      [0, -1],
      [1, 0],
      [-1, 0],
    ],
  }
  let left: [number, number] = [row, col]
  let right: [number, number] = [row, col]
  if (
    (action === 0 && row < 4) ||
    (action === 1 && row > 0) ||
    (action === 2 && wallChar(row, 2 * col + 2) === ':') ||
    (action === 3 && wallChar(row, 2 * col) === ':')
  ) {
    const m = moves[action] ?? []
    const [dr, dc] = m[0] ?? [0, 0]
    newRow = Math.max(0, Math.min(row + dr, 4))
    newCol = Math.max(0, Math.min(col + dc, 4))
    const [lr, lc] = m[1] ?? [0, 0]
    const [rr, rc] = m[2] ?? [0, 0]
    left = calcNewPosition(row, col, lr, lc)
    right = calcNewPosition(row, col, rr, rc)
  } else if (action === 4) [newPass, reward] = pickup(loc, newPass)
  else if (action === 5) [newPass, reward, terminated] = dropoff(loc, newPass, dest)
  const intended = taxiEncode(newRow, newCol, newPass, dest)
  if (action <= 3) {
    return [
      { probability: p, state: intended, reward: -1, terminated },
      { probability: lateral, state: taxiEncode(left[0], left[1], newPass, dest), reward: -1, terminated },
      { probability: lateral, state: taxiEncode(right[0], right[1], newPass, dest), reward: -1, terminated },
    ]
  }
  return [{ probability: 1, state: intended, reward, terminated }]
}

function renderTaxi(s: number): Scene {
  const [row, col, pass, dest] = taxiDecode(s % 500)
  const cell = 80
  const nodes: SceneNode[] = []
  const labels = ['R', 'G', 'Y', 'B']
  const colours = ['#ff0000', '#00aa00', '#cccc00', '#0000ff'] as const
  for (const [i, [r, c]] of TAXI_LOCATIONS.entries()) {
    nodes.push(rect(c * cell, r * cell, cell, cell, { fill: colours[i] ?? '#ffffff', opacity: 0.3 }))
    nodes.push(text(c * cell + 8, r * cell + 20, labels[i] ?? '', { size: 16 }))
  }
  // Interior walls: a '|' between columns c and c + 1 at row r.
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 4; c++) {
      if (wallChar(r, 2 * c + 2) === '|')
        nodes.push(line((c + 1) * cell, r * cell, (c + 1) * cell, (r + 1) * cell, { stroke: 'fg', strokeWidth: 6 }))
    }
  }
  const [dr, dc] = TAXI_LOCATIONS[dest] ?? [0, 0]
  nodes.push(
    rect(dc * cell + 4, dr * cell + 4, cell - 8, cell - 8, {
      stroke: colours[dest] ?? 'fg',
      strokeWidth: 4,
      fill: 'none',
      id: 'destination',
    }),
  )
  nodes.push(rect(col * cell + 15, row * cell + 30, 50, 30, { fill: pass === 4 ? 'tab2' : '#ffcc00', id: 'taxi' }))
  if (pass < 4) {
    const [pr, pc] = TAXI_LOCATIONS[pass] ?? [0, 0]
    nodes.push(circle(pc * cell + cell / 2, pr * cell + cell / 2, 12, { fill: 'tab1', id: 'passenger' }))
  }
  return renderGrid(
    Array.from({ length: 5 }, () => '     '),
    undefined,
    { cell, title: 'Taxi', overlay: nodes },
  )
}

/** Options of Taxi (gymnasium's TaxiEnv). */
export interface TaxiOptions {
  readonly isRainy?: boolean
  readonly rainyProbability?: number
  /** The passenger may change destination once, after the first move with them aboard. */
  readonly ficklePassenger?: boolean
  readonly fickleProbability?: number
}

/**
 * Build Taxi exactly as gymnasium's TaxiEnv (Dietterich 2000): pick up the passenger at one of four locations and drop
 * them at their destination on a 5 x 5 grid with walls. With `ficklePassenger` the state also records whether the
 * passenger will still change their mind (states `s + 500`), so the model stays exact; observations are always
 * `s mod 500`.
 *
 * @param options - Rain and the fickle passenger.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeTaxi } from 'whitebox-engine/methods/rl'
 * makeTaxi().model?.states.length // => 500
 * ```
 */
export function makeTaxi(options: TaxiOptions = {}): Env<number, number, number> {
  const fickle = options.ficklePassenger === true
  const fp = options.fickleProbability ?? 0.3
  const base = (s: number, a: number): Outcome<number>[] => taxiTransitions(s, a, options)
  const transitions = (s: number, a: number): Outcome<number>[] => {
    if (s < 500) return base(s, a)
    // A fickle passenger aboard changes destination (uniformly among the others) the first time the taxi moves.
    const [r0, c0, p0, d0] = taxiDecode(s - 500)
    return base(s - 500, a).flatMap((o) => {
      const [r, c, p] = taxiDecode(o.state)
      if (p0 !== 4 || (r === r0 && c === c0)) return [{ ...o, state: o.state + 500 }]
      return [0, 1, 2, 3]
        .filter((d) => d !== d0)
        .map((d) => ({ ...o, probability: o.probability / 3, state: taxiEncode(r, c, p, d) }))
    })
  }
  const initial = new Array<number>(fickle ? 1000 : 500).fill(0)
  let count = 0
  for (let s = 0; s < 500; s++) {
    const [, , pass, dest] = taxiDecode(s)
    if (pass < 4 && pass !== dest) count += 1
  }
  for (let s = 0; s < 500; s++) {
    const [, , pass, dest] = taxiDecode(s)
    if (pass < 4 && pass !== dest) {
      initial[s] = (fickle ? 1 - fp : 1) / count
      if (fickle) initial[s + 500] = fp / count
    }
  }
  return tabularEnv({
    id: 'Taxi-v4',
    description: 'Pick up a passenger and drop them at their destination on a small grid with walls.',
    nStates: fickle ? 1000 : 500,
    nActions: 6,
    transitions,
    initial,
    render: renderTaxi,
    observe: (s) => s % 500,
    info: (s) => ({ actionMask: taxiActionMask(s % 500) }),
    maxEpisodeSteps: 200,
    rewardThreshold: 8,
    stateLabel: 'taxi row, column, passenger, destination (encoded)',
    actionLabel: 'south (0), north (1), east (2), west (3), pick up (4), drop off (5)',
  })
}

/**
 * Taxi-v4 (gymnasium).
 *
 * @concept environments
 * @example
 * ```ts
 * import { taxi } from 'whitebox-engine/methods/rl'
 * taxi.create().actionSpace.kind // => 'discrete'
 * ```
 */
export const taxi: EnvDefinition<number, number, number> = {
  id: 'Taxi-v4',
  description: 'Pick up a passenger and drop them at their destination on a small grid with walls.',
  options: dict({
    isRainy: bool({ default: false }),
    rainyProbability: real(0, 1, { default: 0.8 }),
    ficklePassenger: bool({ default: false }),
    fickleProbability: real(0, 1, { default: 0.3 }),
  }),
  create: (options) => makeTaxi((options ?? {}) as TaxiOptions),
}

// -------------------------------------------------------------------------------------------------- NChain, RiverSwim

function renderChain(n: number, s: number, title: string, labels: readonly string[] = []): Scene {
  const w = 80
  const nodes: SceneNode[] = []
  for (let i = 0; i < n; i++) {
    nodes.push(circle(w / 2 + i * w, 60, 28, { fill: i === s ? 'tab1' : 'surface', stroke: 'fg', strokeWidth: 1 }))
    nodes.push(text(w / 2 + i * w, 60, labels[i] ?? String(i), { anchor: 'middle', baseline: 'middle', size: 16 }))
    if (i + 1 < n) nodes.push(line(w / 2 + i * w + 28, 60, w / 2 + (i + 1) * w - 28, 60, { stroke: 'muted' }))
  }
  return scene(n * w, 120, nodes, { title })
}

/**
 * Build the n-chain of Strens (2000) as gym's NChain-v0: action 0 moves forward along the chain (reward 0, and `large`
 * for staying at the end), action 1 returns to the start for `small`; with probability `slip` the other action is
 * taken. Myopic agents settle for the small reward.
 *
 * @param options - Chain length (default 5), slip (0.2), small (2) and large (10) rewards.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeNChain } from 'whitebox-engine/methods/rl'
 * makeNChain().model?.transitions(4, 0) // => [{ probability: 0.8, state: 4, reward: 10, terminated: false }, { probability: 0.2, state: 0, reward: 2, terminated: false }]
 * ```
 */
export function makeNChain(
  options: { readonly n?: number; readonly slip?: number; readonly small?: number; readonly large?: number } = {},
): Env<number, number, number> {
  const n = options.n ?? 5
  const slip = options.slip ?? 0.2
  const small = options.small ?? 2
  const large = options.large ?? 10
  const effect = (s: number, a: number): Outcome<number> =>
    a === 1
      ? { probability: 1, state: 0, reward: small, terminated: false }
      : s < n - 1
        ? { probability: 1, state: s + 1, reward: 0, terminated: false }
        : { probability: 1, state: s, reward: large, terminated: false }
  const initial = new Array<number>(n).fill(0)
  initial[0] = 1
  return tabularEnv({
    id: 'NChain-v0',
    description: 'Walk forward along a chain for a large delayed reward or return for a small one.',
    nStates: n,
    nActions: 2,
    transitions: (s, a) =>
      [
        { ...effect(s, a), probability: 1 - slip },
        { ...effect(s, 1 - a), probability: slip },
      ].filter((o) => o.probability > 0),
    initial,
    render: (s) => renderChain(n, s, 'NChain'),
    maxEpisodeSteps: 1000,
    stateLabel: 'position on the chain',
    actionLabel: 'forward (0) or back to the start (1)',
  })
}

/**
 * NChain-v0 (gym's version of Strens 2000).
 *
 * @concept environments
 * @example
 * ```ts
 * import { nChain } from 'whitebox-engine/methods/rl'
 * nChain.create().spec.maxEpisodeSteps // => 1000
 * ```
 */
export const nChain: EnvDefinition<number, number, number> = {
  id: 'NChain-v0',
  description: 'Walk forward along a chain for a large delayed reward or return for a small one.',
  options: dict({ n: int(2, 50, { default: 5 }), slip: real(0, 1, { default: 0.2 }) }),
  create: (options) => makeNChain((options ?? {}) as { readonly n?: number; readonly slip?: number }),
}

/**
 * Build RiverSwim (Strehl and Littman 2008, with the probabilities of Osband, Russo and Van Roy 2013): swimming left
 * (action 0) always succeeds and pays `small` at the left bank; swimming right against the current moves right with
 * probability 0.35, stays with 0.6 and drifts back with 0.05 (0.4 and 0.6 at the banks) and pays `large` at the right
 * bank. A hard-exploration benchmark: the optimal policy always swims right.
 *
 * @param options - Number of states (default 6), small (0.005) and large (1) rewards.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeRiverSwim } from 'whitebox-engine/methods/rl'
 * makeRiverSwim().model?.transitions(0, 0)[0]?.reward // => 0.005
 * ```
 */
export function makeRiverSwim(
  options: { readonly n?: number; readonly small?: number; readonly large?: number } = {},
): Env<number, number, number> {
  const n = options.n ?? 6
  const small = options.small ?? 0.005
  const large = options.large ?? 1
  const transitions = (s: number, a: number): Outcome<number>[] => {
    if (a === 0) return [{ probability: 1, state: Math.max(s - 1, 0), reward: s === 0 ? small : 0, terminated: false }]
    if (s === 0) {
      return [
        { probability: 0.6, state: 0, reward: 0, terminated: false },
        { probability: 0.4, state: 1, reward: 0, terminated: false },
      ]
    }
    if (s === n - 1) {
      return [
        { probability: 0.6, state: s, reward: large, terminated: false },
        { probability: 0.4, state: s - 1, reward: large, terminated: false },
      ]
    }
    return [
      { probability: 0.05, state: s - 1, reward: 0, terminated: false },
      { probability: 0.6, state: s, reward: 0, terminated: false },
      { probability: 0.35, state: s + 1, reward: 0, terminated: false },
    ]
  }
  const initial = new Array<number>(n).fill(0)
  initial[0] = 1
  return tabularEnv({
    id: 'RiverSwim-v0',
    description: 'Swim upstream against the current for a large reward instead of resting on the near bank.',
    nStates: n,
    nActions: 2,
    transitions,
    initial,
    render: (s) => renderChain(n, s, 'RiverSwim'),
    maxEpisodeSteps: 20 * n,
    stateLabel: 'position across the river',
    actionLabel: 'swim left (0) or right (1)',
  })
}

/**
 * RiverSwim-v0 (Strehl and Littman 2008).
 *
 * @concept environments
 * @example
 * ```ts
 * import { riverSwim } from 'whitebox-engine/methods/rl'
 * riverSwim.create().observationSpace // => { kind: 'discrete', n: 6, label: 'position across the river' }
 * ```
 */
export const riverSwim: EnvDefinition<number, number, number> = {
  id: 'RiverSwim-v0',
  description: 'Swim upstream against the current for a large reward instead of resting on the near bank.',
  options: dict({ n: int(3, 20, { default: 6 }) }),
  create: (options) => makeRiverSwim((options ?? {}) as { readonly n?: number }),
}
