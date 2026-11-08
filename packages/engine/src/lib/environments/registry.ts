/**
 * The tagged environment registry: every environment definition of this domain, and the differentiable control models
 * as environments, as one entry whose tags come from its spaces (discrete, continuous or multi actions; vector, grid,
 * image or discrete observations) and from the table (its family, and whether its episodes end at terminal states),
 * built as a value by constructing each definition once with its default options. `whereEnvironments` answers a
 * selector's queries, `environmentEntry` derives an entry from any definition, and `registerEnvironments` returns a
 * new registry with an app's own definitions added.
 */

import { find, matchesText, OmaiError, type Registry, register, registry } from '../base/index.ts'
import type { AecEnv, Env, EnvDefinition, EnvSpec, ParallelEnv } from '../rl/index.ts'
import type { DictSpace, Space } from '../space/index.ts'
import {
  adversarialBandit,
  bernoulliBandit,
  changePointBandit,
  contextualLinearBandit,
  gaussianBandit,
  nonStationaryBandit,
} from './bandits.ts'
import { blockingMaze, shortcutMaze } from './changing-maze.ts'
import { acrobot, cartPole, mountainCar, mountainCarContinuous, pendulum } from './classic-control.ts'
import { cartDoublePendulum, doubleIntegrator, planarQuadrotor, pointMass, reacher2d } from './continuous-control.ts'
import { connectFour, type GameDefinition, kuhnPoker, leducPoker, nim, ticTacToe } from './games.ts'
import { go } from './go.ts'
import { reachGoal } from './goal-reaching.ts'
import {
  cliffWalking,
  cliffWalkingSlippery,
  fourRooms,
  frozenLake,
  frozenLake8x8,
  gridWorld,
  keyMaze,
  windyGridWorld,
} from './gridworlds.ts'
import { asterixMinAtar, breakoutMinAtar, freewayMinAtar, seaquestMinAtar, spaceInvadersMinAtar } from './minatar.ts'
import {
  acrobotModel,
  type ControlModel,
  cartDoublePendulumModel,
  cartPoleModel,
  controlModelEnv,
  doubleIntegratorModel,
  mountainCarContinuousModel,
  pendulumModel,
  planarQuadrotorModel,
  pointMassModel,
  reacher2dModel,
} from './models.ts'
import { cooperativeNavigation, matrixGame, type ParallelDefinition, pursuitEvasion } from './multi-agent.ts'
import { bairdCounterexample, infiniteVariance, randomWalk } from './prediction.ts'
import { keyDoor, proceduralMaze } from './procedural.ts'
import { blackjack, nChain, riverSwim, taxi } from './toy-text.ts'

/** The families environments are grouped by (a hand-written tag). */
export type EnvironmentFamily =
  | 'classicControl'
  | 'toyText'
  | 'gridWorld'
  | 'bandit'
  | 'continuousControl'
  | 'minatar'
  | 'game'
  | 'procedural'
  | 'multiAgent'
  | 'model'

/** Every family, in the order selectors list them. */
export const ENVIRONMENT_FAMILIES: readonly EnvironmentFamily[] = [
  'classicControl',
  'toyText',
  'gridWorld',
  'bandit',
  'continuousControl',
  'minatar',
  'game',
  'procedural',
  'multiAgent',
  'model',
]

/** What an action is: one of finitely many, a real vector, or several coordinates at once (derived from the space). */
export type ActionKind = 'discrete' | 'continuous' | 'multi'

/** What an observation is: a real vector, a grid of cells, an image of pixels, or a state index (derived). */
export type ObservationKind = 'vector' | 'grid' | 'image' | 'discrete'

/** The tags of an environment entry: two by hand, two derived from the spaces. */
export interface EnvironmentTags {
  readonly family: EnvironmentFamily
  /** Episodes end at terminal states of the task (not only by a time limit). */
  readonly episodic: boolean
  readonly actions: ActionKind
  readonly observations: ObservationKind
}

/** A single-agent definition (an `EnvDefinition`, which carries no `multiAgent` mark). */
export type SingleAgentDefinition = EnvDefinition & { readonly multiAgent?: undefined }

/** Any environment definition: single-agent, turn-based (`'aec'`) or simultaneous-move (`'parallel'`). */
export type EnvironmentDefinition =
  | SingleAgentDefinition
  | GameDefinition<unknown, unknown, unknown>
  | ParallelDefinition<unknown, unknown, unknown>

/** What every environment entry carries besides its form. */
export interface EnvironmentEntryBase extends EnvironmentTags {
  /** The entry's name, unique in its registry (the definition's export name in the library's registry). */
  readonly name: string
  readonly title: string
  readonly description: string
  readonly definition: EnvironmentDefinition
  /** The spec of the environment at its default options. */
  readonly spec: EnvSpec
  /** The spaces at the default options (the first agent's, for a multi-agent environment). */
  readonly observationSpace: Space
  readonly actionSpace: Space
  /** The definition's options, for controls. */
  readonly options?: DictSpace
}

/**
 * One registered environment: its tags, spec and spaces, and `make(options)`, which creates it; a multi-agent entry
 * says which form it takes and names its agents.
 *
 * @concept environments
 */
export type EnvironmentEntry = EnvironmentEntryBase &
  (
    | {
        readonly multiAgent?: undefined
        readonly agents?: undefined
        readonly make: (options?: Readonly<Record<string, unknown>>) => Env<unknown, unknown, unknown>
      }
    | {
        readonly multiAgent: 'aec'
        readonly agents: readonly string[]
        readonly make: (options?: Readonly<Record<string, unknown>>) => AecEnv<unknown, unknown, unknown>
      }
    | {
        readonly multiAgent: 'parallel'
        readonly agents: readonly string[]
        readonly make: (options?: Readonly<Record<string, unknown>>) => ParallelEnv<unknown, unknown, unknown>
      }
  )

/** A registry of environments. */
export type EnvironmentRegistry = Registry<EnvironmentEntry>

/**
 * The kind of action a space admits: `discrete` for a discrete, integer, choice or boolean space, `continuous` for a
 * box or real interval, `multi` for several coordinates at once (multi-discrete, multi-binary, tuple, dict).
 *
 * @param space - The action space.
 * @returns The kind.
 * @concept environments
 * @example
 * ```ts
 * import { actionKindOf } from 'omai-engine/environments'
 * import { box, discrete, multiDiscrete } from 'omai-engine/space'
 * actionKindOf(discrete(3)) // => 'discrete'
 * actionKindOf(box(-1, 1, [2])) // => 'continuous'
 * actionKindOf(multiDiscrete([2, 3])) // => 'multi'
 * ```
 */
export function actionKindOf(space: Space): ActionKind {
  switch (space.kind) {
    case 'discrete':
    case 'int':
    case 'choice':
    case 'bool':
      return 'discrete'
    case 'box':
    case 'real':
      return 'continuous'
    default:
      return 'multi'
  }
}

/** Whether a box holds pixel intensities: at least two dimensions and an upper bound of 255 throughout. */
function isPixels(space: Space & { readonly kind: 'box' }): boolean {
  if (space.shape.length < 2) return false
  return typeof space.high === 'number' ? space.high === 255 : space.high.every((h) => h === 255)
}

/**
 * The kind of observation a space describes: `discrete` for a state index or a tuple of them (and a multi-discrete
 * space), `vector` for a one-dimensional box or any mixed composite, `grid` for a box of two or more dimensions, and
 * `image` for such a box of pixel intensities (an upper bound of 255).
 *
 * @param space - The observation space.
 * @returns The kind.
 * @concept environments
 * @example
 * ```ts
 * import { observationKindOf } from 'omai-engine/environments'
 * import { box, discrete, tuple } from 'omai-engine/space'
 * observationKindOf(box(-1, 1, [4])) // => 'vector'
 * observationKindOf(box(0, 1, [10, 10, 4])) // => 'grid'
 * observationKindOf(box(0, 255, [24, 32, 3])) // => 'image'
 * observationKindOf(tuple([discrete(32), discrete(11)])) // => 'discrete'
 * ```
 */
export function observationKindOf(space: Space): ObservationKind {
  switch (space.kind) {
    case 'discrete':
    case 'int':
    case 'choice':
    case 'bool':
    case 'multiDiscrete':
      return 'discrete'
    case 'box':
      if (space.shape.length <= 1) return 'vector'
      return isPixels(space) ? 'image' : 'grid'
    case 'tuple':
      return space.spaces.every((s) => observationKindOf(s) === 'discrete') ? 'discrete' : 'vector'
    case 'dict':
      return Object.values(space.spaces).every((s) => observationKindOf(s) === 'discrete') ? 'discrete' : 'vector'
    default:
      return 'vector'
  }
}

/** What registering an environment takes: the definition and the two hand-written tags, with a title. */
export interface EnvironmentRegistration {
  readonly definition: EnvironmentDefinition
  readonly title: string
  readonly family: EnvironmentFamily
  readonly episodic: boolean
  /** A name other than the definition's id. */
  readonly name?: string
  /** A description other than the definition's. */
  readonly description?: string
}

/**
 * A registry entry derived from a definition: the definition is created once with its default options to read its
 * spec and spaces, and the action and observation kinds are derived from the spaces (the first agent's, for a
 * multi-agent definition).
 *
 * @param registration - The definition, its title, family and whether it is episodic.
 * @returns The entry.
 * @throws OmaiError with code `value` for a multi-agent definition without agents.
 * @concept environments
 * @example
 * ```ts
 * import { cartPole, environmentEntry } from 'omai-engine/environments'
 * const entry = environmentEntry({ definition: cartPole, title: 'Cart-pole', family: 'classicControl', episodic: true })
 * [entry.name, entry.actions, entry.observations] // => ['CartPole-v1', 'discrete', 'vector']
 * ```
 */
export function environmentEntry(registration: EnvironmentRegistration): EnvironmentEntry {
  const { definition, title, family, episodic } = registration
  const name = registration.name ?? definition.id
  const description = registration.description ?? definition.description
  const base = {
    name,
    title,
    description,
    family,
    episodic,
    definition,
    ...(definition.options === undefined ? {} : { options: definition.options }),
  }
  const tagged = (spec: EnvSpec, observationSpace: Space, actionSpace: Space) => ({
    spec,
    observationSpace,
    actionSpace,
    actions: actionKindOf(actionSpace),
    observations: observationKindOf(observationSpace),
  })
  if (definition.multiAgent === undefined) {
    const env = definition.create()
    return { ...base, ...tagged(env.spec, env.observationSpace, env.actionSpace), make: (o) => definition.create(o) }
  }
  const env = definition.create()
  const first = env.possibleAgents[0]
  const observationSpace = first === undefined ? undefined : env.observationSpaces[first]
  const actionSpace = first === undefined ? undefined : env.actionSpaces[first]
  if (first === undefined || observationSpace === undefined || actionSpace === undefined)
    throw new OmaiError('value', `environmentEntry: '${name}' has no agents`)
  const shared = { ...base, ...tagged(env.spec, observationSpace, actionSpace), agents: env.possibleAgents }
  return definition.multiAgent === 'aec'
    ? { ...shared, multiAgent: 'aec', make: (o) => definition.create(o) }
    : { ...shared, multiAgent: 'parallel', make: (o) => definition.create(o) }
}

// ------------------------------------------------------------------------------------------------------ the table

/** A row of the table: name, definition, title, episodic. */
type Row = readonly [name: string, definition: EnvironmentDefinition, title: string, episodic: boolean]

const family = (family: EnvironmentFamily, rows: readonly Row[]): EnvironmentEntry[] =>
  rows.map(([name, definition, title, episodic]) => environmentEntry({ definition, name, title, family, episodic }))

/** A differentiable control model as an environment definition (the reward is minus the cost; nothing terminates). */
const model = (id: string, make: () => ControlModel): SingleAgentDefinition => ({
  id: `${id}-model`,
  description: `A differentiable model of ${id} as an environment: the reward is minus the cost and nothing terminates.`,
  create: () => controlModelEnv(make()),
})

/**
 * The library's environment registry: every definition of this domain and the control models as environments, tagged.
 * Extend it with {@link registerEnvironments}; query it with {@link whereEnvironments}.
 *
 * @concept environments
 * @index table Tagged registry of every environment, with action and observation kinds.
 * @example
 * ```ts
 * import { lookup } from 'omai-engine/base'
 * import { environmentRegistry } from 'omai-engine/environments'
 * environmentRegistry.entries.length // => 59
 * lookup(environmentRegistry, 'frozenLake').observations // => 'discrete'
 * ```
 */
export const environmentRegistry: EnvironmentRegistry = registry<EnvironmentEntry>([
  ...family('classicControl', [
    ['cartPole', cartPole, 'Cart-pole', true],
    ['mountainCar', mountainCar, 'Mountain car', true],
    ['mountainCarContinuous', mountainCarContinuous, 'Mountain car (continuous)', true],
    ['pendulum', pendulum, 'Pendulum', false],
    ['acrobot', acrobot, 'Acrobot', true],
  ]),
  ...family('toyText', [
    ['blackjack', blackjack, 'Blackjack', true],
    ['taxi', taxi, 'Taxi', true],
    ['nChain', nChain, 'NChain', false],
    ['riverSwim', riverSwim, 'RiverSwim', false],
    ['randomWalk', randomWalk, 'Random walk', true],
    ['bairdCounterexample', bairdCounterexample, "Baird's counterexample", false],
    ['infiniteVariance', infiniteVariance, 'Infinite-variance process', true],
  ]),
  ...family('gridWorld', [
    ['frozenLake', frozenLake, 'Frozen lake (4 by 4)', true],
    ['frozenLake8x8', frozenLake8x8, 'Frozen lake (8 by 8)', true],
    ['cliffWalking', cliffWalking, 'Cliff walking', true],
    ['cliffWalkingSlippery', cliffWalkingSlippery, 'Cliff walking (slippery)', true],
    ['gridWorld', gridWorld, 'Grid world (AIMA 4 by 3)', true],
    ['windyGridWorld', windyGridWorld, 'Windy grid world', true],
    ['keyMaze', keyMaze, 'Key maze', true],
    ['fourRooms', fourRooms, 'Four rooms', true],
    ['blockingMaze', blockingMaze, 'Blocking maze', true],
    ['shortcutMaze', shortcutMaze, 'Shortcut maze', true],
  ]),
  ...family('bandit', [
    ['bernoulliBandit', bernoulliBandit, 'Bernoulli bandit', false],
    ['gaussianBandit', gaussianBandit, 'Gaussian bandit (the 10-armed testbed)', false],
    ['contextualLinearBandit', contextualLinearBandit, 'Contextual linear bandit', false],
    ['adversarialBandit', adversarialBandit, 'Adversarial bandit', false],
    ['nonStationaryBandit', nonStationaryBandit, 'Non-stationary bandit', false],
    ['changePointBandit', changePointBandit, 'Change-point bandit', false],
  ]),
  ...family('continuousControl', [
    ['pointMass', pointMass, 'Point mass', false],
    ['reacher2d', reacher2d, 'Reacher (two links)', false],
    ['doubleIntegrator', doubleIntegrator, 'Double integrator', false],
    ['planarQuadrotor', planarQuadrotor, 'Planar quadrotor', false],
    ['cartDoublePendulum', cartDoublePendulum, 'Cart double pendulum', true],
    ['reachGoal', reachGoal, 'Goal reaching', true],
  ]),
  ...family('minatar', [
    ['breakoutMinAtar', breakoutMinAtar, 'Breakout (MinAtar)', true],
    ['freewayMinAtar', freewayMinAtar, 'Freeway (MinAtar)', true],
    ['asterixMinAtar', asterixMinAtar, 'Asterix (MinAtar)', true],
    ['spaceInvadersMinAtar', spaceInvadersMinAtar, 'Space Invaders (MinAtar)', true],
    ['seaquestMinAtar', seaquestMinAtar, 'Seaquest (MinAtar)', true],
  ]),
  ...family('game', [
    ['ticTacToe', ticTacToe, 'Tic-tac-toe', true],
    ['connectFour', connectFour, 'Connect four', true],
    ['nim', nim, 'Nim', true],
    ['kuhnPoker', kuhnPoker, 'Kuhn poker', true],
    ['leducPoker', leducPoker, "Leduc hold'em", true],
    ['go', go, 'Go', true],
  ]),
  ...family('procedural', [
    ['proceduralMaze', proceduralMaze, 'Procedural maze', true],
    ['keyDoor', keyDoor, 'Key-door maze', true],
  ]),
  ...family('multiAgent', [
    ['matrixGame', matrixGame, 'Matrix game', false],
    ['pursuitEvasion', pursuitEvasion, 'Pursuit-evasion', true],
    ['cooperativeNavigation', cooperativeNavigation, 'Cooperative navigation', false],
  ]),
  ...family('model', [
    ['cartPoleModel', model('CartPole-v1', cartPoleModel), 'Cart-pole model', false],
    ['pendulumModel', model('Pendulum-v1', pendulumModel), 'Pendulum model', false],
    [
      'mountainCarContinuousModel',
      model('MountainCarContinuous-v0', mountainCarContinuousModel),
      'Mountain car model',
      false,
    ],
    ['acrobotModel', model('Acrobot-v1', acrobotModel), 'Acrobot model', false],
    ['pointMassModel', model('PointMass-v0', pointMassModel), 'Point mass model', false],
    ['reacher2dModel', model('Reacher2D-v0', reacher2dModel), 'Reacher model', false],
    ['doubleIntegratorModel', model('DoubleIntegrator-v0', doubleIntegratorModel), 'Double integrator model', false],
    ['planarQuadrotorModel', model('PlanarQuadrotor-v0', planarQuadrotorModel), 'Planar quadrotor model', false],
    [
      'cartDoublePendulumModel',
      model('CartDoublePendulum-v0', cartDoublePendulumModel),
      'Cart double pendulum model',
      false,
    ],
  ]),
])

// ------------------------------------------------------------------------------------------------------- queries

/** A query over an environment registry: every given filter must hold (a list means any of its values). */
export interface EnvironmentQuery {
  readonly family?: EnvironmentFamily | readonly EnvironmentFamily[]
  readonly actions?: ActionKind | readonly ActionKind[]
  readonly observations?: ObservationKind | readonly ObservationKind[]
  readonly episodic?: boolean
  /** Multi-agent entries only (`true`), single-agent only (`false`), or both. */
  readonly multiAgent?: boolean
  /** Search-box text: every word must appear in the name, title, description, id or family. */
  readonly text?: string
}

const oneOf = <T>(filter: T | readonly T[] | undefined, value: T): boolean =>
  filter === undefined || (Array.isArray(filter) ? (filter as readonly T[]).includes(value) : filter === value)

/**
 * Whether an entry satisfies a query (what {@link whereEnvironments} tests entry by entry).
 *
 * @param entry - The entry.
 * @param query - The query.
 * @returns Whether every given filter holds.
 * @concept environments
 * @example
 * ```ts
 * import { lookup } from 'omai-engine/base'
 * import { environmentRegistry, matchesEnvironmentQuery } from 'omai-engine/environments'
 * const pendulum = lookup(environmentRegistry, 'pendulum')
 * matchesEnvironmentQuery(pendulum, { actions: 'continuous', episodic: false }) // => true
 * matchesEnvironmentQuery(pendulum, { family: ['gridWorld', 'toyText'] }) // => false
 * ```
 */
export function matchesEnvironmentQuery(entry: EnvironmentEntry, query: EnvironmentQuery): boolean {
  return (
    oneOf(query.family, entry.family) &&
    oneOf(query.actions, entry.actions) &&
    oneOf(query.observations, entry.observations) &&
    (query.episodic === undefined || query.episodic === entry.episodic) &&
    (query.multiAgent === undefined || query.multiAgent === (entry.multiAgent !== undefined)) &&
    matchesText(query.text, [entry.name, entry.title, entry.description, entry.spec.id, entry.family])
  )
}

/**
 * The entries of a registry that satisfy a query, in registration order; no query returns every entry.
 *
 * @param reg - The registry.
 * @param query - The filters.
 * @returns The matching entries.
 * @concept environments
 * @example
 * ```ts
 * import { environmentRegistry, whereEnvironments } from 'omai-engine/environments'
 * whereEnvironments(environmentRegistry, { actions: 'continuous', episodic: true }).map((e) => e.name) // => ['mountainCarContinuous', 'cartDoublePendulum']
 * whereEnvironments(environmentRegistry, { observations: 'grid' }).length // => 5
 * ```
 */
export function whereEnvironments(reg: EnvironmentRegistry, query: EnvironmentQuery = {}): EnvironmentEntry[] {
  return reg.entries.filter((e) => matchesEnvironmentQuery(e, query))
}

/**
 * A new environment registry with more definitions (an app's own), each derived into an entry as
 * {@link environmentEntry} does. A registration with a registered name replaces it; the given registry is left as
 * it was.
 *
 * @param reg - The registry to extend (usually {@link environmentRegistry}).
 * @param registrations - The definitions with their titles and hand-written tags.
 * @returns The extended registry.
 * @throws OmaiError with code `value` for a name added twice.
 * @concept environments
 * @example
 * ```ts
 * import { environmentRegistry, pendulum, registerEnvironments, whereEnvironments } from 'omai-engine/environments'
 * const mine = registerEnvironments(environmentRegistry, [
 *   { definition: pendulum, name: 'myPendulum', title: 'My pendulum', family: 'classicControl', episodic: false },
 * ])
 * mine.entries.length - environmentRegistry.entries.length // => 1
 * whereEnvironments(mine, { text: 'my pendulum' })[0]?.spec.id // => 'Pendulum-v1'
 * ```
 */
export function registerEnvironments(
  reg: EnvironmentRegistry,
  registrations: readonly EnvironmentRegistration[],
): EnvironmentRegistry {
  return register(reg, registrations.map(environmentEntry))
}

/**
 * The entry of a name in an environment registry.
 *
 * @param reg - The registry.
 * @param name - The entry's name.
 * @returns The entry, or `undefined` when nothing of that name is registered.
 * @concept environments
 * @example
 * ```ts
 * import { environmentRegistry, findEnvironment } from 'omai-engine/environments'
 * findEnvironment(environmentRegistry, 'taxi')?.spec.id // => 'Taxi-v4'
 * ```
 */
export function findEnvironment(reg: EnvironmentRegistry, name: string): EnvironmentEntry | undefined {
  return find(reg, name)
}
