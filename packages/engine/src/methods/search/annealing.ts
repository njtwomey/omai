/**
 * Local search on a test function of the plane as a method of nothing but its configuration: simulated annealing, hill
 * climbing or late acceptance walking over Rastrigin, Himmelblau, Ackley, Rosenbrock or Beale, one move per step. The
 * walk's position, its energy, the best energy found and the annealing temperature are on the tape, and the objective
 * itself is a probe (a decision surface on a grid over the search box) for a page to draw the walk over.
 */

import { linspaceNumbers, varianceOf } from '../../lib/base/index.ts'
import { ackley, beale, himmelblau, rastrigin, rosenbrock, type TestFunction } from '../../lib/datasets/index.ts'
import { type DecisionSurface, decisionSurface } from '../../lib/exposition/index.ts'
import {
  geometricSchedule,
  hillClimbingAlgorithm,
  type LocalSearchState,
  lateAcceptanceAlgorithm,
  simulatedAnnealingAlgorithm,
  spaceProblem,
} from '../../lib/search/index.ts'
import { choice, dict, real, type SpaceValue } from '../../lib/space/index.ts'
import { item, tensor, toFlat } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../../lib/trace/index.ts'

/** The test functions {@link annealing} walks over. */
export const ANNEALING_OBJECTIVES = ['rastrigin', 'himmelblau', 'ackley', 'rosenbrock', 'beale'] as const

/** An objective of {@link annealing}. */
export type AnnealingObjective = (typeof ANNEALING_OBJECTIVES)[number]

/** The local searches {@link annealing} can run. */
export const ANNEALING_ALGORITHMS = ['simulated-annealing', 'hill-climbing', 'late-acceptance'] as const

/** A local search of {@link annealing}. */
export type AnnealingAlgorithm = (typeof ANNEALING_ALGORITHMS)[number]

/** The state of the walk: the library's local-search state over points `{ x, y }` of the search box. */
export type AnnealingState = LocalSearchState<SpaceValue>

// Each objective with the points of the grid whose adjacent points are a walk's neighbours: a step of a few tenths,
// fine enough to cross the objective's ripples one at a time.
const FUNCTIONS: Readonly<Record<AnnealingObjective, { readonly make: () => TestFunction; readonly points: number }>> =
  {
    rastrigin: { make: () => rastrigin(2), points: 41 },
    himmelblau: { make: himmelblau, points: 41 },
    ackley: { make: () => ackley(2), points: 129 },
    rosenbrock: { make: () => rosenbrock(2), points: 41 },
    beale: { make: beale, points: 41 },
  }

const RESOLUTION = 60

/** An objective with its surface on a grid over the search box, and the spread of its values (the energy scale). */
interface Landscape {
  readonly fn: TestFunction
  readonly surface: DecisionSurface
  readonly scale: number
}

// The landscape of an objective depends on nothing else, so it is computed once.
const landscapes = new Map<AnnealingObjective, Landscape>()

function landscapeOf(name: AnnealingObjective): Landscape {
  const known = landscapes.get(name)
  if (known !== undefined) return known
  const fn = FUNCTIONS[name].make()
  const [bx, by] = fn.bounds as readonly [readonly [number, number], readonly [number, number]]
  const x = linspaceNumbers(bx[0], bx[1], RESOLUTION)
  const y = linspaceNumbers(by[0], by[1], RESOLUTION)
  // Rows of the grid are y, columns x: values[i][j] is the objective at (x[j], y[i]).
  const flat = toFlat(fn.f(tensor(y.flatMap((v) => x.map((u) => [u, v])))))
  const values = y.map((_, i) => Array.from(flat.slice(i * RESOLUTION, (i + 1) * RESOLUTION)))
  const landscape: Landscape = {
    fn,
    surface: decisionSurface({ features: ['x', 'y'], x, y, values, field: { name: 'objective' } }),
    scale: Math.sqrt(varianceOf(Array.from(flat))) || 1,
  }
  landscapes.set(name, landscape)
  return landscape
}

const xy = (v: SpaceValue): [number, number] => {
  const p = v as { readonly x: number; readonly y: number }
  return [p.x, p.y]
}

/**
 * Local search over a test function of the plane. Simulated annealing takes a random neighbouring point, always when it
 * is lower and otherwise with probability `exp(-rise / temperature)`, cooling geometrically; hill climbing takes the
 * steepest downhill neighbour until none is lower, and ends in the nearest local minimum; late acceptance takes a
 * point no worse than the one fifty moves ago. Neighbours are the four adjacent points of a grid of 41 over the box.
 *
 * Knobs: `objective`, `algorithm`, and for annealing `temperature` (the starting temperature, in units of the
 * objective's spread over the box, so one setting suits every objective) and `cooling` (the factor the temperature is
 * multiplied by each step). The tape has `position` (kept in full), `energy`, `best` and, for annealing, `temperature`;
 * `objective/surface` is the objective on a grid, for a page to draw the walk over.
 *
 * @example
 * ```ts
 * import { annealing } from 'mdml-engine/methods/search'
 * import { session } from 'mdml-engine/trace'
 * const s = session(annealing, { mode: 'eager', steps: 50, config: { objective: 'himmelblau' } })
 * s.t // => 50
 * ```
 */
export const annealing: Method<AnnealingState, Config> = defineMethod<AnnealingState, Config>({
  name: 'search/annealing',
  config: dict({
    objective: choice(ANNEALING_OBJECTIVES, { default: 'rastrigin', label: 'objective' }),
    algorithm: choice(ANNEALING_ALGORITHMS, { default: 'simulated-annealing', label: 'search' }),
    temperature: real(0.01, 10, { log: true, default: 1, label: 'starting temperature' }),
    cooling: real(0.9, 0.9999, { default: 0.998, label: 'cooling per step' }),
  }),
  make: (c) => {
    const name = c['algorithm'] as AnnealingAlgorithm
    const { fn, scale } = landscapeOf(c['objective'] as AnnealingObjective)
    const space = dict({
      x: real(...(fn.bounds[0] as [number, number])),
      y: real(...(fn.bounds[1] as [number, number])),
    })
    // The searches maximise, so the value is minus the energy.
    const problem = spaceProblem(space, (v) => -item(fn.f(tensor(xy(v)))), {
      points: FUNCTIONS[c['objective'] as AnnealingObjective].points,
    })
    const start = (c['temperature'] as number) * scale
    const inner: Algorithm<AnnealingState> =
      name === 'simulated-annealing'
        ? simulatedAnnealingAlgorithm(problem, {
            schedule: geometricSchedule(start, c['cooling'] as number, start * 1e-6),
          })
        : name === 'hill-climbing'
          ? hillClimbingAlgorithm(problem)
          : lateAcceptanceAlgorithm(problem)
    return {
      name: `search/annealing/${name}`,
      config: { objective: c['objective'] as string, algorithm: name },
      init: (key) => inner.init(key),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log('position', xy(next.current))
        ctx.log('energy', -next.value)
        ctx.log('best', -next.bestValue)
        if (name === 'simulated-annealing') ctx.log('temperature', next.temperature / scale)
        return next
      },
      done: (s, t) => inner.done?.(s, t),
      // The walk's own record of values is as long as the run and is on the tape, so checkpoints leave it out.
      serialise: (s) => stateToJson({ ...s, history: [] }),
      deserialise: (json) => stateFromJson(json) as AnnealingState,
    }
  },
  contract: (c) => ({
    position: tensorSource({ role: 'state', shape: [2], description: 'the point the walk stands at, (x, y)' }),
    energy: scalar({ description: 'the objective at the current point', direction: 'lower' }),
    best: scalar({ description: 'the lowest objective found so far', direction: 'lower' }),
    temperature: scalar({ description: 'the annealing temperature, in units of the objective’s spread' }),
    'objective/surface': derived<AnnealingState>({
      description: 'the objective on a grid over the search box, whatever the state',
      cost: 'dear',
      compute: () => landscapeOf(c['objective'] as AnnealingObjective).surface,
    }),
  }),
  // The position is drawn point by point, so it is kept in full (`all` keeps summaries).
  defaults: {
    position: Log({ detail: 'full' }),
    energy: Log(),
    best: Log(),
    temperature: Log(),
  },
})
