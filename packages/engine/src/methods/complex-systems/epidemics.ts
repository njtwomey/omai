/**
 * An epidemic among people who move about a room: the agent-based SEIR simulation recorded day by day (who is
 * susceptible, exposed, infectious or recovered, and where), with the counts in each state beneath, and the SEIR
 * equations for the same disease as a well-mixed comparison.
 */

import { MdmlError } from '../../lib/base/index.ts'
import {
  type EpidemicOptions,
  type EpidemicState,
  epidemicAlgorithm,
  SEIR_STATES,
  seirCounts,
  seirCurves,
} from '../../lib/dynamics/index.ts'
import { type AgentFrame, agentSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'

/** Options of {@link epidemic}: the simulation's, and how long to run. */
export interface EpidemicRunOptions extends EpidemicOptions {
  /** Days to simulate at most (default 120); the run stops early when nobody is exposed or infectious. */
  readonly days?: number
  readonly key?: Key
}

/** The outcome of {@link epidemic}: the final people, the counts per day, the peak and the share ever infected. */
export interface EpidemicResult {
  readonly people: EpidemicState
  readonly counts: readonly (readonly [number, number, number, number])[]
  readonly peakInfectious: number
  readonly everInfected: number
}

// SEIR colours: susceptible grey-blue, exposed amber, infectious red, recovered green.
const COLOURS = ['#8aa4c8', '#f2b63d', '#e5484d', '#46a758']

/**
 * Simulate an epidemic among people moving about a room, recording every step (positions and states) with the shares
 * in each SEIR state as series; and, for comparison, the SEIR equations with the same periods and the transmission
 * rate implied by the room (beta = transmission times the people within reach per day).
 *
 * @param options - The people, room, distancing, reach, transmission, periods, days and key.
 * @returns The final people, the counts per recorded step, the peak and the share ever infected, with an
 *   `agentSteps` record whose series show the simulation (solid) beside the equations (dashed).
 * @throws MdmlError with code `value` for non-positive days.
 */
export function epidemic(options: EpidemicRunOptions = {}): Explained<EpidemicResult> {
  const { days = 120, key = makeKey(0), ...sim } = options
  if (!(days > 0)) throw new MdmlError('value', 'epidemic: days must be positive')
  const perDay = sim.stepsPerDay ?? 4
  const algorithm = epidemicAlgorithm(sim)
  const n = sim.people ?? 300
  const frames: AgentFrame[] = []
  const counts: [number, number, number, number][] = []
  const record = (s: EpidemicState, step: number): void => {
    frames.push({
      t: Math.round((step / perDay) * 100) / 100,
      x: Array.from(s.x),
      y: Array.from(s.y),
      state: Array.from(s.state),
    })
    counts.push(seirCounts(s))
  }
  let state = algorithm.init(key)
  record(state, 0)
  const total = Math.round(days * perDay)
  for (let t = 0; t < total; t++) {
    const run = runAlgorithm(algorithm, { key, maxSteps: 1, start: { state, step: t } })
    state = run.state
    record(state, t + 1)
    if (run.status === 'converged') break
  }
  const share = (c: number): number[] => counts.map((k) => (k[c] ?? 0) / n)
  const peakInfectious = Math.max(...counts.map((k) => k[2]))
  const everInfected = (n - (counts.at(-1)?.[0] ?? n)) / n
  // The well-mixed equations: each infectious person reaches (people - 1) * pi r^2 / area others per step.
  const area = (sim.width ?? 100) * (sim.height ?? 100)
  const reach = ((n - 1) * Math.PI * (sim.radius ?? 2.5) ** 2) / area
  const moving = 1 - (sim.distancing ?? 0)
  const beta = (sim.transmission ?? 0.08) * reach * perDay * moving
  const ode = seirCurves(
    { beta, sigma: 1 / (sim.incubation ?? 3), gamma: 1 / (sim.infectious ?? 7) },
    { infected: (sim.infected ?? 3) / n, days: frames.at(-1)?.t ?? days, points: frames.length },
  )
  return {
    people: state,
    counts,
    peakInfectious,
    everInfected,
    exposition: [
      agentSteps({
        method: 'SEIR agents',
        world: { width: sim.width ?? 100, height: sim.height ?? 100, wrap: false },
        states: [...SEIR_STATES],
        stateColours: COLOURS,
        frames,
        series: [
          ...SEIR_STATES.map((name, c) => ({ name, values: share(c), state: c })),
          ...SEIR_STATES.map((name, c) => ({
            name: `${name}, equations`,
            values: [ode.s, ode.e, ode.i, ode.r][c] ?? [],
            state: c,
            dashed: true,
          })),
        ],
        seriesName: `share of people (equations' R0 = ${ode.r0.toFixed(1)})`,
        timeName: 'day',
      }),
    ],
  }
}
