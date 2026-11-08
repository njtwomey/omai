/**
 * Example "Chaos: nearby paths part": the Lorenz equations solved from seven starting points that differ by one part in a
 * thousand in one coordinate. Each run is solved by `solveIvp` (an adaptive ODE solver) and read back at a fixed time step.
 * At first the seven paths are one; then they part and follow different turns of the butterfly. The figure draws the head
 * of each path with a short fading tail, seen from the front (x against z).
 */
import { lorenzField } from 'omai-engine/dynamics'
import { solveIvp } from 'omai-engine/dynamics/ode'
import { tensor, toFlat } from 'omai-engine/tensor'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { fadingTrail } from '../_shared/trail.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Chaos: nearby paths part',
  summary:
    'The Lorenz equations from seven starting points one part in a thousand apart: the paths separate and follow different turns of the butterfly.',
  readme: false,
  alt: 'Seven paths of the Lorenz system started a hair apart, seen from the front, with fading tails',
  home: { label: 'Chaos', fps: 20, seconds: 28 },
}

const PARTICLES = 7
const END = 40
const STEP = 0.02

interface Result {
  readonly t: number[]
  /** Each particle's x and z over time. */
  readonly paths: readonly { readonly x: number[]; readonly z: number[] }[]
}

export function run(): Outcome<Result> {
  const t = Array.from({ length: Math.round(END / STEP) + 1 }, (_, i) => i * STEP)
  const field = lorenzField()
  const solved = Array.from({ length: PARTICLES }, (_, k) => {
    const start = [1 + k * 1e-3, 1, 1]
    const solution = solveIvp(field, [0, END], tensor(start), { method: 'RK45', rtol: 1e-9, atol: 1e-11, tEval: t })
    return solution.ys.map((s) => Array.from(toFlat(s)))
  })
  const paths = solved.map((states) => ({ x: states.map((s) => s[0] ?? 0), z: states.map((s) => s[2] ?? 0) }))
  const last = (k: number): number[] => solved[k]?.at(-1) ?? []
  const apart = Math.hypot(...last(PARTICLES - 1).map((v, c) => v - (last(0)[c] ?? 0)))
  return {
    result: { t, paths },
    notes: [
      `${PARTICLES} particles over t = 0 to ${END}; the last is ${apart.toFixed(1)} from the first at the end (it started 0.006 away)`,
    ],
  }
}

/**
 * Solved live, as it is watched: each step advances every particle by 0.08 time units, one call of `solveIvp` for each,
 * from where it was left. (`run` solves each particle over the whole time at once.)
 */
export function live(): Live<Result> {
  const field = lorenzField()
  const t = [0]
  const states = Array.from({ length: PARTICLES }, (_, k) => [1 + k * 1e-3, 1, 1])
  const paths = states.map((s) => ({ x: [s[0] ?? 0], z: [s[2] ?? 0] }))
  return {
    step: () => {
      const from = t.at(-1) ?? 0
      states.forEach((state, k) => {
        const solved = solveIvp(field, [from, from + 0.08], tensor(state), { method: 'RK45', rtol: 1e-9, atol: 1e-11 })
        const end = Array.from(toFlat(solved.ys.at(-1) ?? tensor(state)))
        states[k] = end
        paths[k]?.x.push(end[0] ?? 0)
        paths[k]?.z.push(end[2] ?? 0)
      })
      t.push(from + 0.08)
      return from + 0.08 < END
    },
    result: () => ({ t, paths }),
  }
}

/** Played in the browser: each frame shows the paths a little further on. */
export const frames = (result: Result): number => Math.min(260, result.t.length - 1)

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const upTo =
    at === undefined ? result.t.length : Math.max(2, Math.round(((at + 1) / (frames(result) + 1)) * result.t.length))
  const front = cartesian(
    [
      lines(
        result.paths.flatMap((p, k) =>
          fadingTrail(p.x.slice(0, upTo), p.z.slice(0, upTo), {
            length: at === undefined ? 400 : 140,
            name: `particle ${k + 1}`,
            colour: `tab${k}`,
            width: 2,
          }),
        ),
        { mode },
      ),
      points(
        result.paths.map((p, k) => ({
          x: [p.x[upTo - 1] ?? 0],
          y: [p.z[upTo - 1] ?? 0],
          name: `now ${k + 1}`,
          size: 9,
          opacity: 1,
          colour: `tab${k}`,
        })),
        { mode },
      ),
    ],
    {
      mode,
      figure: figure({ width: 760, height: 520, padding: { top: 12, right: 16, bottom: 40, left: 46 } }),
      x: { label: 'x', limits: [-25, 25] },
      y: { label: 'z', limits: [0, 55] },
      legend: false,
      tooltip: false,
    },
  )
  return (
    <Figure width={760} height={576} title={meta.alt} mode={mode}>
      <Panel width={760} height={520} title="seven paths started one part in a thousand apart" still={front} />
    </Figure>
  )
}
