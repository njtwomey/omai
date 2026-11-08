/**
 * Example "A random walk against Hamiltonian Monte Carlo": two samplers draw from the same twisted density (Haario's
 * banana), 300 draws each from one chain. The random walk proposes a blind step from where it is; Hamiltonian Monte Carlo
 * follows the density's slope, so it takes long moves along the curve. The figure draws the draws so far over the true density,
 * and the effective sample size of each (how many independent draws its 300 are worth) is in the titles.
 */
import { foldIn, key } from 'omai-engine/random'
import {
  banana,
  hmc,
  hmcKernel,
  parameterDraws,
  randomWalkMetropolis,
  sampleChains,
  summary,
} from 'omai-engine/sampling'
import { toFlat } from 'omai-engine/tensor'
import { type Run, recorder, tracePolicy } from 'omai-engine/trace'
import { cartesian, contours, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { fadingTrail } from '../_shared/trail.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A random walk against Hamiltonian Monte Carlo',
  summary:
    'Two samplers on a twisted density: a random walk takes blind steps, Hamiltonian Monte Carlo follows the slope.',
  readme: false,
  alt: 'The draws of a random-walk sampler and of Hamiltonian Monte Carlo over the contours of a banana-shaped density',
  home: { label: 'Sampling', fps: 10, seconds: 20 },
}

const target = banana({ curvature: 0.1, scale: 3 })
const DRAWS = 300
const X: [number, number] = [-10, 10]
const Y: [number, number] = [-5, 7]
const GRID = 70

interface Result {
  readonly walk: number[][]
  readonly hamiltonian: number[][]
  /** The effective sample size of the first coordinate for each. */
  readonly ess: { readonly walk: number; readonly hamiltonian: number }
  /** The density on a grid over the window, for the contours. */
  readonly density: number[][]
}

const draws = (c: ReturnType<typeof hmc>): number[][] => {
  const x = toFlat(parameterDraws(c, 0))
  const y = toFlat(parameterDraws(c, 1))
  return Array.from({ length: DRAWS }, (_, i) => [x[i] ?? 0, y[i] ?? 0])
}

/** The density on a grid over the window, for the contours. */
const AXES = {
  x: Array.from({ length: GRID }, (_, i) => X[0] + ((X[1] - X[0]) * i) / (GRID - 1)),
  y: Array.from({ length: GRID }, (_, j) => Y[0] + ((Y[1] - Y[0]) * j) / (GRID - 1)),
}
const DENSITY = AXES.y.map((y) => AXES.x.map((x) => Math.exp(target.logDensity([x, y]))))

export function run(): Outcome<Result> {
  const tape = recorder()
  const walkChain = sampleChains(randomWalkMetropolis(target, { scale: 1.5 }), {
    key: key(1),
    numChains: 1,
    numWarmup: 200,
    numDraws: DRAWS,
    init: [[-6, 3]],
  })
  const hmcChain = hmc(target, {
    key: key(1),
    numChains: 1,
    numWarmup: 300,
    numDraws: DRAWS,
    numIntegrationSteps: 12,
    init: [[-6, 3]],
    tracer: tape,
    policy: tracePolicy({ metrics: ['logDensity'], schedule: { every: 5 } }),
  })
  const xs = Array.from({ length: GRID }, (_, i) => X[0] + ((X[1] - X[0]) * i) / (GRID - 1))
  const ys = Array.from({ length: GRID }, (_, j) => Y[0] + ((Y[1] - Y[0]) * j) / (GRID - 1))
  const density = ys.map((y) => xs.map((x) => Math.exp(target.logDensity([x, y]))))
  const run: Run = tape.run()
  const ess = {
    walk: summary(walkChain)[0]?.essBulk ?? Number.NaN,
    hamiltonian: summary(hmcChain)[0]?.essBulk ?? Number.NaN,
  }
  return {
    result: { walk: draws(walkChain), hamiltonian: draws(hmcChain), ess, density },
    tapes: { run },
    notes: [
      `effective sample size of x[0] from ${DRAWS} draws: random walk ${ess.walk.toFixed(0)}, Hamiltonian ${ess.hamiltonian.toFixed(0)}`,
    ],
  }
}

/**
 * Sampled live, as it is watched: the two kernels of `run` (a random walk and Hamiltonian Monte Carlo, with fixed step sizes)
 * from the same start, three draws of each a step. The effective sample size is left out, for it needs the whole chain.
 */
export function live(): Live<Result> {
  const walkKernel = randomWalkMetropolis(target, { scale: 1.5 })
  const hamiltonianKernel = hmcKernel(target, { stepSize: 0.2, numIntegrationSteps: 12 })
  let walk = walkKernel.init([-6, 3])
  let hamiltonian = hamiltonianKernel.init([-6, 3])
  const draws = { walk: [[-6, 3]], hamiltonian: [[-6, 3]] } as { walk: number[][]; hamiltonian: number[][] }
  let t = 0
  return {
    step: () => {
      for (let i = 0; i < 3; i++) {
        walk = walkKernel.step(foldIn(key(1), 2 * t), walk).state
        hamiltonian = hamiltonianKernel.step(foldIn(key(1), 2 * t + 1), hamiltonian).state
        draws.walk.push([...walk.position])
        draws.hamiltonian.push([...hamiltonian.position])
        t++
      }
      return draws.walk.length <= DRAWS
    },
    result: () => ({
      walk: draws.walk,
      hamiltonian: draws.hamiltonian,
      ess: { walk: Number.NaN, hamiltonian: Number.NaN },
      density: DENSITY,
    }),
  }
}

/** Played in the browser: each frame shows three more draws of each sampler. */
export const frames = (result: Result): number => Math.ceil(result.walk.length / 3) - 1

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const upTo = at === undefined ? result.walk.length : Math.min(result.walk.length, (at + 1) * 3)
  const peak = Math.max(...result.density.flat())
  const xs = Array.from({ length: GRID }, (_, i) => X[0] + ((X[1] - X[0]) * i) / (GRID - 1))
  const ys = Array.from({ length: GRID }, (_, j) => Y[0] + ((Y[1] - Y[0]) * j) / (GRID - 1))
  const panel = (rows: number[][]): unknown => {
    const seen = rows.slice(0, upTo)
    return cartesian(
      [
        contours(result.density, {
          x: xs,
          y: ys,
          levels: [0.011, 0.135, 0.607].map((r) => r * peak),
          labels: false,
          opacity: 0.6,
          mode,
        }),
        points(
          [{ x: seen.map((p) => p[0] ?? 0), y: seen.map((p) => p[1] ?? 0), name: 'draws', size: 3, opacity: 0.3 }],
          { mode },
        ),
        lines(
          fadingTrail(
            seen.map((p) => p[0] ?? 0),
            seen.map((p) => p[1] ?? 0),
            { length: 12, pieces: 6, name: 'last moves', colour: 'tab1', width: 2 },
          ),
          { mode },
        ),
        points(
          [
            {
              x: [seen.at(-1)?.[0] ?? 0],
              y: [seen.at(-1)?.[1] ?? 0],
              name: 'now',
              size: 9,
              opacity: 1,
              colour: 'fg',
              outline: false,
            },
          ],
          { mode },
        ),
      ],
      {
        mode,
        figure: figure({ width: 470, height: 360, padding: { top: 12, right: 12, bottom: 38, left: 42 } }),
        x: { label: 'x[0]', limits: X },
        y: { label: 'x[1]', limits: Y },
        legend: false,
        tooltip: false,
      },
    )
  }
  return (
    <Figure width={940} height={412} title={meta.alt} mode={mode}>
      <Panel
        x={0}
        width={470}
        height={360}
        title={
          Number.isFinite(result.ess.walk)
            ? `random walk: effective sample size ${result.ess.walk.toFixed(0)} of ${DRAWS}`
            : `random walk: ${upTo} draws`
        }
        still={panel(result.walk)}
      />
      <Panel
        x={470}
        width={470}
        height={360}
        title={
          Number.isFinite(result.ess.hamiltonian)
            ? `Hamiltonian: effective sample size ${result.ess.hamiltonian.toFixed(0)} of ${DRAWS}`
            : `Hamiltonian: ${upTo} draws`
        }
        still={panel(result.hamiltonian)}
      />
    </Figure>
  )
}
