/**
 * Example "Tracking a moving target": a target goes round a circle and is seen only through noisy position readings. A
 * Kalman filter (a constant-velocity model) folds in one reading a step and keeps an estimate of where the target is. The
 * figure shows the target, the last few readings and the filter's estimate with a fading tail, and the average distance of
 * the readings and of the estimate from the target so far. The filter runs as a recorded algorithm; on the site it is
 * stepped live.
 */
import { filtering } from 'omai-engine/dynamics'
import { key, normal } from 'omai-engine/random'
import { toArray, toFlat } from 'omai-engine/tensor'
import { iterate, type Run, recorder, tracePolicy } from 'omai-engine/trace'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { fadingTrail } from '../_shared/trail.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Tracking a moving target',
  summary: 'A Kalman filter following a target round a circle through noisy position readings, one reading at a time.',
  readme: false,
  alt: 'A target going round a circle, its noisy readings and the estimate of a Kalman filter following it',
  home: { label: 'Tracking', fps: 10, seconds: 16 },
}

const STEPS = 150
const RADIUS = 5
/** The angle the target moves through in a step. */
const TURN = 0.06
const NOISE = 1

interface Result {
  readonly truth: number[][]
  readonly readings: number[][]
  readonly estimate: number[][]
}

/** The target's path, the noisy readings of it and the model the filter is told (constant velocity, readings of position). */
function scenario() {
  const truth = Array.from({ length: STEPS }, (_, i) => [RADIUS * Math.cos(TURN * i), RADIUS * Math.sin(TURN * i)])
  const noise = toArray(normal(key(3), [STEPS, 2])) as number[][]
  const readings = truth.map((p, i) => [
    (p[0] ?? 0) + NOISE * (noise[i]?.[0] ?? 0),
    (p[1] ?? 0) + NOISE * (noise[i]?.[1] ?? 0),
  ])
  const model = {
    F: [
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    H: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    Q: [
      [0.001, 0, 0, 0],
      [0, 0.001, 0, 0],
      [0, 0, 0.003, 0],
      [0, 0, 0, 0.003],
    ],
    R: [
      [NOISE ** 2, 0],
      [0, NOISE ** 2],
    ],
    x0: [readings[0]?.[0] ?? 0, readings[0]?.[1] ?? 0, 0, 0],
    P0: [
      [NOISE ** 2, 0, 0, 0],
      [0, NOISE ** 2, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
  }
  return { truth, readings, model }
}

export function run(): Outcome<Result> {
  const { truth, readings, model } = scenario()
  const tape = recorder()
  const filtered = filtering.kalmanFilter(model, readings, {
    tracer: tape,
    policy: tracePolicy({ metrics: ['logLikelihood'] }),
  })
  const estimate = (toArray(filtered.means) as number[][]).map((m) => [m[0] ?? 0, m[1] ?? 0])
  const run: Run = tape.run()
  const result = { truth, readings, estimate }
  return { result, tapes: { run }, notes: [`mean distance from the target: ${summary(result, STEPS)}`] }
}

/** The target's distance from the readings and from the estimate, averaged over the first `n` steps. */
function summary(r: Result, n: number): string {
  const mean = (rows: number[][]): number =>
    rows
      .slice(0, n)
      .reduce(
        (s, p, i) => s + Math.hypot((p[0] ?? 0) - (r.truth[i]?.[0] ?? 0), (p[1] ?? 0) - (r.truth[i]?.[1] ?? 0)),
        0,
      ) / Math.max(1, Math.min(n, rows.length))
  return `readings ${mean(r.readings).toFixed(2)}, filter ${mean(r.estimate).toFixed(2)}`
}

/**
 * Filtered live: the readings are made first (they stand for what a sensor sends), and each step the library's Kalman
 * algorithm takes one more of them, so the estimate on screen is the filter at that step.
 */
export function live(): Live<Result> {
  const { truth, readings, model } = scenario()
  const { algorithm, steps } = filtering.kalmanFilterAlgorithm(model, readings)
  const states = iterate(algorithm, { maxSteps: steps })
  const estimate: number[][] = []
  const take = (state: { readonly means: readonly unknown[] }): void => {
    const last = state.means.at(-1)
    if (last === undefined) return
    const m = Array.from(toFlat(last as never))
    estimate.push([m[0] ?? 0, m[1] ?? 0])
  }
  const run: Live<Result> = {
    step: () => {
      const next = states.next()
      if (next.done === true) return false
      take(next.value.state as never)
      return next.value.status === undefined && estimate.length < STEPS
    },
    result: () => ({ truth, readings: readings.slice(0, estimate.length), estimate }),
  }
  run.step() // the first step, so there is always something to draw
  return run
}

/** Played in the browser: one frame per reading. */
export const frames = (result: Result): number => result.estimate.length - 1

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const n = at === undefined ? result.estimate.length : Math.min(at, result.estimate.length - 1) + 1
  const col = (rows: readonly (readonly number[])[], c: number): number[] => rows.slice(0, n).map((r) => r[c] ?? 0)
  const here = result.truth[Math.max(0, n - 1)] ?? [0, 0]
  const track = cartesian(
    [
      lines(
        [
          {
            x: result.truth.map((p) => p[0] ?? 0),
            y: result.truth.map((p) => p[1] ?? 0),
            name: 'circle',
            width: 1,
            colour: 'muted',
            dash: 'dotted',
            opacity: 0.5,
          },
        ],
        { mode },
      ),
      points(
        [
          {
            x: col(result.readings, 0).slice(-25),
            y: col(result.readings, 1).slice(-25),
            name: 'readings',
            size: 6,
            opacity: 0.5,
            colour: 'tab1',
          },
        ],
        { mode },
      ),
      lines(
        fadingTrail(col(result.estimate, 0), col(result.estimate, 1), {
          length: 40,
          name: 'estimate',
          colour: 'tab0',
          width: 3,
        }),
        { mode },
      ),
      points(
        [{ x: [here[0] ?? 0], y: [here[1] ?? 0], name: 'target', size: 11, opacity: 1, colour: 'fg', outline: false }],
        { mode },
      ),
    ],
    {
      mode,
      figure: figure({ width: 620, height: 520, padding: { top: 30, right: 12, bottom: 38, left: 44 } }),
      x: { label: 'x', limits: [-9, 9] },
      y: { label: 'y', limits: [-9, 9] },
      legend: true,
      tooltip: false,
    },
  )
  return (
    <Figure width={620} height={578} title={meta.alt} mode={mode}>
      <Panel
        width={620}
        height={520}
        title={`average distance from the target: ${summary({ ...result }, n)}`}
        still={track}
      />
    </Figure>
  )
}
