/**
 * Example "Finding the moment a stream changes": Bayesian online changepoint detection on a stream whose mean shifts twice.
 * The detector runs as a session, one reading per step, and its tape holds, at every step, the belief about how long the
 * current stretch has lasted (the "run length"), the most likely run length and what it expects the next reading to be.
 * The figure is drawn from that tape: the stream with the true and the detected changes, the prediction beside it, the
 * run-length belief over time with the most likely run length on it, and the belief at the step shown.
 */
import { key, normal } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { bocpdDetector, changepointData } from 'omai-engine/timeseries'
import { metricSeries, type Run, session as sessionOf, tensorSeries } from 'omai-engine/trace'
import { cartesian, heatmap, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { sequentialRamp } from 'omai-render/theme'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Finding the moment a stream changes',
  summary:
    'Bayesian online changepoint detection read back from its tape: the stream, its prediction, and the belief about how long the current stretch has lasted.',
  readme: false,
  alt: 'A stream whose mean shifts twice with the changes a detector found, its prediction, and the run-length belief over time and at one step',
  home: { label: 'Change points', fps: 10, seconds: 12 },
}

/** The stretches of the stream: how many readings, and their mean. */
const SEGMENTS = [
  { length: 50, mean: 0 },
  { length: 45, mean: 3 },
  { length: 55, mean: -1.5 },
] as const
const SIGMA = 1

interface Result {
  readonly series: number[]
  /** Where each stretch really starts, and where the detector put the changes. */
  readonly changes: number[]
  readonly detected: number[]
  /** From the tape, at each step: the run-length belief (the probability of each run length), the most likely run length and the prediction. */
  readonly belief: number[][]
  readonly likely: number[]
  readonly predictionMean: number[]
  readonly predictionStd: number[]
}

/** The stream the detector reads and the detector set up for it. */
function scenario() {
  const readings = SEGMENTS.reduce((n, s) => n + s.length, 0)
  const noise = Array.from(toFlat(normal(key(5), [readings])))
  let t = 0
  const series = SEGMENTS.flatMap((s) => Array.from({ length: s.length }, () => s.mean + SIGMA * (noise[t++] ?? 0)))
  const changes = SEGMENTS.slice(0, -1).map((_, i) => SEGMENTS.slice(0, i + 1).reduce((n, s) => n + s.length, 0))
  const data = changepointData(series, changes)
  const detector = bocpdDetector.with({
    model: { kind: 'mean', sigma: SIGMA, priorMean: 0, priorStd: 5 },
    hazard: { kind: 'constant', rate: 0.02 },
  })
  return { series, changes, data, detector }
}

/** What the figure reads from a tape: the belief at each step, the most likely run length and the prediction. */
function fromTape(tape: Run) {
  return {
    belief: tensorSeries(tape, 'bocpd/run-length').map((e) => Array.from(e.values ?? [])),
    likely: metricSeries(tape, 'bocpd/map-run-length').values,
    predictionMean: metricSeries(tape, 'bocpd/predictive-mean').values,
    predictionStd: metricSeries(tape, 'bocpd/predictive-std').values,
  }
}

export function run(): Outcome<Result> {
  const { series, changes, data, detector } = scenario()
  // The session is made from the method so that a checkpoint is kept every 50 steps (the tape stays small).
  const session = sessionOf(detector.method(data), { key: key(1), steps: 1000, checkpointEvery: 50 })
  session.advance()
  const detected = detector.fit(data).changepoints()
  return {
    result: { series, changes, detected: [...detected], ...fromTape(session.run()) },
    tapes: { session },
    notes: [`${series.length} readings; true changes at ${changes.join(', ')}; detected at ${detected.join(', ')}`],
  }
}

/**
 * Detected live, as it is watched: each step is two more readings of the same session `run` uses, and the belief, the
 * prediction and the changes found so far are read from its tape and its `model/changepoints` record.
 */
export function live(): Live<Result> {
  const { series, changes, data, detector } = scenario()
  const session = sessionOf(detector.method(data), { key: key(1), steps: 1000, checkpointEvery: 50 })
  let detected: number[] = []
  const run: Live<Result> = {
    step: () => {
      session.advance({ steps: 2 })
      const found = session.probe(session.t, 'model/changepoints')
      detected = Array.isArray(found) ? (found as { index: number }[]).map((d) => d.index) : []
      return session.status === undefined
    },
    result: () => ({ series, changes, detected, ...fromTape(session.run()) }),
  }
  run.step() // the first step, so there is always something to draw
  return run
}

/** Played in the browser: one frame per two readings. */
export const frames = (result: Result): number => Math.floor((result.belief.length - 1) / 2)

const gaussian = (v: number, mean: number, sd: number): number =>
  Math.exp(-0.5 * ((v - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))

const WIDE = figure({ width: 640, height: 280, padding: { top: 12, right: 12, bottom: 34, left: 48 } })
const NARROW = figure({ width: 260, height: 280, padding: { top: 12, right: 12, bottom: 34, left: 20 } })

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const n = result.series.length
  const t = at === undefined ? result.belief.length - 1 : Math.min(2 * at, result.belief.length - 1)
  const lo = Math.min(...result.series)
  const hi = Math.max(...result.series)
  const yLimits: [number, number] = [lo - 0.5, hi + 0.5]
  const runMax = 60
  const trueSoFar = result.changes.filter((c) => c <= t)
  const foundSoFar = result.detected.filter((c) => c <= t)
  const mark = (x: number, colour: string, name: string, dash: 'dashed' | 'solid', y: readonly [number, number]) => ({
    x: [x, x],
    y: [...y],
    colour,
    dash,
    width: 1.4,
    ...(name === '' ? {} : { name }),
  })
  const stream = cartesian(
    [
      lines(
        [
          {
            x: Array.from({ length: t + 1 }, (_, i) => i),
            y: result.series.slice(0, t + 1),
            name: 'stream',
            width: 1.8,
            colour: 'tab0',
          },
          ...trueSoFar.map((c, i) => mark(c, 'muted', i === 0 ? 'true change' : '', 'dashed', yLimits)),
          ...foundSoFar.map((c, i) => mark(c, 'tab1', i === 0 ? 'detected change' : '', 'solid', yLimits)),
        ],
        { mode },
      ),
    ],
    {
      mode,
      figure: WIDE,
      x: { label: 'reading', limits: [0, n - 1] },
      y: { label: 'value', limits: yLimits },
      legend: true,
      tooltip: false,
    },
  )
  const mean = result.predictionMean[t] ?? 0
  const sd = result.predictionStd[t] ?? 1
  const grid = Array.from({ length: 120 }, (_, i) => yLimits[0] + ((yLimits[1] - yLimits[0]) * i) / 119)
  const density = grid.map((v) => gaussian(v, mean, sd))
  const top = Math.max(0.05, ...density) * 1.1
  const prediction = cartesian(
    [
      lines([{ x: density, y: grid, name: 'expected next reading', width: 2, colour: 'tab2' }], { mode }),
      lines(
        [
          {
            x: [0, top],
            y: [result.series[t] ?? 0, result.series[t] ?? 0],
            name: 'reading',
            width: 1.6,
            colour: 'fg',
            dash: 'dashed',
          },
        ],
        { mode },
      ),
    ],
    {
      mode,
      figure: NARROW,
      x: { label: 'density', limits: [0, top] },
      y: { limits: yLimits, show: false },
      legend: false,
      tooltip: false,
    },
  )
  // The belief over time: the probability of each run length (rows) at each reading (columns) up to the one shown, on a log scale
  // over six decades so the thin tails show beside the one cell that holds most of the mass.
  const cells: number[][] = Array.from({ length: runMax }, () => new Array<number>(n).fill(Number.NaN))
  for (let s = 0; s <= t; s++) {
    const row = result.belief[s] ?? []
    for (let r = 0; r < Math.min(runMax, row.length); r++) {
      const p = row[r] ?? 0
      const target = cells[r]
      if (target !== undefined && p >= 1e-6) target[s] = 1 + Math.log10(p) / 6
    }
  }
  const believed = cartesian(
    [
      heatmap(cells, {
        x: Array.from({ length: n }, (_, i) => i),
        y: Array.from({ length: runMax }, (_, i) => i),
        smooth: 'cells',
        ramp: sequentialRamp('blues', mode),
        range: [0, 1],
        colourBar: false,
        mode,
      }),
      lines(
        [
          {
            x: Array.from({ length: t + 1 }, (_, i) => i),
            y: result.likely.slice(0, t + 1),
            name: 'most likely run length',
            width: 2,
            colour: 'tab1',
          },
          ...trueSoFar.map((c, i) => mark(c, 'fg', i === 0 ? 'true change' : '', 'dashed', [0, runMax])),
        ],
        { mode },
      ),
    ],
    {
      mode,
      figure: WIDE,
      x: { label: 'reading', limits: [0, n - 1] },
      y: { label: 'run length', limits: [0, runMax] },
      legend: false,
      tooltip: false,
    },
  )
  const row = (result.belief[t] ?? []).slice(0, runMax)
  const now = cartesian(
    [lines([{ x: row, y: row.map((_, i) => i), name: 'belief', width: 2, colour: 'tab1' }], { mode })],
    {
      mode,
      figure: NARROW,
      x: { label: 'probability now', limits: [0, 1] },
      y: { limits: [0, runMax], show: false },
      legend: false,
      tooltip: false,
    },
  )
  return (
    <Figure width={900} height={660} title={meta.alt} mode={mode}>
      <Panel x={0} width={640} height={280} title="the stream, the true changes and the ones found" still={stream} />
      <Panel x={640} width={260} height={280} title="what it expects next" still={prediction} />
      <g transform="translate(0, 320)">
        <Panel
          x={0}
          width={640}
          height={280}
          title="belief about how long the current stretch has lasted"
          still={believed}
        />
        <Panel x={640} width={260} height={280} title="the belief at this reading" still={now} />
      </g>
    </Figure>
  )
}
