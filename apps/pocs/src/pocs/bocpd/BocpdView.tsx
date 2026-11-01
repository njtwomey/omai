import { metricSeries, tensorSeries } from 'mdml-engine/trace'
import { cartesian, heatmap, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, type LiveSession, useColourMode, useProbe, useRun, useScrubClick } from 'mdml-render/react'
import { type ReactNode, useMemo } from 'react'

const LEFT = figure({ width: 900, height: 300 })
const RIGHT = figure({ width: 300, height: 300 })
const FG = '#e6edf3'
const ORANGE = '#e8743b'
const BLUE = '#4a7fd8'
const GREY = '#7d8590'
const GREEN = '#3fb27f'

/** What the view needs of a series: the observations and where its segments truly start. */
export interface BocpdViewProps {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly y: readonly number[]
  readonly truth: readonly number[]
  /** Moves the viewed step: a click on the series or on the run-length plot goes to that step. */
  readonly onScrub: (t: number | null) => void
}

const gaussian = (v: number, mean: number, sd: number): number =>
  Math.exp(-0.5 * ((v - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))

/**
 * The default view of the detector: the series with the true and detected changepoints and the predictive density
 * beside it, and under it the run-length posterior over time (the MAP run length drawn on it) with the posterior now
 * beside that. Everything is read from the session's tape at the viewed step, so scrubbing moves all four together.
 */
export function BocpdView(props: BocpdViewProps): ReactNode {
  const { live, y, truth, onScrub } = props
  const mode = useColourMode()
  const run = useRun(live)
  const n = y.length
  const t = Math.max(0, Math.min(props.at ?? live.head, live.head, n - 1))
  const steps = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(steps, live.head, onScrub)
  const detections = useProbe(live, t, 'model/changepoints')
  const rows = useMemo(() => tensorSeries(run, 'bocpd/run-length'), [run])
  const map = useMemo(() => metricSeries(run, 'bocpd/map-run-length'), [run])
  const mean = useMemo(() => metricSeries(run, 'bocpd/predictive-mean'), [run])
  const sd = useMemo(() => metricSeries(run, 'bocpd/predictive-std'), [run])
  const lo = Math.min(...y)
  const hi = Math.max(...y)
  const pad = (hi - lo) * 0.08 || 1
  const yLimits = [lo - pad, hi + pad] as const
  // The vertical range of the run-length plots: the longest segment that truly occurs, with room, or the series.
  const bounds = [0, ...truth, n]
  const longest = Math.max(...bounds.slice(1).map((b, i) => b - (bounds[i] ?? 0)))
  const runMax = truth.length > 0 ? Math.min(n, Math.ceil(longest * 1.3)) : n

  const detected = (Array.isArray(detections.value) ? (detections.value as { index: number }[]) : []).map(
    (d) => d.index,
  )
  const shownTruth = truth.filter((c) => c <= t)

  const series = useMemo(() => {
    const x = y.map((_, i) => i)
    const mark = (at: number, colour: string, dash: 'dashed' | 'solid', name: string) => ({
      x: [at, at],
      y: [...yLimits],
      colour,
      dash,
      width: 1.6,
      ...(name === '' ? {} : { name }),
    })
    return cartesian(
      [
        lines(
          [
            { x: x.slice(t), y: y.slice(t), colour: GREY, width: 1.2, opacity: 0.45, name: 'to come' },
            { x: x.slice(0, t + 1), y: y.slice(0, t + 1), colour: BLUE, width: 1.8, name: 'series' },
            ...shownTruth.map((c, i) => mark(c, FG, 'dashed', i === 0 ? 'true changepoints' : '')),
            ...detected.map((c, i) => mark(c, ORANGE, 'solid', i === 0 ? 'detected changepoints' : '')),
            mark(t, FG, 'dashed', ''),
          ],
          { mode },
        ),
      ],
      { mode, figure: LEFT, x: { label: 'time', limits: [0, n - 1] }, y: { label: 'x', limits: yLimits } },
    )
  }, [y, t, shownTruth, detected, mode, n, yLimits[0], yLimits[1]])

  const predictive = useMemo(() => {
    const m = mean.values[t]
    const s = sd.values[t]
    const grid = Array.from({ length: 120 }, (_, i) => yLimits[0] + ((yLimits[1] - yLimits[0]) * i) / 119)
    const density =
      m === undefined || s === undefined || !(s > 0) ? grid.map(() => 0) : grid.map((v) => gaussian(v, m, s))
    const top = Math.max(0.05, ...density) * 1.1
    return cartesian(
      [
        lines([{ x: density, y: grid, colour: GREEN, width: 2, name: 'predictive' }], { mode }),
        lines([{ x: [0, top], y: [y[t] ?? 0, y[t] ?? 0], colour: FG, dash: 'dashed', width: 1.6, name: 'observed' }], {
          mode,
        }),
      ],
      { mode, figure: RIGHT, x: { label: 'density', limits: [0, top] }, y: { limits: yLimits, show: false } },
    )
  }, [mean, sd, t, y, mode, yLimits[0], yLimits[1]])

  const posterior = useMemo(() => {
    // Rows of the posterior by step: z[r][s] is p(run length r | the data to step s) on a log scale (six decades), so the
    // thin tails that carry the story show beside the one cell that holds most of the mass.
    const z: number[][] = Array.from({ length: runMax }, () => new Array<number>(n).fill(Number.NaN))
    for (const e of rows) {
      if (e.step > t || e.values === undefined) continue
      for (let r = 0; r < Math.min(runMax, e.values.length); r++) {
        const row = z[r]
        const p = e.values[r] ?? 0
        // Below the floor a cell is left empty (transparent), not painted the background colour, which would hide the
        // grid and marks beneath it.
        if (row !== undefined && p >= 1e-6) row[e.step] = 1 + Math.log10(p) / 6
      }
    }
    const x = y.map((_, i) => i)
    const marks = [
      ...shownTruth.map((c, i) => ({
        x: [c, c],
        y: [0, runMax],
        colour: FG,
        dash: 'dashed' as const,
        width: 1.6,
        ...(i === 0 ? { name: 'true changepoints' } : {}),
      })),
      { x: [t, t], y: [0, runMax], colour: FG, dash: 'dashed' as const, width: 1.2 },
    ]
    return cartesian(
      [
        heatmap(z, {
          x,
          y: Array.from({ length: runMax }, (_, i) => i),
          smooth: 'cells',
          ramp: ['#0d1117', '#17306b', '#2457b8', '#4a86e8', '#9cc4ff'],
          range: [0, 1],
          colourBar: false,
          mode,
        }),
        lines(
          [
            {
              x: map.steps.filter((s) => s <= t),
              y: map.values.filter((_, i) => (map.steps[i] ?? 0) <= t),
              colour: ORANGE,
              width: 2,
              name: 'MAP run length',
            },
            ...marks,
          ],
          { mode },
        ),
      ],
      { mode, figure: LEFT, x: { label: 'time', limits: [0, n - 1] }, y: { label: 'run length', limits: [0, runMax] } },
    )
  }, [rows, map, t, n, runMax, shownTruth, mode, y])

  const marginal = useMemo(() => {
    const row = rows.filter((e) => e.step <= t && e.values !== undefined).at(-1)?.values
    const p = Array.from(row ?? []).slice(0, runMax)
    const r = p.map((_, i) => i)
    const best = p.reduce((b, v, i) => (v > (p[b] ?? 0) ? i : b), 0)
    return cartesian(
      [
        lines([{ x: p, y: r, colour: ORANGE, width: 2, name: 'p(run length | data)' }], { mode }),
        lines([{ x: [0, 1], y: [best, best], colour: FG, dash: 'dashed', width: 1, opacity: 0.5 }], { mode }),
      ],
      {
        mode,
        figure: RIGHT,
        x: { label: 'p(run length | data to t)', limits: [0, 1] },
        y: { limits: [0, runMax], show: false },
      },
    )
  }, [rows, t, runMax, mode])

  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-2 lg:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
      <Chart option={series} figure={LEFT} onInit={seek} />
      <Chart option={predictive} figure={RIGHT} />
      <Chart option={posterior} figure={LEFT} onInit={seek} />
      <Chart option={marginal} figure={RIGHT} />
    </div>
  )
}
