import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result, Stage } from './compute.ts'
import { type Points, TRUTH, X_RANGE } from './data.ts'

const X_AXIS = { label: 'x', limits: [X_RANGE[0], X_RANGE[1]] as [number, number] }
const Y_AXIS = { label: 'y', limits: [-1.5, 3.5] as [number, number] }

/** A vertical line at `x` from the bottom to the top of the y axis. */
const cut = (
  x: number,
  name: string,
  dash: 'dashed' | 'dotted',
  colour: string,
  y: readonly [number, number] = Y_AXIS.limits,
) => ({
  x: [x, x],
  y: [...y],
  name,
  colour,
  dash,
  width: 1.4,
})

/**
 * The sixty points on axes that already span the range the trees will cover, so nothing rescales when the result arrives.
 *
 * @param props - The points.
 * @returns The chart.
 */
export function DataFigure(props: { readonly data: Points }): ReactNode {
  const { data } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const dots = points([{ x: data.x, y: data.y, groups: data.x.map(() => 'the data') }], { mode, colourOffset: 1 })
      return cartesian([dots], { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false })
    },
    [data, mode],
  )
  return <ResponsiveChart aspect={1.7} narrowAspect={1.2} build={build} />
}

/** One panel of the greedy growth: the data, the tree's curve with its band, and its cuts. */
function StagePanel(props: {
  readonly data: Points
  readonly grid: readonly number[]
  readonly stage: Stage
}): ReactNode {
  const { data, grid, stage } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          band([{ x: grid, lower: stage.lower, upper: stage.upper, opacity: 0.3 }], { mode }),
          lines(
            stage.cuts.map((c) => cut(c, 'cut', 'dashed', 'fgSecondary')),
            { mode },
          ),
          lines([{ x: grid, y: stage.mean, width: 2.6 }], { mode }),
          points([{ x: data.x, y: data.y, groups: data.x.map(() => 'the data') }], { mode, colourOffset: 1 }),
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      ),
    [data, grid, stage, mode],
  )
  return <ResponsiveChart aspect={1.15} narrowAspect={1.15} build={build} />
}

/**
 * The tree before any split and after each split, side by side: the data, the leaf means with a band of two standard
 * deviations, and the cuts so far.
 *
 * @param props - The points and the result.
 * @returns The panels.
 */
export function GreedyPanels(props: { readonly data: Points; readonly result: Result }): ReactNode {
  const { data, result } = props
  const labels = ['No split', ...result.greedy.cuts.map((_, k) => `After split ${k + 1}`)]
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))' }}>
      {result.greedy.stages.map((stage, k) => (
        <div key={labels[k]}>
          <div className="mb-1 px-1 font-sans text-xs text-muted">{labels[k]}</div>
          <StagePanel data={data} grid={result.grid} stage={stage} />
        </div>
      ))}
    </div>
  )
}

/**
 * The greedy tree's curve against the average over all trees: the average with its band, some trees drawn from the
 * posterior, the greedy tree's step curve (dashed), and the data.
 *
 * @param props - The points and the result.
 * @returns The chart.
 */
export function AverageFigure(props: { readonly data: Points; readonly result: Result }): ReactNode {
  const { data, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const greedy = result.greedy.stages.at(-1) as Stage
      return cartesian(
        [
          band([{ x: result.grid, lower: result.posterior.lower, upper: result.posterior.upper, opacity: 0.25 }], {
            mode,
          }),
          lines(
            result.sampled.curves.map((y) => ({ x: result.grid, y, width: 1, opacity: 0.25, colour: 'tab0' })),
            { mode },
          ),
          lines([{ x: result.grid, y: result.posterior.mean, name: 'average over trees', width: 2.8 }], { mode }),
          lines(
            [
              {
                x: result.grid,
                y: greedy.mean,
                name: 'the greedy tree',
                width: 2,
                dash: 'dashed',
                colour: 'fgSecondary',
              },
            ],
            { mode },
          ),
          points([{ x: data.x, y: data.y, groups: data.x.map(() => 'the data') }], { mode, colourOffset: 1 }),
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      )
    },
    [data, result, mode],
  )
  return <ResponsiveChart aspect={1.55} narrowAspect={1.1} build={build} />
}

/**
 * The chance that a cut is at each candidate position, as a profile along x, with the greedy tree's cuts and the cuts the
 * data were made with marked.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CutsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const top = Math.max(...result.posterior.usage) * 1.15
      const span = [0, top] as const
      return cartesian(
        [
          lines([{ x: result.posterior.thresholds, y: result.posterior.usage, step: 'middle', area: 0.3, width: 2 }], {
            mode,
          }),
          lines(
            TRUTH.cuts.map((c) => cut(c, 'the true cut', 'dotted', 'fgSecondary', span)),
            { mode },
          ),
          lines(
            result.greedy.cuts.map((c) => cut(c.threshold, 'greedy cut', 'dashed', 'tab1', span)),
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: X_AXIS,
          y: { label: 'chance of a cut here', limits: [0, top] as [number, number] },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.3} build={build} />
}
