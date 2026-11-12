import { cartesian, errorBars, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'

/** The colour of each measurement, the same wherever it is drawn. */
const COLOURS = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5'] as const

/** One column of a table of rows. */
const column = (rows: readonly (readonly number[])[], j: number): number[] => rows.map((r) => r[j] ?? Number.NaN)

/** One view: a measurement against another, one dot per point. */
function View(props: {
  readonly y: readonly (readonly number[])[]
  readonly a: number
  readonly b: number
}): ReactNode {
  const { y, a, b } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian([points([{ x: column(y, a), y: column(y, b), name: 'points', colour: 'tab0', size: 9 }], { mode })], {
        mode,
        figure,
        x: { label: `measurement ${a + 1}`, limits: [-1.6, 1.6] },
        y: { label: `measurement ${b + 1}`, limits: [-1.6, 1.6] },
        legend: false,
        tooltip: false,
      }),
    [y, a, b, mode],
  )
  return <ResponsiveChart aspect={1.1} narrowAspect={1.1} build={build} />
}

/**
 * Two views of the measurements before anything runs: the first measurement against the second, and the third against the
 * fourth, one dot per point.
 *
 * @param props - The measurements, one row per point.
 * @returns The two charts side by side.
 */
export function PairFigure(props: { readonly y: readonly (readonly number[])[] }): ReactNode {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <View y={props.y} a={0} b={1} />
      <View y={props.y} a={2} b={3} />
    </div>
  )
}

/**
 * The coordinate the method found for each point against the true one, with a bar of one spread each side, and the line of a
 * perfect recovery.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function RecoveryFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          lines([{ x: [-1.9, 1.9], y: [-1.9, 1.9], name: 'a perfect recovery', colour: 'muted', width: 1.2 }], {
            mode,
          }),
          errorBars(
            result.truth.map((t, i) => ({ x: t, y: result.recovered[i] ?? Number.NaN, yErr: result.spread[i] ?? 0 })),
            { mode, name: 'recovered coordinate', colour: 'tab0' },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'true coordinate', limits: [-1.9, 1.9] },
          y: { label: 'coordinate found by the method', limits: [-2.4, 2.4] },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={1.35} narrowAspect={1} build={build} />
}

/** The vertical distance between the measurements' panels in the figure of the learned functions. */
const STEP = 3

/**
 * The learned functions: each measurement against the coordinate the method found for it, one above the other. The dots are the
 * measurements, the thick line is the function the method learned and the thin line is the function the data were made from.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CurvesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const outputs = result.curves.length
      // the lowest measurement at the bottom, the first at the top
      const offset = (d: number): number => (outputs - 1 - d) * STEP
      const order = result.truth.map((_, i) => i).sort((a, b) => (result.truth[a] ?? 0) - (result.truth[b] ?? 0))
      const outputsList = result.curves.map((_, d) => d)
      return cartesian(
        [
          lines(
            outputsList.map((d) => ({
              x: order.map((i) => result.truth[i] ?? Number.NaN),
              y: order.map((i) => (result.clean[i]?.[d] ?? Number.NaN) + offset(d)),
              name: `__true ${d + 1}`,
              colour: 'fg',
              width: 1.4,
              dash: 'dashed',
            })),
            { mode },
          ),
          lines(
            outputsList.map((d) => ({
              x: [...result.grid],
              y: (result.curves[d] ?? []).map((v) => v + offset(d)),
              name: `__learned ${d + 1}`,
              colour: COLOURS[d % COLOURS.length] as string,
              width: 2.6,
            })),
            { mode },
          ),
          points(
            outputsList.map((d) => ({
              x: [...result.recovered],
              y: column(result.y, d).map((v) => v + offset(d)),
              name: `measurement ${d + 1}`,
              colour: COLOURS[d % COLOURS.length] as string,
              size: 8,
            })),
            { mode },
          ),
          points(
            [
              {
                x: outputsList.map(() => -1.85),
                y: outputsList.map((d) => offset(d) + 1.35),
                labels: outputsList.map((d) => `measurement ${d + 1}`),
                labelPosition: 'right',
                labelColour: 'fg',
                labelSize: 13,
                symbol: 'none',
                name: '__names',
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'coordinate found by the method', limits: [-1.9, 1.9] },
          y: { limits: [-1.6, (outputs - 1) * STEP + 1.8], show: false, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={0.95} narrowAspect={0.7} build={build} />
}
