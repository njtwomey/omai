import { bars, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import { CLEAR, REAL, type Result } from './compute.ts'
import { LENGTH_RANGE, WAIT_RANGE } from './data.ts'

/** The axes, fixed so a figure does not rescale when the result arrives. */
const X_AXIS = { label: 'eruption length (minutes)', limits: [LENGTH_RANGE[0], LENGTH_RANGE[1]] as [number, number] }
const Y_AXIS = { label: 'wait before it (minutes)', limits: [WAIT_RANGE[0], WAIT_RANGE[1]] as [number, number] }

/** The colour of a group by its place in the order of weight (the tab colours without the grey one). */
const COLOURS = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6', 'tab8'] as const
export const colourOf = (index: number): string => COLOURS[index % COLOURS.length] as string

/**
 * One dot per eruption, all grey: the data before anything has run.
 *
 * @param props - The eruption lengths and the waits.
 * @returns The chart.
 */
export function DataFigure(props: { readonly length: readonly number[]; readonly wait: readonly number[] }): ReactNode {
  const { length, wait } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [points([{ x: length, y: wait, name: 'eruptions', colour: 'fgSecondary', size: 6, opacity: 0.8 }], { mode })],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      ),
    [length, wait, mode],
  )
  return <ResponsiveChart aspect={1.5} narrowAspect={1.1} build={build} />
}

/**
 * The fit: each eruption a dot in the colour of the group it most likely belongs to (grey where two groups could own it), and
 * each group that holds real weight drawn as an ellipse two spreads out from its centre.
 *
 * @param props - The data and the result.
 * @returns The chart.
 */
export function GroupsFigure(props: {
  readonly length: readonly number[]
  readonly wait: readonly number[]
  readonly result: Result
}): ReactNode {
  const { length, wait, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const members = (keep: (i: number) => boolean): readonly number[] => length.map((_, i) => i).filter(keep)
      const dots = (indices: readonly number[], colour: string, name: string): ReturnType<typeof points> =>
        points(
          [
            {
              x: indices.map((i) => length[i] as number),
              y: indices.map((i) => wait[i] as number),
              name,
              colour,
              size: 6,
              opacity: 0.85,
            },
          ],
          { mode },
        )
      const clear = (i: number): boolean => (result.sure[i] as number) >= CLEAR
      const heavy = result.groups.flatMap((g, index) => (g.weight >= REAL ? [index] : []))
      const layers = [
        ...heavy.map((g) =>
          dots(
            members((i) => clear(i) && result.group[i] === g),
            colourOf(g),
            `group ${g + 1}`,
          ),
        ),
        dots(
          members((i) => !clear(i) || !heavy.includes(result.group[i] as number)),
          'fgSecondary',
          'unclear',
        ),
        ...heavy.map((g) => {
          const { ellipse } = result.groups[g] as Result['groups'][number]
          return lines([{ x: ellipse.x, y: ellipse.y, name: `ellipse ${g + 1}`, colour: colourOf(g), width: 2.6 }], {
            mode,
          })
        }),
      ]
      return cartesian(layers, { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false })
    },
    [length, wait, result, mode],
  )
  return <ResponsiveChart aspect={1.5} narrowAspect={1.1} build={build} />
}

/**
 * The share of the eruptions each piece of the model holds, heaviest first, in the colour of its group (grey when it holds
 * less than the real weight).
 *
 * @param props - The result.
 * @returns The chart.
 */
export function WeightsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const names = result.groups.map((_, i) => String(i + 1))
      // one stacked series per bar, so each bar has its own colour
      const series = result.groups.map((g, i) => ({
        name: `group ${i + 1}`,
        values: result.groups.map((_, j) => (j === i ? 100 * g.weight : 0)),
        colour: g.weight >= REAL ? colourOf(i) : 'fgSecondary',
      }))
      return cartesian([bars(names, series, { mode, stacked: true })], {
        mode,
        figure,
        x: { label: 'group (heaviest first)' },
        y: { label: 'share of the eruptions (%)', limits: [0, 100] },
        legend: false,
        tooltip: false,
      })
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2.4} narrowAspect={1.5} build={build} />
}
