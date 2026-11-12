import { bars, cartesian, histogram, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { PIECES, RANGE } from './data.ts'
import { REAL, trueShares } from './reading.ts'

/** The width of a slot of the histogram. */
const SLOT = 1

/** The colour of piece `k` (counting from 0). */
const colourOf = (k: number): string => `tab${k}`

const X_AXIS = { label: 'value', limits: [RANGE[0], RANGE[1]] as [number, number] }
const Y_AXIS = {
  label: 'numbers in each slot',
  limits: [-3, 15] as [number, number],
  // the strip of dots sits below zero and has no scale
  format: (v: number): string => (v < 0 ? '' : String(v)),
  splitLine: false,
}

/** A fixed scatter for the strip of dots: each dot's height is a fraction, so the dots do not overlap exactly. */
const strip = (values: readonly number[]): number[] => values.map((_, i) => -2.6 + 2 * ((i * 0.6180339887) % 1))

/**
 * Five sticks broken at random from the prior, as stacked bars: one bar per stick, one colour per piece.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function PriorFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          bars(
            result.prior.map((_, i) => `stick ${i + 1}`).reverse(),
            Array.from({ length: PIECES }, (_, k) => ({
              name: `piece ${k + 1}`,
              values: result.prior.map((row) => row[k] as number).reverse(),
              colour: colourOf(k),
            })),
            { mode, stacked: true, horizontal: true },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'share of the stick', limits: [0, 1] as [number, number] },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.3} build={build} />
}

/**
 * The numbers counted in slots, with one grey dot for each under the bars; with a result, the curve of each piece that holds real
 * weight and the sum of all the curves are drawn over them.
 *
 * @param props - The numbers, and the result once there is one.
 * @returns The chart.
 */
export function NumbersFigure(props: {
  readonly numbers: readonly number[]
  readonly result: Result | undefined
}): ReactNode {
  const { numbers, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const bins = histogram(numbers, {
        bins: Math.round((RANGE[1] - RANGE[0]) / SLOT),
        range: [RANGE[0], RANGE[1]],
        colour: 'fgSecondary',
        opacity: 0.4,
        mode,
      })
      const dots = points(
        [{ x: [...numbers], y: strip(numbers), name: 'numbers', colour: 'fgSecondary', size: 5, opacity: 0.8 }],
        {
          mode,
        },
      )
      if (result === undefined)
        return cartesian([bins, dots], { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false })
      // A curve is drawn as a count: its density times the number of numbers and the width of a slot.
      const scale = result.rows * SLOT
      const counts = (density: readonly number[]): number[] => density.map((v) => v * scale)
      const kept = result.weights.map((w, k) => ({ w, k })).filter((p) => p.w >= REAL)
      return cartesian(
        [
          bins,
          lines(
            kept.map((p) => ({
              x: [...result.grid],
              y: counts(result.curves[p.k] as number[]),
              name: `piece ${p.k + 1}`,
              colour: colourOf(p.k),
              width: 2,
              area: 0.18,
            })),
            { mode },
          ),
          lines([{ x: [...result.grid], y: counts(result.total), name: 'all pieces', colour: 'fg', width: 2.4 }], {
            mode,
          }),
          dots,
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      )
    },
    [numbers, result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}

/**
 * The weight of each piece as a bar, with a line of one standard deviation of the belief about it.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function WeightsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          bars(
            result.weights.map((_, k) => `${k + 1}`),
            [
              { name: 'weight found', values: [...result.weights], errors: [...result.spread], colour: 'tab0' },
              { name: 'true share', values: trueShares(result).map((v) => v ?? 0), colour: 'fgSecondary' },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'piece' },
          y: { label: 'weight', limits: [0, 0.5] as [number, number] },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.3} build={build} />
}
