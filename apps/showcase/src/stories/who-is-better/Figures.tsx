import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { PLAYERS } from './data.ts'
import { readResult } from './reading.ts'

/** The colour of each player, the same in both figures. */
const COLOURS = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4'] as const

/** The density of a bell curve with a mean and a standard deviation. */
const bell = (x: number, mean: number, sd: number): number =>
  Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))

/**
 * The ladder: one bell curve for each player, the best at the top, drawn at the rating's mean with its spread, and a ring where the
 * player's true skill is (the skill the games were made from).
 *
 * @param props - The result.
 * @returns The chart.
 */
export function LadderFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const reading = readResult(result)
      // the lowest rating at the bottom, so the best player is on top
      const rows = [...reading.ladder].reverse()
      const grid = Array.from({ length: 161 }, (_, i) => -3.5 + (7 * i) / 160)
      const tallest = Math.max(...rows.map((r) => bell(r.mean, r.mean, r.sd)))
      const height = 0.85 / tallest
      return cartesian(
        [
          band(
            rows.map((r, k) => ({
              x: grid,
              lower: grid.map(() => k),
              upper: grid.map((x) => k + height * bell(x, r.mean, r.sd)),
              name: r.name,
              colour: COLOURS[r.index] as string,
              opacity: 0.45,
            })),
            { mode },
          ),
          lines(
            rows.map((r, k) => ({
              x: grid,
              y: grid.map((x) => k + height * bell(x, r.mean, r.sd)),
              name: `__${r.name}`,
              colour: COLOURS[r.index] as string,
              width: 1.6,
            })),
            { mode },
          ),
          points(
            [
              {
                x: rows.map(() => -3.45),
                y: rows.map((_, k) => k + 0.5),
                labels: rows.map((r) => r.name),
                labelPosition: 'right',
                labelColour: 'fg',
                labelSize: 14,
                symbol: 'none',
                name: '__names',
              },
            ],
            { mode },
          ),
          points(
            [
              {
                x: rows.map((r) => r.truth),
                y: rows.map((_, k) => k),
                name: 'true skill',
                colour: 'fg',
                symbol: 'diamond',
                size: 11,
                outline: true,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'skill (higher is better)', limits: [-3.5, 3.5] },
          y: { limits: [-0.2, rows.length], show: false, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.7} narrowAspect={1.1} build={build} />
}

/**
 * How each player's rating changed as games came in: a line for the mean of the rating after the first game, the second and so
 * on, with a band of one standard deviation each side, starting from the same bell curve for everyone at zero games.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function TrajectoryFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const x = result.history.map((h) => h.after)
      const series = PLAYERS.map((name, p) => ({
        name,
        mean: result.history.map((h) => h.ratings[p]?.mean ?? Number.NaN),
        sd: result.history.map((h) => h.ratings[p]?.sd ?? Number.NaN),
      }))
      return cartesian(
        [
          band(
            series.map((s, p) => ({
              x,
              lower: s.mean.map((m, i) => m - (s.sd[i] ?? 0)),
              upper: s.mean.map((m, i) => m + (s.sd[i] ?? 0)),
              name: `__${s.name}-band`,
              colour: COLOURS[p] as string,
              opacity: 0.12,
            })),
            { mode },
          ),
          lines(
            series.map((s, p) => ({ x, y: s.mean, name: s.name, colour: COLOURS[p] as string, width: 2 })),
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'games played', limits: [0, Math.max(...x)] },
          y: { label: 'rating (mean, with one spread each side)', limits: [-2.6, 2.6] },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}
