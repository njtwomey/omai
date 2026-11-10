import { kde, type Panel, points, subplots } from 'omai-render/charts'
import { type Figure, gridSpec } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'

/** One method's draws as the pair plot is given them: the draws visible at the playhead, one array per coefficient. */
export interface PairMethod {
  readonly name: string
  /** The colour index (`tab<i>`), stable across the page. */
  readonly colour: number
  /** `columns[j][i]` is coefficient `j` of draw `i`, for the coefficients shown. */
  readonly columns: Readonly<Record<number, readonly number[]>>
}

const MAX_POINTS = 150

const thin = (v: readonly number[]): number[] =>
  v.length <= MAX_POINTS
    ? [...v]
    : Array.from({ length: MAX_POINTS }, (_, i) => v[Math.floor((i * (v.length - 1)) / (MAX_POINTS - 1))] as number)

/**
 * The pair plot of the chosen coefficients with the methods overlaid, one colour each: scatter plots of the draws below the
 * diagonal and each method's density on it. This is the page's one adapter between its results and a pair-plot view: it
 * draws with the library's chart layers the way `pairView` does (one colour per method where that view has one per
 * chain), and is the single place to replace with `PosteriorPairsView` of omai-render once that view is in.
 *
 * @param props - The methods with their visible draws, the coefficient indices to show and their labels.
 * @returns The chart.
 */
export function PairPlots(props: {
  readonly methods: readonly PairMethod[]
  readonly coordinates: readonly number[]
  readonly labels: readonly string[]
}): ReactNode {
  const { methods, coordinates, labels } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const k = coordinates.length
      const grid = gridSpec(k, k, {
        hspace: 12,
        wspace: 12,
        cells: coordinates.flatMap((_, r) =>
          coordinates.slice(0, r + 1).map((__, c) => ({ id: `p${r}_${c}`, row: r, col: c })),
        ),
      })
      const panels: Record<string, Panel> = {}
      for (const [r, jy] of coordinates.entries())
        for (const [c, jx] of coordinates.slice(0, r + 1).entries()) {
          const edge = {
            ...(r === k - 1 ? { x: { label: labels[jx] ?? `w[${jx}]` } } : {}),
            ...(c === 0 && r > 0 ? { y: { label: labels[jy] ?? `w[${jy}]` } } : {}),
          }
          const layers = methods.flatMap((m) => {
            const xs = m.columns[jx] ?? []
            const ys = m.columns[jy] ?? []
            if (xs.length === 0) return []
            if (r === c) {
              const spread = new Set(xs.map((v) => v.toPrecision(6))).size
              return spread < 3 ? [] : [kde(xs, { mode, colour: `tab${m.colour}`, fill: 0.12, name: m.name })]
            }
            return [
              points([{ x: thin(xs), y: thin(ys), name: m.name, colour: `tab${m.colour}`, size: 4, opacity: 0.5 }], {
                mode,
              }),
            ]
          })
          panels[`p${r}_${c}`] = { layers, options: edge }
        }
      return subplots(grid, panels, { figure, legend: false, tooltip: 'item' })
    },
    [methods, coordinates, labels, mode],
  )
  return <ResponsiveChart aspect={1} narrowAspect={1} build={build} />
}
