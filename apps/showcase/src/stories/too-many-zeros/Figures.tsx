import { linspace, toFlat } from 'omai-engine/tensor'
import { annotations, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { TRUTH, X_RANGE } from './data.ts'
import { BANDS, rateAt, readZeros, trueRateAt } from './reading.ts'

/** The colours of the three bands of a zero's chance of being structural, and of the counts above zero. */
const BAND_COLOUR = ['tab0', 'tab4', 'tab1'] as const
const COUNT_COLOUR = 'fgSecondary'

/** How far below the axis the rows of zeros sit: zeros take turns in three rows, so neighbouring zeros do not cover each other. */
const LANES = [-0.55, -1.1, -1.65] as const

/** The tick labels of the count axis: whole numbers, and nothing in the rows of zeros. */
const countTick = (v: number): string => (v < -0.2 || !Number.isInteger(v) ? '' : String(v))

/** The values of the covariate a curve is drawn at. */
const GRID: readonly number[] = toFlat(linspace(X_RANGE[0], X_RANGE[1], 41))

/** The part of a key that names a colour. */
function Swatch(props: {
  readonly colour: string
  readonly label: string
  readonly shape?: 'dot' | 'diamond' | 'line' | 'dash' | 'dot-line'
}): ReactNode {
  const { colour, label, shape = 'dot' } = props
  const paint = `var(--wb-${colour.replace(/([A-Z])/g, '-$1').toLowerCase()})`
  const mark = (() => {
    if (shape === 'line') return <span className="inline-block h-0.5 w-5" style={{ background: paint }} />
    if (shape === 'dash')
      return <span className="inline-block h-0 w-5 border-t-2 border-dashed" style={{ borderColor: paint }} />
    if (shape === 'dot-line')
      return <span className="inline-block h-0 w-5 border-t-2 border-dotted" style={{ borderColor: paint }} />
    if (shape === 'diamond') return <span className="inline-block size-2.5 rotate-45" style={{ background: paint }} />
    return <span className="inline-block size-2.5 rounded-full" style={{ background: paint }} />
  })()
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      {mark}
      {label}
    </span>
  )
}

/** A row of keys above a chart. */
function Key(props: { readonly children: ReactNode }): ReactNode {
  return <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 pb-2 font-sans text-xs text-muted">{props.children}</div>
}

/**
 * The counts alone, before anything has run: each count above zero as a dot at its height, and the zeros in rows under the axis.
 *
 * @param props - The covariate and the counts.
 * @returns The chart.
 */
export function CountsFigure(props: { readonly x: readonly number[]; readonly y: readonly number[] }): ReactNode {
  const { x, y } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const up = y.map((_, i) => i).filter((i) => (y[i] as number) > 0)
      const zero = y
        .map((_, i) => i)
        .filter((i) => y[i] === 0)
        .sort((a, b) => (x[a] as number) - (x[b] as number))
      return cartesian(
        [
          points(
            [
              {
                x: up.map((i) => x[i] as number),
                y: up.map((i) => y[i] as number),
                name: 'counts above zero',
                colour: COUNT_COLOUR,
                size: 8,
                opacity: 0.85,
              },
              {
                x: zero.map((i) => x[i] as number),
                y: zero.map((_, k) => LANES[k % LANES.length] as number),
                name: 'zeros',
                colour: COUNT_COLOUR,
                size: 8,
                opacity: 0.85,
                symbol: 'rect',
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'x', limits: [X_RANGE[0] - 0.1, X_RANGE[1] + 0.1] },
          y: { label: 'count', limits: [-2.0, Math.max(...y) + 1], format: countTick, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [x, y, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}

/**
 * The counts after the run: the counts above zero in grey, each zero in a row under the axis coloured by the chance the model
 * gives it of being structural (a diamond where it really was), and three curves of the Poisson rate against x: the one the
 * counts were made from, the one EP fits with the switch and the one it fits without.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function ChancesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const { x, y } = result
      const up = y.map((_, i) => i).filter((i) => (y[i] as number) > 0)
      const { zeros } = readZeros(result)
      const band = (c: number): number => (c < BANDS[0] ? 0 : c < BANDS[1] ? 1 : 2)
      const sets = [0, 1, 2].flatMap((b) =>
        [false, true].flatMap((structural) => {
          const mine = zeros
            .map((z, k) => ({ z, k }))
            .filter(({ z }) => band(z.chance) === b && z.structural === structural)
          if (mine.length === 0) return []
          return [
            {
              x: mine.map(({ z }) => z.x),
              y: mine.map(({ k }) => LANES[k % LANES.length] as number),
              name: `zeros, band ${b}${structural ? ', structural' : ''}`,
              colour: BAND_COLOUR[b] as string,
              size: structural ? 11 : 9,
              opacity: 0.95,
              symbol: structural ? 'diamond' : 'circle',
            },
          ]
        }),
      )
      const curve = (
        rate: (x: number) => number,
        name: string,
        colour: string,
        dash: 'solid' | 'dashed' | 'dotted',
        width: number,
      ) => lines([{ x: [...GRID], y: GRID.map(rate), name, colour, dash, width }], { mode })
      return cartesian(
        [
          curve(trueRateAt, 'the rate the counts were made from', 'fgSecondary', 'dotted', 2),
          curve((v) => rateAt(result.withoutSwitch, v), 'without the switch', 'tab3', 'dashed', 2),
          curve((v) => rateAt(result.withSwitch, v), 'with the switch', 'fg', 'solid', 2.6),
          points(
            [
              {
                x: up.map((i) => x[i] as number),
                y: up.map((i) => y[i] as number),
                name: 'counts above zero',
                colour: COUNT_COLOUR,
                size: 7,
                opacity: 0.55,
              },
            ],
            { mode },
          ),
          points(sets, { mode }),
        ],
        {
          mode,
          figure,
          x: { label: 'x', limits: [X_RANGE[0] - 0.1, X_RANGE[1] + 0.1] },
          y: { label: 'count', limits: [-2.0, Math.max(...y) + 1], format: countTick, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return (
    <>
      <Key>
        <Swatch colour="tab0" label={`zero, chance under ${Math.round(BANDS[0] * 100)}%`} />
        <Swatch colour="tab4" label={`${Math.round(BANDS[0] * 100)}% to ${Math.round(BANDS[1] * 100)}%`} />
        <Swatch colour="tab1" label={`over ${Math.round(BANDS[1] * 100)}%`} />
        <Swatch colour="fg" shape="diamond" label="really structural" />
      </Key>
      <Key>
        <Swatch colour="fg" shape="line" label="rate with the switch" />
        <Swatch colour="tab3" shape="dash" label="rate without it" />
        <Swatch colour="fgSecondary" shape="dot-line" label="true rate" />
      </Key>
      <ResponsiveChart aspect={1.7} narrowAspect={0.8} build={build} />
    </>
  )
}

/**
 * The belief about the chance that a switch is structural after each number of counts, a curve for each, with the true chance as
 * a dashed line. The curves are Beta curves from the Dirichlet belief EP holds.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function BeliefsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const top = Math.max(...result.beliefs.flatMap((b) => b.density)) * 1.12
      const labels = result.beliefs.map((b, i) => {
        const peak = Math.max(...b.density)
        return {
          at: [result.grid[b.density.indexOf(peak)] as number, peak * 1.05] as [number, number],
          text: `${b.rows}`,
          i,
        }
      })
      return cartesian(
        [
          ...result.beliefs.map((b, i) =>
            lines(
              [
                {
                  x: [...result.grid],
                  y: [...b.density],
                  name: `${b.rows} counts`,
                  colour: 'tab0',
                  width: 1.4 + i * 0.4,
                  area: 0.06 + i * 0.06,
                  opacity: 0.4 + (0.6 * i) / Math.max(1, result.beliefs.length - 1),
                },
              ],
              { mode },
            ),
          ),
          lines(
            [
              {
                x: [TRUTH.inflation, TRUTH.inflation],
                y: [0, top],
                name: 'the chance the counts were made with',
                colour: 'tab1',
                dash: 'dashed',
                width: 1.8,
              },
            ],
            { mode },
          ),
          annotations({ labels: labels.map(({ at, text }) => ({ at, text })) }, { mode }),
        ],
        {
          mode,
          figure,
          x: { label: 'chance that a switch is structural', limits: [0, 0.8], format: 'percent' },
          y: { label: '', limits: [0, top], show: false, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.4} build={build} />
}
