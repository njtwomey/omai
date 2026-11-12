import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result, Taken } from './compute.ts'
import { ROWS, TOLERANCE } from './data.ts'

const finite = (v: number): boolean => Number.isFinite(v)

/** A tick label for whole numbers only. */
const whole = (v: number): string => (Number.isInteger(v) ? String(v) : '')

/**
 * The factor graph of the model, drawn before anything runs: one circle for the unknown, one square for the prior and one for each
 * of the measurements, and a line from each factor to the unknown it mentions. It shows no values, so it does not change when the
 * measurements are drawn.
 *
 * @returns The picture.
 */
export function GraphArt(): ReactNode {
  const width = 560
  const height = 290
  const cx = width / 2
  const cy = 118
  const factors = ['prior', ...Array.from({ length: ROWS }, (_, i) => `y${i + 1}`)]
  // the prior at the top, the measurements in a row below
  const spot = (i: number): readonly [number, number] => {
    if (i === 0) return [cx, 30]
    return [60 + (440 * (i - 1)) / (ROWS - 1), 232]
  }
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="A factor graph: one circle for the unknown location and one square for each of the prior and the six measurements, each joined to the circle"
      className="block h-auto w-full text-fg"
    >
      <title>The factor graph of the model</title>
      <g stroke="var(--wb-muted)" strokeWidth="1.5">
        {factors.map((name, i) => {
          const [x, y] = spot(i)
          return <line key={name} x1={x} y1={y} x2={cx} y2={cy} />
        })}
      </g>
      <g fontFamily="system-ui, sans-serif" fontSize="13" fill="var(--wb-fg)" textAnchor="middle">
        {factors.map((name, i) => {
          const [x, y] = spot(i)
          return (
            <g key={name}>
              <rect
                x={x - 13}
                y={y - 13}
                width="26"
                height="26"
                rx="3"
                fill="var(--wb-surface)"
                stroke="var(--wb-accent)"
                strokeWidth="2"
              />
              <text x={x} y={y + 4} fontSize="12">
                {i === 0 ? 'f0' : `f${i}`}
              </text>
              <text
                x={i === 0 ? x + 20 : x}
                y={i === 0 ? y + 4 : y + 30}
                fill="var(--wb-muted)"
                textAnchor={i === 0 ? 'start' : 'middle'}
              >
                {i === 0 ? 'the prior' : `measurement ${i}`}
              </text>
            </g>
          )
        })}
        <circle cx={cx} cy={cy} r="27" fill="var(--wb-surface)" stroke="var(--wb-fg)" strokeWidth="2.5" />
        <text x={cx} y={cy + 6} fontSize="20" fontFamily="Georgia, serif" fontStyle="italic">
          {'θ'}
        </text>
        <text x={cx + 40} y={cy + 5} fill="var(--wb-muted)" textAnchor="start">
          the unknown location
        </text>
      </g>
    </svg>
  )
}

const THETA = { label: 'the location θ' }

/**
 * One update taken apart, on one axis: the cavity (what everything except this factor says), the real factor as a function of the
 * location (scaled to the height of the cavity, since it is not a density), the tilted curve (their product) and the bell with
 * the tilted curve's mean and variance.
 *
 * @param props - The update.
 * @returns The chart.
 */
export function UpdateFigure(props: { readonly taken: Taken }): ReactNode {
  const { taken } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const top = Math.max(...taken.cavityDensity, ...taken.tiltedDensity)
      const scale = top / Math.max(...taken.factorValues)
      const x = taken.grid
      const centre = taken.cavity.shift / taken.cavity.precision
      const reach = 1 / Math.sqrt(taken.cavity.precision)
      return cartesian(
        [
          lines(
            [
              { x, y: taken.cavityDensity, name: 'cavity', colour: 'tab0', width: 2.2, area: 0.12 },
              {
                x,
                y: taken.factorValues.map((v) => v * scale),
                name: 'the real factor (scaled)',
                colour: 'fgSecondary',
                width: 1.6,
                dash: 'dotted',
              },
              { x, y: taken.tiltedDensity, name: 'tilted: cavity times factor', colour: 'tab1', width: 3, area: 0.18 },
              {
                x,
                y: taken.matchedDensity,
                name: 'bell with the same mean and variance',
                colour: 'fg',
                width: 2.2,
                dash: 'dashed',
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { ...THETA, limits: [centre - 3.8 * reach, centre + 3.8 * reach] },
          y: { label: 'density (the factor: scaled)', limits: [0, top * 1.15], show: false, splitLine: false },
          legend: true,
          tooltip: false,
        },
      )
    },
    [taken, mode],
  )
  return <ResponsiveChart aspect={1.8} narrowAspect={1.05} build={build} />
}

/**
 * The belief after every sweep: its mean as a line and one standard deviation either side as a band, with the exact mean dashed.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function SweepsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const sweep = result.beliefs.sweep.filter(
        (_, i) => finite(result.beliefs.mean[i] as number) && finite(result.beliefs.sd[i] as number),
      )
      const mean = result.beliefs.mean.filter((m, i) => finite(m) && finite(result.beliefs.sd[i] as number))
      const sd = result.beliefs.sd.filter((s, i) => finite(s) && finite(result.beliefs.mean[i] as number))
      const centre = result.exact.mean
      const reach = Math.max(1.5, 4 * result.exact.sd)
      const lo = Math.max(centre - 6, Math.min(centre - reach, ...mean.map((m, i) => m - (sd[i] as number)).slice(1)))
      const hi = Math.min(centre + 6, Math.max(centre + reach, ...mean.map((m, i) => m + (sd[i] as number)).slice(1)))
      const last = result.beliefs.sweep.at(-1) ?? 1
      return cartesian(
        [
          band(
            [
              {
                x: sweep,
                lower: mean.map((m, i) => m - (sd[i] as number)),
                upper: mean.map((m, i) => m + (sd[i] as number)),
                opacity: 0.22,
              },
            ],
            { mode },
          ),
          lines(
            [
              {
                x: [1, last],
                y: [centre, centre],
                name: 'exact mean',
                colour: 'fgSecondary',
                dash: 'dashed',
                width: 1.5,
              },
              { x: sweep, y: mean, name: 'EP: mean', colour: 'tab0', width: 2.6, markers: true },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'sweeps', limits: [1, Math.max(2, last)], format: whole },
          y: { label: 'belief about θ (mean ± one sd)', limits: [lo, hi] },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2} narrowAspect={1.25} build={build} />
}

/**
 * The largest change of any belief over each sweep, on a log axis, with the tolerance the run stops at as a dashed line.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function ChangeFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const sweep = result.beliefs.sweep.filter(
        (_, i) => (result.beliefs.change[i] as number) > 0 && finite(result.beliefs.change[i] as number),
      )
      const change = result.beliefs.change.filter((c) => c > 0 && finite(c))
      const last = result.beliefs.sweep.at(-1) ?? 1
      return cartesian(
        [
          lines(
            [
              {
                x: [1, last],
                y: [TOLERANCE, TOLERANCE],
                name: 'tolerance',
                colour: 'fgSecondary',
                dash: 'dashed',
                width: 1.5,
              },
              { x: sweep, y: change, name: 'largest change', colour: 'tab1', width: 2.6, markers: true },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'sweeps', limits: [1, Math.max(2, last)], format: whole },
          y: { label: 'largest change of a belief (in standard deviations)', scale: 'log' },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2.2} narrowAspect={1.3} build={build} />
}

/**
 * The end of the run: the exact posterior of the location (from a grid), EP's bell, and the six measurements as dots on the axis,
 * coloured by whether the generator made them clutter (EP is not told). The location the data were made around is dashed.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function AnswerFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const peak = Math.max(...result.exact.density, ...result.exact.epDensity.filter(finite))
      const below = -0.1 * peak
      const x = result.exact.grid
      const lo = Math.min(x[0] as number, ...result.points) - 0.4
      const hi = Math.max(x.at(-1) as number, ...result.points) + 0.4
      const at = (clutter: boolean): number[] =>
        result.points.flatMap((p, i) => (result.clutter[i] === clutter ? [p] : []))
      const dots = (clutter: boolean, name: string): ReturnType<typeof points> =>
        points(
          [
            {
              x: at(clutter),
              y: at(clutter).map((_, k) => below * (0.35 + 0.55 * ((k * 0.618) % 1))),
              name,
              colour: clutter ? 'tab2' : 'tab0',
              size: 7,
              opacity: 0.9,
            },
          ],
          { mode },
        )
      return cartesian(
        [
          lines(
            [
              { x, y: [...result.exact.density], name: 'exact posterior', colour: 'tab1', width: 2.6, area: 0.2 },
              { x, y: [...result.exact.epDensity], name: 'EP: one bell', colour: 'fg', width: 2.4, dash: 'dashed' },
              {
                x: [result.truth, result.truth],
                y: [0, peak * 1.05],
                name: 'location used to make the data',
                colour: 'fgSecondary',
                width: 1.4,
                dash: 'dotted',
              },
            ],
            { mode },
          ),
          dots(false, 'a reading of the location'),
          dots(true, 'clutter'),
        ],
        {
          mode,
          figure,
          x: { ...THETA, limits: [lo, hi] },
          y: { label: 'density', limits: [below * 1.25, peak * 1.15], show: false, splitLine: false },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.1} build={build} />
}
