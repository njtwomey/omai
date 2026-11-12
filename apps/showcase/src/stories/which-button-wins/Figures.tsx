import { annotations, cartesian, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { CALL } from './data.ts'
import { rate } from './reading.ts'

/** The colour of version A and of version B, the same in every figure of the page. */
const COLOUR = { A: 'tab0', B: 'tab1' } as const

/**
 * What the visitors did, as plain counts, before anything is computed: for each version the clicks out of the visitors, as a bar
 * (from zero) and as words. Nothing here is inferred; it is the raw tally.
 *
 * @param props - The visitors each version was shown and the clicks each got.
 * @returns The two bars.
 */
export function TallyBars(props: {
  readonly visitors: number
  readonly clicksA: number
  readonly clicksB: number
}): ReactNode {
  const { visitors, clicksA, clicksB } = props
  const rows = [
    { name: 'A', clicks: clicksA, colour: 'var(--wb-tab0)' },
    { name: 'B', clicks: clicksB, colour: 'var(--wb-tab1)' },
  ]
  const top = Math.max(clicksA, clicksB, 1) * 1.3
  return (
    <div className="flex flex-col gap-3 px-1 py-2 font-sans">
      {rows.map((row) => (
        <div key={row.name} className="grid grid-cols-[4.5rem_1fr] items-center gap-3">
          <span className="text-sm font-semibold">Version {row.name}</span>
          <div className="flex flex-col gap-1">
            <div className="h-5 rounded-sm bg-line/40">
              <div
                className="h-5 rounded-sm"
                style={{ width: `${(100 * row.clicks) / top}%`, background: row.colour }}
                role="img"
                aria-label={`${row.clicks} clicks`}
              />
            </div>
            <span className="text-sm text-muted">
              {row.clicks} of {visitors.toLocaleString('en-GB')} visitors clicked ({rate(row.clicks / visitors)})
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

/** The part of a curve inside a range of x (for the shaded middle 90%). */
function within(
  x: readonly number[],
  y: readonly number[],
  lo: number,
  hi: number,
): { readonly x: number[]; readonly y: number[] } {
  const keep = x.map((v) => v >= lo && v <= hi)
  return { x: x.filter((_, i) => keep[i]), y: y.filter((_, i) => keep[i]) }
}

/**
 * The two posteriors of the click rates on one axis: a curve for each version, shaded where it holds the middle 90% of the
 * belief, and a dashed mark at the rate each version really had.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function RatesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const one = (name: 'A' | 'B', density: readonly number[], lo: number, hi: number) => {
        const core = within(result.grid, density, lo, hi)
        return [
          lines([{ x: result.grid, y: density, area: 0.1, width: 2.4, colour: COLOUR[name] }], { mode }),
          lines([{ ...core, area: 0.4, width: 0, colour: COLOUR[name] }], { mode }),
        ]
      }
      const top = Math.max(...result.densityA, ...result.densityB) * 1.1
      const truth = (name: 'A' | 'B', at: number) =>
        lines([{ x: [at, at], y: [0, top * 0.92], dash: 'dashed', colour: COLOUR[name], width: 1.6 }], { mode })
      return cartesian(
        [
          ...one('A', result.densityA, result.rateA.lo, result.rateA.hi),
          ...one('B', result.densityB, result.rateB.lo, result.rateB.hi),
          truth('A', result.truth.rateA),
          truth('B', result.truth.rateB),
          annotations(
            {
              labels: [
                { at: [result.rateA.mean, top * 0.97], text: 'A' },
                { at: [result.rateB.mean, top * 0.97], text: 'B' },
              ],
            },
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'click rate', format: 'percent' },
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

/**
 * The posterior of the difference of the two rates (B minus A): a histogram of draws in which the bars to the right of zero (B is
 * better) and to the left (A is) are told apart by colour, with zero and the true difference marked.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function DifferenceFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const { centres, density } = result.difference
      const top = Math.max(...density) * 1.12
      const side = (positive: boolean, name: 'A' | 'B') => {
        const keep = centres.map((c) => c > 0 === positive)
        return lines(
          [
            {
              x: centres.filter((_, i) => keep[i]),
              y: density.filter((_, i) => keep[i]),
              step: 'middle',
              area: 0.55,
              width: 1.2,
              colour: COLOUR[name],
            },
          ],
          { mode },
        )
      }
      const gap = result.truth.rateB - result.truth.rateA
      return cartesian(
        [
          side(false, 'A'),
          side(true, 'B'),
          lines([{ x: [0, 0], y: [0, top], colour: 'fg', width: 1.4 }], { mode }),
          lines([{ x: [gap, gap], y: [0, top * 0.92], dash: 'dashed', colour: 'fgSecondary', width: 1.6 }], { mode }),
          annotations({ labels: [{ at: [gap, top * 0.97], text: 'the real gap' }] }, { mode }),
        ],
        {
          mode,
          figure,
          x: {
            label: 'click rate of B minus click rate of A',
            format: (v) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(0)} pts`,
          },
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

/**
 * The chance that B is better as the visitors accumulate: a line with a mark at each look, on a log axis so the first few dozen
 * visitors, where it moves most, have room, with the bands beyond 95% either way (where most teams stop) shaded.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function StripFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const { visitors, better } = result.strip
      const first = visitors[0] as number
      const last = visitors.at(-1) as number
      return cartesian(
        [
          annotations(
            {
              regions: [
                { y: [CALL, 1], colour: COLOUR.B },
                { y: [0, 1 - CALL], colour: COLOUR.A },
              ],
            },
            { mode },
          ),
          lines([{ x: [first, last], y: [0.5, 0.5], dash: 'dotted', colour: 'fgSecondary', width: 1.2 }], { mode }),
          lines([{ x: visitors, y: better, markers: true, width: 2.4, colour: 'fg' }], { mode }),
          annotations(
            {
              labels: [
                { at: [Math.sqrt(first * last), 0.975], text: 'call it for B' },
                { at: [Math.sqrt(first * last), 0.025], text: 'call it for A' },
              ],
            },
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: {
            label: 'visitors shown each version',
            scale: 'log',
            limits: [15, 1300] as [number, number],
            format: (v) => v.toLocaleString('en-GB'),
          },
          y: { label: 'chance that B is better', limits: [0, 1], format: 'percent' },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2} narrowAspect={1.3} build={build} />
}
