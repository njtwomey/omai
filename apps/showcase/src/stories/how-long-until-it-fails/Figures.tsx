import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Case, Result, Update } from './compute.ts'
import { HORIZON } from './data.ts'

/** The height of a strip of marks under the survival curve, so that the marks do not touch the curve. */
const FAILED_STRIP = -0.05
const CENSORED_STRIP = -0.11

/**
 * One case of the update that is taken apart: the belief about the scale before the unit (the cavity), after it (the tilted
 * curve), and the bell curve that replaces the tilted curve, with a dotted line at the time of the unit.
 *
 * @param props - The case, the update it belongs to and its title.
 * @returns The chart.
 */
function CaseFigure(props: { readonly case: Case; readonly update: Update; readonly title: string }): ReactNode {
  const { case: c, update, title } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const top = 1.15 * Math.max(...c.tilted, ...c.cavity, ...c.matched)
      return cartesian(
        [
          lines(
            [
              { x: c.scale, y: c.cavity, name: 'before this unit', colour: 'fgSecondary', width: 2, dash: 'dashed' },
              { x: c.scale, y: c.tilted, name: 'with this unit', colour: 'tab0', width: 3 },
              { x: c.scale, y: c.matched, name: 'the bell curve for it', colour: 'tab1', width: 2, dash: 'dotted' },
              {
                x: [update.time, update.time],
                y: [0, top],
                name: 'time of the unit',
                colour: 'fg',
                width: 1,
                opacity: 0.6,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          title,
          x: { label: 'scale', limits: [update.range[0], update.range[1]] },
          y: { label: 'how likely each scale is', limits: [0, top], format: () => '', splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [c, update, title, mode],
  )
  return <ResponsiveChart aspect={1.25} narrowAspect={1.0} build={build} />
}

/**
 * The same update twice, side by side: the unit taken as a failure and as a censoring (stacked on a narrow screen).
 *
 * @param props - The update.
 * @returns The two charts.
 */
export function UpdateFigure(props: { readonly update: Update }): ReactNode {
  const { update } = props
  const time = update.time.toFixed(1)
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <CaseFigure case={update.failed} update={update} title={`Failed at time ${time}`} />
      <CaseFigure case={update.censored} update={update} title={`Still working at time ${time}`} />
    </div>
  )
}

/**
 * The survival curves: the band and the middle of the fit with every unit, the true curve, the fit without the censored units, and
 * under the axis one mark for each unit (a dot where it failed, a cross where it was still working).
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CurvesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const { curves } = result
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const failed = result.time.filter((_, i) => result.event[i] === 1)
      const censored = result.time.filter((_, i) => result.event[i] === 0)
      return cartesian(
        [
          band([{ x: curves.time, lower: curves.lower, upper: curves.upper, colour: 'tab0', opacity: 0.25 }], { mode }),
          lines(
            [
              { x: curves.time, y: curves.middle, name: 'fitted, every unit used', colour: 'tab0', width: 3 },
              {
                x: curves.time,
                y: curves.dropped,
                name: 'fitted, censored units dropped',
                colour: 'tab1',
                width: 2,
                dash: 'dotted',
              },
              { x: curves.time, y: curves.truth, name: 'true curve', colour: 'fg', width: 1.6, dash: 'dashed' },
            ],
            { mode },
          ),
          points(
            [
              { x: failed, y: failed.map(() => FAILED_STRIP), name: 'failed', colour: 'tab0', size: 8, opacity: 0.85 },
              {
                x: censored,
                y: censored.map(() => CENSORED_STRIP),
                name: 'still working',
                colour: 'fgSecondary',
                symbol: 'cross',
                size: 11,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'time', limits: [0, HORIZON] },
          y: {
            label: 'chance of lasting this long',
            limits: [-0.16, 1.02],
            format: (v: number) => (v < 0 ? '' : v.toFixed(1)),
          },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, curves, mode],
  )
  return <ResponsiveChart aspect={1.6} narrowAspect={1.05} build={build} />
}
