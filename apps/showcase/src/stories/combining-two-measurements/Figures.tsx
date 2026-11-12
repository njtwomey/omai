import { cartesian, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { PRIOR } from './data.ts'
import { bell } from './reading.ts'

const points = 241

/** Evenly spaced numbers from lo to hi, for the x axis of a curve. */
const spaced = (lo: number, hi: number): number[] =>
  Array.from({ length: points }, (_, i) => lo + ((hi - lo) * i) / (points - 1))

const X_LABEL = 'temperature (°C)'

/**
 * What is believed before any reading, drawn from the numbers the page fixes: one bell curve over the temperatures from 8 to 32
 * degrees. It is there before play, because it needs no computation.
 *
 * @returns The chart.
 */
export function PriorFigure(): ReactNode {
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const x = spaced(PRIOR.mean - 3 * PRIOR.sd, PRIOR.mean + 3 * PRIOR.sd)
      return cartesian(
        [
          lines(
            [
              {
                x,
                y: x.map((v) => bell(PRIOR.mean, PRIOR.sd ** 2, v)),
                area: 0.18,
                width: 2.4,
                name: 'before any reading',
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: X_LABEL, limits: [x[0] as number, x.at(-1) as number] },
          y: { label: '', show: false, splitLine: false },
          legend: false,
          tooltip: false,
        },
      )
    },
    [mode],
  )
  return <ResponsiveChart aspect={2.6} narrowAspect={1.7} build={build} />
}

/**
 * The messages and the belief as bell curves on one axis: the three messages of the first sweep thin and in their own colours
 * (the prior, the digital reading, the dial reading) and the belief, which is their product, thick and filled. The curves are
 * drawn at their true heights, so the narrowest is the tallest.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function MessagesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const sds = result.messages.map((m) => Math.sqrt(m.variance))
      const lo = Math.min(...result.messages.map((m, i) => m.mean - 3 * (sds[i] as number)))
      const hi = Math.max(...result.messages.map((m, i) => m.mean + 3 * (sds[i] as number)))
      const x = spaced(lo, hi)
      const curve = (mean: number, variance: number): number[] => x.map((v) => bell(mean, variance, v))
      const names = { prior: 'prior', digital: 'digital reading', dial: 'dial reading' }
      return cartesian(
        [
          lines(
            result.messages.map((m, i) => ({
              x,
              y: curve(m.mean, m.variance),
              name: `message from the ${names[m.from]}`,
              colour: `tab${i + 1}`,
              width: 1.6,
            })),
            { mode },
          ),
          lines(
            [
              {
                x,
                y: curve(result.belief.mean, result.belief.sd ** 2),
                name: 'the belief (their product)',
                area: 0.22,
                width: 3,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: X_LABEL, limits: [lo, hi] },
          y: { label: '', show: false, splitLine: false },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}
