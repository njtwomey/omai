import { band, bars, cartesian, errorBars, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { SCORE_NAMES, Z90 } from './reading.ts'
import { REFERENCE } from './reference.ts'

/** The colour of each cut point, the same wherever it appears. */
const COLOURS = ['tab0', 'tab1', 'tab2', 'tab3'] as const

/**
 * How many times each score was given, before any model: one bar for each of the scores 1 to 5.
 *
 * @param props - The counts, from the score 1 to the score 5.
 * @returns The chart.
 */
export function GivenFigure(props: { readonly counts: readonly number[] }): ReactNode {
  const { counts } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian([bars(SCORE_NAMES, [{ name: 'ratings', values: counts, colour: 'tab0' }], { mode, labels: true })], {
        mode,
        figure,
        x: { label: 'score given' },
        y: { label: 'number of ratings' },
        legend: false,
        tooltip: false,
      }),
    [counts, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.5} build={build} />
}

/**
 * The cut points as the ratings arrive: for each of the four cut points a line for the mean of its belief after the first 10
 * ratings, the first 25 and so on, with a 90% band, a dotted line at the true cut point, and at the right the sampler's 90%
 * interval for the same cut point on all the ratings.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CutsFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const stops = result.history.map((h) => h.after)
      const last = Math.max(...stops)
      // the sampler's interval sits just right of the last stop
      const right = last * 1.25
      const cutNames = result.ratings.cuts.map((_, k) => `${k + 1}|${k + 2}`)
      const series = cutNames.map((name, k) => ({
        name,
        mean: result.history.map((h) => h.cuts.mean[k] as number),
        sd: result.history.map((h) => h.cuts.sd[k] as number),
        truth: result.ratings.cuts[k] as number,
      }))
      return cartesian(
        [
          band(
            series.map((s, k) => ({
              x: stops,
              lower: s.mean.map((m, i) => m - Z90 * (s.sd[i] as number)),
              upper: s.mean.map((m, i) => m + Z90 * (s.sd[i] as number)),
              name: `__${s.name}-band`,
              colour: COLOURS[k] as string,
              opacity: 0.16,
            })),
            { mode },
          ),
          lines(
            series.map((s, k) => ({
              x: [0, right],
              y: [s.truth, s.truth],
              name: `__${s.name}-true`,
              colour: COLOURS[k] as string,
              width: 1.4,
              dash: 'dotted' as const,
            })),
            { mode },
          ),
          lines(
            series.map((s, k) => ({
              x: stops,
              y: s.mean,
              name: `cut ${s.name}`,
              colour: COLOURS[k] as string,
              width: 2,
              markers: true,
            })),
            { mode },
          ),
          ...series.map((s, k) =>
            errorBars([{ x: right, y: REFERENCE.cutMean[k] as number, yErr: Z90 * (REFERENCE.cutSd[k] as number) }], {
              mode,
              name: `__${s.name}-sampler`,
              colour: COLOURS[k] as string,
              cap: right * 0.04,
            }),
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'ratings given to the model', limits: [0, right * 1.1] },
          y: { label: 'cut point (score units)', limits: [-5, 3] },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.6} narrowAspect={1.1} build={build} />
}

/**
 * The scores the ratings have, as bars, and the number of each the fitted model expects, as a second bar beside it.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function ExpectedFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const observed = SCORE_NAMES.map((_, k) => result.ratings.score.filter((s) => s === k).length)
      return cartesian(
        [
          bars(
            SCORE_NAMES,
            [
              { name: 'given', values: observed, colour: 'tab0' },
              { name: 'expected by the fitted model', values: [...result.final.expected], colour: 'tab1' },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'score' },
          y: { label: 'number of ratings' },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.5} build={build} />
}
