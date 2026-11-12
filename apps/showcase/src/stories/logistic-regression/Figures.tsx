import { cartesian, contours, heatmap, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { categoricalColour, mixColours, tokens } from 'omai-render/theme'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { LEVELS, LIMIT, type Points } from './data.ts'

/** Where along its line the label of each level of {@link LEVELS} sits. */
const LABEL_AT = [0.3, 0.2, 0.5, 0.8, 0.7] as const

const AXES = {
  x: { label: 'measurement 1', limits: [-LIMIT, LIMIT] as [number, number] },
  y: { label: 'measurement 2', limits: [-LIMIT, LIMIT] as [number, number] },
}

/** The points as one layer: no in the first colour, yes in the second, the colours of the two ends of the chance ramp. */
const dots = (p: Points, mode: 'light' | 'dark'): ReturnType<typeof points> =>
  points(
    [
      {
        x: p.x.map((q) => q[0]),
        y: p.x.map((q) => q[1]),
        groups: p.y,
        groupNames: { 0: 'no', 1: 'yes' },
        size: 9,
        outline: true,
      },
    ],
    { mode },
  )

/**
 * The marked points alone, in the plane (before anything has run).
 *
 * @param props - The points.
 * @returns The chart.
 */
export function PointsFigure(props: { readonly points: Points }): ReactNode {
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => cartesian([dots(props.points, mode)], { mode, figure, ...AXES, legend: true, tooltip: false }),
    [props.points, mode],
  )
  return <ResponsiveChart aspect={1.35} narrowAspect={1.05} build={build} />
}

/**
 * The expected chance of yes at each place, from blue (no) through grey (one half) to orange (yes), with a solid line where the
 * chance is one half and the points on top.
 *
 * @param props - The points and the result.
 * @returns The chart.
 */
export function FieldFigure(props: { readonly points: Points; readonly result: Result }): ReactNode {
  const { points: data, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const middle = mixColours(tokens(mode).bg, tokens(mode).fgSecondary, 0.25)
      return cartesian(
        [
          heatmap(result.chance as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            ramp: [categoricalColour(0), middle, categoricalColour(1)],
            range: [0, 1],
            colourBar: false,
            mode,
          }),
          // one layer per level, so that the labels sit at different places along their lines and do not touch
          ...LEVELS.map((level, k) =>
            contours(result.chance as number[][], {
              x: [...result.xs],
              y: [...result.ys],
              levels: [level],
              colour: 'fg',
              width: level === 0.5 ? 2.6 : 1.2,
              labels: true,
              labelAt: LABEL_AT[k] as number,
              mode,
            }),
          ),
          dots(data, mode),
        ],
        { mode, figure, ...AXES, legend: false, tooltip: false },
      )
    },
    [data, result, mode],
  )
  return <ResponsiveChart aspect={1.2} narrowAspect={0.95} build={build} />
}

/**
 * How unsure the model is about the score at each place (its standard deviation), darker where it is surer, with the points on top.
 *
 * @param props - The points and the result.
 * @returns The chart.
 */
export function DoubtFigure(props: { readonly points: Points; readonly result: Result }): ReactNode {
  const { points: data, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          heatmap(result.doubt as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            ramp: 'blues',
            colourBar: false,
            mode,
          }),
          dots(data, mode),
        ],
        { mode, figure, ...AXES, legend: false, tooltip: false },
      ),
    [data, result, mode],
  )
  return <ResponsiveChart aspect={1.2} narrowAspect={0.95} build={build} />
}

/**
 * The joint density of the two weights as filled contours, with the mean marked.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function JointFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          heatmap(result.joint.density as number[][], {
            x: [...result.joint.first],
            y: [...result.joint.second],
            smooth: 'contours',
            levels: 8,
            ramp: 'blues',
            colourBar: false,
            mode,
          }),
          points(
            [
              {
                x: [result.weights.first.mean],
                y: [result.weights.second.mean],
                groups: ['the mean'],
                size: 11,
                labels: ['the mean'],
                labelPosition: 'top',
                outline: true,
              },
            ],
            { mode, colourOffset: 1 },
          ),
          points(
            [
              {
                x: [result.truth.first],
                y: [result.truth.second],
                groups: ['true'],
                size: 13,
                symbol: 'diamond',
                labels: ['true'],
                labelPosition: 'bottom',
                outline: true,
              },
            ],
            { mode, colourOffset: 2 },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'weight of measurement 1' },
          y: { label: 'weight of measurement 2' },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={1.4} narrowAspect={1.05} build={build} />
}

/**
 * The density of the offset and of each weight, with the 90% range of each filled.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CurvesFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const { axis } = result.curves
      const parts = [
        {
          name: 'offset',
          density: result.curves.offset,
          interval: result.weights.offset,
          truth: result.truth.offset,
          colour: 'fgSecondary',
        },
        {
          name: 'weight of measurement 1',
          density: result.curves.first,
          interval: result.weights.first,
          truth: result.truth.first,
          colour: 'tab2',
        },
        {
          name: 'weight of measurement 2',
          density: result.curves.second,
          interval: result.weights.second,
          truth: result.truth.second,
          colour: 'tab4',
        },
      ]
      const top = 1.05 * Math.max(...parts.flatMap((p) => p.density))
      return cartesian(
        [
          lines(
            parts.map((p) => ({ x: [...axis], y: [...p.density], name: p.name, colour: p.colour, width: 2.2 })),
            { mode },
          ),
          lines(
            parts.map((p) => ({
              x: [...axis],
              y: p.density.map((v, i) =>
                (axis[i] as number) >= p.interval.lo && (axis[i] as number) <= p.interval.hi ? v : Number.NaN,
              ),
              name: `__range ${p.name}`,
              colour: p.colour,
              width: 0,
              area: 0.3,
            })),
            { mode },
          ),
          lines(
            parts.map((p) => ({
              x: [p.truth, p.truth],
              y: [0, top],
              name: `__true ${p.name}`,
              colour: p.colour,
              width: 1.8,
              dash: 'dashed' as const,
            })),
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'value of the weight' },
          y: { label: 'density', format: () => '' },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}
