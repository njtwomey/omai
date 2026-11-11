/**
 * The rows panel of a session on a dataset: the training rows (by assignment, class or target) with an overlay probed
 * at the viewed step (assignments and centres, mixture components with their ellipses, a 1-D fit with or without its
 * predictive band, a 1-D mixture of experts, a decision surface), plus the probe hook and the figure size every
 * session panel shares.
 */

import type { Json } from 'omai-engine/base'
import { maxOf } from 'omai-engine/base'
import { covarianceEllipse } from 'omai-engine/numerics/geometry'
import { type Tensor, toFlat } from 'omai-engine/tensor'
import type { Derived, Run, Session } from 'omai-engine/trace'
import { createContext, type ReactElement, type ReactNode, useContext, useMemo } from 'react'
import { band, lines } from '../../charts/lines.ts'
import { cartesian, type Layer } from '../../charts/plot.ts'
import { points } from '../../charts/points.ts'
import { type Figure, figure } from '../../figure/index.ts'
import { canProbeAsync } from '../../producer/remote.ts'
import { categoricalColour } from '../../theme/index.ts'
import { useByContent } from '../_content.ts'
import { Chart } from '../Chart.tsx'
import { FigureChromeProvider, FigureFrame } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'
import { ExpositionView } from '../views/ExpositionView.tsx'
import type { Panel, RowSource } from './layout.ts'
import {
  channelFocus,
  channelLayers,
  channelMarks,
  channelNumbers,
  channelRegions,
  isProbed,
  readChannel,
} from './overlays.ts'
import type { LiveSession } from './useSession.ts'

/** The figure every session panel is drawn at unless a {@link PanelFigureProvider} says another: 440 by 320 pixels. */
export const FIG: Figure = figure({ width: 440, height: 320 })

const PanelFigureContext = createContext<Figure | undefined>(undefined)

/**
 * Set the figure the session panels below it are drawn at: its width and height, type size (`fontSize`), padding and
 * medium, for every panel of `SessionPanels` and of the specialised panels at once (the tape views too, where the
 * default differs). Panels that derive their own shape (a strip's aspect, a grid of images, the tall equation figure)
 * keep the width, type size and padding and change the height as they need. With `chrome={false}` the same provider
 * takes the expand and export buttons off every panel's frame. Either is optional. Nested, the innermost wins.
 *
 * @param props - The figure and the panels.
 * @returns The panels, with the figure in context.
 * @example
 * ```tsx
 * import { figure } from 'omai-render/figure'
 * import { PanelFigureProvider, SessionPanels } from 'omai-render/react'
 * const large = figure({ width: 640, height: 400, fontSize: 14 })
 * const panels = (
 *   <PanelFigureProvider figure={large} chrome={false}>
 *     <SessionPanels live={live} at={at} layout={layout} />
 *   </PanelFigureProvider>
 * )
 * ```
 */
export function PanelFigureProvider(props: {
  readonly figure?: Figure
  /**
   * Whether the panels draw their toolbar (the expand, export and settings buttons of each figure's frame; default
   * true), as `chrome` of a `FigureFrame`: `false` leaves them all out, for panels placed in a slide. A panel's own
   * `chrome` prop wins; not given, the enclosing provider's setting stays.
   */
  readonly chrome?: boolean | undefined
  readonly children?: ReactNode
}): ReactElement {
  const inner = <FigureChromeProvider chrome={props.chrome}>{props.children}</FigureChromeProvider>
  return props.figure === undefined ? (
    inner
  ) : (
    <PanelFigureContext.Provider value={props.figure}>{inner}</PanelFigureContext.Provider>
  )
}

/**
 * The figure a session panel is drawn at: the one a {@link PanelFigureProvider} above it gives, else `fallback`.
 *
 * @param fallback - The figure without a provider (default {@link FIG}, 440 by 320; the tape views pass their own).
 * @returns The figure.
 */
export function usePanelFigure(fallback: Figure = FIG): Figure {
  return useContext(PanelFigureContext) ?? fallback
}

/**
 * A figure of another shape for a panel that needs one (a strip's aspect, a grid of images, the tall equation figure),
 * with the type size, padding, medium and colour mode of the panel figure it comes from.
 *
 * @param base - The panel figure.
 * @param shape - The width, the height or the aspect to change (`width` defaults to the base's).
 * @returns The figure.
 */
export function reshaped(base: Figure, shape: { readonly height?: number; readonly aspect?: number }): Figure {
  return figure({
    medium: base.medium,
    width: base.width,
    padding: base.padding,
    fontSize: base.fontSize,
    mode: base.mode,
    ...shape,
  })
}

/** The outcome of a probe: its value, or the message of the error it threw. */
export type Probed = { readonly value?: unknown; readonly error?: string }

// The answers of a session's probes at steps the tape already has (a probe there never changes), the most recent kept
// per session: scrubbing back to a step, or looping over a run, does not compute its grid again. Held by the session,
// so they go with it. Answers are not copied: a reader must not change what a probe returned.
const PROBES_KEPT = 48
const answers = new WeakMap<object, Map<string, unknown>>()

function remembered(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  session: Session<any, any>,
  t: number,
  key: string,
  compute: () => unknown,
): unknown {
  if (t > session.t) return compute()
  const kept = answers.get(session) ?? new Map<string, unknown>()
  answers.set(session, kept)
  if (kept.has(key)) {
    const hit = kept.get(key)
    kept.delete(key)
    kept.set(key, hit)
    return hit
  }
  const value = compute()
  kept.set(key, value)
  if (kept.size > PROBES_KEPT) kept.delete(kept.keys().next().value as string)
  return value
}

/**
 * One probe of the session at a step, computed once per (session, step) and never throwing: a failure is returned as
 * its message so a panel can say why and still draw its rows. The answers at the latest steps are kept, so a step
 * visited again is not computed again.
 *
 * @param live - The session.
 * @param t - The step.
 * @param path - The derived path (`undefined` probes nothing).
 * @param args - The probe's arguments.
 * @returns The value or the error.
 */
export function useProbe(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  live: LiveSession<any, any>,
  t: number,
  path: string | undefined,
  args?: Json,
): Probed {
  const held = useByContent(args)
  // A session computed in a worker answers a probe later: the answers that have come (`live.probed`) remake the memo, and
  // its own producer keeps the answers, so nothing here may keep the stand-in a pending probe returns.
  const remote = canProbeAsync(live.session)
  const answered = remote ? (live.probed ?? 0) : 0
  // biome-ignore lint/correctness/useExhaustiveDependencies: `answered` is what a worker's late answer changes
  return useMemo(() => {
    if (path === undefined) return {}
    try {
      const read = (): Derived => live.session.probe(t, path, held)
      return {
        value: remote ? read() : remembered(live.session, t, `${path}|${t}|${JSON.stringify(held ?? null)}`, read),
      }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [live.session, t, path, held, remote, answered])
}

/**
 * The tape of a session, re-read whenever it grows.
 *
 * @param live - The session.
 * @returns The tape, a new one each time the session's version changes.
 */
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
export function useRun(live: LiveSession<any, any>): Run {
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  return useMemo(() => live.session.run(), [live.session, live.version])
}

/** The channels of the rows panel (see the layout's `rows` panel): each read from a path at the viewed step. */
export type RowChannelSpecs = Pick<Extract<Panel, { kind: 'rows' }>, 'values' | 'size' | 'marks' | 'regions' | 'focus'>

/**
 * One channel of the rows panel at a step, never throwing: a derived path is probed (once per step), a tensor path is
 * read from the tape (and re-read as it grows).
 *
 * @param live - The session.
 * @param t - The step.
 * @param source - The path and arguments (`undefined` reads nothing).
 * @returns The value or the error.
 */
export function useChannel(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  live: LiveSession<any, any>,
  t: number,
  source: RowSource | undefined,
): Probed {
  const argsKey = JSON.stringify(source?.args ?? null)
  const version = isProbed(live.session, source) ? 0 : live.version
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the path and arguments, not the source object
  return useMemo(() => {
    if (source === undefined) return {}
    try {
      const read = (): unknown => readChannel(live.session, t, source)
      return {
        value: isProbed(live.session, source)
          ? remembered(live.session, t, `${source.path}|${t}|${argsKey}`, read)
          : read(),
      }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [live.session, t, source?.path, argsKey, version])
}

// Whether a column looks like class labels: a few distinct whole numbers.
function looksLikeClasses(ys: ArrayLike<number>): boolean {
  const seen = new Set<number>()
  for (let i = 0; i < ys.length; i++) {
    const v = ys[i] as number
    if (!Number.isInteger(v)) return false
    seen.add(v)
    if (seen.size > 12) return false
  }
  return true
}

/** The one-line note a panel shows in its frame when its probe failed. */
export const Note = (props: { readonly children: ReactNode }): ReactNode => (
  <span className="text-xs text-muted">{props.children}</span>
)

/**
 * The training rows (by assignment, class or target) with the probe overlay at the scrubbed step: `assignments` the
 * assignments and centres, `components` those with each component's 1σ and 2σ ellipses, `fit` the 1-D fit, `band`
 * the 1-D fit with its predictive band, `surface` the decision surface, `experts` the rows by expert with each expert's
 * line and the mixture's. Each probe is independent and the chart stays mounted across frames.
 *
 * @param props - The overlay, title and file name, the session, the viewed step, the training rows and the per-row
 *   channels.
 * @returns The framed chart.
 */
export function RowsPanel(props: {
  readonly overlay: NonNullable<Extract<Panel, { kind: 'rows' }>['overlay']> | undefined
  readonly title: string
  readonly filename: string
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly t: number
  readonly x: Tensor
  readonly y?: Tensor
  /** The per-row channels: values, size, marks, regions and focus. */
  readonly channels?: RowChannelSpecs
  /** Draw the frame without its buttons (default true; a `PanelFigureProvider` above may say). */
  readonly chrome?: boolean
}): ReactNode {
  const base = usePanelFigure()
  const { overlay, live, t, x, y, channels = {} } = props
  const mode = useColourMode()
  const withCentres = overlay === 'assignments' || overlay === 'components'
  const byExpert = overlay === 'experts'
  const assignments = useProbe(live, t, withCentres || byExpert ? 'model/assignments' : undefined)
  const centres = useProbe(live, t, withCentres ? 'model/centres' : undefined)
  // A mixture's parameters carry covariances: drawn as each component's 1 and 2 sigma ellipses.
  const params = useProbe(live, t, overlay === 'components' ? 'model/params' : undefined)
  const fit = useProbe(live, t, overlay === 'fit' || byExpert ? 'model/predictions' : undefined)
  // A mixture of experts: each expert's own line over the training rows.
  const expertLines = useProbe(live, t, byExpert ? 'model/experts' : undefined)
  // A predictive band: the mean and the band's edges per training row (a Bayesian model's posterior predictive).
  const predictive = useProbe(live, t, overlay === 'band' ? 'model/predictive' : undefined)
  const surface = useProbe(live, t, overlay === 'surface' ? 'model/decision-surface' : undefined)
  const valuesIn = useChannel(live, t, channels.values)
  const sizeIn = useChannel(live, t, channels.size)
  const marksIn = useChannel(live, t, channels.marks)
  const regionsIn = useChannel(live, t, channels.regions)
  const focusIn = useChannel(live, t, channels.focus)
  const hasChannels = Object.values(channels).some((c) => c !== undefined)
  const xs = useMemo(() => toFlat(x), [x])
  const ys = useMemo(() => (y === undefined ? undefined : toFlat(y)), [y])
  const d = x.shape[1] ?? 1
  const n = x.shape[0] ?? 0
  const rowX = useMemo(() => Array.from({ length: n }, (_, i) => xs[i * d] ?? 0), [xs, n, d])
  const rowY = useMemo(
    () => Array.from({ length: n }, (_, i) => (d > 1 ? (xs[i * d + 1] ?? 0) : (ys?.[i] ?? 0))),
    [xs, ys, n, d],
  )
  const labels = assignments.value as number[] | undefined
  const classes = useMemo(
    () =>
      ys !== undefined && ((overlay === 'surface' && d > 1) || (hasChannels && d > 1 && looksLikeClasses(ys)))
        ? Array.from(ys)
        : undefined,
    [ys, overlay, d, hasChannels],
  )
  const groups = labels ?? classes
  const centreRows = centres.value as number[][] | undefined
  const covs = (params.value as { readonly covariances?: number[][][] } | undefined)?.covariances
  const interval = predictive.value as
    | { readonly mean: number[]; readonly lower: number[]; readonly upper: number[] }
    | undefined
  const fitted = (fit.value as number[] | undefined) ?? interval?.mean
  const expertRows = (expertLines.value as { readonly lines?: number[][] } | undefined)?.lines
  const error = [
    assignments,
    centres,
    params,
    fit,
    expertLines,
    predictive,
    surface,
    valuesIn,
    sizeIn,
    marksIn,
    regionsIn,
    focusIn,
  ].find((p) => p.error !== undefined)?.error
  // The channels' layers: regions beneath, the rows coloured and sized, the focus and the marks over them.
  const extra = useMemo(() => {
    if (!hasChannels || d < 2 || byExpert) return undefined
    const numbers = channelNumbers(valuesIn.value)
    const sizes = channelNumbers(sizeIn.value)
    const categories = channelMarks(marksIn.value, n)
    const focus = channelFocus(focusIn.value)
    return channelLayers({
      mode,
      x: rowX,
      y: rowY,
      ...(groups === undefined ? {} : { groups }),
      ...(numbers === undefined
        ? {}
        : {
            values: {
              numbers,
              diverging: channels.values?.diverging === true,
              ...(channels.values?.range === undefined ? {} : { range: channels.values.range }),
              ...(channels.values?.label === undefined ? {} : { label: channels.values.label }),
            },
          }),
      ...(sizes === undefined
        ? {}
        : { sizes: { numbers: sizes, ...(channels.size?.range === undefined ? {} : { range: channels.size.range }) } }),
      ...(categories === undefined
        ? {}
        : { marks: { categories, ...(channels.marks?.names === undefined ? {} : { names: channels.marks.names }) } }),
      regions: channelRegions(regionsIn.value),
      ...(focus === undefined ? {} : { focus }),
    })
  }, [
    hasChannels,
    d,
    byExpert,
    n,
    mode,
    rowX,
    rowY,
    groups,
    valuesIn.value,
    sizeIn.value,
    marksIn.value,
    regionsIn.value,
    focusIn.value,
    channels.values,
    channels.size,
    channels.marks,
  ])
  const option = useMemo(() => {
    const layers: Layer[] =
      byExpert && labels !== undefined
        ? // One set per expert so that an expert keeps its colour whichever experts own rows.
          Array.from(
            { length: Math.max(expertRows?.length ?? 0, Math.max(-1, maxOf(labels)) + 1) },
            (_, k) => k,
          ).flatMap((k) => {
            const own = labels.flatMap((l, i) => (l === k ? [i] : []))
            return own.length === 0
              ? []
              : [
                  points(
                    [
                      {
                        x: own.map((i) => rowX[i] ?? 0),
                        y: own.map((i) => rowY[i] ?? 0),
                        name: `expert ${k}`,
                        colour: categoricalColour(k),
                        opacity: 0.7,
                      },
                    ],
                    { mode },
                  ),
                ]
          })
        : [points([{ x: rowX, y: rowY, ...(groups === undefined ? {} : { groups }), opacity: 0.7 }], { mode })]
    if (extra !== undefined) {
      if (extra.rows !== undefined) layers.splice(0, 1, extra.rows)
      layers.unshift(...extra.under)
      layers.push(...extra.over)
    }
    if (centreRows !== undefined)
      layers.push(
        points(
          [
            {
              x: centreRows.map((c) => c[0] ?? 0),
              y: centreRows.map((c) => c[1] ?? 0),
              name: 'centres',
              size: 16,
              symbol: 'diamond',
            },
          ],
          { mode, colourOffset: 9 },
        ),
      )
    if (centreRows !== undefined && covs !== undefined && d >= 2) {
      centreRows.forEach((m, k) => {
        const c = covs[k]
        if (c === undefined) return
        const sigma = [
          [c[0]?.[0] ?? 0, c[0]?.[1] ?? 0],
          [c[1]?.[0] ?? 0, c[1]?.[1] ?? 0],
        ]
        for (const nStd of [1, 2]) {
          const flat = toFlat(covarianceEllipse([m[0] ?? 0, m[1] ?? 0], sigma, { nStd, points: 72 }).points)
          const ex = Array.from({ length: 73 }, (_, i) => flat[(2 * i) % flat.length] ?? 0)
          const ey = Array.from({ length: 73 }, (_, i) => flat[(2 * i + 1) % flat.length] ?? 0)
          layers.push(
            lines(
              [
                {
                  x: ex,
                  y: ey,
                  name: `__ring${k}-${nStd}`,
                  colour: categoricalColour(k),
                  width: nStd === 1 ? 2 : 1,
                  dash: nStd === 1 ? 'solid' : 'dashed',
                },
              ],
              { mode },
            ),
          )
        }
      })
    }
    if (d === 1) {
      const order = rowX.map((_, i) => i).sort((a, b) => (rowX[a] ?? 0) - (rowX[b] ?? 0))
      const xs = order.map((i) => rowX[i] ?? 0)
      expertRows?.forEach((line, k) => {
        layers.push(
          lines(
            [
              {
                x: xs,
                y: order.map((i) => line[i] ?? 0),
                name: `__expert${k}`,
                colour: categoricalColour(k),
                width: 1.5,
              },
            ],
            { mode },
          ),
        )
      })
      if (interval !== undefined)
        layers.push(
          band(
            [
              {
                x: xs,
                lower: order.map((i) => interval.lower[i] ?? 0),
                upper: order.map((i) => interval.upper[i] ?? 0),
                name: 'band',
              },
            ],
            { mode, colourOffset: 1 },
          ),
        )
      if (fitted !== undefined)
        layers.push(
          lines(
            [
              {
                x: xs,
                y: order.map((i) => fitted[i] ?? 0),
                name: byExpert ? 'mixture' : 'fit',
                ...(byExpert ? { colour: 'fg', width: 3.5 } : {}),
              },
            ],
            { mode, colourOffset: 1 },
          ),
        )
    }
    return { ...cartesian(layers, { mode, figure: base }), animation: false }
  }, [rowX, rowY, groups, labels, byExpert, expertRows, centreRows, covs, fitted, interval, d, mode, extra, base])
  // The decision surface is an exposition view (its own frame); the rest is one chart in this panel's frame.
  if (surface.value !== undefined)
    return (
      <FigureChromeProvider chrome={props.chrome}>
        <ExpositionView
          record={surface.value as never}
          figure={base}
          title={props.title}
          filename={props.filename}
          points={rowsOf(xs, d)}
          {...(groups === undefined ? {} : { labels: groups })}
        />
      </FigureChromeProvider>
    )
  return (
    <FigureChromeProvider chrome={props.chrome}>
      <FigureFrame
        figure={base}
        title={props.title}
        filename={props.filename}
        {...(error === undefined ? {} : { caption: <Note>unavailable: {error}</Note> })}
      >
        <Chart figure={base} option={option} />
      </FigureFrame>
    </FigureChromeProvider>
  )
}

function rowsOf(flat: ArrayLike<number>, width: number): number[][] {
  const out: number[][] = []
  for (let i = 0; i + width <= flat.length; i += width)
    out.push(Array.from({ length: width }, (_, j) => flat[i + j] ?? 0))
  return out
}
