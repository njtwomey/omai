import type { Json } from 'mdml-engine/base'
import { covarianceEllipse } from 'mdml-engine/numerics/geometry'
import { type Tensor, toFlat } from 'mdml-engine/tensor'
import { cartesian, type Layer, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ExpositionView, FigureFrame, useColourMode } from 'mdml-render/react'
import { categoricalColour } from 'mdml-render/theme'
import { type ReactNode, useMemo } from 'react'
import type { LiveSession } from '../session/useSession.ts'
import type { Panel } from './entries.ts'

export const FIG = figure({ width: 440, height: 320 })

export type Probed = { readonly value?: unknown; readonly error?: string }

// One probe of the session at a step, computed once per (session, step) and never throwing: a failure is returned as
// its message so a panel can say why and still draw its rows.
export function useProbe(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  live: LiveSession<any, any>,
  t: number,
  path: string | undefined,
  args?: Json,
): Probed {
  const argsKey = JSON.stringify(args ?? null)
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the viewed step, not on every tape version
  return useMemo(() => {
    if (path === undefined) return {}
    try {
      return { value: live.session.probe(t, path, args) }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [live.session, t, path, argsKey])
}

/** The one-line note a panel shows in its frame when its probe failed. */
export const Note = (props: { readonly children: ReactNode }): ReactNode => (
  <span className="text-xs text-muted">{props.children}</span>
)

/**
 * The training rows (by assignment, class or target) with the probe overlay at the scrubbed step: `assignments` the
 * assignments and centres, `components` those with each component's 1σ and 2σ ellipses, `fit` the 1-D fit, `surface`
 * the decision surface. Each probe is independent and the chart stays mounted across frames.
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
}): ReactNode {
  const { overlay, live, t, x, y } = props
  const mode = useColourMode()
  const withCentres = overlay === 'assignments' || overlay === 'components'
  const assignments = useProbe(live, t, withCentres ? 'model/assignments' : undefined)
  const centres = useProbe(live, t, withCentres ? 'model/centres' : undefined)
  // A mixture's parameters carry covariances: drawn as each component's 1 and 2 sigma ellipses.
  const params = useProbe(live, t, overlay === 'components' ? 'model/params' : undefined)
  const fit = useProbe(live, t, overlay === 'fit' ? 'model/predictions' : undefined)
  const surface = useProbe(live, t, overlay === 'surface' ? 'model/decision-surface' : undefined)
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
  const classes = overlay === 'surface' && ys !== undefined ? Array.from(ys) : undefined
  const groups = labels ?? classes
  const centreRows = centres.value as number[][] | undefined
  const covs = (params.value as { readonly covariances?: number[][][] } | undefined)?.covariances
  const fitted = fit.value as number[] | undefined
  const error = [assignments, centres, params, fit, surface].find((p) => p.error !== undefined)?.error
  const option = useMemo(() => {
    const layers: Layer[] = [
      points([{ x: rowX, y: rowY, ...(groups === undefined ? {} : { groups }), opacity: 0.7 }], { mode }),
    ]
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
    if (fitted !== undefined && d === 1) {
      const order = rowX.map((_, i) => i).sort((a, b) => (rowX[a] ?? 0) - (rowX[b] ?? 0))
      layers.push(
        lines([{ x: order.map((i) => rowX[i] ?? 0), y: order.map((i) => fitted[i] ?? 0), name: 'fit' }], {
          mode,
          colourOffset: 1,
        }),
      )
    }
    return { ...cartesian(layers, { mode, figure: FIG }), animation: false }
  }, [rowX, rowY, groups, centreRows, covs, fitted, d, mode])
  // The decision surface is an exposition view (its own frame); the rest is one chart in this panel's frame.
  if (surface.value !== undefined)
    return (
      <ExpositionView
        record={surface.value as never}
        figure={FIG}
        points={rowsOf(xs, d)}
        {...(groups === undefined ? {} : { labels: groups })}
      />
    )
  return (
    <FigureFrame
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(error === undefined ? {} : { caption: <Note>unavailable: {error}</Note> })}
    >
      <Chart figure={FIG} option={option} />
    </FigureFrame>
  )
}

function rowsOf(flat: ArrayLike<number>, width: number): number[][] {
  const out: number[][] = []
  for (let i = 0; i + width <= flat.length; i += width)
    out.push(Array.from({ length: width }, (_, j) => flat[i + j] ?? 0))
  return out
}
