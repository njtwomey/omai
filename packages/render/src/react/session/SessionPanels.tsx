/**
 * The panels of a session layout: curves and per-element series of the tape (which scrub the shared step cursor on
 * click), the rows, a probed derived path, a tensor as a heatmap and trajectories in the plane at the viewed step, a
 * projection of the tape so far, and the whole tape; laid out in a grid under one step cursor.
 */

import { isExposition } from 'mdml-engine/exposition'
import { covarianceEllipse } from 'mdml-engine/numerics/geometry'
import { type Tensor, toFlat } from 'mdml-engine/tensor'
import { type Run, type TensorEvent, tensorSeries } from 'mdml-engine/trace'
import { type ReactNode, useMemo } from 'react'
import { contours } from '../../charts/fields.ts'
import { lines } from '../../charts/lines.ts'
import { heatmapChart } from '../../charts/matrices.ts'
import { cartesian, type Layer } from '../../charts/plot.ts'
import { points } from '../../charts/points.ts'
import { metricCurves } from '../../trace/metrics.ts'
import { parameterTrajectories, stepPositions } from '../../trace/tensors.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'
import { ExpositionView } from '../views/ExpositionView.tsx'
import { StepCursorProvider } from './cursor.tsx'
import type { Panel } from './layout.ts'
import { FIG, Note, RowsPanel, useProbe } from './RowsPanel.tsx'
import { recordedSteps, stepMarker, useScrubClick } from './scrub.ts'
import { TapeChannels } from './TapeView.tsx'
import type { LiveSession } from './useSession.ts'

/** What a panel needs of the page: a name for its files, the session, the scrubbed step and the training data (if any). */
export interface PanelContext {
  /** The name exported files start with (an entry's id, say). */
  readonly id: string
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly at: number | null
  /** Sets the scrubbed step (`null` follows the head). */
  readonly onScrub?: (t: number | null) => void
  readonly x?: Tensor
  readonly y?: Tensor
}

const ROWS_TITLES = {
  assignments: 'assignments and centres',
  components: 'components (1σ solid, 2σ dashed)',
  fit: 'the fit',
  band: 'the fit and its band',
  surface: 'decision surface',
  experts: 'experts and the mixture',
} as const

const slug = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]+/g, '-')
const leaf = (path: string): string => path.split('/').at(-1) ?? path

/**
 * The panels of a layout over a session, in a responsive two-column grid; nothing else is drawn. The viewed step is one
 * cursor: panels of the tape scrub it on click, panels of a step follow it, and a custom panel inside reads it with
 * `useStepCursor`.
 *
 * @param props - The session, the scrubbed step and its setter, the training data if any, and the layout.
 * @returns The grid of panels.
 */
export function SessionPanels(props: PanelContext & { readonly layout: readonly Panel[] }): ReactNode {
  const { layout, ...ctx } = props
  const head = ctx.live.session.t
  const viewed = ctx.at === null ? head : Math.min(ctx.at, head)
  const cursor = useMemo(() => ({ viewed, head, scrub: ctx.onScrub ?? (() => undefined) }), [viewed, head, ctx.onScrub])
  return (
    <StepCursorProvider value={cursor}>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2" data-testid="panels">
        {layout.map((panel) => (
          <div key={JSON.stringify(panel)} className="min-w-0">
            <PanelView panel={panel} {...ctx} />
          </div>
        ))}
      </div>
    </StepCursorProvider>
  )
}

function PanelView(props: PanelContext & { readonly panel: Panel }): ReactNode {
  const { panel, id, live, at, onScrub } = props
  const t = at === null ? live.session.t : Math.min(at, live.session.t)
  const file = (title: string): string => `${slug(id)}-${slug(title)}`
  switch (panel.kind) {
    case 'rows': {
      if (props.x === undefined) return null
      const title = panel.title ?? ROWS_TITLES[panel.overlay ?? 'assignments']
      return (
        <RowsPanel
          overlay={panel.overlay}
          title={`${title} at step ${t}`}
          filename={file(title)}
          live={live}
          t={t}
          x={props.x}
          {...(props.y === undefined ? {} : { y: props.y })}
        />
      )
    }
    case 'probe':
      return <ProbeView {...props} panel={panel} t={t} file={file} />
    case 'curves': {
      const title = panel.title ?? panel.paths.join(', ')
      return (
        <CurvesView
          live={live}
          paths={panel.paths}
          title={title}
          filename={file(title)}
          t={t}
          {...(onScrub === undefined ? {} : { onScrub })}
        />
      )
    }
    case 'series': {
      const title = panel.title ?? leaf(panel.path)
      return (
        <SeriesView
          live={live}
          panel={panel}
          title={title}
          filename={file(title)}
          t={t}
          {...(onScrub === undefined ? {} : { onScrub })}
        />
      )
    }
    case 'trajectory': {
      const title = panel.title ?? panel.paths.map(leaf).join(', ')
      return <TrajectoryView live={live} panel={panel} title={`${title} to step ${t}`} filename={file(title)} t={t} />
    }
    case 'tensor': {
      const title = panel.title ?? leaf(panel.path)
      return <TensorView live={live} path={panel.path} title={title} filename={file(title)} t={t} />
    }
    case 'projection':
      return <ProjectionView live={live} name={panel.name} t={at === null ? undefined : t} />
    case 'tape':
      return <TapeChannels live={live} at={at} {...(onScrub === undefined ? {} : { onScrub })} />
  }
}

// The tape, re-read whenever it grows.
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
function useRun(live: LiveSession<any, any>): Run {
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  return useMemo(() => live.session.run(), [live.session, live.version])
}

function CurvesView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly paths: readonly string[]
  readonly title: string
  readonly filename: string
  /** The viewed step, marked on the chart. */
  readonly t: number
  readonly onScrub?: (t: number | null) => void
}): ReactNode {
  const { live, paths, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const option = useMemo(() => {
    const curves = metricCurves(run, paths, { mode })
    return cartesian([curves, ...stepMarker(curves, t, mode)], {
      mode,
      figure: FIG,
      x: { label: 'step' },
      legend: true,
    })
  }, [run, paths, mode, t])
  const steps = useMemo(() => recordedSteps(run, paths), [run, paths])
  const onInit = useScrubClick(steps, live.session.t, props.onScrub)
  return (
    <FigureFrame figure={FIG} title={props.title} filename={props.filename}>
      <Chart figure={FIG} option={option} onInit={onInit} />
    </FigureFrame>
  )
}

/** The tensor events of a path with full values (subsampled or summarised ones cannot be drawn element by element). */
function fullTensors(run: Run, path: string): TensorEvent[] {
  return tensorSeries(run, path).filter((e) => e.values !== undefined && e.indices === undefined)
}

/** The latest tensor event of a path at or before step `t`, with full values. */
function tensorAt(run: Run, path: string, t: number): TensorEvent | undefined {
  return fullTensors(run, path)
    .filter((e) => e.step <= t)
    .at(-1)
}

// Each element of a tensor path as a curve against the step or a scalar path; a click scrubs to the nearest step.
function SeriesView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'series' }>
  readonly title: string
  readonly filename: string
  readonly t: number
  readonly onScrub?: (t: number | null) => void
}): ReactNode {
  const { live, panel, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const { steps, xs } = useMemo(() => stepPositions(run, panel.path, panel.x), [run, panel.path, panel.x])
  const option = useMemo(() => {
    const layer = parameterTrajectories(run, panel.path, {
      mode,
      max: panel.max ?? 32,
      ...(panel.x === undefined ? {} : { x: panel.x }),
      ...(panel.names === undefined ? {} : { names: panel.names }),
    })
    const i = steps.findIndex((s) => s >= t)
    const xAt = xs[i < 0 ? xs.length - 1 : i] ?? t
    return cartesian([layer, ...stepMarker(layer, xAt, mode)], {
      mode,
      figure: FIG,
      x: { label: panel.x === undefined ? 'step' : leaf(panel.x) },
      legend: layer.series.length <= 8,
    })
  }, [run, steps, xs, t, mode, panel])
  const onInit = useScrubClick(steps, live.session.t, props.onScrub, panel.x === undefined ? undefined : xs)
  return (
    <FigureFrame figure={FIG} title={props.title} filename={props.filename}>
      <Chart figure={FIG} option={option} onInit={onInit} />
    </FigureFrame>
  )
}

// Two elements of tensor paths as paths in the plane up to the viewed step, over a probed surface if asked.
function TrajectoryView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'trajectory' }>
  readonly title: string
  readonly filename: string
  readonly t: number
}): ReactNode {
  const { live, panel, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const over = useProbe(live, t, panel.over?.path, panel.over?.args)
  const [i, j] = panel.dims ?? [0, 1]
  const option = useMemo(() => {
    const layers: Layer[] = []
    const surface = over.value as
      | { readonly x?: number[]; readonly y?: number[]; readonly values?: number[][] }
      | undefined
    if (surface?.values !== undefined)
      layers.push(
        contours(surface.values, {
          ...(surface.x === undefined ? {} : { x: surface.x }),
          ...(surface.y === undefined ? {} : { y: surface.y }),
          colour: 'ramp',
          labels: false,
          opacity: 0.6,
          mode,
          name: '__surface',
        }),
      )
    const heads: { x: number; y: number }[] = []
    panel.paths.forEach((path, k) => {
      const events = fullTensors(run, path).filter((e) => e.step <= t)
      const trail = panel.trail === undefined ? events : events.slice(-panel.trail)
      const x = trail.map((e) => e.values?.[i] ?? Number.NaN)
      const y = trail.map((e) => e.values?.[j] ?? Number.NaN)
      const name = leaf(path)
      if (x.length === 0) return
      layers.push(
        panel.points === true
          ? points([{ x, y, name, opacity: 0.6, size: 5 }], { mode, colourOffset: k })
          : lines([{ x, y, name, width: 1.5, opacity: 0.9 }], { mode, colourOffset: k }),
      )
      heads.push({ x: x.at(-1) ?? Number.NaN, y: y.at(-1) ?? Number.NaN })
    })
    if (heads.length > 0)
      layers.push(
        points(
          heads.map((h, k) => ({ x: [h.x], y: [h.y], name: `__head${k}`, size: 12, symbol: 'circle' })),
          { mode },
        ),
      )
    if (panel.ellipse !== undefined && heads[0] !== undefined) {
      const cov = tensorAt(run, panel.ellipse, t)
      const d = cov?.shape[0] ?? 0
      if (cov?.values !== undefined && d > Math.max(i, j)) {
        const sigma = [
          [cov.values[i * d + i] ?? 0, cov.values[i * d + j] ?? 0],
          [cov.values[j * d + i] ?? 0, cov.values[j * d + j] ?? 0],
        ]
        const flat = toFlat(covarianceEllipse([heads[0].x, heads[0].y], sigma, { nStd: 1, points: 72 }).points)
        layers.push(
          lines(
            [
              {
                x: Array.from({ length: 73 }, (_, k) => flat[(2 * k) % flat.length] ?? 0),
                y: Array.from({ length: 73 }, (_, k) => flat[(2 * k + 1) % flat.length] ?? 0),
                name: '__ellipse',
                colour: 'muted',
                width: 1,
                dash: 'dashed',
              },
            ],
            { mode },
          ),
        )
      }
    }
    return { ...cartesian(layers, { mode, figure: FIG, legend: panel.paths.length > 1 }), animation: false }
  }, [run, over.value, panel, t, i, j, mode])
  return (
    <FigureFrame
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(over.error === undefined ? {} : { caption: <Note>surface unavailable: {over.error}</Note> })}
    >
      <Chart figure={FIG} option={option} />
    </FigureFrame>
  )
}

// A tensor path at the viewed step as a heatmap: a vector as one row, a matrix as is, a deeper tensor by its last axis.
function TensorView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly path: string
  readonly title: string
  readonly filename: string
  readonly t: number
}): ReactNode {
  const { live, path, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const at = useMemo(() => tensorAt(run, path, t), [run, path, t])
  const option = useMemo(() => {
    if (at?.values === undefined) return undefined
    const width = at.shape.length >= 2 ? (at.shape[at.shape.length - 1] ?? 1) : at.values.length
    const z: number[][] = []
    for (let r = 0; r < at.values.length; r += width) z.push([...at.values.slice(r, r + width)])
    // Row 0 at the top, as a matrix is read.
    return { ...heatmapChart(z.reverse(), { smooth: 'cells' }, { mode, figure: FIG }), animation: false }
  }, [at, mode])
  return (
    <FigureFrame
      figure={FIG}
      title={`${props.title} at step ${at?.step ?? t}${at === undefined ? '' : ` (${at.shape.join('×')})`}`}
      filename={props.filename}
      {...(option === undefined ? { caption: <Note>nothing recorded yet</Note> } : {})}
    >
      {option === undefined ? <div className="h-full" /> : <Chart figure={FIG} option={option} />}
    </FigureFrame>
  )
}

function ProbeView(
  props: PanelContext & {
    readonly panel: Extract<Panel, { kind: 'probe' }>
    readonly t: number
    readonly file: (title: string) => string
  },
): ReactNode {
  const { panel, live, t } = props
  const probed = useProbe(live, t, panel.path, panel.args)
  const title = `${panel.title ?? panel.path} at step ${t}`
  const filename = props.file(panel.title ?? panel.path)
  // A record at a step is a still: views that would play it along x are told not to.
  if (probed.value !== undefined && isExposition(probed.value))
    return <ExpositionView record={probed.value} figure={FIG} title={title} filename={filename} play={false} />
  const text = probed.value === undefined ? undefined : JSON.stringify(probed.value, null, 1)
  return (
    <FigureFrame
      figure={FIG}
      title={title}
      filename={filename}
      exportFormats={['json']}
      onExport={() => {
        const a = document.createElement('a')
        a.href = URL.createObjectURL(new Blob([text ?? ''], { type: 'application/json' }))
        a.download = `${filename}.json`
        a.click()
      }}
      {...(probed.error === undefined ? {} : { caption: <Note>unavailable: {probed.error}</Note> })}
    >
      <pre className="h-full overflow-auto p-2 font-mono text-xs">{text}</pre>
    </FigureFrame>
  )
}

// A projection of the tape so far; hidden while it has nothing to project.
function ProjectionView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly name: string
  /** The scrubbed step: project the tape up to it (`undefined` follows the head). */
  readonly t: number | undefined
}): ReactNode {
  const { live, name, t } = props
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const record = useMemo((): unknown => {
    try {
      const s = live.session
      const p = s.method.projections?.[name]
      if (t === undefined || p === undefined) return s.project(name)
      const run = s.run()
      return p({ ...run, events: run.events.filter((e) => !('step' in e) || e.step <= t) })
    } catch {
      return undefined
    }
  }, [live.session, live.version, name, t])
  return record !== undefined && isExposition(record) ? <ExpositionView record={record} figure={FIG} /> : null
}
