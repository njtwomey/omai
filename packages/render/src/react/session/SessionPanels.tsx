/**
 * The panels of a session layout: curves and per-element series of the tape (which scrub the shared step cursor on
 * click), the trials of a sweep, the rows, a probed derived path, a tensor as a heatmap and trajectories in the plane
 * at the viewed step, a projection of the tape so far, and the whole tape; laid out in a grid under one step cursor.
 * Over a comparison, curves and sweeps overlay the variants (one colour each, the step marker shared) and every other
 * panel is drawn once per variant, side by side, titled with the variant's label.
 */

import { isExposition } from 'mdml-engine/exposition'
import type { Tensor } from 'mdml-engine/tensor'
import type { Run } from 'mdml-engine/trace'
import { type ReactNode, useMemo } from 'react'
import { contours } from '../../charts/fields.ts'
import { lines } from '../../charts/lines.ts'
import { heatmapChart } from '../../charts/matrices.ts'
import { cartesian, type Layer } from '../../charts/plot.ts'
import { type PointSet, points } from '../../charts/points.ts'
import { categoricalColour } from '../../theme/palettes.ts'
import { parameterTrajectories, stepPositions } from '../../trace/tensors.ts'
import { type MatrixMark, matrixMarksView } from '../../views/matrix-marks.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'
import { ExpositionView } from '../views/ExpositionView.tsx'
import { leaf, overlayCurves, sweepTrials } from './compare.ts'
import { StepCursorProvider } from './cursor.tsx'
import { FieldPanel } from './FieldPanel.tsx'
import { type FamilyOptions, familyOption, hasFamily, intervalOption, logAxes, sweepFamily } from './families.ts'
import type { Panel } from './layout.ts'
import { Multiples } from './Multiples.tsx'
import { GraphPanel, ImagesPanel, PointsPanel, ScenePanel, StripPanel, TablePanel } from './RecordPanels.tsx'
import { FIG, Note, RowsPanel, useProbe, useRun } from './RowsPanel.tsx'
import { SpecialisedPanel } from './SpecialisedPanels.tsx'
import { recordedSteps, stepMarker, useScrubClick } from './scrub.ts'
import { TapeChannels } from './TapeView.tsx'
import { tensorAt, trajectoryLayers } from './trajectory.ts'
import { type LiveSession, type LiveSessions, type LiveVariant, variantsOf } from './useSession.ts'

/**
 * What a panel needs of the page: a name for its files, the session (or a comparison of several), the scrubbed step
 * and the training data (if any).
 */
export interface PanelContext {
  /** The name exported files start with (an entry's id, say). */
  readonly id: string
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any> | LiveSessions<any, any>
  readonly at: number | null
  /** Sets the scrubbed step (`null` follows the head). */
  readonly onScrub?: (t: number | null) => void
  readonly x?: Tensor
  readonly y?: Tensor
}

// What a panel of one variant gets: the page's context with that variant's session and label.
type VariantContext = Omit<PanelContext, 'live'> & {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  /** The variant's label in a comparison; absent for a session on its own. */
  readonly label?: string
  /** The comparison's shared head (the step every variant has reached); the variant's own without one. */
  readonly head?: number
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

/**
 * The panels of a layout over a session or a comparison, in a responsive two-column grid; nothing else is drawn. The
 * viewed step is one cursor: panels of the tape scrub it on click, panels of a step follow it, and a custom panel
 * inside reads it with `useStepCursor`. Over a comparison, curves and sweeps overlay the variants and every other
 * panel takes a full row with one figure per variant.
 *
 * @param props - The session or comparison, the scrubbed step and its setter, the training data if any, and the
 *   layout.
 * @returns The grid of panels.
 */
export function SessionPanels(props: PanelContext & { readonly layout: readonly Panel[] }): ReactNode {
  const { layout, live, ...ctx } = props
  const variants = variantsOf(live)
  const single = variants.length === 1 ? variants[0] : undefined
  const head = live.head
  const viewed = ctx.at === null ? head : Math.min(ctx.at, head)
  const cursor = useMemo(() => ({ viewed, head, scrub: ctx.onScrub ?? (() => undefined) }), [viewed, head, ctx.onScrub])
  return (
    <StepCursorProvider value={cursor}>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2" data-testid="panels">
        {layout.map((panel) => {
          const key = JSON.stringify(panel)
          if (panel.kind === 'multiples')
            return (
              <div key={key} className="col-span-full min-w-0">
                <Multiples
                  spec={panel}
                  variants={variants}
                  renderCell={(cell, limits) => {
                    const cellVariants =
                      cell.variant === undefined ? variants : variants.filter((_, i) => i === cell.variant)
                    const inner = cell.panel
                    if (inner.kind === 'curves' || inner.kind === 'sweep')
                      return (
                        <TapePanel
                          panel={
                            limits !== undefined && inner.kind === 'curves' ? { ...inner, yLimits: limits } : inner
                          }
                          variants={cellVariants}
                          head={head}
                          {...ctx}
                        />
                      )
                    const v = cellVariants[0]
                    return v === undefined ? null : (
                      <PanelView
                        panel={inner}
                        live={v.live}
                        {...(cell.variant === undefined ? {} : { head })}
                        {...ctx}
                      />
                    )
                  }}
                />
              </div>
            )
          if (panel.kind === 'curves' || panel.kind === 'sweep')
            return (
              <div key={key} className="min-w-0">
                <TapePanel panel={panel} variants={variants} head={head} {...ctx} />
              </div>
            )
          if (single !== undefined)
            return (
              <div key={key} className="min-w-0">
                <PanelView panel={panel} live={single.live} {...ctx} />
              </div>
            )
          return (
            <div key={key} className="col-span-full min-w-0" data-testid="compared">
              <div className={panel.kind === 'tape' ? 'flex flex-col gap-4' : 'flex flex-wrap gap-4'}>
                {variants.map((v) => (
                  <div key={v.label} className="min-w-0">
                    <PanelView panel={panel} live={v.live} label={v.label} head={head} {...ctx} />
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </StepCursorProvider>
  )
}

// A panel of the tape over every variant at once: curves or a sweep, overlaid.
function TapePanel(
  props: Omit<PanelContext, 'live'> & {
    readonly panel: Extract<Panel, { kind: 'curves' | 'sweep' }>
    // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
    readonly variants: readonly LiveVariant<any, any>[]
    readonly head: number
  },
): ReactNode {
  const { panel, variants, head, id, at, onScrub } = props
  const t = at === null ? head : Math.min(at, head)
  const file = (title: string): string => `${slug(id)}-${slug(title)}`
  if (panel.kind === 'curves') {
    const title = panel.title ?? panel.paths.join(', ')
    return (
      <CurvesView
        variants={variants}
        head={head}
        paths={panel.paths}
        title={title}
        filename={file(title)}
        t={t}
        family={panel}
        {...(onScrub === undefined ? {} : { onScrub })}
      />
    )
  }
  const title =
    panel.title ?? `${leaf(panel.path)} against ${panel.over === undefined ? 'the trial' : leaf(panel.over)}`
  return (
    <SweepView
      variants={variants}
      head={head}
      panel={panel}
      title={title}
      filename={file(title)}
      t={t}
      {...(onScrub === undefined ? {} : { onScrub })}
    />
  )
}

function PanelView(
  props: VariantContext & { readonly panel: Exclude<Panel, { kind: 'curves' | 'sweep' | 'multiples' }> },
): ReactNode {
  const { panel, id, live, at, onScrub, label } = props
  // The viewed step, never past this variant's head nor the shared one (a variant may run a few steps ahead).
  const limit = Math.min(props.head ?? live.head, live.head)
  const t = at === null ? limit : Math.min(at, limit)
  const titled = (title: string): string => (label === undefined ? title : `${label} · ${title}`)
  const file = (title: string): string => `${slug(id)}-${label === undefined ? '' : `${slug(label)}-`}${slug(title)}`
  switch (panel.kind) {
    case 'rows': {
      if (props.x === undefined) return null
      const channelled = [panel.values, panel.size, panel.marks, panel.regions, panel.focus].some(
        (c) => c !== undefined,
      )
      const title =
        panel.title ??
        (panel.overlay === undefined && channelled ? 'rows' : ROWS_TITLES[panel.overlay ?? 'assignments'])
      return (
        <RowsPanel
          overlay={panel.overlay}
          title={titled(`${title} at step ${t}`)}
          filename={file(title)}
          live={live}
          t={t}
          x={props.x}
          {...(props.y === undefined ? {} : { y: props.y })}
          channels={{
            ...(panel.values === undefined ? {} : { values: panel.values }),
            ...(panel.size === undefined ? {} : { size: panel.size }),
            ...(panel.marks === undefined ? {} : { marks: panel.marks }),
            ...(panel.regions === undefined ? {} : { regions: panel.regions }),
            ...(panel.focus === undefined ? {} : { focus: panel.focus }),
          }}
        />
      )
    }
    case 'probe':
      return <ProbeView {...props} panel={panel} t={t} file={file} titled={titled} />
    case 'series': {
      const title = panel.title ?? leaf(panel.path)
      return (
        <SeriesView
          live={live}
          panel={panel}
          title={titled(title)}
          filename={file(title)}
          t={t}
          {...(onScrub === undefined ? {} : { onScrub })}
        />
      )
    }
    case 'trajectory': {
      const title = panel.title ?? panel.paths.map(leaf).join(', ')
      return (
        <TrajectoryView
          live={live}
          panel={panel}
          title={titled(`${title} to step ${t}`)}
          filename={file(title)}
          t={t}
        />
      )
    }
    case 'tensor': {
      const title = panel.title ?? leaf(panel.path)
      return <TensorView live={live} panel={panel} title={titled(title)} filename={file(title)} t={t} />
    }
    case 'field': {
      const title = panel.title ?? panel.path
      return (
        <FieldPanel live={live} panel={panel} t={t} title={titled(`${title} at step ${t}`)} filename={file(title)} />
      )
    }
    case 'strip':
    case 'table':
    case 'images':
    case 'points':
    case 'scene':
    case 'graph': {
      const title = panel.title ?? panel.path
      const common = { live, t, title: titled(`${title} at step ${t}`), filename: file(title) }
      if (panel.kind === 'strip') return <StripPanel {...common} panel={panel} />
      if (panel.kind === 'table') return <TablePanel {...common} panel={panel} />
      if (panel.kind === 'images') return <ImagesPanel {...common} panel={panel} />
      if (panel.kind === 'points') return <PointsPanel {...common} panel={panel} />
      if (panel.kind === 'scene') return <ScenePanel {...common} panel={panel} />
      return <GraphPanel {...common} panel={panel} />
    }
    case 'zoom':
    case 'landscape':
    case 'stability':
    case 'equation':
    case 'audio':
    case 'transcript':
    case 'samples': {
      const title = panel.title ?? panel.path
      return (
        <SpecialisedPanel
          live={live}
          panel={panel}
          t={t}
          title={titled(`${title} at step ${t}`)}
          filename={file(title)}
        />
      )
    }
    case 'projection': {
      const title = panel.title ?? panel.name
      return (
        <ProjectionView
          live={live}
          name={panel.name}
          t={at === null ? undefined : t}
          title={titled(`${title} at step ${t}`)}
          filename={file(title)}
        />
      )
    }
    case 'tape':
      return label === undefined ? (
        <TapeChannels live={live} at={at} {...(onScrub === undefined ? {} : { onScrub })} />
      ) : (
        <div className="flex flex-col gap-1">
          <div className="text-sm text-muted">{label}</div>
          <TapeChannels live={live} at={at} {...(onScrub === undefined ? {} : { onScrub })} />
        </div>
      )
  }
}

// The tapes of every variant, re-read whenever they grow (they share one version and are remade together).
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
function useRuns(variants: readonly LiveVariant<any, any>[]): readonly Run[] {
  const first = variants[0]?.live.session
  const version = variants.reduce((v, x) => Math.max(v, x.live.version), 0)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tapes change with the version
  return useMemo(() => variants.map((v) => v.live.session.run()), [first, version, variants.length])
}

function CurvesView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly variants: readonly LiveVariant<any, any>[]
  readonly head: number
  readonly paths: readonly string[]
  readonly title: string
  readonly filename: string
  /** The viewed step, marked on the chart. */
  readonly t: number
  /** The curve families the panel asks for (many runs, bands, a truth line, log axes, slopes, events). */
  readonly family?: FamilyOptions
  readonly onScrub?: (t: number | null) => void
}): ReactNode {
  const { variants, paths, t, family } = props
  const mode = useColourMode()
  const runs = useRuns(variants)
  const option = useMemo(() => {
    if (family !== undefined && hasFamily(family))
      return familyOption(
        runs.map((run, i) => ({ label: variants[i]?.label ?? '', run })),
        paths,
        { ...family, t, mode, figure: FIG },
      )
    const curves = overlayCurves(
      runs.map((run, i) => ({ label: variants[i]?.label ?? '', run })),
      paths,
      { mode },
    )
    return cartesian([curves, ...stepMarker(curves, t, mode)], {
      mode,
      figure: FIG,
      x: { label: 'step' },
      legend: true,
    })
  }, [runs, variants, paths, mode, t, family])
  const steps = useMemo(
    () => [...new Set(runs.flatMap((run) => recordedSteps(run, paths)))].sort((a, b) => a - b),
    [runs, paths],
  )
  const onInit = useScrubClick(steps, props.head, props.onScrub)
  return (
    <FigureFrame figure={FIG} title={props.title} filename={props.filename}>
      <Chart figure={FIG} option={option} onInit={onInit} />
    </FigureFrame>
  )
}

// The trials of a sweep: the metric against the knob (or the step), one point set per variant, those after the
// viewed step faint and the viewed one ringed; a click scrubs to the nearest trial.
function SweepView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly variants: readonly LiveVariant<any, any>[]
  readonly head: number
  readonly panel: Extract<Panel, { kind: 'sweep' }>
  readonly title: string
  readonly filename: string
  readonly t: number
  readonly onScrub?: (t: number | null) => void
}): ReactNode {
  const { variants, panel, t } = props
  const mode = useColourMode()
  const runs = useRuns(variants)
  const log = logAxes(panel.log)
  const trials = useMemo(
    () =>
      runs.map((run) => {
        const all = sweepTrials(run, panel.path, panel.over)
        if (!log.x && !log.y) return all
        // A log axis shows only the positive.
        const keep = all.steps.flatMap((_, k) =>
          (!log.x || (all.x[k] ?? 0) > 0) && (!log.y || (all.y[k] ?? 0) > 0) ? [k] : [],
        )
        return {
          steps: keep.map((k) => all.steps[k] ?? 0),
          x: keep.map((k) => all.x[k] ?? 0),
          y: keep.map((k) => all.y[k] ?? 0),
        }
      }),
    [runs, panel, log.x, log.y],
  )
  const option = useMemo(() => {
    if (panel.intervals !== undefined) {
      const run = runs[0]
      if (run === undefined) return {}
      const lower = sweepTrials(run, panel.intervals.lower, panel.over)
      const upper = sweepTrials(run, panel.intervals.upper, panel.over)
      const centre = sweepTrials(run, panel.path, panel.over)
      const truth =
        panel.truth?.value ?? (panel.truth?.path === undefined ? undefined : sweepTrials(run, panel.truth.path).y)
      return intervalOption(
        {
          steps: centre.steps,
          centre: centre.y,
          lower: lower.y,
          upper: upper.y,
          ...(truth === undefined ? {} : { truth }),
          t,
        },
        { mode, figure: FIG, label: leaf(panel.path) },
      )
    }
    const compared = trials.length > 1
    const sets: PointSet[] = []
    trials.forEach((trial, i) => {
      const name = compared ? (variants[i]?.label ?? `${i}`) : leaf(panel.path)
      const colour = categoricalColour(i)
      const past = trial.steps.flatMap((s, k) => (s <= t ? [k] : []))
      const future = trial.steps.flatMap((s, k) => (s > t ? [k] : []))
      if (past.length > 0)
        sets.push({ x: past.map((k) => trial.x[k] ?? 0), y: past.map((k) => trial.y[k] ?? 0), name, colour, size: 7 })
      if (future.length > 0)
        sets.push({
          x: future.map((k) => trial.x[k] ?? 0),
          y: future.map((k) => trial.y[k] ?? 0),
          name: `__ahead${i}`,
          colour,
          size: 6,
          opacity: 0.2,
        })
      const k = past.at(-1)
      if (k !== undefined)
        sets.push({
          x: [trial.x[k] ?? 0],
          y: [trial.y[k] ?? 0],
          name: `__at${i}`,
          colour,
          size: 16,
          symbol: 'emptyCircle',
          outline: false,
        })
    })
    const layer = sets.length === 0 ? lines([], { mode }) : points(sets, { mode })
    const extra = sweepFamily(
      { x: trials.flatMap((trial) => trial.x), y: trials.flatMap((trial) => trial.y) },
      {
        mode,
        ...(panel.truth === undefined ? {} : { truth: panel.truth }),
        ...(panel.slopes === undefined ? {} : { slopes: panel.slopes }),
      },
    )
    return {
      ...cartesian([layer, ...extra], {
        mode,
        figure: FIG,
        x: {
          label: panel.over === undefined ? 'trial (step)' : leaf(panel.over),
          ...(log.x ? { scale: 'log' as const } : {}),
        },
        y: { label: leaf(panel.path), ...(log.y ? { scale: 'log' as const } : {}) },
        legend: compared || extra.length > 0,
      }),
      animation: false,
    }
  }, [trials, runs, variants, panel, t, mode, log.x, log.y])
  // The interval strip has a trial per row, not per x: a click there scrubs nowhere.
  const { steps, xs } = useMemo(
    () =>
      panel.intervals === undefined
        ? { steps: trials.flatMap((trial) => trial.steps), xs: trials.flatMap((trial) => trial.x) }
        : { steps: [], xs: [] },
    [trials, panel.intervals],
  )
  const onInit = useScrubClick(steps, props.head, props.onScrub, xs)
  return (
    <FigureFrame figure={FIG} title={props.title} filename={props.filename}>
      <Chart figure={FIG} option={option} onInit={onInit} />
    </FigureFrame>
  )
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
  const onInit = useScrubClick(steps, live.head, props.onScrub, panel.x === undefined ? undefined : xs)
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
    layers.push(
      ...trajectoryLayers(run, t, {
        paths: panel.paths,
        dims: [i, j],
        mode,
        ...(panel.trail === undefined ? {} : { trail: panel.trail }),
        ...(panel.points === undefined ? {} : { points: panel.points }),
        ...(panel.ellipse === undefined ? {} : { ellipse: panel.ellipse }),
      }),
    )
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
// With labels or marks (fixed, or probed at the viewed step) it is drawn as labelled cells with the marks over them.
function TensorView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'tensor' }>
  readonly title: string
  readonly filename: string
  readonly t: number
}): ReactNode {
  const { live, panel, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const at = useMemo(() => tensorAt(run, panel.path, t), [run, panel.path, t])
  const probedMarks = useProbe(live, t, panel.marksPath, panel.marksArgs)
  const marked =
    panel.rowLabels !== undefined ||
    panel.colLabels !== undefined ||
    panel.marks !== undefined ||
    panel.marksPath !== undefined
  const option = useMemo(() => {
    if (at?.values === undefined) return undefined
    const width = at.shape.length >= 2 ? (at.shape[at.shape.length - 1] ?? 1) : at.values.length
    const z: number[][] = []
    for (let r = 0; r < at.values.length; r += width) z.push([...at.values.slice(r, r + width)])
    if (marked) {
      const probed = Array.isArray(probedMarks.value) ? (probedMarks.value as MatrixMark[]) : []
      return matrixMarksView(z, {
        mode,
        figure: FIG,
        marks: [...(panel.marks ?? []), ...probed],
        ...(panel.rowLabels === undefined ? {} : { rowLabels: panel.rowLabels }),
        ...(panel.colLabels === undefined ? {} : { colLabels: panel.colLabels }),
      })
    }
    // Row 0 at the top, as a matrix is read.
    return { ...heatmapChart(z.reverse(), { smooth: 'cells' }, { mode, figure: FIG }), animation: false }
  }, [at, mode, marked, panel, probedMarks.value])
  return (
    <FigureFrame
      figure={FIG}
      title={`${props.title} at step ${at?.step ?? t}${at === undefined ? '' : ` (${at.shape.join('×')})`}`}
      filename={props.filename}
      {...(option === undefined
        ? { caption: <Note>nothing recorded yet</Note> }
        : probedMarks.error === undefined
          ? {}
          : { caption: <Note>marks unavailable: {probedMarks.error}</Note> })}
    >
      {option === undefined ? <div className="h-full" /> : <Chart figure={FIG} option={option} />}
    </FigureFrame>
  )
}

function ProbeView(
  props: VariantContext & {
    readonly panel: Extract<Panel, { kind: 'probe' }>
    readonly t: number
    readonly file: (title: string) => string
    readonly titled: (title: string) => string
  },
): ReactNode {
  const { panel, live, t } = props
  const probed = useProbe(live, t, panel.path, panel.args)
  const title = props.titled(`${panel.title ?? panel.path} at step ${t}`)
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
  readonly title: string
  readonly filename: string
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
  // The record up to the viewed step is a still: views that would play it are told not to.
  return record !== undefined && isExposition(record) ? (
    <ExpositionView record={record} figure={FIG} title={props.title} filename={props.filename} play={false} />
  ) : null
}
