/**
 * The panels of a session that draw one record probed at the viewed step with its own view: a strip, a marked table,
 * a grid of small images, a graph whose looks follow the run. Each says in its frame why when the probe fails or is not
 * a record of the kind it draws.
 */

import type { Exposition, ExpositionKind } from 'omai-engine/exposition'
import { type ReactNode, useMemo } from 'react'
import { figure } from '../../figure/figure.ts'
import { imageGridFigure } from '../../views/images.ts'
import { stripAspect } from '../../views/strips.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { AudioView } from '../views/audio.tsx'
import { EquationFitView } from '../views/equation-fit.tsx'
import { ImageGridView } from '../views/images.tsx'
import { LandscapeSliceView } from '../views/landscape-slice.tsx'
import { MarkedTableView } from '../views/marked-tables.tsx'
import { PointsStepsView } from '../views/points-steps.tsx'
import { SampleGridView } from '../views/sample-grid.tsx'
import { SceneStepsView } from '../views/scene-steps.tsx'
import { StabilityRegionView } from '../views/stability-region.tsx'
import { StripView } from '../views/strips.tsx'
import { StructureStepsView } from '../views/structure-steps.tsx'
import { TranscriptView } from '../views/transcript.tsx'
import { useStepCursor } from './cursor.tsx'
import type { Panel } from './layout.ts'
import { FIG, Note, useProbe } from './RowsPanel.tsx'
import type { LiveSession } from './useSession.ts'

type Recorded<K extends ExpositionKind> = Extract<Exposition, { kind: K }>

// The probe at the viewed step as a record of one kind, or why not.
function useRecord<K extends ExpositionKind>(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  live: LiveSession<any, any>,
  t: number,
  kind: K,
  path: string,
  args: Parameters<typeof useProbe>[3],
): { readonly record?: Recorded<K>; readonly problem?: string } {
  const probed = useProbe(live, t, path, args)
  if ((probed.value as { kind?: unknown } | undefined)?.kind === kind) return { record: probed.value as Recorded<K> }
  if (probed.error !== undefined) return { problem: probed.error }
  return probed.value === undefined ? {} : { problem: `the probe is not a ${kind} record` }
}

// A frame saying why nothing is drawn yet.
function Nothing(props: { readonly title: string; readonly filename: string; readonly problem?: string }): ReactNode {
  return (
    <FigureFrame
      figure={FIG}
      title={props.title}
      filename={props.filename}
      caption={<Note>{props.problem === undefined ? 'nothing recorded yet' : `unavailable: ${props.problem}`}</Note>}
    >
      <div className="h-full" />
    </FigureFrame>
  )
}

/** The width the record panels draw at: the session panels' figure width, whatever the record's shape. */
const WIDTH = FIG.width

// The figure of the equation panel: the expressions, the table of terms, the curve and the residual strip stacked.
const TALL = figure({ width: WIDTH, height: 500 })

/**
 * A strip probed at the viewed step: tokens on x, a row of cells per track, tags, the chosen path and marks.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function StripPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'strip' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'strip', props.panel.path, props.panel.args)
  const fig = useMemo(
    () => (record === undefined ? FIG : figure({ width: WIDTH, aspect: stripAspect(record) })),
    [record],
  )
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <StripView record={record} figure={fig} title={props.title} filename={props.filename} />
}

/**
 * A marked table probed at the viewed step.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function TablePanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'table' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'markedTable', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <MarkedTableView record={record} figure={FIG} title={props.title} filename={props.filename} />
}

/**
 * Small images probed at the viewed step as a grid: a row per group, captions under the images.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function ImagesPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'images' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { panel, t } = props
  const { record, problem } = useRecord(props.live, t, 'imageSteps', panel.path, panel.args)
  const fig = useMemo(
    () => (record === undefined ? FIG : imageGridFigure(record, WIDTH, panel.columns)),
    [record, panel.columns],
  )
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return (
    <ImageGridView
      record={record}
      step={t}
      figure={fig}
      title={props.title}
      filename={props.filename}
      {...(panel.columns === undefined ? {} : { columns: panel.columns })}
    />
  )
}

/**
 * A graph probed at the viewed step: its nodes and edges looking as the latest recorded step at or before it says.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function GraphPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'graph' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { panel, t } = props
  const { record, problem } = useRecord(props.live, t, 'structureSteps', panel.path, panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return (
    <StructureStepsView
      record={record}
      step={t}
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(panel.layout === undefined ? {} : { layout: panel.layout })}
    />
  )
}

/**
 * A point set probed at the viewed step: the points so far on axes that hold still, or a cloud to orbit.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function PointsPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'points' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'pointsSteps', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  const { colour, fade, camera } = props.panel
  return (
    <PointsStepsView
      record={record}
      step={props.t}
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(colour === undefined ? {} : { colour })}
      {...(fade === undefined ? {} : { fade })}
      {...(camera === undefined ? {} : { camera })}
    />
  )
}

/**
 * A scene probed at the viewed step: the latest scene recorded at or before it.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function ScenePanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'scene' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'sceneSteps', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <SceneStepsView record={record} step={props.t} figure={FIG} title={props.title} filename={props.filename} />
}

/**
 * A slice through a loss landscape probed at the viewed step: a curve or contours with the iterates over them.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function LandscapePanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'landscape' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { panel, t } = props
  const { record, problem } = useRecord(props.live, t, 'landscapeSlice', panel.path, panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return (
    <LandscapeSliceView
      record={record}
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(panel.surface === undefined ? {} : { surface: panel.surface })}
    />
  )
}

/**
 * The region of absolute stability probed at the viewed step, with the eigenvalues times the step size marked.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function StabilityPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'stability' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'stabilityRegion', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <StabilityRegionView record={record} figure={FIG} title={props.title} filename={props.filename} />
}

/**
 * A fitted equation probed at the viewed step, beside the truth with its residual.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function EquationPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'equation' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'equationFit', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <EquationFitView record={record} figure={TALL} title={props.title} filename={props.filename} />
}

/**
 * An audio clip probed at the viewed step: waveform, spectrogram and a playhead locked to the step cursor, which a click
 * on the picture and the sound playing both move.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function AudioPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'audio' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'audio', props.panel.path, props.panel.args)
  const cursor = useStepCursor()
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return (
    <AudioView
      record={record}
      step={props.t}
      title={props.title}
      filename={props.filename}
      onStep={(s) => cursor?.scrub(Math.min(s, cursor.head))}
    />
  )
}

/**
 * A transcript probed at the viewed step: tokens in turns with the highlight of a track and the emitted token marked.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function TranscriptPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'transcript' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { record, problem } = useRecord(props.live, props.t, 'transcript', props.panel.path, props.panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return <TranscriptView record={record} figure={FIG} title={props.title} filename={props.filename} />
}

/**
 * Samples over training probed at the viewed step: the grid of the viewed step with a filmstrip of earlier ones.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function SamplesPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'samples' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { panel, t } = props
  const { record, problem } = useRecord(props.live, t, 'sampleGrid', panel.path, panel.args)
  if (record === undefined) return <Nothing {...props} {...(problem === undefined ? {} : { problem })} />
  return (
    <SampleGridView
      record={record}
      step={t}
      title={props.title}
      filename={props.filename}
      {...(panel.filmstrip === undefined ? {} : { filmstrip: panel.filmstrip })}
    />
  )
}
