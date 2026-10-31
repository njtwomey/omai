/**
 * The React view of a point set that grows over a run: a scatter plot with fixed axes in two dimensions, a cloud to
 * orbit (drag, arrow keys, camera presets) in three, at a given step, coloured by class or by a continuous channel
 * and with the trail of older points fading when asked.
 */

import type { PointsSteps } from 'mdml-engine/exposition'
import { fitCamera, camera as makeCamera } from 'mdml-engine/scene'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { sequentialRamp } from '../../theme/palettes.ts'
import { short } from '../../views/_view.ts'
import {
  CAMERA_PRESETS,
  type CameraPreset,
  cameraPreset,
  type PointsLook,
  pointsStepsScene3d,
  pointsStepsValueRange,
  pointsStepsView,
} from '../../views/points-steps.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { Scene3DView } from '../Scene3DView.tsx'
import { ToolButton } from '../session/parts.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/** Props of {@link PointsStepsView}: the record, the tape step to show, the look and the frame settings. */
export interface PointsStepsViewProps extends RecordViewProps<PointsSteps>, PointsLook {
  /** The tape step to show: the points up to it, or the latest set at or before it (default the last recorded). */
  readonly step?: number
  /** The viewpoint a three-dimensional cloud starts from (default `iso`); the buttons choose another. */
  readonly camera?: CameraPreset
}

const DEFAULTS = { aspect: 1.25 } as const

// The colour bar of a continuous channel: its name, the ramp as a gradient and the two ends.
function ValueBar(props: { readonly record: PointsSteps; readonly mode: Figure['mode'] }): ReactElement {
  const [lo, hi] = pointsStepsValueRange(props.record)
  const gradient = `linear-gradient(to right, ${sequentialRamp('viridis', props.mode).join(', ')})`
  return (
    <span className="flex items-center gap-2 text-xs text-muted tabular-nums">
      <span>{props.record.valueName ?? 'value'}</span>
      <span>{short(lo)}</span>
      <span aria-hidden="true" className="h-2 w-24 rounded-sm" style={{ background: gradient }} />
      <span>{short(hi)}</span>
    </span>
  )
}

/**
 * Points at a step: a scatter plot of the points so far with the latest step's points set apart (or the colour
 * channel and the fading trail the look asks for), or, for three dimensions, the cloud in a box to orbit with buttons
 * for the viewpoints (isometric, front, side, top).
 *
 * @param props - The record, the step, the look and the figure and frame settings.
 * @returns The figure.
 */
export function PointsStepsView(props: PointsStepsViewProps): ReactElement {
  const { record, step } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const look = useMemo<PointsLook>(
    () => ({
      ...(props.colour === undefined ? {} : { colour: props.colour }),
      ...(props.fade === undefined ? {} : { fade: props.fade }),
    }),
    [props.colour, props.fade],
  )
  const build = useCallback(
    (fig: Figure) => pointsStepsView(record, step, { figure: fig, ...look }),
    [record, step, look],
  )
  const scene3 = useMemo(
    () => (record.dims === 3 ? pointsStepsScene3d(record, step, { ...look, mode: figure.mode }) : undefined),
    [record, step, look, figure.mode],
  )
  const [preset, setPreset] = useState<CameraPreset>(props.camera ?? 'iso')
  const start = useMemo(
    () => (scene3 === undefined ? undefined : fitCamera(makeCamera(cameraPreset(preset)), scene3)),
    [scene3, preset],
  )
  const channel =
    props.colour === 'values' || (props.colour === undefined && record.steps.some((s) => s.values !== undefined))
  const bar =
    channel && record.steps.some((s) => s.values !== undefined) ? <ValueBar record={record} mode={figure.mode} /> : null
  if (scene3 === undefined)
    return (
      <RecordFigure
        {...props}
        figure={figure}
        build={build}
        filename={props.filename ?? 'points'}
        caption={props.caption ?? bar ?? undefined}
      />
    )
  return (
    <FigureFrame
      width={figure.width}
      height={figure.height}
      filename={props.filename ?? 'points'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={
        <div className="flex flex-col gap-1">
          <div className="flex flex-wrap items-center gap-1" role="toolbar" aria-label="Viewpoint">
            {CAMERA_PRESETS.map((name) => (
              <ToolButton
                key={name}
                label={name}
                title={`View from the ${name} viewpoint`}
                pressed={name === preset}
                onClick={() => setPreset(name)}
              />
            ))}
            {bar}
          </div>
          <span className="text-xs text-muted">
            {props.caption ?? 'Drag to orbit; double-click to return to the viewpoint.'}
          </span>
        </div>
      }
    >
      <Scene3DView
        key={preset}
        scene={scene3}
        {...(start === undefined ? {} : { camera: start })}
        width={figure.width}
        height={figure.height}
      />
    </FigureFrame>
  )
}
