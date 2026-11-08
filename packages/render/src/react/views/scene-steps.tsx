/**
 * The React view of a scene per step: the scene recorded at or before the viewed step, as one framed figure.
 */

import type { SceneSteps } from 'omai-engine/exposition'
import { type ReactElement, useMemo } from 'react'
import { sceneStepsScene } from '../../views/scene-steps.ts'
import { SceneFrame } from '../SceneFrame.tsx'
import { useColourMode } from '../theme.tsx'
import type { RecordViewProps } from './_frame.tsx'

/** Props of {@link SceneStepsView}: the record, the tape step to show and the frame settings. */
export interface SceneStepsViewProps extends RecordViewProps<SceneSteps> {
  /** The tape step to show: the latest recorded scene at or before it (default the last recorded). */
  readonly step?: number
}

/**
 * A scene at a step: agents, a plant with its horizon, a point cloud being drawn, as the record's latest scene at or
 * before the viewed step.
 *
 * @param props - The record, the step and the frame settings.
 * @returns The figure.
 */
export function SceneStepsView(props: SceneStepsViewProps): ReactElement {
  const { record, step } = props
  const mode = useColourMode()
  const scene = useMemo(() => sceneStepsScene(record, step), [record, step])
  return (
    <SceneFrame
      scene={scene}
      mode={mode}
      maxWidth={props.figure?.width ?? 520}
      maxHeight={props.figure?.height ?? 520}
      filename={props.filename ?? 'scene'}
      {...(props.title === undefined ? {} : { title: props.title })}
      {...(props.caption === undefined ? {} : { caption: props.caption })}
    />
  )
}
