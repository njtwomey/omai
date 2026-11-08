/**
 * The React view of a structure whose node and edge looks follow a run: the graph at a step, drawn as a scene.
 */

import type { StructureSteps } from 'omai-engine/exposition'
import { type ReactElement, useMemo } from 'react'
import { structureStepsScene } from '../../views/structure-steps.ts'
import { SceneFrame } from '../SceneFrame.tsx'
import { useColourMode } from '../theme.tsx'
import type { RecordViewProps } from './_frame.tsx'

/** Props of {@link StructureStepsView}: the record, the tape step to show and the frame settings. */
export interface StructureStepsViewProps extends RecordViewProps<StructureSteps> {
  /** The tape step to show: the latest recorded one at or before it (default the last recorded). */
  readonly step?: number
  /** The layout of nodes the steps give no position (default `force`). */
  readonly layout?: 'force' | 'circular'
}

/**
 * A graph at a step: nodes coloured by value or class, sized and labelled, edges thick, dashed or labelled, as the
 * record's latest step at or before the viewed one says.
 *
 * @param props - The record, the step and the frame settings.
 * @returns The figure.
 */
export function StructureStepsView(props: StructureStepsViewProps): ReactElement {
  const { record, step, layout } = props
  const mode = useColourMode()
  const width = props.figure?.width ?? 440
  const height = props.figure?.height ?? 320
  const scene = useMemo(
    () => structureStepsScene(record, step, { mode, width, height, ...(layout === undefined ? {} : { layout }) }),
    [record, step, mode, width, height, layout],
  )
  return (
    <SceneFrame
      scene={scene}
      mode={mode}
      maxWidth={width}
      maxHeight={height}
      filename={props.filename ?? 'graph'}
      {...(props.title === undefined ? {} : { title: props.title })}
      {...(props.caption === undefined ? {} : { caption: props.caption })}
    />
  )
}
