/**
 * The React view of samples over training: the grid of the viewed step, with a switch for a filmstrip of earlier steps.
 */

import type { SampleGrid } from 'omai-engine/exposition'
import { type ReactElement, useMemo, useState } from 'react'
import { sampleGridScene } from '../../views/sample-grid.ts'
import { SceneFrame } from '../SceneFrame.tsx'
import { ToolButton } from '../session/parts.tsx'
import { useColourMode } from '../theme.tsx'
import type { RecordViewProps } from './_frame.tsx'

/** Props of {@link SampleGridView}: the record, the tape step and the filmstrip. */
export interface SampleGridViewProps extends RecordViewProps<SampleGrid> {
  /** The tape step: the latest recorded step at or before it (default the last recorded). */
  readonly step?: number
  /** Rows of earlier steps in the filmstrip (default 4; a button switches it off). */
  readonly filmstrip?: number
}

/**
 * Samples at a step as a grid, with the filmstrip of earlier steps beneath when switched on.
 *
 * @param props - The record, the step, the filmstrip rows and the frame settings.
 * @returns The figure.
 */
export function SampleGridView(props: SampleGridViewProps): ReactElement {
  const { record, step } = props
  const mode = useColourMode()
  const rows = props.filmstrip ?? 4
  const [strip, setStrip] = useState(rows > 0)
  const scene = useMemo(
    () => sampleGridScene(record, step, { filmstrip: strip ? rows : 0 }),
    [record, step, strip, rows],
  )
  return (
    <SceneFrame
      scene={scene}
      mode={mode}
      maxWidth={props.figure?.width ?? 440}
      maxHeight={props.figure?.height ?? 560}
      filename={props.filename ?? 'samples'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={
        <div className="flex flex-col gap-1">
          <div className="flex">
            <ToolButton
              label="filmstrip"
              title="Show earlier steps beneath the grid"
              pressed={strip}
              onClick={() => setStrip(!strip)}
            />
          </div>
          {props.caption ?? record.caption ?? null}
        </div>
      }
    />
  )
}
