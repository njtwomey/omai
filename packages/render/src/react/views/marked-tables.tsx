/**
 * The React view of a marked table: its cells tinted by status and the decision outlined, drawn as a scene.
 */

import type { MarkedTable } from 'mdml-engine/exposition'
import { type ReactElement, useMemo } from 'react'
import { markedTableScene } from '../../views/marked-tables.ts'
import { SceneFrame } from '../SceneFrame.tsx'
import { useColourMode } from '../theme.tsx'
import type { RecordViewProps } from './_frame.tsx'

/**
 * A small table with marks: status tints on cells, rows and columns, and the decision outlined.
 *
 * @param props - The record and the frame settings (the figure's width bounds the table's drawn width).
 * @returns The figure.
 */
export function MarkedTableView(props: RecordViewProps<MarkedTable>): ReactElement {
  const { record } = props
  const mode = useColourMode()
  const scene = useMemo(() => markedTableScene(record), [record])
  return (
    <SceneFrame
      scene={scene}
      mode={mode}
      maxWidth={props.figure?.width ?? 520}
      maxHeight={props.figure?.height ?? 720}
      filename={props.filename ?? 'table'}
      {...(props.title === undefined ? {} : { title: props.title })}
      {...(props.caption === undefined ? {} : { caption: props.caption })}
    />
  )
}
