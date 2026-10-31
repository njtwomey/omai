/**
 * The React view of a strip: tokens on x with a row of coloured cells per track beneath them, rows of tags, the chosen
 * path and the marked positions.
 */

import type { Strip } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { stripAspect, stripView } from '../../views/strips.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * A strip: the tokens along x, a row of cells per track (probabilities, attention, gates) coloured by value, rows of
 * tags, the chosen path as a line through its cells and the marked positions outlined.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function StripView(props: RecordViewProps<Strip>): ReactElement {
  const { record } = props
  const defaults = useMemo(() => ({ aspect: stripAspect(record) }), [record])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback((fig: Figure) => stripView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'strip'} />
}
