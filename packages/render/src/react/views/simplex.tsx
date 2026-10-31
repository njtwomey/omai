/**
 * The React view of a simplex: the triangle of three strategies, or the square of two populations, with its field,
 * paths and fixed points.
 */

import type { Simplex } from 'mdml-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { simplexView } from '../../views/simplex.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { aspect: 1.15 } as const

/**
 * A simplex with the paths of the state over it, each ending in its current point.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function SimplexView(props: RecordViewProps<Simplex>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const build = useCallback((fig: Figure) => simplexView(record, { figure: fig }), [record])
  const caption = props.caption ?? record.caption
  return (
    <RecordFigure
      {...props}
      figure={figure}
      build={build}
      {...(caption === undefined ? {} : { caption })}
      filename={props.filename ?? 'simplex'}
    />
  )
}
