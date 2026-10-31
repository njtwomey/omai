/**
 * The React view of a vector field: arrows or streamlines over the plane with its nullclines and classified fixed
 * points, in one figure.
 */

import type { VectorField } from 'mdml-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { type VectorFieldViewOptions, vectorFieldView } from '../../views/fields.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const FIELD_FIGURE = { aspect: 1.25 } as const

/** Props of {@link VectorFieldView}: the record, the figure and frame settings, and how the field is drawn. */
export interface VectorFieldViewProps
  extends RecordViewProps<VectorField>,
    Pick<VectorFieldViewOptions, 'style' | 'length' | 'density' | 'colour'> {}

/**
 * A vector field: arrows (equal length, coloured by magnitude) or streamlines over the plane, its nullclines dashed and
 * its fixed points marked by their stability.
 *
 * @param props - The record, how to draw the field, and the figure and frame settings.
 * @returns The figure.
 */
export function VectorFieldView(props: VectorFieldViewProps): ReactElement {
  const figure = useViewFigure(props.figure, FIELD_FIGURE)
  const { record, style, length, density, colour } = props
  const build = useCallback(
    (fig: Figure) =>
      vectorFieldView(record, {
        figure: fig,
        ...(style === undefined ? {} : { style }),
        ...(length === undefined ? {} : { length }),
        ...(density === undefined ? {} : { density }),
        ...(colour === undefined ? {} : { colour }),
      }),
    [record, style, length, density, colour],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'vector-field'} />
}
