/**
 * The React view of a linear decomposition (PCA and kin): the biplot of the scores with the loadings as arrows, the
 * scree with the cumulative share of variance, and the loadings as a matrix.
 */

import { type ReactElement, useCallback } from 'react'
import type { ComponentsRecord } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type ComponentsViewOptions, componentsView } from '../../views/components.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const COMPONENTS_FIGURE = { width: 720, aspect: 1.75 } as const

/**
 * A linear decomposition: the biplot of the scores with the loadings as arrows, the scree with the cumulative share of
 * variance, and the loadings as a matrix.
 *
 * @param props - The record, labels, the biplot's pair and the number of components shown, and the figure settings.
 * @returns The figure.
 */
export function ComponentsView(
  props: RecordViewProps<ComponentsRecord> & Omit<ComponentsViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, COMPONENTS_FIGURE)
  const { record, labels, pair, maxComponents } = props
  const build = useCallback(
    (fig: Figure) =>
      componentsView(record, {
        figure: fig,
        ...(labels === undefined ? {} : { labels }),
        ...(pair === undefined ? {} : { pair }),
        ...(maxComponents === undefined ? {} : { maxComponents }),
      }),
    [record, labels, pair, maxComponents],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'components'} />
}
