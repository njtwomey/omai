/**
 * React views of value grids (feature maps, filters, hidden states) as blocky small multiples on one colour range, and
 * of a classifier's confusion matrix, both played through training when the record has steps.
 */

import type { ConfusionMatrix, ValueGrids } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { cellsLayout, confusionFrames, valueGridFrames } from '../../views/cells.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * Grids of values as cells (feature maps, filters, an image, hidden states over time), never smoothed, in small
 * multiples on one colour range; played through training when the record has the steps.
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure, with its player when it has steps.
 */
export function ValueGridView(props: RecordViewProps<ValueGrids> & PlayerViewProps): ReactElement {
  const { record } = props
  // The figure's shape follows the grids' (a single tall grid of hidden states, a row of filters).
  const defaults = useMemo(() => {
    const { columns, rows } = cellsLayout(record)
    const first = record.grids[0]?.values ?? []
    const h = Math.max(1, first.length)
    const w = Math.max(1, first[0]?.length ?? 1)
    const single = record.grids.length === 1
    return { aspect: Math.min(2.4, Math.max(0.9, (columns * w) / (rows * h) + (single ? 0.3 : 0.15))) }
  }, [record])
  const figure = useViewFigure(props.figure, defaults)
  const frames = useCallback((fig: Figure) => valueGridFrames(record, { figure: fig }), [record])
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'value-grids'} />
}

/**
 * A classifier's confusion matrix: true classes down, predicted across, counts (or row shares) in the cells and the
 * accuracy above; played through training when the record has the steps.
 *
 * @param props - The record, row normalisation, the playback rate, and the figure settings.
 * @returns The figure, with its player when it has steps.
 */
export function ConfusionMatrixView(
  props: RecordViewProps<ConfusionMatrix> & PlayerViewProps & { readonly normalise?: boolean },
): ReactElement {
  const { record, normalise } = props
  const defaults = useMemo(() => ({ aspect: record.classes.length > 6 ? 1.15 : 1.25 }), [record])
  const figure = useViewFigure(props.figure, defaults)
  const frames = useCallback(
    (fig: Figure) => confusionFrames(record, { figure: fig, ...(normalise === undefined ? {} : { normalise }) }),
    [record, normalise],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'confusion-matrix'} />
}
