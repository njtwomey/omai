/**
 * React views of example-based and contrastive records: the counterfactual table and the prototypes and criticisms.
 */

import { type ReactElement, useCallback, useMemo } from 'react'
import type { CounterfactualSet, PrototypeSet } from 'whitebox-engine/exposition'
import { type Figure, figure as makeFigure } from '../../figure/figure.ts'
import { counterfactualTableHeight, counterfactualTableView, prototypeView } from '../../views/examples.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const PROTOTYPE_FIGURE = { aspect: 1.35 } as const

/**
 * Counterfactuals as a table: the original beside each counterfactual, changed cells coloured by direction, with each
 * column's prediction, distance and validity.
 *
 * @param props - The record, whether to show every feature, and the figure settings.
 * @returns The figure.
 */
export function CounterfactualView(
  props: RecordViewProps<CounterfactualSet> & { readonly all?: boolean },
): ReactElement {
  const { record, all } = props
  const defaults = useMemo(
    () => ({ height: counterfactualTableHeight(record, makeFigure({}), all ?? false) }),
    [record, all],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => counterfactualTableView(record, { figure: fig, ...(all === undefined ? {} : { all }) }),
    [record, all],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'counterfactuals'} />
}

/**
 * Prototypes and criticisms: marked on a scatter of the data, or drawn as images.
 *
 * @param props - The record and the figure settings.
 * @returns The figure.
 */
export function PrototypeView(props: RecordViewProps<PrototypeSet>): ReactElement {
  const { record } = props
  const defaults = useMemo(() => {
    if (record.imageShape === undefined) return PROTOTYPE_FIGURE
    const cols = Math.max(record.prototypes.length, record.criticisms.length, 1)
    return { aspect: Math.min(3, Math.max(1, cols / 2)) }
  }, [record])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback((fig: Figure) => prototypeView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'prototypes'} />
}
