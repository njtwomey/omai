/**
 * The React view of a hidden-state model's reading of a sequence: the states' posteriors over time stacked to one,
 * above the decoded path and the true states.
 */

import { type ReactElement, useCallback } from 'react'
import type { StateSequence } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { stateSequenceView } from '../../views/sequences.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SEQUENCE_FIGURE = { aspect: 2.4 } as const

/**
 * A hidden-state model's reading of a sequence: the states' posteriors over time stacked to one, above the decoded
 * path and the true states as coloured strips.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function StateSequenceView(props: RecordViewProps<StateSequence>): ReactElement {
  const figure = useViewFigure(props.figure, SEQUENCE_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => stateSequenceView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'hidden-states'} />
}
