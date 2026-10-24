import { type ReactElement, useCallback } from 'react'
import type { StateSequence } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { stateSequenceView } from '../../views/sequences.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SEQUENCE_FIGURE = { aspect: 2 } as const

/**
 * A hidden-state sequence: every state's posterior over time as stacked bands, the Viterbi path and the true states
 * as strips beneath, and the observations above.
 *
 * @param props - The record and the figure settings.
 * @returns The figure.
 */
export function StateSequenceView(props: RecordViewProps<StateSequence>): ReactElement {
  const figure = useViewFigure(props.figure, SEQUENCE_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => stateSequenceView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'state-sequence'} />
}
