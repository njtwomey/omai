/**
 * The React view of a curve set: several curves on shared axes (an optimiser race, rating trajectories with bands),
 * coloured by group, revealed along x as a player.
 */

import type { CurveSet } from 'mdml-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { curveSetFrames, curveSetView } from '../../views/curves.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const CURVES_FIGURE = { aspect: 1.6 } as const

/**
 * Several curves on shared axes (an optimiser race, a study of schemes, rating trajectories with deviation bands,
 * coefficients over the samples seen), coloured by group, a second quantity on a right-hand axis; played along x, so
 * the race runs again.
 *
 * @param props - The record, whether to play (default true), the playback rate, and the figure settings.
 * @returns The figure, with its player when it plays.
 */
export function CurvesView(
  props: RecordViewProps<CurveSet> & PlayerViewProps & { readonly play?: boolean },
): ReactElement {
  const figure = useViewFigure(props.figure, CURVES_FIGURE)
  const { record } = props
  const frames = useCallback((fig: Figure) => curveSetFrames(record, { figure: fig }), [record])
  const build = useCallback((fig: Figure) => curveSetView(record, { figure: fig }), [record])
  if (props.play === false)
    return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'curves'} />
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'curves'} />
}
