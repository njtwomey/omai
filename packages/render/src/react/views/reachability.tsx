/**
 * The React view of a reachability plot: bars in cluster order coloured by cluster, with the cut as a line.
 */

import type { Reachability } from 'omai-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { reachabilityPlot } from '../../views/reachability.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { aspect: 2.2 } as const

/**
 * A reachability plot: valleys are clusters, peaks separate them, the dashed line is the cut.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function ReachabilityPlotView(props: RecordViewProps<Reachability>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const build = useCallback((fig: Figure) => reachabilityPlot(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'reachability'} />
}
