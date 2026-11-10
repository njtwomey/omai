/**
 * The React view of a variational run: the ELBO, the gradient, the guide's parameters, the beliefs over the steps or the
 * final posterior over the truth, one at a time under a choice (only the charts the record has the data for), each in a
 * figure frame. A run in progress redraws as its record grows.
 */

import type { ElboRun } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { type ElboChart, elboRunView, hasElboChart } from '../../views/elbo-run.ts'
import { Choice } from '../controls/Choice.tsx'
import { Select } from '../controls/Select.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const ORDER: readonly ElboChart[] = ['elbo', 'gradient', 'parameters', 'beliefs', 'posterior']

/**
 * A variational run: the evidence lower bound over the steps (with Pathfinder's paths and k-hat), the gradient's norm and
 * variance, the guide's parameters, the posterior's mean and sd as it sharpens, or the final posterior over the truth.
 * The choice offers the charts the record has the data for, and a record that changes (a run in progress) keeps the chart
 * open while it can still be drawn.
 *
 * @param props - The record and the frame settings.
 * @returns The figure with its choice.
 * @index function React component: ELBO, gradient, parameters, beliefs and final posterior of a variational run.
 */
export function ElboRunView(props: RecordViewProps<ElboRun>): ReactElement {
  const { record } = props
  const available = useMemo(() => ORDER.filter((c) => hasElboChart(record, c)), [record])
  const [chosen, setChart] = useState<ElboChart>('elbo')
  const chart = available.includes(chosen) ? chosen : (available[0] ?? 'elbo')
  const defaults = useMemo(() => ({ aspect: chart === 'elbo' ? 1.8 : 1.5 }), [chart])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback((fig: Figure) => elboRunView(record, { figure: fig, chart }), [record, chart])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      {available.length > 1 &&
        (available.length <= 5 ? (
          <Choice label="chart" value={chart} options={available} onChange={setChart} />
        ) : (
          <Select label="chart" value={chart} options={available} onChange={setChart} />
        ))}
      <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? `elbo-${chart}`} />
    </div>
  )
}
