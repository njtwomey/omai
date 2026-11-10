/**
 * The React view of the diagnostics of Markov chains: trace plots, rank plots and the R-hat and ESS table, one at a time
 * under a segmented choice, each in a figure frame.
 */

import { type ChainDiagnostics, chainTable } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { rankView, traceView } from '../../views/chain-diagnostics.ts'
import { Choice } from '../controls/Choice.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'
import { MarkedTableView } from './marked-tables.tsx'

type Which = 'trace' | 'rank' | 'table'

/**
 * Chain diagnostics: the draws of every chain over the iterations (divergent transitions marked), the rank histogram of
 * each chain (flat when the chains mix), or the table of mean, sd, R-hat and effective sample size with the worrying
 * cells marked.
 *
 * @param props - The record and the frame settings.
 * @returns The figure with its choice.
 * @index function React component: trace plots, rank plots and the R-hat/ESS table of MCMC chains.
 */
export function ChainDiagnosticsView(props: RecordViewProps<ChainDiagnostics>): ReactElement {
  const { record } = props
  const [which, setWhich] = useState<Which>('trace')
  const defaults = useMemo(() => ({ aspect: record.quantities.length > 3 ? 1.5 : 1.8 }), [record])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => (which === 'rank' ? rankView(record, { figure: fig }) : traceView(record, { figure: fig })),
    [record, which],
  )
  const table = useMemo(() => chainTable(record), [record])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      <Choice label="chart" value={which} options={['trace', 'rank', 'table'] as const} onChange={setWhich} />
      {which === 'table' ? (
        <MarkedTableView record={table} {...(props.title === undefined ? {} : { title: props.title })} />
      ) : (
        <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? `chains-${which}`} />
      )}
    </div>
  )
}
