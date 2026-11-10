/**
 * The React view of the diagnostics of Markov chains: trace plots, rank plots, ECDF-difference plots, autocorrelation,
 * effective sample size by quantile and as the draws accumulate, the energy plot, the pair plot with the divergent
 * transitions and the R-hat and ESS table, one at a time under a choice (only the charts the record can draw), each in a
 * figure frame.
 */

import { type ChainDiagnostics, chainTable } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import {
  autocorrelationView,
  type ChainChart,
  ecdfView,
  energyView,
  essView,
  hasChart,
  pairView,
} from '../../views/chain-charts.ts'
import { rankView, traceView } from '../../views/chain-diagnostics.ts'
import { Choice } from '../controls/Choice.tsx'
import { Select } from '../controls/Select.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'
import { MarkedTableView } from './marked-tables.tsx'

type Which =
  | 'trace'
  | 'rank'
  | 'ecdf'
  | 'autocorrelation'
  | 'ess by quantile'
  | 'ess growth'
  | 'energy'
  | 'pairs'
  | 'table'

const CHART_OF: Readonly<Partial<Record<Which, ChainChart>>> = {
  ecdf: 'ecdf',
  autocorrelation: 'autocorrelation',
  'ess by quantile': 'essQuantile',
  'ess growth': 'essEvolution',
  energy: 'energy',
  pairs: 'pairs',
}

const ORDER: readonly Which[] = [
  'trace',
  'rank',
  'ecdf',
  'autocorrelation',
  'ess by quantile',
  'ess growth',
  'energy',
  'pairs',
  'table',
]

/**
 * The charts of the record's data, in the order a reader looks at them: the trace plots, the rank plots, the ECDF
 * difference with its band, the autocorrelation, the ESS by quantile and its growth, the energy, the pair plot and the
 * table.
 *
 * @param record - The diagnostics.
 * @returns The names of the charts the record can draw.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { chainChartsOf } from 'omai-render/react'
 * chainChartsOf(chainDiagnostics({ mu: [[1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]] })) // => ['trace', 'rank', 'ecdf', 'autocorrelation', 'table']
 * ```
 */
export function chainChartsOf(record: ChainDiagnostics): string[] {
  return ORDER.filter((w) => {
    const chart = CHART_OF[w]
    return chart === undefined || hasChart(record, chart)
  })
}

/**
 * Chain diagnostics: the draws of every chain over the iterations (divergent transitions marked), the rank histogram and the
 * ECDF difference of each chain (flat, inside the band, when the chains mix), the autocorrelation, the effective sample
 * size by quantile and as the draws accumulate, the energy with the E-BFMI, the pair plot with the divergences, or the table
 * of mean, sd, R-hat and effective sample size with the worrying cells marked. The choice lists the charts the record has
 * the data for.
 *
 * @param props - The record and the frame settings.
 * @returns The figure with its choice.
 * @index function React component: trace, rank, ECDF, ACF, ESS, energy, pair plots and R-hat table of MCMC chains.
 */
export function ChainDiagnosticsView(props: RecordViewProps<ChainDiagnostics>): ReactElement {
  const { record } = props
  const [chosen, setWhich] = useState<Which>('trace')
  const available = useMemo(() => chainChartsOf(record) as Which[], [record])
  // A record that changes (a run in progress) may no longer have the chart that was open: fall back to the first.
  const which = available.includes(chosen) ? chosen : 'trace'
  const defaults = useMemo(
    () => ({
      aspect:
        which === 'pairs'
          ? 1
          : which === 'energy'
            ? 1.8
            : record.quantities.length > 3
              ? 1.5
              : record.quantities.length > 1
                ? 1.15
                : 1.8,
    }),
    [record, which],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => {
      switch (which) {
        case 'rank':
          return rankView(record, { figure: fig })
        case 'ecdf':
          return ecdfView(record, { figure: fig })
        case 'autocorrelation':
          return autocorrelationView(record, { figure: fig })
        case 'ess by quantile':
          return essView(record, { figure: fig, curve: 'quantile' })
        case 'ess growth':
          return essView(record, { figure: fig, curve: 'evolution' })
        case 'energy':
          return energyView(record, { figure: fig })
        case 'pairs':
          return pairView(record, { figure: fig })
        default:
          return traceView(record, { figure: fig })
      }
    },
    [record, which],
  )
  const table = useMemo(() => chainTable(record), [record])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      {available.length <= 5 ? (
        <Choice label="chart" value={which} options={available as Which[]} onChange={setWhich} />
      ) : (
        <Select label="chart" value={which} options={available as Which[]} onChange={setWhich} />
      )}
      {which === 'table' ? (
        <MarkedTableView record={table} {...(props.title === undefined ? {} : { title: props.title })} />
      ) : (
        <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? `chains-${which}`} />
      )}
    </div>
  )
}
