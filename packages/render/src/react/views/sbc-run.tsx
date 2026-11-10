/**
 * The React view of a simulation-based calibration run: the rank histograms with their band, the ECDF-difference plots
 * with the simultaneous band, or the verdict table, one at a time under a segmented choice, each in a figure frame.
 */

import { type SbcRun, sbcTable } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { sbcEcdfView, sbcRankView } from '../../views/sbc-run.ts'
import { Choice } from '../controls/Choice.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'
import { MarkedTableView } from './marked-tables.tsx'

type Which = 'ecdf' | 'ranks' | 'table'

/**
 * Simulation-based calibration: the ECDF of the ranks of the true value minus the uniform CDF inside its simultaneous
 * band (the default: one look says whether the method is calibrated), the rank histograms with their binomial band, or
 * the table of p-values and verdicts with the failing quantities marked.
 *
 * @param props - The record and the frame settings.
 * @returns The figure with its choice.
 * @index function React component: ECDF-difference and rank plots with exact bands, verdict table.
 */
export function SbcRunView(props: RecordViewProps<SbcRun>): ReactElement {
  const { record } = props
  const [which, setWhich] = useState<Which>('ecdf')
  const defaults = useMemo(
    () => ({ aspect: record.quantities.length > 3 ? 1.5 : record.quantities.length > 1 ? 1.15 : 1.8 }),
    [record],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => (which === 'ranks' ? sbcRankView(record, { figure: fig }) : sbcEcdfView(record, { figure: fig })),
    [record, which],
  )
  const table = useMemo(() => sbcTable(record), [record])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      <Choice label="chart" value={which} options={['ecdf', 'ranks', 'table'] as const} onChange={setWhich} />
      {which === 'table' ? (
        <MarkedTableView record={table} {...(props.title === undefined ? {} : { title: props.title })} />
      ) : (
        <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? `calibration-${which}`} />
      )}
    </div>
  )
}
