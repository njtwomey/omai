/**
 * The React view of a comparison of posterior approximations: the methods' densities, ECDFs or ECDF differences over the
 * oracle's, or the agreement table with the cells beyond tolerance flagged, one at a time under a choice, each in a
 * figure frame.
 */

import { comparisonTable, type PosteriorComparison } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { posteriorComparisonView } from '../../views/posterior-comparison.ts'
import { Choice } from '../controls/Choice.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'
import { MarkedTableView } from './marked-tables.tsx'

type Which = 'density' | 'ecdf' | 'difference' | 'table'

/**
 * A comparison of posterior approximations with an oracle: the densities over one another (the default), the ECDFs, the
 * difference of each ECDF from the oracle's CDF (whose largest gap is the Kolmogorov distance), or the agreement table
 * with the mean error, the sd ratio and the distances, the cells beyond tolerance flagged. The choice offers only what
 * the record has: no difference without an oracle, no table without the kernel's rows.
 *
 * @param props - The record and the frame settings.
 * @returns The figure with its choice.
 * @index function React component: methods' posteriors over the oracle's, and the agreement table.
 */
export function PosteriorComparisonView(props: RecordViewProps<PosteriorComparison>): ReactElement {
  const { record } = props
  const options = useMemo(
    (): Which[] => [
      'density',
      'ecdf',
      ...(record.quantities.some((q) => q.truth !== undefined) ? (['difference'] as const) : []),
      ...(record.agreement.length > 0 ? (['table'] as const) : []),
    ],
    [record],
  )
  const [chosen, setWhich] = useState<Which>('density')
  const which = options.includes(chosen) ? chosen : 'density'
  const defaults = useMemo(
    () => ({ aspect: record.quantities.length > 3 ? 1.5 : record.quantities.length > 1 ? 1.15 : 1.8 }),
    [record],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => posteriorComparisonView(record, { figure: fig, chart: which === 'table' ? 'density' : which }),
    [record, which],
  )
  const table = useMemo(() => comparisonTable(record), [record])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      <Choice label="chart" value={which} options={options} onChange={setWhich} />
      {which === 'table' ? (
        <MarkedTableView record={table} {...(props.title === undefined ? {} : { title: props.title })} />
      ) : (
        <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? `comparison-${which}`} />
      )}
    </div>
  )
}
