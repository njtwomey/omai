import { type ReactElement, useCallback, useMemo } from 'react'
import type { CoefficientTable, LikelihoodTable, NeighbourRecord, SearchTrials } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import {
  coefficientRows,
  coefficientView,
  isIntercept,
  likelihoodView,
  type NeighbourViewOptions,
  neighbourView,
  type SearchViewOptions,
  searchTrialsView,
} from '../../views/tables.ts'
import { DataTable, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const LIKELIHOOD_FIGURE = { aspect: 1.9 } as const
const NEIGHBOUR_FIGURE = { aspect: 1.75 } as const
const SEARCH_FIGURE = { aspect: 2 } as const

/**
 * A coefficient table: the estimates with their intervals as a forest plot, and the table (estimate, standard error,
 * interval, p-value) beneath it.
 *
 * @param props - The record, whether to show the table (default true), and the figure and frame settings.
 * @returns The figure and the table.
 */
export function CoefficientTableView(
  props: RecordViewProps<CoefficientTable> & { readonly table?: boolean },
): ReactElement {
  const defaults = useMemo(
    () => ({ width: 640, height: Math.max(160, 64 + 26 * props.record.names.filter((n) => !isIntercept(n)).length) }),
    [props.record],
  )
  const figure = useViewFigure(props.figure, defaults)
  const { record } = props
  const build = useCallback((fig: Figure) => coefficientView(record, { figure: fig }), [record])
  const rows = useMemo(() => coefficientRows(record), [record])
  return (
    <RecordFigure
      {...props}
      figure={figure}
      build={build}
      filename={props.filename ?? 'coefficients'}
      footer={props.table === false ? undefined : <DataTable table={rows} />}
    />
  )
}

/**
 * A likelihood table (naive Bayes, discriminant analysis): each feature's contribution to each class's score, and the
 * classes' totals.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function LikelihoodView(props: RecordViewProps<LikelihoodTable>): ReactElement {
  const figure = useViewFigure(props.figure, LIKELIHOOD_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => likelihoodView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'likelihoods'} />
}

/**
 * The neighbours of a query: spokes to each neighbour over the rows, and their distances.
 *
 * @param props - The record, rows, queries, the query shown and labels, and the figure and frame settings.
 * @returns The figure.
 */
export function NeighbourView(
  props: RecordViewProps<NeighbourRecord> & Omit<NeighbourViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, NEIGHBOUR_FIGURE)
  const { record, points, columns, queries, query, labels } = props
  const build = useCallback(
    (fig: Figure) =>
      neighbourView(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(columns === undefined ? {} : { columns }),
        ...(queries === undefined ? {} : { queries }),
        ...(query === undefined ? {} : { query }),
        ...(labels === undefined ? {} : { labels }),
      }),
    [record, points, columns, queries, query, labels],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'neighbours'} />
}

/**
 * The trials of a search: values in order with the best so far, and the value against the parameters.
 *
 * @param props - The record, the parameters drawn and the objective's name, and the figure and frame settings.
 * @returns The figure.
 */
export function SearchTrialsView(
  props: RecordViewProps<SearchTrials> & Omit<SearchViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, SEARCH_FIGURE)
  const { record, parameters, label } = props
  const build = useCallback(
    (fig: Figure) =>
      searchTrialsView(record, {
        figure: fig,
        ...(parameters === undefined ? {} : { parameters }),
        ...(label === undefined ? {} : { label }),
      }),
    [record, parameters, label],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'search'} />
}
