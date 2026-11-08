/**
 * React views of a group fairness report: the groups' rates, gaps to the reference group and ROC curves with operating
 * points, a table of summary disparities (with intervals and p-values) against their ideals, each group's contingency
 * table, the rates over a threshold sweep, and a threshold explorer per group.
 */

import { maxOf, minOf } from 'omai-engine/base'
import type { GroupRateName, GroupReport } from 'omai-engine/exposition'
import { analyseThresholds } from 'omai-engine/metrics'
import { type ReactElement, type ReactNode, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { formatNumber } from '../../theme/format.ts'
import { categoricalColour } from '../../theme/palettes.ts'
import { groupReportView, groupSweepView, reportAt } from '../../views/fairness.ts'
import { ContingencyTable } from '../ContingencyTable.tsx'
import { ThresholdExplorer, type ThresholdFigure } from '../ThresholdExplorer.tsx'
import { useTokens } from '../theme.tsx'
import { DataTable, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const REPORT_FIGURE = { aspect: 1.75 } as const
const SWEEP_FIGURE = { aspect: 1.6 } as const

/**
 * Each group's contingency table (actual by predicted at its threshold) side by side, headed by the group, its
 * threshold and its size.
 *
 * @param props - The report (already at the thresholds to show).
 * @returns The tables.
 */
export function GroupContingencyTables(props: { readonly record: GroupReport }): ReactNode {
  const { record } = props
  const t = useTokens()
  const classes = record.classes ?? ['negative', 'positive']
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
      {record.groups.map((name, g) => (
        <div key={name} style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
          <div style={{ font: `600 13px ${t.font}`, color: t.fg, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 5,
                background: categoricalColour(g),
                display: 'inline-block',
              }}
            />
            {name}
            <span style={{ color: t.fgSecondary, fontWeight: 400 }}>
              threshold {formatNumber(record.thresholds[g] ?? 0.5, { digits: 3 })}, n ={' '}
              {formatNumber(record.counts[g] ?? 0, { digits: 4 })}
            </span>
          </div>
          <ContingencyTable
            table={{ labels: [0, 1], matrix: record.tables[g] ?? [] }}
            classNames={classes}
            positive={1}
          />
        </div>
      ))}
    </div>
  )
}

/**
 * The summary disparities of a report as a table: each with its value and its ideal.
 *
 * @param props - The report.
 * @returns The table.
 */
export function GroupDisparityTable(props: { readonly record: GroupReport }): ReactElement {
  const iv = props.record.intervals
  const num = (v: number): string => (Number.isFinite(v) ? formatNumber(v, { digits: 3 }) : '–')
  const level = iv === undefined ? '' : `${Math.round(iv.level * 100)}% interval`
  const table = {
    columns: [
      'disparity',
      'value',
      ...(iv?.disparities === undefined ? [] : [level]),
      ...(iv?.pValues === undefined ? [] : ['p (permutation)']),
      'ideal',
    ],
    rows: props.record.disparities.map((d, j) => {
      const range = iv?.disparities?.[j]
      const p = iv?.pValues?.[j]
      return [
        d.name,
        num(d.value),
        ...(iv?.disparities === undefined
          ? []
          : [range === null || range === undefined ? '–' : `${num(range[0])} to ${num(range[1])}`]),
        ...(iv?.pValues === undefined ? [] : [p === null || p === undefined ? '–' : num(p)]),
        String(d.ideal),
      ]
    }),
  }
  return <DataTable table={table} />
}

/**
 * A report's warnings (groups too small to measure), as a short list; nothing without warnings.
 *
 * @param props - The report.
 * @returns The list.
 */
export function GroupReportWarnings(props: { readonly record: GroupReport }): ReactNode {
  const t = useTokens()
  const w = props.record.warnings ?? []
  if (w.length === 0) return null
  return (
    <ul style={{ margin: 0, paddingLeft: 18, font: `12px ${t.font}`, color: t.fgSecondary }}>
      {w.map((m) => (
        <li key={m}>{m}</li>
      ))}
    </ul>
  )
}

/** Props of {@link GroupReportView}. */
export interface GroupReportViewProps extends RecordViewProps<GroupReport> {
  /** A threshold per group to show the report at (bind it to controls or dragged lines). */
  readonly thresholds?: readonly number[]
  readonly rates?: readonly GroupRateName[]
  /** Show each group's contingency table and the disparities beneath the figure (default true). */
  readonly tables?: boolean
}

/**
 * A group report: the rates of every group side by side, the gaps to the reference group and each group's ROC curve
 * with its operating point, then (by default) the disparities and each group's contingency table.
 *
 * @param props - The record, thresholds, rates, whether to show the tables, and the figure and frame settings.
 * @returns The figure.
 * @index function React component: per-group rates, gaps and ROC curves: `<GroupReportView record={report} />`
 */
export function GroupReportView(props: GroupReportViewProps): ReactElement {
  const figure = useViewFigure(props.figure, REPORT_FIGURE)
  const { record, thresholds, rates } = props
  const at = useMemo(() => reportAt(record, thresholds), [record, thresholds])
  const build = useCallback(
    (fig: Figure) => groupReportView(at, { figure: fig, ...(rates === undefined ? {} : { rates }) }),
    [at, rates],
  )
  const footer =
    props.tables === false ? undefined : (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
        <GroupContingencyTables record={at} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <GroupDisparityTable record={at} />
          <GroupReportWarnings record={record} />
        </div>
      </div>
    )
  const title = props.title ?? (record.label === undefined ? undefined : `${record.label}: ${record.attribute}`)
  return (
    <RecordFigure
      {...props}
      {...(title === undefined ? {} : { title })}
      figure={figure}
      build={build}
      filename={props.filename ?? 'group-report'}
      {...(footer === undefined ? {} : { footer })}
    />
  )
}

/**
 * A group report's rates over a shared threshold: selection, true- and false-positive rates of every group.
 *
 * @param props - The record (with a `sweep`) and the figure and frame settings.
 * @returns The figure.
 */
export function GroupSweepView(props: RecordViewProps<GroupReport>): ReactElement {
  const figure = useViewFigure(props.figure, SWEEP_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => groupSweepView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'group-sweep'} />
}

/** Props of {@link GroupThresholds}. */
export interface GroupThresholdsProps {
  /** A report with its scored rows. */
  readonly record: GroupReport
  /** A threshold per group (default the report's own). */
  readonly thresholds?: readonly number[]
  /** Called with every group's thresholds when one is dragged. */
  readonly onThresholds?: (thresholds: readonly number[]) => void
  /** The figures of each group (default its score distributions). */
  readonly figures?: readonly ThresholdFigure[]
  /** The size of each figure. */
  readonly figure?: Figure
}

/**
 * Each group's scores by class with its own threshold to drag: a threshold explorer per group of a report's scored rows
 * (`analyseThresholds` of the group's rows), so a page can move one group's threshold and show the report there.
 *
 * @param props - The report, the thresholds and their handler, the figures and their size.
 * @returns A headed explorer per group (nothing without scored rows).
 */
export function GroupThresholds(props: GroupThresholdsProps): ReactNode {
  const { record } = props
  const t = useTokens()
  const rows = record.scored
  const sweeps = useMemo(
    () =>
      rows === undefined
        ? []
        : record.groups.map((_, g) => {
            const idx = rows.groups.flatMap((v, i) => (v === g ? [i] : []))
            return analyseThresholds(
              idx.map((i) => rows.labels[i] ?? 0),
              idx.map((i) => rows.scores[i] ?? 0),
              { posLabel: 1, ...(record.classes === undefined ? {} : { classNames: record.classes }) },
            )
          }),
    [record, rows],
  )
  const [own, setOwn] = useState<readonly number[]>(record.thresholds)
  if (rows === undefined) return null
  const probability = rows.scores.every((s) => s >= 0 && s <= 1)
  const range: [number, number] = probability ? [0, 1] : [minOf(rows.scores), maxOf(rows.scores)]
  const thresholds = props.thresholds ?? own
  const setGroup = (g: number, v: number): void => {
    const next = thresholds.map((x, i) => (i === g ? v : x))
    setOwn(next)
    props.onThresholds?.(next)
  }
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
      {sweeps.map((sweep, g) => (
        <div key={record.groups[g]} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ font: `600 13px ${t.font}`, color: t.fg, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 5,
                background: categoricalColour(g),
                display: 'inline-block',
              }}
            />
            {record.groups[g]}
            <span style={{ color: t.fgSecondary, fontWeight: 400 }}>
              {formatNumber(sweep.positives + sweep.negatives, { digits: 4 })} rows
            </span>
          </div>
          <ThresholdExplorer
            record={sweep}
            threshold={thresholds[g] ?? 0.5}
            onThreshold={(v) => setGroup(g, v)}
            figures={props.figures ?? ['scores']}
            table={false}
            controls={false}
            range={range}
            filename={`scores-${record.groups[g] ?? g}`}
            {...(props.figure === undefined ? {} : { figure: props.figure })}
          />
        </div>
      ))}
    </div>
  )
}
