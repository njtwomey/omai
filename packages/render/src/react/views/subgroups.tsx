/**
 * React views of subgroup fairness and trade-offs: the subgroup report (one rate as a heatmap over the attributes beside
 * every subgroup's rate, with a choice of rate and the entropy and rich-subgroup audit beneath), the data-bias record
 * (representation, base rates, proxies and label flips, with the label disparities), and the Pareto front of accuracy
 * against a fairness violation.
 */

import {
  type DataBias,
  GROUP_RATES,
  type GroupRateName,
  type ParetoFront,
  type SubgroupReport,
} from 'omai-engine/exposition'
import { type ReactElement, useCallback, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { formatNumber } from '../../theme/format.ts'
import { dataBiasView } from '../../views/data-bias.ts'
import { GROUP_RATE_LABELS } from '../../views/fairness.ts'
import { paretoFrontView } from '../../views/frontiers.ts'
import { subgroupView } from '../../views/subgroups.ts'
import { Select } from '../controls/Select.tsx'
import { DataTable, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SUBGROUP_FIGURE = { width: 900, aspect: 2.2 } as const
const BIAS_FIGURE = { width: 900, aspect: 1.6 } as const
const num = (v: number): string => (Number.isFinite(v) ? formatNumber(v, { digits: 3 }) : '–')

/** Props of {@link SubgroupView}. */
export interface SubgroupViewProps extends RecordViewProps<SubgroupReport> {
  /** The rate shown (uncontrolled when absent: the reader picks it). */
  readonly rate?: GroupRateName
}

/**
 * A subgroup report: the chosen rate as a heatmap over the attributes beside every subgroup's rate (sorted, with sizes
 * and the overall rate), a choice of rate, and beneath it the generalised entropy decomposition and the rich-subgroup
 * audit when the report has them.
 *
 * @param props - The record, the rate and the figure and frame settings.
 * @returns The figure.
 */
export function SubgroupView(props: SubgroupViewProps): ReactElement {
  const figure = useViewFigure(props.figure, SUBGROUP_FIGURE)
  const { record } = props
  const [own, setOwn] = useState<GroupRateName>('selectionRate')
  const rate = props.rate ?? own
  const build = useCallback((fig: Figure) => subgroupView(record, { figure: fig, rate }), [record, rate])
  const e = record.entropy
  const a = record.audit
  const rows: string[][] = [
    ...(e === undefined
      ? []
      : [
          [`generalised entropy (α = ${e.alpha})`, num(e.total)],
          ['  between subgroups', num(e.between)],
          ['  within subgroups', num(e.within)],
        ]),
    ...(a === undefined
      ? []
      : [
          [
            `rich subgroup (${a.rate === 'falsePositive' ? 'false positives' : 'false negatives'})`,
            num(a.weightedDisparity),
          ],
          ['  its rate against the baseline', `${num(a.subgroupRate)} vs ${num(a.baseline)}`],
          ['  its size', num(a.size)],
          ...a.features.map((f, j) => [`  weight on ${f}`, num(a.weights[j] ?? 0)]),
        ]),
  ]
  const footer = (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
      {props.rate === undefined && (
        <div style={{ minWidth: 220 }}>
          <Select
            label="rate"
            value={rate}
            options={GROUP_RATES.map((r) => ({ value: r, label: GROUP_RATE_LABELS[r] }))}
            onChange={(v) => setOwn(v as GroupRateName)}
          />
        </div>
      )}
      {rows.length > 0 && <DataTable table={{ columns: ['measure', 'value'], rows }} />}
    </div>
  )
  return (
    <RecordFigure
      {...props}
      figure={figure}
      build={build}
      filename={props.filename ?? 'subgroup-report'}
      footer={footer}
    />
  )
}

/**
 * A data-bias record: representation, base rates (with a model's selection rates), proxies and label flips, with the
 * label disparities beneath.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function DataBiasView(props: RecordViewProps<DataBias>): ReactElement {
  const figure = useViewFigure(props.figure, BIAS_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => dataBiasView(record, { figure: fig }), [record])
  const footer = (
    <DataTable
      table={{
        columns: ['measure', 'value', 'ideal'],
        rows: [
          ...record.disparities.map((d) => [d.name, num(d.value), String(d.ideal)]),
          ...(record.amplification === undefined
            ? []
            : [['bias amplification (model)', num(record.amplification.value), '0']]),
          ...(record.proxies?.combinedAuc === undefined
            ? []
            : [['AUC of all features for the group', num(record.proxies.combinedAuc), '0.5']]),
        ],
      }}
    />
  )
  return (
    <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'data-bias'} footer={footer} />
  )
}

const FRONT_FIGURE = { aspect: 1.5 } as const

/**
 * A Pareto-front record: candidates by accuracy and violation, each family's front, reference marks and the choice.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function ParetoFrontView(props: RecordViewProps<ParetoFront>): ReactElement {
  const figure = useViewFigure(props.figure, FRONT_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => paretoFrontView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'pareto-front'} />
}
