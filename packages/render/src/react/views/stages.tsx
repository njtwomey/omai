/**
 * React views of staged models: an ensemble played stage by stage (scores, member weights, sample weights or residuals
 * over the rows), and a fitted pipeline as a strip of step cards with the selected step's output shown large in the
 * form that suits it, beside its data and fitted parameters.
 */

import { type CSSProperties, type ReactElement, type ReactNode, useCallback, useMemo, useState } from 'react'
import type { EnsembleStages, PipelineStages, StageData, StageParameter } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { formatNumber } from '../../theme/format.ts'
import { mixColours } from '../../theme/palettes.ts'
import type { Tokens } from '../../theme/tokens.ts'
import {
  type EnsembleViewOptions,
  ensembleFrames,
  type PipelineStep,
  type PipelineSurfaceField,
  pipelinePanelOf,
  pipelineProjectionView,
  pipelineScatterView,
  pipelineStepSummary,
  pipelineSteps,
  pipelineSurfaceView,
} from '../../views/stages.ts'
import { Choice } from '../controls/Choice.tsx'
import { cx, iconButtonClass } from '../controls/classes.ts'
import { ChevronLeftIcon, ChevronRightIcon } from '../controls/icons.tsx'
import { useTokens } from '../theme.tsx'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const ENSEMBLE_FIGURE = { aspect: 1.75 } as const

/**
 * An ensemble stage by stage: training and validation scores, member weights, and what each member was fitted to
 * (AdaBoost's sample weights as sizes, boosting residuals as colours) over the training rows, played stage by stage.
 *
 * @param props - The record, the training rows, labels and columns, the score label, the playback rate, and the
 * figure settings.
 * @returns The figure with its player.
 */
export function EnsembleView(
  props: RecordViewProps<EnsembleStages> & Omit<EnsembleViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, ENSEMBLE_FIGURE)
  const { record, points, labels, columns, scoreLabel } = props
  const frames = useCallback(
    (fig: Figure) =>
      ensembleFrames(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(labels === undefined ? {} : { labels }),
        ...(columns === undefined ? {} : { columns }),
        ...(scoreLabel === undefined ? {} : { scoreLabel }),
      }),
    [record, points, labels, columns, scoreLabel],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'ensemble'} />
}

const num = (v: number): string => formatNumber(v, { digits: 3 })

/** A cell's background: the colour mixed into the page by the value's share of the largest magnitude. */
function tint(t: Tokens, value: number, max: number): string | undefined {
  // Values that are all rounding noise (a mean of standardised columns) are not tinted.
  if (!(max > 1e-9) || !Number.isFinite(value)) return undefined
  return mixColours(t.bg, value >= 0 ? t.tab0 : t.tab3, 0.45 * Math.min(1, Math.abs(value) / max))
}

/** The pipeline as a strip of cards, one per step, with previous and next; the selected card is outlined. */
function StageStrip(props: {
  readonly steps: readonly PipelineStep[]
  readonly classes: readonly string[] | undefined
  readonly selected: number
  readonly onSelect: (i: number) => void
}): ReactElement {
  const t = useTokens()
  const { steps, selected, onSelect } = props
  const last = steps.length - 1
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 8 }}>
      <button
        type="button"
        className={cx(iconButtonClass, 'self-center rounded-md')}
        aria-label="Previous stage"
        disabled={selected <= 0}
        onClick={() => onSelect(Math.max(0, selected - 1))}
      >
        <ChevronLeftIcon />
      </button>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, flex: 1, minWidth: 0 }}>
        {steps.map((step, i) => {
          const { shape, detail } = pipelineStepSummary(step, props.classes)
          const on = i === selected
          const card: CSSProperties = {
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            minWidth: 128,
            maxWidth: 210,
            padding: '6px 10px',
            borderRadius: 6,
            textAlign: 'left',
            cursor: 'pointer',
            background: on ? t.surface : t.bg,
            border: `1.5px solid ${on ? t.accent : t.grid}`,
            color: t.fg,
            font: `12px ${t.font}`,
            overflowWrap: 'anywhere',
          }
          return (
            <div
              key={step.kind === 'input' ? '__input' : step.name}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <button type="button" style={card} aria-pressed={on} onClick={() => onSelect(i)}>
                <span style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <strong style={{ fontWeight: 600 }}>{step.name}</strong>
                  <span style={{ color: t.muted, fontSize: 11 }}>{step.kind}</span>
                </span>
                <span style={{ color: t.fgSecondary, font: `11px ${t.monoFont}` }}>{shape}</span>
                <span style={{ color: t.fgSecondary, fontSize: 11 }}>{detail}</span>
              </button>
              {i < last && (
                <span aria-hidden="true" style={{ color: t.muted }}>
                  →
                </span>
              )}
            </div>
          )
        })}
      </div>
      <button
        type="button"
        className={cx(iconButtonClass, 'self-center rounded-md')}
        aria-label="Next stage"
        disabled={selected >= last}
        onClick={() => onSelect(Math.min(last, selected + 1))}
      >
        <ChevronRightIcon />
      </button>
    </div>
  )
}

const TABLE_ROWS = 6

/**
 * A step's data as a compact table: its first rows, then each column's mean, standard deviation (with a bar on one
 * scale across the columns, and the data's before the step when the step keeps its columns, so a change of scale
 * shows), minimum and maximum.
 */
function StageTable(props: {
  readonly data: StageData
  readonly before?: StageData
  readonly classes?: readonly string[]
}): ReactElement {
  const t = useTokens()
  const { data } = props
  const before =
    props.before !== undefined &&
    props.before.columns.length === data.columns.length &&
    props.before.columns.every((c, j) => c === data.columns[j])
      ? props.before
      : undefined
  const stats = data.stats ?? []
  const stdMax = Math.max(1e-300, ...stats.map((s) => s.std), ...(before?.stats ?? []).map((s) => s.std))
  const cell: CSSProperties = {
    padding: '3px 8px',
    textAlign: 'right',
    borderBottom: `1px solid ${t.grid}`,
    whiteSpace: 'nowrap',
  }
  const head: CSSProperties = { ...cell, color: t.fgSecondary, fontWeight: 600, fontFamily: t.font }
  const label: CSSProperties = { ...cell, textAlign: 'left', color: t.fgSecondary, fontFamily: t.font }
  const stdCell = (std: number | undefined, colour: string): ReactNode => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
      <span>{std === undefined ? '' : num(std)}</span>
      <span
        style={{
          display: 'block',
          height: 4,
          minWidth: 1,
          width: `${Math.max(2, Math.round(56 * ((std ?? 0) / stdMax)))}px`,
          background: colour,
          borderRadius: 2,
        }}
      />
    </div>
  )
  const labelled = data.labels !== undefined
  return (
    <div
      style={{
        border: `1px solid ${t.grid}`,
        borderRadius: 6,
        padding: 10,
        background: t.bg,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        maxWidth: '100%',
      }}
    >
      <span style={{ color: t.fgSecondary, font: `12px ${t.font}` }}>
        first {Math.min(TABLE_ROWS, data.rows.length)} of {data.n} rows, {data.columns.length} columns; statistics over
        every row
      </span>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
        <table style={{ borderCollapse: 'collapse', font: `12px ${t.monoFont}`, color: t.fg }}>
          <thead>
            <tr>
              <th style={{ ...head, textAlign: 'left' }}>row</th>
              {labelled && <th style={head}>class</th>}
              {data.columns.map((c) => (
                <th key={c} style={head}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.slice(0, TABLE_ROWS).map((r, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: rows are positional
              <tr key={i}>
                <td style={label}>{i}</td>
                {labelled && <td style={cell}>{props.classes?.[data.labels?.[i] ?? -1] ?? data.labels?.[i]}</td>}
                {r.map((v, j) => (
                  <td key={data.columns[j] ?? j} style={cell}>
                    {num(v)}
                  </td>
                ))}
              </tr>
            ))}
            {(['mean', 'std', 'min', 'max'] as const).map((stat) => (
              <tr key={stat} style={{ background: t.surface }}>
                <td style={label}>{stat}</td>
                {labelled && <td style={cell} />}
                {stats.map((s, j) => (
                  <td key={data.columns[j] ?? j} style={cell}>
                    {stat === 'std' ? stdCell(s.std, t.accent) : num(s[stat])}
                  </td>
                ))}
              </tr>
            ))}
            {before?.stats !== undefined && (
              <tr style={{ background: t.surface }}>
                <td style={label}>std before</td>
                {labelled && <td style={cell} />}
                {before.stats.map((s, j) => (
                  <td key={data.columns[j] ?? j} style={cell}>
                    {stdCell(s.std, t.muted)}
                  </td>
                ))}
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** A stage's fitted parameters: numbers and names in a line, vectors and matrices as tables tinted by value. */
function ParameterTables(props: { readonly parameters: readonly StageParameter[] }): ReactElement | null {
  const t = useTokens()
  const scalars = props.parameters.filter((p) => typeof p.value !== 'object')
  const arrays = props.parameters.filter((p) => typeof p.value === 'object')
  if (props.parameters.length === 0) return null
  const cell: CSSProperties = {
    padding: '3px 8px',
    textAlign: 'right',
    whiteSpace: 'nowrap',
    borderBottom: `1px solid ${t.grid}`,
  }
  const head: CSSProperties = { ...cell, color: t.fgSecondary, fontWeight: 600, fontFamily: t.font }
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        color: t.fg,
        font: `12px ${t.font}`,
        maxWidth: '100%',
      }}
    >
      <span style={{ color: t.fgSecondary, fontWeight: 600 }}>fitted parameters</span>
      {scalars.length > 0 && (
        <span style={{ color: t.fgSecondary, overflowWrap: 'anywhere' }}>
          {scalars.map((p) => `${p.name} ${typeof p.value === 'number' ? num(p.value) : p.value}`).join(' · ')}
        </span>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        {arrays.map((p) => {
          const value = p.value
          if (typeof value !== 'object') return null
          const rows: readonly (readonly number[])[] = Array.isArray(value[0])
            ? (value as readonly (readonly number[])[])
            : [value as readonly number[]]
          const width = rows[0]?.length ?? 0
          const columns = p.columns ?? Array.from({ length: width }, (_, j) => String(j))
          const max = Math.max(0, ...rows.flat().map((v) => (Number.isFinite(v) ? Math.abs(v) : 0)))
          return (
            <div key={p.name} style={{ display: 'flex', flexDirection: 'column', gap: 4, maxWidth: '100%' }}>
              <span style={{ color: t.fgSecondary }}>{p.name}</span>
              <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
                <table style={{ borderCollapse: 'collapse', font: `12px ${t.monoFont}` }}>
                  <thead
                    style={
                      p.columns === undefined && rows.length === 1 && width === 1 ? { display: 'none' } : undefined
                    }
                  >
                    <tr>
                      {p.rows !== undefined && <th style={head} />}
                      {columns.map((c) => (
                        <th key={c} style={head}>
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: rows are positional
                      <tr key={i}>
                        {p.rows !== undefined && <td style={{ ...head, textAlign: 'left' }}>{p.rows[i]}</td>}
                        {r.map((v, j) => {
                          const bg = tint(t, v, max)
                          return (
                            <td key={columns[j] ?? j} style={bg === undefined ? cell : { ...cell, background: bg }}>
                              {num(v)}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const PANEL_FIGURES = {
  scatter: { width: 600, aspect: 1.35 },
  projection: { width: 940, aspect: 2.1 },
  surface: { width: 600, aspect: 1.2 },
  table: { width: 600, aspect: 1.35 },
} as const

/** The widest the view grows (the strip and the tables; a figure keeps its own width). */
const VIEW_WIDTH = 960

/** Props of {@link PipelineView}. */
export interface PipelineViewProps extends RecordViewProps<PipelineStages> {
  /** The step shown large (0 is the input when the record carries it; default the last, the final estimator). */
  readonly selected?: number
  readonly onSelect?: (step: number) => void
  /** What the final estimator's panel shades (default the probability). */
  readonly field?: PipelineSurfaceField
}

/**
 * A fitted pipeline: a strip of cards, one per step (the input, then each stage, with its shape change and what it
 * did), and the selected step's output large, in the form that suits it: two columns as a scatter coloured by class,
 * wider data as a table of its first rows with each column's statistics (the standard deviations as bars on one
 * scale, beside the input's, so standardisation shows), a projection's first two components with the variance each
 * explains, and the final classifier's probability over the original two features with the training rows, the 0.5
 * boundary, the misclassified rows ringed and the mean loss (or, switched, the loss the model expects with each row
 * sized by its own loss); then the step's fitted parameters. Click a card, or use previous and next.
 *
 * @param props - The record, the selected step, the surface's field, and the figure and frame settings.
 * @returns The strip and the panel.
 */
export function PipelineView(props: PipelineViewProps): ReactElement {
  const { record } = props
  const steps = useMemo(() => pipelineSteps(record), [record])
  const [own, setOwn] = useState<number | undefined>(undefined)
  const [ownField, setField] = useState<PipelineSurfaceField>('probability')
  const last = Math.max(0, steps.length - 1)
  const selected = Math.min(last, Math.max(0, props.selected ?? own ?? last))
  const select = (i: number): void => {
    setOwn(i)
    props.onSelect?.(i)
  }
  const field = props.field ?? ownField
  const step = steps[selected]
  const panel = step === undefined ? 'table' : pipelinePanelOf(step)
  const figure = useViewFigure(props.figure, PANEL_FIGURES[panel])
  const classes = record.classes
  const build = useCallback(
    (fig: Figure) => {
      if (step === undefined) return {}
      const options = { figure: fig, ...(classes === undefined ? {} : { classes }) }
      if (panel === 'surface') return pipelineSurfaceView(step, { ...options, field })
      if (panel === 'projection') return pipelineProjectionView(step, options)
      return pipelineScatterView(step, options)
    },
    [step, panel, classes, field],
  )
  const t = useTokens()
  const parameters = step?.stage?.parameters ?? []
  const filename = props.filename ?? `pipeline-${step?.name ?? 'stage'}`
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        width: Math.max(VIEW_WIDTH, figure.width),
        maxWidth: '100%',
      }}
    >
      <StageStrip steps={steps} classes={classes} selected={selected} onSelect={select} />
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', gap: 16 }}>
        {step === undefined ? (
          <span style={{ color: t.muted }}>no stages</span>
        ) : panel === 'table' ? (
          step.data === undefined ? (
            <span style={{ color: t.muted, font: `12px ${t.font}` }}>this stage recorded no rows</span>
          ) : (
            <StageTable
              data={step.data}
              {...(step.before === undefined ? {} : { before: step.before })}
              {...(classes === undefined ? {} : { classes })}
            />
          )
        ) : (
          <RecordFigure
            {...(props.title === undefined ? {} : { title: props.title })}
            {...(props.caption === undefined ? {} : { caption: props.caption })}
            figure={figure}
            build={build}
            filename={filename}
            footer={
              panel === 'surface' && step.stage?.loss?.expected !== undefined && props.field === undefined ? (
                <Choice
                  label="shade"
                  value={field}
                  onChange={setField}
                  options={[
                    { value: 'probability', label: 'probability' },
                    { value: 'loss', label: 'expected loss' },
                  ]}
                />
              ) : undefined
            }
          />
        )}
        <div style={{ flex: '1 1 280px', minWidth: 0, maxWidth: '100%' }}>
          <ParameterTables parameters={parameters} />
        </div>
      </div>
    </div>
  )
}
