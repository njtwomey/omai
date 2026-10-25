/**
 * The threshold explorer: a binary classifier's scores cut at one threshold the reader drags, shown in every figure of
 * the evaluation at once (each class's scores, the cost against the threshold, ROC and precision-recall with the
 * operating point) and in the contingency table with its rates and costs.
 */

import { type ReactElement, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { sweepAt, type ThresholdSweep } from 'whitebox-engine/exposition'
import { analyseThresholds, type MetricInput } from 'whitebox-engine/metrics'
import type { TensorLike } from 'whitebox-engine/tensor'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { formatNumber } from '../theme/format.ts'
import {
  defaultThreshold,
  scoreRange,
  thresholdCostPanel,
  thresholdPanelOption,
  thresholdPrPanel,
  thresholdRocPanel,
  thresholdScoresPanel,
} from '../views/thresholds.ts'
import { Chart } from './Chart.tsx'
import { ContingencyTable } from './ContingencyTable.tsx'
import { ControlPanel } from './controls/ControlPanel.tsx'
import { NumberField } from './controls/NumberField.tsx'
import { Toggle } from './controls/Toggle.tsx'
import { FigureFrame } from './FigureFrame.tsx'
import { DraggableLine } from './interaction/DraggableLine.tsx'
import { InteractionOverlay, Overlaid } from './interaction/InteractionOverlay.tsx'
import { useChartProjection } from './interaction/useChartProjection.ts'
import { useColourMode } from './theme.tsx'

/** The figures the explorer can show. */
export type ThresholdFigure = 'scores' | 'cost' | 'roc' | 'pr'

/** Props of {@link ThresholdExplorer}. */
export interface ThresholdExplorerProps {
  /** Binary labels (with `scores`), or give a `record` instead. */
  readonly yTrue?: MetricInput
  readonly scores?: TensorLike
  /** A sweep from `analyseThresholds` (instead of `yTrue` and `scores`). */
  readonly record?: ThresholdSweep
  /** `[[cost of TN, cost of FP], [cost of FN, cost of TP]]`: adds the cost figure and the table's costs. */
  readonly costs?: readonly (readonly number[])[]
  readonly posLabel?: number | string
  readonly sampleWeight?: TensorLike
  /** Names of the negative and positive class. */
  readonly classNames?: readonly [string, string]
  /** The threshold (controlled); without it the explorer keeps its own. */
  readonly threshold?: number
  /** Called whenever the threshold changes (dragged, typed, or followed to the cheapest). */
  readonly onThreshold?: (threshold: number) => void
  /** The threshold to start from when uncontrolled (default the cheapest, else the most accurate). */
  readonly initialThreshold?: number
  /** Start by following the cheapest threshold (needs costs); dragging stops following. Default false. */
  readonly followCheapest?: boolean
  /** The score axis (default `[0, 1]` for probabilities, else the scores' span). */
  readonly range?: readonly [number, number]
  /** The threshold's step when dragged or typed (default a hundredth of the range, rounded). */
  readonly step?: number
  readonly bins?: number
  /** The figures to show, in order (default every one that applies). */
  readonly figures?: readonly ThresholdFigure[]
  /** Show the contingency table (default true). */
  readonly table?: boolean
  /** Show the threshold control and the follow toggle (default true). */
  readonly controls?: boolean
  /** The size of each figure (default 460 by 300). */
  readonly figure?: Figure
  /** The prefix of the export file names (default `threshold`). */
  readonly filename?: string
}

const FIGURE = { width: 460, height: 300 } as const

/** A round step: a power of ten near a hundredth of the range. */
function niceStep(range: readonly [number, number]): number {
  const w = range[1] - range[0]
  return w > 0 ? 10 ** Math.floor(Math.log10(w / 100)) : 0.01
}

/**
 * A threshold on a binary classifier's scores that the reader drags, shown everywhere at once: each class's score
 * distribution and (with costs) the total cost against the threshold, both with the threshold as a line to drag; the
 * ROC and precision-recall curves with the operating point; and the contingency table with its rates and costs. One
 * threshold drives every figure. Controlled (`threshold` and `onThreshold`) or uncontrolled, and with costs it can follow
 * the cheapest threshold until the reader moves it.
 *
 * @param props - The labels and scores (or a sweep), costs, the threshold and its handler, and display options.
 * @returns The figures, the controls and the table.
 * @example
 * ```tsx
 * <ThresholdExplorer yTrue={y} scores={scores} costs={[[0, 1], [20, 0]]} followCheapest />
 * ```
 */
export function ThresholdExplorer(props: ThresholdExplorerProps): ReactElement {
  const mode = useColourMode()
  const { yTrue, scores, costs, posLabel, sampleWeight, classNames } = props
  const record = useMemo(() => {
    if (props.record !== undefined) return props.record
    return analyseThresholds(yTrue ?? [], scores ?? [], {
      ...(costs === undefined ? {} : { costs }),
      ...(posLabel === undefined ? {} : { posLabel }),
      ...(sampleWeight === undefined ? {} : { sampleWeight }),
      ...(classNames === undefined ? {} : { classNames }),
    })
  }, [props.record, yTrue, scores, costs, posLabel, sampleWeight, classNames])
  const range = useMemo(() => props.range ?? scoreRange(record), [props.range, record])
  const step = props.step ?? niceStep(range)
  const hasCosts = record.cost !== undefined
  const cheapest = hasCosts ? Math.min(range[1], record.thresholds[record.cheapest ?? 0] ?? range[1]) : undefined
  const [follow, setFollow] = useState(props.followCheapest === true && hasCosts)
  const [own, setOwn] = useState(() => props.initialThreshold ?? defaultThreshold(record))
  const following = follow && cheapest !== undefined
  const threshold = following ? cheapest : (props.threshold ?? own)
  const onThreshold = useRef(props.onThreshold)
  onThreshold.current = props.onThreshold
  // Tell a controlled parent where following the cheapest has moved the threshold.
  useEffect(() => {
    if (following) onThreshold.current?.(cheapest)
  }, [following, cheapest])
  const setThreshold = (v: number): void => {
    setFollow(false)
    setOwn(v)
    onThreshold.current?.(v)
  }

  const fig = useMemo(() => ({ ...(props.figure ?? makeFigure(FIGURE)), mode }), [props.figure, mode])
  const panelOptions = { threshold, range, ...(props.bins === undefined ? {} : { bins: props.bins }) }
  const scoresPanel = thresholdScoresPanel(record, mode, panelOptions)
  const costPanel = thresholdCostPanel(record, mode, panelOptions)
  const shown = (props.figures ?? ['scores', 'cost', 'roc', 'pr']).filter((f) => f !== 'cost' || hasCosts)
  const projections = { scores: useChartProjection(), cost: useChartProjection() }
  const name = props.filename ?? 'threshold'

  const draggable = (which: 'scores' | 'cost', top: number, bottom: number): ReactNode => {
    const projection = projections[which].projection
    return projection === null ? null : (
      <InteractionOverlay projection={projection} width={fig.width} height={fig.height}>
        <DraggableLine
          axis="x"
          value={Math.min(range[1], Math.max(range[0], threshold))}
          onChange={setThreshold}
          span={[bottom, top]}
          min={range[0]}
          max={range[1]}
          step={step}
          label={following ? 'threshold (cheapest)' : 'threshold'}
        />
      </InteractionOverlay>
    )
  }
  const figureOf = (f: ThresholdFigure): ReactNode => {
    if (f === 'scores' || f === 'cost') {
      const panel = f === 'scores' ? scoresPanel : costPanel
      const bottom = f === 'scores' ? 0 : Math.min(0, ...(record.cost ?? [0]))
      return (
        <FigureFrame key={f} figure={fig} filename={`${name}-${f}`}>
          <Overlaid width={fig.width} height={fig.height} overlay={draggable(f, panel.top, bottom)}>
            <Chart option={thresholdPanelOption(panel, fig)} figure={fig} onInit={projections[f].onInit} />
          </Overlaid>
        </FigureFrame>
      )
    }
    const panel = f === 'roc' ? thresholdRocPanel(record, mode, threshold) : thresholdPrPanel(record, mode, threshold)
    return (
      <FigureFrame key={f} figure={fig} filename={`${name}-${f}`}>
        <Chart option={thresholdPanelOption(panel, fig)} figure={fig} />
      </FigureFrame>
    )
  }

  const at = sweepAt(record, threshold)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {props.controls !== false && (
        <ControlPanel columns={hasCosts ? 2 : 1}>
          <NumberField
            label={following ? 'threshold (the cheapest)' : 'threshold'}
            value={threshold}
            onChange={setThreshold}
            min={range[0]}
            max={range[1]}
            step={step}
          />
          {hasCosts && <Toggle label="follow the cheapest threshold" value={follow} onChange={setFollow} />}
        </ControlPanel>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>{shown.map(figureOf)}</div>
      {props.table !== false && (
        <ContingencyTable
          table={{ labels: [0, 1], matrix: at.table }}
          classNames={record.classes}
          positive={1}
          {...(record.costs === undefined ? {} : { costs: record.costs })}
        />
      )}
      {hasCosts && (
        <div style={{ font: '12px var(--wb-font, sans-serif)', color: 'var(--wb-muted)' }}>
          cost at this threshold {formatNumber(at.cost ?? Number.NaN, { digits: 4 })}; the cheapest threshold is{' '}
          {formatNumber(record.thresholds[record.cheapest ?? 0] ?? Number.NaN, { digits: 3 })} at{' '}
          {formatNumber(record.cost?.[record.cheapest ?? 0] ?? Number.NaN, { digits: 4 })}
        </div>
      )}
    </div>
  )
}
