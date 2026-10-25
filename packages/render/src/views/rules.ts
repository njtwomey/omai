/**
 * Views of rule and concept records: a rule list as bars (importance, or an anchor's precision and coverage) labelled
 * by each rule's text, and TCAV scores per concept against the random baseline, with each run as a dot and the
 * significant concepts starred.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { type ConceptScores, type RuleSet, ruleText } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { referenceLines } from '../charts/functions.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import type { Figure } from '../figure/figure.ts'
import { tokens } from '../theme/tokens.ts'
import { short, spanOf, type ViewOptions, viewFigure } from './_view.ts'
import { ATTRIBUTION_DOWN, ATTRIBUTION_UP } from './attributions.ts'

/** Options of {@link ruleListView}. */
export interface RuleListOptions extends ViewOptions {
  /** Rules shown (default 12). */
  readonly maxRules?: number
}

// Room on the left for category labels of this many characters.
function padded(fig: Figure, labels: readonly string[]): Figure {
  const longest = Math.max(4, ...labels.map((l) => l.length))
  const left = Math.min(360, Math.round(16 + longest * fig.fontSize * 0.58))
  return { ...fig, padding: { ...fig.padding, left: Math.max(fig.padding.left, left) } }
}

/**
 * A rule list (RuleFit's rules, an anchor): each rule's text on the left; RuleFit's importance as a bar coloured by the
 * sign of its coefficient (red raises the output), or an anchor's precision and coverage side by side.
 *
 * @param record - The record.
 * @param options - maxRules, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { ruleSet } from 'whitebox-engine/exposition'
 * import { ruleListView } from 'whitebox-render/views'
 * const r = ruleSet({ method: 'anchors', features: ['age'], output: 'P(y)', rules: [{ clauses: [{ feature: 0, op: '>', threshold: 40 }], precision: 0.97, coverage: 0.3 }] })
 * ruleListView(r).series.length > 0 // => true
 * ```
 */
export function ruleListView(record: RuleSet, options: RuleListOptions = {}): EChartsOption {
  const shown = record.rules.slice(0, options.maxRules ?? 12)
  const cats = [...shown].reverse()
  const labels = cats.map((r) => ruleText(r, record.features, short))
  const anchors = cats.every((r) => r.precision !== undefined)
  const base = viewFigure(options, { height: Math.max(180, 80 + (anchors ? 44 : 28) * cats.length) })
  const fig = padded(base, labels)
  const t = tokens(fig.mode)
  const layers: Layer[] = []
  if (anchors) {
    layers.push(
      bars(
        labels,
        [
          { name: 'precision', values: cats.map((r) => r.precision ?? 0), colour: 'accent' },
          { name: 'coverage', values: cats.map((r) => r.coverage ?? 0), colour: t.muted },
        ],
        { mode: fig.mode, horizontal: true, labels: true },
      ),
    )
  } else {
    const value = cats.map((r) => r.importance ?? Math.abs(r.weight ?? 0))
    const series = {
      type: 'bar',
      name: '__importance',
      data: value.map((v, i) => ({
        value: [v, labels[i] ?? ''],
        itemStyle: { color: (cats[i]?.weight ?? 0) >= 0 ? ATTRIBUTION_UP : ATTRIBUTION_DOWN },
      })),
      barMaxWidth: 22,
      encode: { x: 0, y: 1 },
      label: {
        show: true,
        position: 'right',
        color: t.fgSecondary,
        formatter: (p: { dataIndex: number }) => {
          const r = cats[p.dataIndex]
          return r === undefined
            ? ''
            : `${r.weight === undefined ? '' : `${r.weight >= 0 ? '+' : ''}${short(r.weight)}`}${r.support === undefined ? '' : ` · ${Math.round(r.support * 100)}%`}`
        },
      },
    } as unknown as SeriesOption
    layers.push(layer([series], { x: spanOf([0, ...value]) }, { yCategories: labels }))
  }
  return cartesian(layers, {
    figure: { ...fig, padding: { ...fig.padding, right: Math.max(fig.padding.right, 90) } },
    mode: fig.mode,
    x: {
      label: anchors ? 'share' : `importance to ${record.output}`,
      ...(anchors ? { limits: [0, 1] as [number, number] } : {}),
    },
    legend: anchors,
    tooltip: 'item',
  })
}

/**
 * TCAV scores (Kim et al. 2018's bar plots): per concept the mean score over runs as a bar, the random directions'
 * mean beside it, every run as a dot, a line at one half (chance), and a star on the concepts whose scores differ
 * significantly from random.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { conceptScores } from 'whitebox-engine/exposition'
 * import { conceptView } from 'whitebox-render/views'
 * const r = conceptScores({ method: 'TCAV', target: 'zebra', layer: 'l', concepts: [{ name: 'striped', scores: [0.9, 0.8], randomScores: [0.5, 0.4], mean: 0.85, pValue: 0.01 }] })
 * conceptView(r).series.length > 2 // => true
 * ```
 */
export function conceptView(record: ConceptScores, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const t = tokens(fig.mode)
  const alpha = record.alpha ?? 0.05
  const labels = record.concepts.map((c) => `${c.name}${c.pValue < alpha ? ' *' : ''}`)
  const mean = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
  const dots: [string, number][] = record.concepts.flatMap((c, i) =>
    c.scores.map((s) => [labels[i] ?? '', s] as [string, number]),
  )
  return cartesian(
    [
      bars(
        labels,
        [
          { name: 'concept', values: record.concepts.map((c) => c.mean), colour: 'accent' },
          { name: 'random', values: record.concepts.map((c) => mean(c.randomScores)), colour: t.muted },
        ],
        { mode: fig.mode },
      ),
      layer(
        [
          {
            type: 'scatter',
            name: '__runs',
            data: dots,
            symbolSize: 5,
            itemStyle: { color: t.fg, opacity: 0.6 },
            z: 5,
          } as SeriesOption,
        ],
        { y: [0, 1] },
      ),
      referenceLines({ y: [0.5] }, { mode: fig.mode, colour: t.muted, dash: 'dashed' }),
    ],
    {
      figure: fig,
      mode: fig.mode,
      title: `${record.method} · ${record.target} · ${record.layer}`,
      y: { label: 'TCAV score', limits: [0, 1] },
      legend: true,
      tooltip: 'item',
    },
  )
}
