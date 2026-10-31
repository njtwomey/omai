/**
 * The view of a `Reachability` record: each position's reachability distance in the order the points were visited, as
 * bars coloured by cluster (noise muted), with the cut drawn as a line. Without cluster labels the cut at the record's
 * threshold finds them (`reachabilityClusters`).
 */

import type { EChartsOption } from 'echarts'
import type { Reachability } from 'mdml-engine/exposition'
import { bars } from '../charts/bars.ts'
import { referenceLines } from '../charts/functions.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { short, spanOf, type ViewOptions, viewFigure } from './_view.ts'

/**
 * The clusters a reachability plot cuts out at its threshold: a valley is a run of positions whose distance is at most
 * the threshold, together with the position before it that rose above (the point the run was reached from); positions
 * above the threshold that open no valley are noise (-1). The record's own `labels` win when it has them.
 *
 * @param record - The record.
 * @returns The cluster of each position (0, 1, ...; -1 for noise).
 * @example
 * ```ts
 * import { reachabilityRecord } from 'mdml-engine/exposition'
 * import { reachabilityClusters } from 'mdml-render/views'
 * const r = reachabilityRecord({ reach: [null, 0.1, 0.1, 0.9, 0.1, 0.1, 2], threshold: 0.5 })
 * reachabilityClusters(r) // => [0, 0, 0, 1, 1, 1, -1]
 * ```
 */
export function reachabilityClusters(record: Reachability): number[] {
  if (record.labels !== undefined) return [...record.labels]
  const threshold = record.threshold
  const n = record.reach.length
  if (threshold === undefined) return new Array<number>(n).fill(0)
  const low = (i: number): boolean => {
    const r = record.reach[i]
    return r !== null && r !== undefined && r <= threshold
  }
  const out = new Array<number>(n).fill(-1)
  let cluster = -1
  for (let i = 0; i < n; i++) {
    if (low(i)) {
      // A low position continues the valley it follows.
      if (cluster < 0 || out[i - 1] === -1) cluster++
      out[i] = cluster
    } else if (low(i + 1)) {
      // A high position followed by a low one starts a valley and belongs to it.
      cluster++
      out[i] = cluster
    }
  }
  return out
}

/**
 * A reachability plot as a figure: bars in visiting order coloured by cluster, the cut as a dashed line.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { reachabilityRecord } from 'mdml-engine/exposition'
 * import { reachabilityPlot } from 'mdml-render/views'
 * reachabilityPlot(reachabilityRecord({ reach: [null, 0.1, 0.9, 0.1], threshold: 0.5 })).series.length // => 3
 * ```
 */
export function reachabilityPlot(record: Reachability, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2.2 })
  const mode = fig.mode
  const defined = record.reach.filter((r): r is number => r !== null && Number.isFinite(r))
  const top = defined.length > 0 ? spanOf(defined)[1] * 1.08 : 1
  // An undefined distance (the first point of a component) is drawn as a full bar.
  const values = record.reach.map((r) => (r === null || !Number.isFinite(r) ? top : r))
  const clusters = reachabilityClusters(record)
  const ids = [...new Set(clusters)].sort((a, b) => a - b)
  const groups = ids.map((c) => ({
    name: c < 0 ? 'noise' : `cluster ${c + 1}`,
    colour: c < 0 ? 'muted' : `tab${c % 10}`,
    values: values.map((v, i) => (clusters[i] === c ? v : 0)),
  }))
  const layers: Layer[] = [
    bars(
      values.map((_, i) => `${record.order?.[i] ?? i}`),
      groups,
      { mode, stacked: true },
    ),
  ]
  if (record.threshold !== undefined)
    layers.push(referenceLines({ y: [{ value: record.threshold, label: `cut ${short(record.threshold)}` }] }, { mode }))
  return {
    ...cartesian(layers, {
      figure: fig,
      mode,
      x: { label: 'cluster order', show: false },
      y: { label: 'reachability distance' },
      legend: ids.length > 1,
      tooltip: 'item',
    }),
    animation: false,
  }
}
