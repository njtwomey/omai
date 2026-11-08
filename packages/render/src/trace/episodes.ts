/**
 * Figures of reinforcement-learning and bandit runs: episode returns or lengths as faint points with a rolling mean and
 * a band of one rolling standard deviation, and cumulative regret (the running sum of the best possible reward minus
 * the reward received).
 */

import type { EChartsOption } from 'echarts'
import { episodes, type Run } from 'omai-engine/trace'
import { modeOf } from '../charts/_data.ts'
import { band, lines } from '../charts/lines.ts'
import { cartesian, type Layer, type PlotOptions } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import type { ColourMode } from '../theme/tokens.ts'

/**
 * A trailing rolling mean and standard deviation over a window (shorter at the start).
 *
 * @param values - The values.
 * @param window - The window length (default 20).
 * @returns The means and standard deviations.
 * @example
 * ```ts
 * import { rolling } from 'omai-render/trace'
 * rolling([1, 3, 5], 2).mean // => [1, 2, 4]
 * ```
 */
export function rolling(values: readonly number[], window = 20): { readonly mean: number[]; readonly std: number[] } {
  const mean: number[] = []
  const std: number[] = []
  values.forEach((_, i) => {
    const w = values.slice(Math.max(0, i - window + 1), i + 1)
    const m = w.reduce((a, b) => a + b, 0) / w.length
    mean.push(m)
    std.push(Math.sqrt(w.reduce((a, b) => a + (b - m) ** 2, 0) / w.length))
  })
  return { mean, std }
}

/**
 * Episode returns (or lengths) of one or more runs: each episode as a faint point, the rolling mean as a line and one
 * rolling standard deviation as a band.
 *
 * @param runs - Named runs (or one run).
 * @param options - Which quantity (`return` or `length`), the window and colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { episodeCurves } from 'omai-render/trace'
 * const ep = (i: number) => ({ type: 'episode' as const, runId: 'r', episode: i, return: i, length: 10, terminated: true })
 * episodeCurves({ schema: 'omai.run/1', events: [ep(0), ep(1), ep(2)] }, { window: 2 }).length // => 3
 * ```
 */
export function episodeCurves(
  runs: Run | readonly { readonly name: string; readonly run: Run }[],
  options: { readonly quantity?: 'return' | 'length'; readonly window?: number; readonly mode?: ColourMode } = {},
): Layer[] {
  const mode = options.mode ?? 'light'
  const list = Array.isArray(runs) ? runs : [{ name: options.quantity ?? 'return', run: runs as Run }]
  return list.flatMap((r, i) => {
    const eps = episodes(r.run)
    const x = eps.map((e) => e.episode)
    const y = eps.map((e) => (options.quantity === 'length' ? e.length : e.return))
    const { mean, std } = rolling(y, options.window ?? 20)
    const colour = `tab${i % 10}`
    return [
      points([{ x, y, colour, size: 3, opacity: 0.35, outline: false, name: `__episodes-${r.name}` }], { mode }),
      band(
        [
          {
            x,
            lower: mean.map((m, k) => m - (std[k] ?? 0)),
            upper: mean.map((m, k) => m + (std[k] ?? 0)),
            colour,
            opacity: 0.15,
          },
        ],
        { mode },
      ),
      lines([{ x, y: mean, colour, name: r.name }], { mode }),
    ]
  })
}

/**
 * An episode chart: returns (or lengths) per episode with rolling means and bands.
 *
 * @param runs - Named runs (or one run).
 * @param options - Plot options, the quantity and the window.
 * @returns The ECharts option.
 * @index function Episode returns with a rolling mean and band: `episodeChart(run)`.
 * @example
 * ```ts
 * import { episodeChart } from 'omai-render/trace'
 * episodeChart({ schema: 'omai.run/1', events: [] }).xAxis[0].name // => 'episode'
 * ```
 */
export function episodeChart(
  runs: Run | readonly { readonly name: string; readonly run: Run }[],
  options: PlotOptions & { readonly quantity?: 'return' | 'length'; readonly window?: number } = {},
): EChartsOption {
  const { quantity, window, ...plot } = options
  const layers = episodeCurves(runs, {
    mode: modeOf(options),
    ...(quantity === undefined ? {} : { quantity }),
    ...(window === undefined ? {} : { window }),
  })
  return cartesian(layers, { x: { label: 'episode' }, y: { label: quantity ?? 'return' }, ...plot })
}

/**
 * Cumulative regret: the running sum of (best possible reward minus reward received), for bandits and online learning.
 *
 * @param rewards - Rewards received per round.
 * @param best - The best expected reward per round (a constant or one per round).
 * @returns The cumulative regret per round.
 * @example
 * ```ts
 * import { cumulativeRegret } from 'omai-render/trace'
 * cumulativeRegret([1, 0, 1], 1) // => [0, 1, 1]
 * ```
 */
export function cumulativeRegret(rewards: readonly number[], best: number | readonly number[]): number[] {
  let acc = 0
  return rewards.map((r, i) => {
    acc += (typeof best === 'number' ? best : (best[i] ?? 0)) - r
    return acc
  })
}
