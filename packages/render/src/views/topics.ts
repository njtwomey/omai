import type { EChartsOption } from 'echarts'
import type { ComponentsRecord } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { gridSpec } from '../figure/grid.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link topicWordsView}. */
export interface TopicWordsViewOptions extends ViewOptions {
  /** Words per topic (default 8). */
  readonly top?: number
  /** Topics across (default up to 4). */
  readonly columns?: number
  /** Topic names (default `topic 1`, `topic 2`, ...). */
  readonly names?: readonly string[]
}

/**
 * The top words of every topic: each component's largest weights (the topic's word probabilities) with the words they
 * belong to, best first.
 *
 * @param record - A components record whose features are words (LDA's topics, NMF's parts).
 * @param top - Words per topic.
 * @returns Per component, its top words and weights.
 * @example
 * ```ts
 * import { componentsRecord } from 'whitebox-engine/exposition'
 * import { topWords } from 'whitebox-render/views'
 * topWords(componentsRecord({ method: 'lda', features: ['a', 'b', 'c'], components: [[0.1, 0.7, 0.2]] }), 2) // => [[{ word: 'b', weight: 0.7 }, { word: 'c', weight: 0.2 }]]
 * ```
 */
export function topWords(record: ComponentsRecord, top: number): { word: string; weight: number }[][] {
  return record.components.map((row) =>
    row
      .map((weight, j) => ({ word: record.features[j] ?? `${j}`, weight }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, Math.max(1, top)),
  )
}

/**
 * Topics by their top words as one figure: a small horizontal bar chart per topic (its most probable words, best at
 * the top, in the topic's colour), on one weight axis so topics compare.
 *
 * @param record - A components record whose features are words.
 * @param options - Mode, figure, words per topic, columns and topic names.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { componentsRecord } from 'whitebox-engine/exposition'
 * import { topicWordsView } from 'whitebox-render/views'
 * const r = componentsRecord({ method: 'lda', features: ['a', 'b', 'c'], components: [[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]] })
 * topicWordsView(r).grid.length // => 2
 * ```
 */
export function topicWordsView(record: ComponentsRecord, options: TopicWordsViewOptions = {}): EChartsOption {
  const top = options.top ?? 8
  const words = topWords(record, top)
  const k = Math.max(1, words.length)
  const columns = Math.min(k, options.columns ?? Math.min(4, k))
  const rows = Math.ceil(k / columns)
  const fig = viewFigure(options, { aspect: Math.min(2.2, Math.max(1.1, (columns * 1.25) / rows)) })
  const mode = fig.mode
  const max = Math.max(1e-12, ...words.flat().map((w) => w.weight))
  const panels: Record<string, Panel> = {}
  for (const [t, list] of words.entries()) {
    // Bars run bottom-up: reverse so the best word is at the top.
    const shown = [...list].reverse()
    panels[`t${t}`] = {
      layers: [
        bars(
          shown.map((w) => w.word),
          [{ name: '__weight', values: shown.map((w) => w.weight), colour: categoricalColour(t) }],
          { mode, horizontal: true },
        ),
      ],
      options: {
        title: options.names?.[t] ?? `topic ${t + 1}`,
        x: { limits: [0, max * 1.05], show: Math.floor(t / columns) === rows - 1 },
        y: {},
      },
    }
  }
  const grid = gridSpec(rows, columns, {
    wspace: 70,
    hspace: 34,
    cells: words.map((_, t) => ({ id: `t${t}`, row: Math.floor(t / columns), col: t % columns })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}
