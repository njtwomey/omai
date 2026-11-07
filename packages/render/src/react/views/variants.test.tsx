import {
  clusteringSteps,
  type Dependence,
  dependence,
  type Exposition,
  type FeatureAttributions,
  featureAttributions,
  groupReport,
  groupReportFromTables,
  importances,
  type PipelineStage,
  pipelineStages,
  type StageData,
} from 'mdml-engine/exposition'
import type { ReactElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { figure } from '../../figure/figure.ts'
import { tokenStripHeight } from '../../views/modalities.ts'
import {
  AttributionBarView,
  AttributionImageView,
  AttributionSeriesView,
  AttributionView,
  BeeswarmView,
  DendrogramView,
  DependenceScatterView,
  DependenceSurfaceView,
  ExpositionView,
  ForcePlotView,
  GroupSweepView,
  GroupThresholds,
  ImportanceView,
  PipelineView,
  ReachabilityView,
  TokenStripView,
  WaterfallView,
} from './index.ts'

/** The drawing area of the first framed figure in the markup: the size the view asked its frame for. */
function chartSize(html: string): [number, number] {
  const m = /style="width:(\d+)px;height:(\d+)px;max-width:100%" data-mdml-chart/.exec(html)
  return [Number(m?.[1]), Number(m?.[2])]
}
const heightFor = (aspect: number): number => Math.round(640 / aspect)

/** An attribution record of `rows` rows and `n` features with the given layout. */
function attributions(rows: number, n: number, layout?: FeatureAttributions['layout']): FeatureAttributions {
  const features = Array.from({ length: n }, (_, j) => `f${j}`)
  const grid = (f: (i: number, j: number) => number): number[][] =>
    Array.from({ length: rows }, (_, i) => features.map((_, j) => f(i, j)))
  return featureAttributions({
    method: 'exact',
    features,
    output: 'y',
    baseValue: 0,
    values: grid((i, j) => ((i + j) % 3) - 1),
    data: grid((i, j) => i + j),
    predictions: Array.from({ length: rows }, (_, i) => i),
    ...(layout === undefined ? {} : { layout }),
  })
}

describe('record views beyond the one sample of each kind', () => {
  it('give each attribution view its own frame size: a waterfall or a beeswarm by the number of rows, a map, strip, series or force plot by layout', () => {
    const many = attributions(5, 14)
    const one = attributions(1, 14)
    const small = attributions(5, 3)
    const tokens = attributions(1, 6, { kind: 'tokens' })
    const rows: [string, () => ReactElement, number][] = [
      ['waterfall of one row', () => <WaterfallView record={one} />, heightFor(1.5)],
      ['AttributionView of one row', () => <AttributionView record={one} />, heightFor(1.5)],
      // A beeswarm has a row per feature, up to 12: 70 + 30 per row, at least 220.
      ['beeswarm of 14 features', () => <BeeswarmView record={many} />, 70 + 30 * 12],
      ['AttributionView of many rows', () => <AttributionView record={many} />, 70 + 30 * 12],
      ['beeswarm of 3 features', () => <BeeswarmView record={small} />, 220],
      ['beeswarm limited to 4', () => <BeeswarmView record={many} maxFeatures={4} />, 220],
      ['bars of 14 features', () => <AttributionBarView record={many} />, 70 + 26 * 12],
      ['bars of 3 features', () => <AttributionBarView record={small} />, 200],
      ['dependence scatter', () => <DependenceScatterView record={many} feature={1} />, heightFor(1.5)],
      ['force plot', () => <ForcePlotView record={one} />, heightFor(3.2)],
      [
        'image map',
        () => <AttributionImageView record={attributions(1, 4, { kind: 'image', shape: [2, 2] })} />,
        heightFor(2.1),
      ],
      [
        'AttributionView with an image layout',
        () => <AttributionView record={attributions(1, 4, { kind: 'image', shape: [2, 2] })} />,
        heightFor(2.1),
      ],
      ['token strip', () => <TokenStripView record={tokens} />, tokenStripHeight(tokens.features, figure({}))],
      [
        'AttributionView with a token layout',
        () => <AttributionView record={tokens} />,
        tokenStripHeight(tokens.features, figure({})),
      ],
      ['series', () => <AttributionSeriesView record={attributions(1, 5, { kind: 'series' })} />, heightFor(1.6)],
      [
        'importances of 20 features',
        () => (
          <ImportanceView
            record={importances({
              method: 'permutation',
              features: Array.from({ length: 20 }, (_, j) => `f${j}`),
              metric: 'accuracy',
              mean: Array.from({ length: 20 }, () => 0.1),
              spread: Array.from({ length: 20 }, () => 0.01),
              direction: 'higher',
            })}
          />
        ),
        70 + 26 * 15,
      ],
    ]
    for (const [name, element, height] of rows) {
      const html = renderToString(element())
      expect(chartSize(html), name).toEqual([640, height])
      // Each is framed with export and expand, and takes a title and a caption through the frame.
      expect(html, name).toContain('aria-label="Expand"')
    }
    const titled = renderToString(<WaterfallView record={one} title="Why 4" caption="one prediction" />)
    expect([titled.includes('Why 4'), titled.includes('one prediction')]).toEqual([true, true])
  })

  it('draw the group, clustering, dependence and pipeline views the samples do not reach, from the records they read', () => {
    const scored = { scores: [0.1, 0.4, 0.6, 0.9, 0.3, 0.7], labels: [0, 1, 0, 1, 0, 1], groups: [0, 0, 0, 1, 1, 1] }
    const report = groupReport(
      groupReportFromTables({
        attribute: 'group',
        groups: ['majority', 'minority'],
        tables: [
          [
            [1, 1],
            [1, 0],
          ],
          [
            [1, 0],
            [0, 2],
          ],
        ],
        scored,
        sweep: {
          thresholds: [0, 1],
          selectionRate: [
            [1, 0],
            [1, 0],
          ],
          truePositiveRate: [
            [1, 0],
            [1, 0],
          ],
          falsePositiveRate: [
            [1, 0],
            [1, 0],
          ],
        },
      }),
    )
    const sweep = renderToString(<GroupSweepView record={report} title="Rates" />)
    expect([sweep.includes('Rates'), chartSize(sweep)[0]]).toEqual([true, 640])
    // One explorer per group, headed by the group's name and its row count.
    const thresholds = renderToString(<GroupThresholds record={report} />)
    expect(thresholds).toContain('majority')
    expect(thresholds).toContain('minority')
    expect(thresholds.replaceAll('<!-- -->', '').match(/3 rows/g)).toHaveLength(2)
    expect(renderToString(<GroupThresholds record={{ ...report, scored: undefined } as never} />)).toBe('')

    // A merge tree is a dendrogram, and a reachability order a reachability plot; both through the standard view.
    const tree = clusteringSteps({
      method: 'ward',
      assignments: [[0, 1, 2]],
      merges: [
        [0, 1, 1, 2],
        [2, 3, 2, 3],
      ],
    })
    const reach = clusteringSteps({
      method: 'optics',
      assignments: [[0, 0, 1]],
      reachability: [0, 0.2, 0.9],
      ordering: [0, 1, 2],
    })
    for (const [name, direct, record] of [
      ['dendrogram', () => <DendrogramView record={tree} title="Tree" />, tree],
      ['reachability', () => <ReachabilityView record={reach} title="Reach" />, reach],
    ] as const) {
      const html = renderToString(direct())
      expect(chartSize(html)[0], name).toBeGreaterThan(0)
      expect(html, name).toContain(name === 'dendrogram' ? 'Tree' : 'Reach')
      // The standard view of the record is this one: the same frame size.
      expect(chartSize(renderToString(<ExpositionView record={record as Exposition} />)), name).toEqual(chartSize(html))
    }

    // A pair of features' surface is one framed heatmap.
    const joint: Dependence = dependence({
      method: 'partial dependence',
      features: ['a', 'b'],
      output: 'y',
      curves: [],
      surfaces: [
        {
          features: [0, 1],
          gridX: [0, 1],
          gridY: [0, 1],
          values: [
            [0, 1],
            [1, 2],
          ],
        },
      ],
    })
    expect(chartSize(renderToString(<DependenceSurfaceView record={joint} />))[0]).toBe(640)

    // A pipeline whose last step is wide shows that step as a table: its columns, the statistics, and the step cards.
    const data = (columns: string[]): StageData => ({
      columns,
      rows: [columns.map(() => 0), columns.map(() => 1)],
      n: 2,
      stats: columns.map(() => ({ mean: 0.5, std: 1.5, min: -1, max: 2 })),
    })
    const wide: PipelineStage = {
      name: 'expand',
      kind: 'transformer',
      inputColumns: ['x0', 'x1'],
      outputColumns: ['u', 'v', 'w'],
      inputSample: [],
      outputSample: [],
      inputShape: [2, 2],
      outputShape: [2, 3],
      learned: [],
      output: data(['u', 'v', 'w']),
    }
    const html = renderToString(<PipelineView record={pipelineStages({ input: data(['x0', 'x1']), stages: [wide] })} />)
    for (const text of ['expand', 'input', '>u<', '>v<', '>w<', 'mean', 'std', 'Previous stage', 'Next stage'])
      expect(html, text).toContain(text)
    expect(html).toContain('1.5')
  })
})
