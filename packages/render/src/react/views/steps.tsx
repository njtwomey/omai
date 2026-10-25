/**
 * React players and views of unsupervised methods: clustering iterations, a mixture's EM steps, an embedding's
 * optimisation, a hierarchical clustering's dendrogram cut into clusters, and the OPTICS reachability plot.
 */

import { type ReactElement, useCallback } from 'react'
import type { ClusteringSteps, EmbeddingSteps, MixtureSteps } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import {
  clusteringFrames,
  type DendrogramViewOptions,
  dendrogramView,
  type EmbeddingViewOptions,
  embeddingFrames,
  mixtureFrames,
  type ReachabilityViewOptions,
  reachabilityView,
  type StepsViewOptions,
} from '../../views/steps.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const STEPS_FIGURE = { aspect: 1.75 } as const
const DENDROGRAM_FIGURE = { aspect: 2 } as const
const REACHABILITY_FIGURE = { aspect: 2.2 } as const

/**
 * A clustering's iterations played back: rows coloured by cluster at each step, centres with their trails, roles for
 * density methods, and the objective.
 *
 * @param props - The record, the input rows and columns, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function ClusteringPlayer(
  props: RecordViewProps<ClusteringSteps> & Omit<StepsViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, STEPS_FIGURE)
  const { record, points, columns } = props
  const frames = useCallback(
    (fig: Figure) =>
      clusteringFrames(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(columns === undefined ? {} : { columns }),
      }),
    [record, points, columns],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'clustering'} />
}

/**
 * A mixture's EM steps played back: rows coloured by their most responsible component, the components' ellipses and
 * means, the objective and the weights.
 *
 * @param props - The record, the input rows and columns, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function MixturePlayer(
  props: RecordViewProps<MixtureSteps> & Omit<StepsViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, STEPS_FIGURE)
  const { record, points, columns } = props
  const frames = useCallback(
    (fig: Figure) =>
      mixtureFrames(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(columns === undefined ? {} : { columns }),
      }),
    [record, points, columns],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'mixture'} />
}

/**
 * An embedding's optimisation played back: positions at each recorded step, coloured by label, and the objective.
 *
 * @param props - The record, labels and graph, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function EmbeddingPlayer(
  props: RecordViewProps<EmbeddingSteps> & Omit<EmbeddingViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, STEPS_FIGURE)
  const { record, labels, values, graph } = props
  const frames = useCallback(
    (fig: Figure) =>
      embeddingFrames(record, {
        figure: fig,
        ...(labels === undefined ? {} : { labels }),
        ...(values === undefined ? {} : { values }),
        ...(graph === undefined ? {} : { graph }),
      }),
    [record, labels, values, graph],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'embedding'} />
}

/**
 * A hierarchical clustering's dendrogram cut into clusters, beside the rows coloured by those clusters.
 *
 * @param props - The record (with merges), the number of clusters, the input rows, and the figure settings.
 * @returns The figure.
 */
export function DendrogramView(
  props: RecordViewProps<ClusteringSteps> & Omit<DendrogramViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, DENDROGRAM_FIGURE)
  const { record, points, columns, clusters } = props
  const build = useCallback(
    (fig: Figure) =>
      dendrogramView(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(columns === undefined ? {} : { columns }),
        ...(clusters === undefined ? {} : { clusters }),
      }),
    [record, points, columns, clusters],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'dendrogram'} />
}

/**
 * The OPTICS reachability plot: reachability distances in cluster order, coloured by cluster.
 *
 * @param props - The record (with reachability and ordering), a threshold, and the figure settings.
 * @returns The figure.
 */
export function ReachabilityView(
  props: RecordViewProps<ClusteringSteps> & Omit<ReachabilityViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, REACHABILITY_FIGURE)
  const { record, threshold } = props
  const build = useCallback(
    (fig: Figure) => reachabilityView(record, { figure: fig, ...(threshold === undefined ? {} : { threshold }) }),
    [record, threshold],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'reachability'} />
}
