import { type ReactElement, useCallback, useMemo } from 'react'
import type { FeatureAttributions, Importances } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import {
  attributionBarView,
  attributionFrames,
  beeswarmView,
  type DependenceScatterOptions,
  dependenceScatterView,
  importanceView,
} from '../../views/attributions.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const WATERFALL_FIGURE = { aspect: 1.5 } as const
const DEPENDENCE_FIGURE = { aspect: 1.5 } as const

function shownRows(record: FeatureAttributions, max: number): number {
  return Math.min(max, record.features.length)
}

/**
 * One prediction's attributions as a waterfall from the expected output to the prediction; with a sampled estimate's
 * convergence it plays the running estimate settling.
 *
 * @param props - The record, the row, how many features, the playback rate, and the figure settings.
 * @returns The figure, with its player when the record has convergence.
 */
export function WaterfallView(
  props: RecordViewProps<FeatureAttributions> &
    PlayerViewProps & { readonly row?: number; readonly maxFeatures?: number },
): ReactElement {
  const { record, row, maxFeatures } = props
  const figure = useViewFigure(props.figure, WATERFALL_FIGURE)
  const frames = useCallback(
    (fig: Figure) =>
      attributionFrames(record, {
        figure: fig,
        ...(row === undefined ? {} : { row }),
        ...(maxFeatures === undefined ? {} : { maxFeatures }),
      }),
    [record, row, maxFeatures],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'waterfall'} />
}

/**
 * Many rows' attributions as a beeswarm: a row per feature, a point per explained row at its attribution, coloured by
 * the feature's value.
 *
 * @param props - The record, how many features, and the figure settings.
 * @returns The figure.
 */
export function BeeswarmView(
  props: RecordViewProps<FeatureAttributions> & { readonly maxFeatures?: number },
): ReactElement {
  const { record, maxFeatures } = props
  const defaults = useMemo(
    () => ({ height: Math.max(220, 70 + 30 * shownRows(record, maxFeatures ?? 12)) }),
    [record, maxFeatures],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => beeswarmView(record, { figure: fig, ...(maxFeatures === undefined ? {} : { maxFeatures }) }),
    [record, maxFeatures],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'beeswarm'} />
}

/**
 * The mean |attribution| of each feature as bars.
 *
 * @param props - The record, how many features, and the figure settings.
 * @returns The figure.
 */
export function AttributionBarView(
  props: RecordViewProps<FeatureAttributions> & { readonly maxFeatures?: number },
): ReactElement {
  const { record, maxFeatures } = props
  const defaults = useMemo(
    () => ({ height: Math.max(200, 70 + 26 * shownRows(record, maxFeatures ?? 12)) }),
    [record, maxFeatures],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => attributionBarView(record, { figure: fig, ...(maxFeatures === undefined ? {} : { maxFeatures }) }),
    [record, maxFeatures],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'mean-attributions'} />
}

/**
 * A feature's dependence scatter: its values against its attributions, coloured by the feature it most interacts
 * with.
 *
 * @param props - The record, the feature and the colouring feature, and the figure settings.
 * @returns The figure.
 */
export function DependenceScatterView(
  props: RecordViewProps<FeatureAttributions> & Omit<DependenceScatterOptions, 'mode' | 'figure'>,
): ReactElement {
  const { record, feature, colourBy } = props
  const figure = useViewFigure(props.figure, DEPENDENCE_FIGURE)
  const build = useCallback(
    (fig: Figure) =>
      dependenceScatterView(record, {
        figure: fig,
        ...(feature === undefined ? {} : { feature }),
        ...(colourBy === undefined ? {} : { colourBy }),
      }),
    [record, feature, colourBy],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'dependence'} />
}

/**
 * The standard view of an attribution record: the waterfall of a single row, the beeswarm of many.
 *
 * @param props - The record and the figure settings.
 * @returns The figure.
 */
export function AttributionView(props: RecordViewProps<FeatureAttributions> & PlayerViewProps): ReactElement {
  return props.record.values.length > 1 ? <BeeswarmView {...props} /> : <WaterfallView {...props} />
}

/**
 * Feature importances as bars with their spread, every repeat a dot.
 *
 * @param props - The record, how many features, and the figure settings.
 * @returns The figure.
 */
export function ImportanceView(props: RecordViewProps<Importances> & { readonly maxFeatures?: number }): ReactElement {
  const { record, maxFeatures } = props
  const defaults = useMemo(
    () => ({ height: Math.max(200, 70 + 26 * Math.min(maxFeatures ?? 15, record.features.length)) }),
    [record, maxFeatures],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => importanceView(record, { figure: fig, ...(maxFeatures === undefined ? {} : { maxFeatures }) }),
    [record, maxFeatures],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'importances'} />
}
