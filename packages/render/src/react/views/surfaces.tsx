import { type ReactElement, useCallback } from 'react'
import type { DecisionSurface, HyperparameterSurface, PosteriorGrid } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import {
  decisionSurfaceFrames,
  decisionSurfaceView,
  type HyperSurfaceViewOptions,
  hyperparameterSurfaceView,
  type PosteriorViewOptions,
  posteriorView,
  type SurfaceViewOptions,
} from '../../views/surfaces.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SURFACE_FIGURE = { aspect: 1.25 } as const
const SURFACE_STEPS_FIGURE = { aspect: 1.9 } as const

/**
 * A decision surface: a classifier's smooth regions (faded where it is unsure) with boundaries and probability
 * contours, or a regressor's or density's field, with the training rows over it (the record's own unless others are
 * given). A record with steps through training plays them, beside the training objective, opening on the fitted
 * model.
 *
 * @param props - The record, training rows and labels, contours and aspect, the playback rate, and the figure and
 * frame settings.
 * @returns The figure (with its player when the record has steps).
 */
export function DecisionSurfaceView(
  props: RecordViewProps<DecisionSurface> & Omit<SurfaceViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const { record, points, labels, columns, contours, aspect, highlight, rowWeights } = props
  const plays = record.steps !== undefined && record.steps.length > 1
  const figure = useViewFigure(props.figure, plays ? SURFACE_STEPS_FIGURE : SURFACE_FIGURE)
  const options = useCallback(
    (fig: Figure): SurfaceViewOptions => ({
      figure: fig,
      ...(points === undefined ? {} : { points }),
      ...(labels === undefined ? {} : { labels }),
      ...(columns === undefined ? {} : { columns }),
      ...(contours === undefined ? {} : { contours }),
      ...(aspect === undefined ? {} : { aspect }),
      ...(highlight === undefined ? {} : { highlight }),
      ...(rowWeights === undefined ? {} : { rowWeights }),
    }),
    [points, labels, columns, contours, aspect, highlight, rowWeights],
  )
  const build = useCallback((fig: Figure) => decisionSurfaceView(record, options(fig)), [record, options])
  const frames = useCallback((fig: Figure) => decisionSurfaceFrames(record, options(fig)), [record, options])
  const filename = props.filename ?? 'decision-surface'
  if (plays) return <FramesPlayer {...props} figure={figure} frames={frames} filename={filename} />
  return <RecordFigure {...props} figure={figure} build={build} filename={filename} />
}

const POSTERIOR_FIGURE = { aspect: 1.6 } as const
const HYPER_FIGURE = { aspect: 1.35 } as const

/**
 * A predictive distribution on a grid (a GP, Bayesian regression): mean, one and two standard-deviation bands, samples
 * and observations.
 *
 * @param props - The record, axis labels and samples, and the figure and frame settings.
 * @returns The figure.
 */
export function PosteriorView(
  props: RecordViewProps<PosteriorGrid> & Omit<PosteriorViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, POSTERIOR_FIGURE)
  const { record, inputLabel, outputLabel, samples } = props
  const build = useCallback(
    (fig: Figure) =>
      posteriorView(record, {
        figure: fig,
        ...(inputLabel === undefined ? {} : { inputLabel }),
        ...(outputLabel === undefined ? {} : { outputLabel }),
        ...(samples === undefined ? {} : { samples }),
      }),
    [record, inputLabel, outputLabel, samples],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'posterior'} />
}

/**
 * An objective over two hyperparameters with the optimiser's path across it.
 *
 * @param props - The record, the objective's name, a selected point, and the figure and frame settings.
 * @returns The figure.
 */
export function HyperparameterSurfaceView(
  props: RecordViewProps<HyperparameterSurface> & Omit<HyperSurfaceViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, HYPER_FIGURE)
  const { record, label, selected } = props
  const build = useCallback(
    (fig: Figure) =>
      hyperparameterSurfaceView(record, {
        figure: fig,
        ...(label === undefined ? {} : { label }),
        ...(selected === undefined ? {} : { selected }),
      }),
    [record, label, selected],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'hyperparameter-surface'} />
}
