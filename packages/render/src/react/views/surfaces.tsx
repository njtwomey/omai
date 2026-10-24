import { type ReactElement, useCallback } from 'react'
import type { DecisionSurface, HyperparameterSurface, PosteriorGrid } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import {
  decisionSurfaceView,
  type HyperSurfaceViewOptions,
  hyperparameterSurfaceView,
  type PosteriorViewOptions,
  posteriorView,
  type SurfaceViewOptions,
} from '../../views/surfaces.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SURFACE_FIGURE = { aspect: 1.25 } as const

/**
 * A decision surface: a classifier's smooth regions (faded where it is unsure) with boundaries and probability
 * contours, or a regressor's or density's field, with the training rows over it.
 *
 * @param props - The record, the training rows and labels, contours and aspect, and the figure and frame settings.
 * @returns The figure.
 */
export function DecisionSurfaceView(
  props: RecordViewProps<DecisionSurface> & Omit<SurfaceViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, SURFACE_FIGURE)
  const { record, points, labels, columns, contours, aspect } = props
  const build = useCallback(
    (fig: Figure) =>
      decisionSurfaceView(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(labels === undefined ? {} : { labels }),
        ...(columns === undefined ? {} : { columns }),
        ...(contours === undefined ? {} : { contours }),
        ...(aspect === undefined ? {} : { aspect }),
      }),
    [record, points, labels, columns, contours, aspect],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'decision-surface'} />
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
