/**
 * React views of paths over a parameter: regularisation paths with a selected penalty marked, cross-validation curves
 * with standard-error bands, learning curves, and cost-complexity pruning paths.
 */

import type { CrossValidationCurve, LearningCurve, PruningPath, RegularisationPath } from 'mdml-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import {
  type CrossValidationViewOptions,
  crossValidationView,
  learningCurveView,
  pruningPathView,
  regularisationPathView,
} from '../../views/paths.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const PATH_FIGURE = { aspect: 1.45, padding: { right: 72 } } as const
const CURVE_FIGURE = { aspect: 1.6 } as const

/**
 * A regularisation path (lasso, ridge, elastic net): coefficients against the penalty, the active-set size beneath,
 * the chosen penalty dashed, and a selected penalty (a control's value) marked with the coefficients there.
 *
 * @param props - The record, the selected penalty, and the figure and frame settings.
 * @returns The figure.
 * @index function React component: coefficients along a regularisation path: `<PathView record={path} />`
 */
export function PathView(props: RecordViewProps<RegularisationPath> & { readonly selected?: number }): ReactElement {
  const figure = useViewFigure(props.figure, PATH_FIGURE)
  const { record, selected } = props
  const build = useCallback(
    (fig: Figure) => regularisationPathView(record, { figure: fig, ...(selected === undefined ? {} : { selected }) }),
    [record, selected],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'regularisation-path'} />
}

/**
 * A cross-validation curve: mean scores with a standard-error band, the folds, and the best and one-standard-error
 * choices.
 *
 * @param props - The record, a selected value, axis direction and label, and the figure and frame settings.
 * @returns The figure.
 */
export function CrossValidationView(
  props: RecordViewProps<CrossValidationCurve> & Omit<CrossValidationViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, CURVE_FIGURE)
  const { record, selected, inverse, scoreLabel } = props
  const build = useCallback(
    (fig: Figure) =>
      crossValidationView(record, {
        figure: fig,
        ...(selected === undefined ? {} : { selected }),
        ...(inverse === undefined ? {} : { inverse }),
        ...(scoreLabel === undefined ? {} : { scoreLabel }),
      }),
    [record, selected, inverse, scoreLabel],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'cross-validation'} />
}

/**
 * A learning curve: training and validation scores against the training size, with their spreads.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function LearningCurveView(props: RecordViewProps<LearningCurve>): ReactElement {
  const figure = useViewFigure(props.figure, CURVE_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => learningCurveView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'learning-curve'} />
}

/**
 * A cost-complexity pruning path: leaf impurity and the number of leaves against the effective alpha.
 *
 * @param props - The record, a selected alpha, and the figure and frame settings.
 * @returns The figure.
 */
export function PruningPathView(props: RecordViewProps<PruningPath> & { readonly selected?: number }): ReactElement {
  const figure = useViewFigure(props.figure, CURVE_FIGURE)
  const { record, selected } = props
  const build = useCallback(
    (fig: Figure) => pruningPathView(record, { figure: fig, ...(selected === undefined ? {} : { selected }) }),
    [record, selected],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'pruning-path'} />
}
