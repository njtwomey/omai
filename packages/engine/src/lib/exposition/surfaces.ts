import { rowsOf, sameLength } from './_check.ts'

/** A model's predictions on a grid over two input features (decision regions, probability maps, regression surfaces). */
export interface DecisionSurface {
  readonly kind: 'decisionSurface'
  readonly features: readonly [string, string]
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** labels[i][j]: the predicted class index at (x[j], y[i]), for classifiers. */
  readonly labels?: readonly (readonly number[])[]
  /** probabilities[c][i][j]: the probability of class c, for probabilistic classifiers. */
  readonly probabilities?: readonly (readonly (readonly number[])[])[]
  /** values[i][j]: the prediction or score, for regressors and scorers. */
  readonly values?: readonly (readonly number[])[]
  readonly classes?: readonly string[]
}

/**
 * A decision surface, with its shapes checked.
 *
 * @param surface - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a grid does not match x and y.
 * @example
 * ```ts
 * import { decisionSurface } from 'whitebox-engine/exposition'
 * decisionSurface({ features: ['x0', 'x1'], x: [0, 1], y: [0], labels: [[0, 1]] }).labels // => [[0, 1]]
 * ```
 */
export function decisionSurface(surface: Omit<DecisionSurface, 'kind'>): DecisionSurface {
  const check = (field: string, grid: readonly (readonly number[])[]): void => {
    sameLength('decisionSurface', field, grid.length, surface.y.length)
    rowsOf('decisionSurface', field, grid, surface.x.length)
  }
  if (surface.labels !== undefined) check('labels', surface.labels)
  if (surface.values !== undefined) check('values', surface.values)
  if (surface.probabilities !== undefined)
    for (const [c, grid] of surface.probabilities.entries()) check(`probabilities[${c}]`, grid)
  return { kind: 'decisionSurface', ...surface }
}
