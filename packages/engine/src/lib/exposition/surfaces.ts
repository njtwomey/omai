import { rowsOf, sameLength } from './_check.ts'
import type { InputRows } from './rows.ts'

/** A grid of values over the surface's two features: grid[i][j] at (x[j], y[i]). */
export type SurfaceGrid = readonly (readonly number[])[]

/**
 * The surface at one recorded step of training (an SMO step, an IRLS iteration, an epoch, a boosting stage): the
 * same fields as the final surface, with what the step did to the training rows.
 */
export interface SurfaceStep {
  /** The iteration (solver step, epoch, stage) the snapshot was taken after; 0 is the initial model. */
  readonly iteration: number
  readonly labels?: SurfaceGrid
  readonly probabilities?: readonly SurfaceGrid[]
  readonly decision?: SurfaceGrid
  readonly values?: SurfaceGrid
  /** The training objective at this step (a loss, the dual objective, a log likelihood). */
  readonly objective?: number
  /** The decision value (or prediction) at each of the record's `rows` at this step. */
  readonly rowValues?: readonly number[]
  /** Rows that matter at this step (indices into `rows`): the support vectors, the rows just seen, the misclassified. */
  readonly highlight?: readonly number[]
  /** A weight per row at this step: AdaBoost's sample weights, the residuals the next member fits. */
  readonly rowWeights?: readonly number[]
}

/** A model's predictions on a grid over two input features (decision regions, probability maps, regression surfaces). */
export interface DecisionSurface {
  readonly kind: 'decisionSurface'
  readonly features: readonly [string, string]
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** labels[i][j]: the predicted class index at (x[j], y[i]), for classifiers. */
  readonly labels?: SurfaceGrid
  /** probabilities[c][i][j]: the probability of class c, for probabilistic classifiers. */
  readonly probabilities?: readonly SurfaceGrid[]
  /** decision[i][j]: a binary classifier's decision function f (an SVM's margin, a linear model's log-odds); f = 0 is the boundary. */
  readonly decision?: SurfaceGrid
  /** Levels of the decision function worth marking besides the boundary (an SVM's margins, -1 and 1). */
  readonly levels?: readonly number[]
  /** values[i][j]: the prediction or score, for regressors and scorers. */
  readonly values?: SurfaceGrid
  readonly classes?: readonly string[]
  /** The training rows (the two features of the grid), when the data is small. */
  readonly rows?: InputRows
  /** The surface at recorded steps of training, in order, the last one the fitted model. */
  readonly steps?: readonly SurfaceStep[]
  /** What a step is (default 'step'): 'SMO step', 'IRLS iteration', 'epoch', 'stage'. */
  readonly stepName?: string
  /** What the steps' highlighted rows are ('support vectors', 'misclassified', 'updated row'). */
  readonly highlightName?: string
  /** The training objective at every iteration it was recorded (finer than `steps`), with its name. */
  readonly objective?: {
    readonly name: string
    readonly iterations: readonly number[]
    readonly values: readonly number[]
    /** Whether lower is better (a loss) rather than higher (a likelihood, a dual objective); default true. */
    readonly lowerIsBetter?: boolean
  }
}

/**
 * A decision surface, with its shapes checked.
 *
 * @param surface - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a grid does not match x and y, or a step's rows do not match the record's rows.
 * @example
 * ```ts
 * import { decisionSurface } from 'whitebox-engine/exposition'
 * decisionSurface({ features: ['x0', 'x1'], x: [0, 1], y: [0], labels: [[0, 1]] }).labels // => [[0, 1]]
 * ```
 */
export function decisionSurface(surface: Omit<DecisionSurface, 'kind'>): DecisionSurface {
  const check = (field: string, grid: SurfaceGrid): void => {
    sameLength('decisionSurface', field, grid.length, surface.y.length)
    rowsOf('decisionSurface', field, grid, surface.x.length)
  }
  const fields = (prefix: string, s: Pick<SurfaceStep, 'labels' | 'probabilities' | 'decision' | 'values'>): void => {
    if (s.labels !== undefined) check(`${prefix}labels`, s.labels)
    if (s.values !== undefined) check(`${prefix}values`, s.values)
    if (s.decision !== undefined) check(`${prefix}decision`, s.decision)
    if (s.probabilities !== undefined)
      for (const [c, grid] of s.probabilities.entries()) check(`${prefix}probabilities[${c}]`, grid)
  }
  fields('', surface)
  const n = surface.rows?.points.length
  for (const [t, step] of (surface.steps ?? []).entries()) {
    fields(`steps[${t}].`, step)
    for (const field of ['rowValues', 'rowWeights'] as const) {
      const column = step[field]
      if (column !== undefined && n !== undefined)
        sameLength('decisionSurface', `steps[${t}].${field}`, column.length, n)
    }
  }
  if (surface.objective !== undefined)
    sameLength(
      'decisionSurface',
      'objective.values',
      surface.objective.values.length,
      surface.objective.iterations.length,
    )
  return { kind: 'decisionSurface', ...surface }
}

/**
 * The surface as it was at one recorded step: the record with that step's fields in place of the final ones (and no
 * steps), for drawing a frame. Steps out of range clamp to the first or last; a record without steps is returned as is.
 *
 * @param surface - The record.
 * @param step - The step's index into `surface.steps`.
 * @returns The surface at that step.
 * @example
 * ```ts
 * import { decisionSurface, surfaceAt } from 'whitebox-engine/exposition'
 * const s = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0], labels: [[1, 1]], steps: [{ iteration: 0, labels: [[0, 0]] }, { iteration: 5, labels: [[1, 1]] }] })
 * surfaceAt(s, 0).labels // => [[0, 0]]
 * ```
 */
export function surfaceAt(surface: DecisionSurface, step: number): DecisionSurface {
  const steps = surface.steps
  if (steps === undefined || steps.length === 0) return surface
  const s = steps[Math.min(Math.max(step, 0), steps.length - 1)] as SurfaceStep
  const { steps: _steps, labels: _l, probabilities: _p, decision: _d, values: _v, ...rest } = surface
  return {
    ...rest,
    ...(s.labels === undefined ? {} : { labels: s.labels }),
    ...(s.probabilities === undefined ? {} : { probabilities: s.probabilities }),
    ...(s.decision === undefined ? {} : { decision: s.decision }),
    ...(s.values === undefined ? {} : { values: s.values }),
  }
}
