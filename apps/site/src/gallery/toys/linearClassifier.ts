/**
 * A linear classifier on two features with every slot open: the loss and the penalty are the model (the objective),
 * the solver is the mode. The tables of `lib/losses` and `lib/optim` fill the slots with the smooth options (a loss
 * that is not smooth, or an l1 penalty, would stall a gradient method, and the tables' properties say so), and the
 * solver is the union of the minimisers of `smoothSolvers` in `lib/optim`, each with the knobs of its own. Run one
 * variant per solver on one data set and the same objective is minimised every way: the objective curves overlay, the
 * decision surfaces sit side by side, and what differs is how each got there.
 */

import { type DecisionSurface, decisionSurface } from 'mdml-engine/exposition'
import { classificationLosses, penalties } from 'mdml-engine/losses'
import {
  type GradientPath,
  gradientContract,
  gradientDefaults,
  loggedSolver,
  type SolverPath,
  smoothSolvers,
  solverContract,
  solverDefaults,
} from 'mdml-engine/optim'
import { type DictSpace, dict, union } from 'mdml-engine/space'
import { add, asTensor, matmul, type TensorLike, tensor, toFlat, type Value } from 'mdml-engine/tensor'
import { type Config, defineMethod, derived, Log, type Method, scalar, tensor as tensorSource } from 'mdml-engine/trace'

/** The paths of the linear classifier: the solver's, and the model's own. */
export type LinearClassifierPath =
  | SolverPath
  | GradientPath
  | 'model/accuracy'
  | 'model/weights'
  | 'model/decision-surface'
  | 'model/margins'
  | 'model/errors'

/** The data: two features per row and a class (0 or 1) each. */
export interface TwoClassData {
  readonly x: TensorLike
  readonly y: TensorLike
}

/** The options of a table that are smooth, as the slot's union (the others would stall a gradient method). */
function smoothOnly(
  table: {
    readonly space: { readonly options: Readonly<Record<string, DictSpace>> }
    where(p: (props: { smooth: boolean }) => boolean): string[]
  },
  label: string,
  defaultOption?: string,
): ReturnType<typeof union> {
  const names = table.where((p) => p.smooth)
  return union(Object.fromEntries(names.map((n) => [n, table.space.options[n] as DictSpace])), {
    label,
    ...(defaultOption === undefined ? {} : { default: defaultOption }),
  })
}

const SPACE: DictSpace = dict({
  loss: smoothOnly(classificationLosses, 'loss'),
  penalty: smoothOnly(penalties, 'penalty', 'l2'),
  solver: smoothSolvers.space,
})

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v))

// The penalty defaults to ridge: without one, an exact line search along a coordinate of separable data runs the weights
// to infinity (the loss underflows to zero and the solver reports convergence).
/**
 * The linear classifier on a data set, as a method of its configuration: `loss`, `penalty` and `solver`.
 *
 * Paths: `solver/objective`, `solver/gradNorm`, `solver/step` (the solver's, and `solver/lr` in the stochastic mode), `model/accuracy` (on the rows),
 * `model/weights` (the two weights and the intercept, kept in full), and the probes `model/decision-surface` (the
 * squashed score over the plane with the rows, at any step), `model/margins` (the signed score of each row, positive when
 * the row is on its class's side: a value per row for the rows panel, diverging) and `model/errors` (the rows on the
 * wrong side, as the named set `misclassified`, for the rows panel's marks).
 *
 * @param data - Two features per row and a class each.
 * @returns The method.
 */
export function linearClassifier(data: TwoClassData): Method<unknown, Config, LinearClassifierPath> {
  const flat = toFlat(asTensor(data.x))
  const n = flat.length / 2
  const labels = Array.from(toFlat(asTensor(data.y)), (v) => (v > 0 ? 1 : 0))
  const rows = Array.from({ length: n }, (_, i): [number, number] => [flat[2 * i] as number, flat[2 * i + 1] as number])
  const design = tensor(rows.map(([a, b]) => [a, b, 1]))
  const signs = tensor(labels.map((c) => (c === 1 ? 1 : -1)))
  const score = (w: ArrayLike<number>, a: number, b: number): number =>
    (w[0] as number) * a + (w[1] as number) * b + (w[2] as number)
  const lo = [Math.min(...rows.map((r) => r[0])), Math.min(...rows.map((r) => r[1]))] as const
  const hi = [Math.max(...rows.map((r) => r[0])), Math.max(...rows.map((r) => r[1]))] as const
  const grid = (a: number, b: number, k: number): number[] =>
    Array.from({ length: k }, (_, i) => a - 0.15 * (b - a) + (i * 1.3 * (b - a)) / (k - 1))
  return defineMethod<unknown, Config, LinearClassifierPath>({
    name: 'linear/linear-classifier',
    config: SPACE,
    make: (c, key) => {
      const loss = classificationLosses.make(c['loss'])
      const penalty = penalties.make(c['penalty'])
      const objective = (w: Value): Value => add(loss(matmul(design, w), signs), penalty.value(w))
      return loggedSolver({ objective, x0: [0, 0, 0] }, c['solver'] as Config, key, (x, ctx) => {
        ctx.log('model/weights', Array.from(x))
        ctx.log('model/accuracy', rows.filter(([a, b], i) => (score(x, a, b) > 0 ? 1 : 0) === labels[i]).length / n)
      })
    },
    contract: {
      ...solverContract,
      ...gradientContract,
      'model/accuracy': scalar({ description: 'the share of the rows classified correctly' }),
      'model/weights': tensorSource({ role: 'state', shape: [3], description: 'the two weights and the intercept' }),
      'model/margins': derived<unknown>({
        description: 'the signed score of each row, positive when it is on its class’s side of the line',
        compute: (s) => {
          const w = (s as { readonly x: Float64Array }).x
          return rows.map(([a, b], i) => (labels[i] === 1 ? 1 : -1) * score(w, a, b))
        },
      }),
      'model/errors': derived<unknown>({
        description: 'the rows on the wrong side of the line, as a named set for the rows panel’s marks',
        compute: (s) => {
          const w = (s as { readonly x: Float64Array }).x
          return {
            sets: {
              misclassified: rows.flatMap(([a, b], i) => ((score(w, a, b) > 0 ? 1 : 0) === labels[i] ? [] : [i])),
            },
          }
        },
      }),
      'model/decision-surface': derived<unknown>({
        description: 'the squashed score over the plane of the two features, with the rows',
        cost: 'dear',
        compute: (s): DecisionSurface => {
          const w = (s as { readonly x: Float64Array }).x
          const x = grid(lo[0], hi[0], 48)
          const y = grid(lo[1], hi[1], 48)
          const p = y.map((v) => x.map((u) => sigmoid(score(w, u, v))))
          return decisionSurface({
            features: ['x₀', 'x₁'],
            x,
            y,
            probabilities: [p.map((r) => r.map((q) => 1 - q)), p],
            labels: p.map((r) => r.map((q) => (q > 0.5 ? 1 : 0))),
            classes: ['0', '1'],
            rows: { points: rows, features: ['x₀', 'x₁'], labels, classes: ['0', '1'] },
          })
        },
      }),
    },
    defaults: {
      ...solverDefaults,
      ...gradientDefaults,
      'model/accuracy': Log(),
      'model/weights': Log({ detail: 'full' }),
    },
  })
}
