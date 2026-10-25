// Shared plumbing of the neural agents (private to methods/rl/agents): spaces to sizes, batches to tensors, and one
// optimiser step on a loss of a parameter tree.
import { valueAndGrad } from '../../../lib/autodiff/index.ts'
import { MdmlError } from '../../../lib/base/index.ts'
import {
  adam,
  applyUpdates,
  chain,
  clipByGlobalNorm,
  type GradientTransformation,
  type Tree,
} from '../../../lib/optim/index.ts'
import { flatDim, flatten, type Space, type SpaceValue } from '../../../lib/space/index.ts'
import {
  fromStorage,
  item,
  mul,
  primalOf,
  type Tensor,
  toFlat,
  treeMap,
  type Value,
} from '../../../lib/tensor/index.ts'

/** What an agent factory needs from an environment: its spaces. */
export interface Spaces {
  readonly observationSpace: Space
  readonly actionSpace: Space
}

/** The length of flat vector observations. */
export function observationSize(space: Space): number {
  return flatDim(space)
}

/** The number of actions of a discrete action space. */
export function actionCount(space: Space, name: string): number {
  if (space.kind !== 'discrete') throw new MdmlError('value', `${name}: needs a discrete action space`)
  return space.n
}

/** The size and bounds of a box action space. */
export function actionBox(
  space: Space,
  name: string,
): { readonly size: number; readonly low: number[]; readonly high: number[] } {
  if (space.kind !== 'box') throw new MdmlError('value', `${name}: needs a box action space`)
  const size = space.shape.reduce((a, b) => a * b, 1)
  const at = (b: number | readonly number[], i: number): number => (typeof b === 'number' ? b : (b[i] ?? 0))
  return {
    size,
    low: Array.from({ length: size }, (_, i) => at(space.low, i)),
    high: Array.from({ length: size }, (_, i) => at(space.high, i)),
  }
}

/** Rows of numbers as a `[n, width]` float64 tensor. */
export function rowsTensor(rows: readonly (readonly number[])[], width: number): Tensor {
  const out = new Float64Array(rows.length * width)
  for (const [i, r] of rows.entries()) for (let j = 0; j < width; j++) out[i * width + j] = r[j] ?? 0
  return fromStorage(out, [rows.length, width], 'float64')
}

/** A vector of numbers as a `[n]` float64 tensor. */
export function vectorTensor(xs: readonly number[]): Tensor {
  return fromStorage(Float64Array.from(xs), [xs.length], 'float64')
}

/** One optimiser step: the loss and its gradient at `params`, and the transformed update applied. */
export function gradientStep(
  loss: (params: Tree) => Value,
  params: Tree,
  transformation: GradientTransformation,
  optState: unknown,
): { readonly params: Tree; readonly optState: unknown; readonly loss: number } {
  const { value, grad } = valueAndGrad(loss)(params)
  const { updates, state } = transformation.update(grad, optState, params)
  return { params: applyUpdates(params, updates), optState: state, loss: item(value) }
}

/**
 * One optimiser step on a loss that also returns per-sample terms (`[loss, terms]`): the terms are read from the same
 * forward pass (for replay priorities and statistics) instead of a second one.
 */
export function gradientStepWithTerms(
  loss: (params: Tree) => readonly [Value, Value],
  params: Tree,
  transformation: GradientTransformation,
  optState: unknown,
): { readonly params: Tree; readonly optState: unknown; readonly loss: number; readonly terms: number[] } {
  const { value, grad, aux } = valueAndGrad(loss, { hasAux: true })(params)
  const { updates, state } = transformation.update(grad, optState, params)
  return {
    params: applyUpdates(params, updates),
    optState: state,
    loss: item(value),
    terms: toFlat(primalOf(aux as Value)),
  }
}

/** Adam with a learning rate (or a given optimiser), behind global-norm clipping when `maxGradNorm` is set. */
export function optimiserOf(
  options: {
    readonly optimizer?: GradientTransformation
    readonly learningRate?: number
    readonly maxGradNorm?: number
  },
  learningRate = 1e-3,
): GradientTransformation {
  const base = options.optimizer ?? adam({ learningRate: options.learningRate ?? learningRate })
  return options.maxGradNorm === undefined ? base : chain(clipByGlobalNorm(options.maxGradNorm), base)
}

/** Flat numeric features of a space's values for networks: discrete values one-hot, everything else flattened. */
export function spaceFeatures(space: Space): { readonly size: number; readonly of: (value: unknown) => number[] } {
  if (space.kind === 'discrete') {
    const n = space.n
    const start = space.start ?? 0
    return {
      size: n,
      of: (v) => {
        const x = new Array<number>(n).fill(0)
        x[(v as number) - start] = 1
        return x
      },
    }
  }
  return { size: flatDim(space), of: (v) => flatten(space, v as SpaceValue) }
}

/**
 * Apply a gradient through an optimiser, with the update scaled by `scale` (learning-rate annealing on top of any
 * optimiser: a factor of 1 leaves the optimiser's own step unchanged).
 */
export function applyGradient(
  grad: Tree,
  params: Tree,
  transformation: GradientTransformation,
  optState: unknown,
  scale = 1,
): { readonly params: Tree; readonly optState: unknown } {
  const { updates, state } = transformation.update(grad, optState, params)
  const scaled = scale === 1 ? updates : treeMap((u) => mul(u, scale), updates)
  return { params: applyUpdates(params, scaled), optState: state }
}
