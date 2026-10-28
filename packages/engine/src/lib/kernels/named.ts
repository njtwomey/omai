/**
 * Kernels chosen the way scikit-learn's estimators choose them: a metric name (`rbf`, `linear`, `poly`, `sigmoid`,
 * `laplacian`, `cosine`) with `gamma`, `degree` and `coef0`, or any {@link Kernel}. `gamma` may be `'scale'` or `'auto'`,
 * resolved against the training data to a number that the fitted model keeps, so a model made from a name is plain
 * JSON. Support vector machines and kernel ridge regression use these.
 */

import { choice, real, type Space } from '../space/index.ts'
import { asTensor, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Kernel } from './kernel.ts'
import { type PairwiseKernelMetric, pairwiseKernels } from './matrices.ts'

/** A kernel by scikit-learn's names: a metric (or a {@link Kernel}) and its parameters. */
export interface KernelChoice {
  readonly kernel: PairwiseKernelMetric | Kernel
  /** The scale: a number, `'scale'` (1 / (d Var(X))) or `'auto'` (1 / d), scikit-learn's. */
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
}

/** A resolved kernel choice, plain JSON: the metric's name (`custom` for a {@link Kernel} object) and numeric parameters. */
export type ResolvedKernel = {
  readonly kernel: PairwiseKernelMetric | 'custom'
  readonly gamma: number
  readonly degree: number
  readonly coef0: number
}

/**
 * Resolve a kernel choice against the training rows: `'scale'` becomes 1 / (d Var(X)) with the variance over all
 * entries (1 for constant data), `'auto'` becomes 1 / d.
 *
 * @param choice - The kernel choice.
 * @param x - Training rows `[n, d]`.
 * @returns The same kernel with a numeric gamma, as plain JSON.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { resolveKernel } from 'mdml-engine/kernels'
 * resolveKernel({ kernel: 'rbf', gamma: 'auto', degree: 3, coef0: 1 }, [[0, 1], [2, 3]]).gamma // => 0.5
 * ```
 */
export function resolveKernel(choice: KernelChoice, x: TensorLike): ResolvedKernel {
  const t = primalOf(asTensor(x))
  const d = t.shape[1] ?? 1
  let gamma: number
  if (choice.gamma === 'auto') gamma = 1 / d
  else if (choice.gamma === 'scale') {
    const values = toFlat(t)
    let s = 0
    let s2 = 0
    for (const v of values) {
      s += v
      s2 += v * v
    }
    const variance = s2 / values.length - (s / values.length) ** 2
    gamma = variance > 0 ? 1 / (d * variance) : 1
  } else gamma = choice.gamma
  return {
    kernel: typeof choice.kernel === 'string' ? choice.kernel : 'custom',
    gamma,
    degree: choice.degree,
    coef0: choice.coef0,
  }
}

/**
 * The kernel matrix between rows of x and y (or of x with itself) as a flat row-major array.
 *
 * @param choice - The kernel choice (its Kernel object, if it has one, is used when `resolved.kernel` is `custom`).
 * @param resolved - Its resolved parameters.
 * @param x - Rows `[m, d]`.
 * @param y - Rows `[n, d]` (default x).
 * @returns The `m * n` entries.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { kernelMatrix } from 'mdml-engine/kernels'
 * const choice = { kernel: 'linear' as const, gamma: 1, degree: 3, coef0: 0 }
 * Array.from(kernelMatrix(choice, { kernel: 'linear', gamma: 1, degree: 3, coef0: 0 }, [[1, 2]], [[3, 4], [0, 1]])) // => [11, 2]
 * ```
 */
export function kernelMatrix(
  choice: Pick<KernelChoice, 'kernel'>,
  resolved: ResolvedKernel,
  x: TensorLike,
  y?: TensorLike,
): Float64Array {
  const metric = typeof choice.kernel === 'string' ? (resolved.kernel as PairwiseKernelMetric) : choice.kernel
  const m = pairwiseKernels(x, y, metric, { gamma: resolved.gamma, degree: resolved.degree, coef0: resolved.coef0 })
  return Float64Array.from(toFlat(primalOf(m)))
}

/** The kernel names {@link kernelSpace} offers. */
const NAMES = ['rbf', 'linear', 'poly', 'sigmoid', 'laplacian'] as const

/**
 * The knobs of a kernel choice as a space: the kernel's name, and gamma (when it is a number), the degree and the
 * offset where the kernel uses them. Empty for a {@link Kernel} object, whose hyperparameters are its own.
 *
 * @param choice - The current kernel choice.
 * @returns Spaces by knob name, to spread into an estimator's `hyperSpace`.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { kernelSpace } from 'mdml-engine/kernels'
 * Object.keys(kernelSpace({ kernel: 'rbf', gamma: 0.5, degree: 3, coef0: 1 })) // => ['kernel', 'gamma', 'degree', 'coef0']
 * ```
 */
export function kernelSpace(current: KernelChoice): Readonly<Record<string, Space>> {
  if (typeof current.kernel !== 'string') return {}
  const name = (NAMES as readonly string[]).includes(current.kernel) ? current.kernel : 'rbf'
  return {
    kernel: choice([...NAMES], {
      label: 'kernel',
      description: 'The kernel measuring similarity between rows.',
      default: name,
    }),
    ...(typeof current.gamma === 'number'
      ? {
          gamma: real(1e-3, 10, {
            label: 'kernel gamma',
            description: 'The kernel scale: larger makes similarity more local.',
            log: true,
            default: Math.min(Math.max(current.gamma, 1e-3), 10),
            when: { key: 'kernel', in: ['rbf', 'poly', 'sigmoid', 'laplacian'] },
          }),
        }
      : {}),
    degree: real(2, 6, {
      label: 'polynomial degree',
      step: 1,
      default: Math.min(Math.max(current.degree, 2), 6),
      when: { key: 'kernel', in: ['poly'] },
    }),
    coef0: real(0, 5, {
      label: 'kernel offset',
      default: Math.min(Math.max(current.coef0, 0), 5),
      when: { key: 'kernel', in: ['poly', 'sigmoid'] },
    }),
  }
}
