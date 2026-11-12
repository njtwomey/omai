// What the multiclass sites share: the parameters that name the class and the log probabilities of the softmax and of the multinomial
// probit. `multiclass.ts` (one Gaussian message per class score) and `multiclass-row.ts` (one joint message on the vector the scores are
// made of) both read them, and integrate with the adaptive Gauss-Hermite rule of `numerics/_product.ts`.

import { ndtr } from '../../../numerics/special/_kernels.ts'
import { numberParam } from '../../core/args.ts'
import { ParameterError } from '../../core/errors.ts'
import type { Params } from '../../core/types.ts'
import { gaussianNodes } from '../../numerics/_quadrature.ts'

/** The classes and the outcome from the parameters, checked. */
export function classesOf(params: Params): { readonly classes: number; readonly y: number; readonly probit: boolean } {
  const classes = numberParam(params, 'classes')
  const y = numberParam(params, 'y')
  if (!(Number.isInteger(classes) && classes >= 2))
    throw new ParameterError('multiclass-site: classes is an integer of at least 2')
  if (!(Number.isInteger(y) && y >= 0 && y < classes))
    throw new ParameterError(`multiclass-site: the class must be an integer from 0 to ${classes - 1}`)
  const link = params['link'] ?? 'logit'
  if (link !== 'logit' && link !== 'probit')
    throw new ParameterError("multiclass-site: the link is 'logit' or 'probit'")
  return { classes, y, probit: link === 'probit' }
}

/** log of the sum of `exp(v)` (`-Infinity` for none). */
export function logSumExp(v: readonly number[]): number {
  let top = Number.NEGATIVE_INFINITY
  for (const x of v) if (x > top) top = x
  if (!Number.isFinite(top)) return top
  let s = 0
  for (const x of v) s += Math.exp(x - top)
  return top + Math.log(s)
}

/** log P(y | z) under the softmax. */
export function logSoftmax(z: ArrayLike<number>, y: number): number {
  let top = Number.NEGATIVE_INFINITY
  for (let k = 0; k < z.length; k++) if ((z[k] as number) > top) top = z[k] as number
  let s = 0
  for (let k = 0; k < z.length; k++) s += Math.exp((z[k] as number) - top)
  return (z[y] as number) - top - Math.log(s)
}

const E_NODES = gaussianNodes(0, 1, 20)

/** log P(y | z) under the multinomial probit: the largest of `z_k + e_k`, `e_k` standard normal; one integral over `e_y`. */
export function logProbit(z: ArrayLike<number>, y: number): number {
  let total = 0
  for (let i = 0; i < E_NODES.x.length; i++) {
    let p = 1
    for (let j = 0; j < z.length; j++)
      if (j !== y) p *= ndtr((z[y] as number) - (z[j] as number) + (E_NODES.x[i] as number))
    total += (E_NODES.w[i] as number) * p
  }
  return Math.log(total)
}

/** The log probability of the observed class at the scores: the softmax's or the multinomial probit's. */
export function scoreLog(params: Params): (z: readonly number[]) => number {
  const { y, probit } = classesOf(params)
  return probit ? (z) => logProbit(z, y) : (z) => logSoftmax(z, y)
}
