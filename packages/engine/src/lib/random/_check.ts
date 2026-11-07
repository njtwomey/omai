import { MdmlError } from '../base/index.ts'
import { asTensor, concreteOrThrow, type TensorLike } from '../tensor/index.ts'

/** Throw `value` unless `x` is a finite number of at least 0 (a scale, which may be 0: a point mass). */
export function assertScale(name: string, what: string, x: number): void {
  if (!(x >= 0 && x < Infinity))
    throw new MdmlError('value', `${name}: ${what} must be finite and non-negative, got ${x}`)
}

/** Throw `value` unless `x` is a finite number above 0 (a rate or degrees of freedom). */
export function assertPositive(name: string, what: string, x: number): void {
  if (!(x > 0 && x < Infinity)) throw new MdmlError('value', `${name}: ${what} must be finite and positive, got ${x}`)
}

/**
 * The numbers of a weight or concentration vector, read once at the entry of a sampler: any `TensorLike` of rank 1
 * (a tensor, a typed array or a nested array), as plain numbers for the sampler's own loop.
 */
export function vectorNumbers(name: string, what: string, x: TensorLike): number[] {
  const t = concreteOrThrow(asTensor(x), name)
  if (t.shape.length !== 1) {
    throw new MdmlError('shape', `${name}: ${what} must be a vector, got shape [${t.shape.join(', ')}]`)
  }
  return Array.from(t.data)
}
