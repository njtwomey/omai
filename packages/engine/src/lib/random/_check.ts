import { MdmlError } from '../base/index.ts'

/** Throw `value` unless `x` is a finite number of at least 0 (a scale, which may be 0: a point mass). */
export function assertScale(name: string, what: string, x: number): void {
  if (!(x >= 0 && x < Infinity))
    throw new MdmlError('value', `${name}: ${what} must be finite and non-negative, got ${x}`)
}

/** Throw `value` unless `x` is a finite number above 0 (a rate or degrees of freedom). */
export function assertPositive(name: string, what: string, x: number): void {
  if (!(x > 0 && x < Infinity)) throw new MdmlError('value', `${name}: ${what} must be finite and positive, got ${x}`)
}
