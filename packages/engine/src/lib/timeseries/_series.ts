/**
 * Shared by the time-series files: a series argument as a plain Float64Array of primal values.
 */
import { OmaiError } from '../base/index.ts'
import { asTensor, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'

/** A 1-D (or single-column) series as a Float64Array. */
export function asSeries(y: TensorLike, who: string): Float64Array {
  const t = primalOf(asTensor(y, 'float64'))
  if (t.shape.length > 2 || (t.shape.length === 2 && t.shape[1] !== 1))
    throw new OmaiError('shape', `${who}: expected a 1-D series, got shape [${t.shape.join(', ')}]`)
  return Float64Array.from(toFlat(t))
}
