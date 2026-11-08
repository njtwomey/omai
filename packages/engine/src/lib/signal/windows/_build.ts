import { OmaiError } from '../../base/index.ts'
import type { Tensor } from '../../tensor/index.ts'
import { constant } from '../_shared.ts'

// Window functions as scipy.signal.windows: every window takes a length M and `sym` (true, the default, for a
// symmetric window used in filter design; false for a periodic window used in spectral analysis, which is the
// symmetric window of length M + 1 without its last sample).

/** A window of length m from its symmetric formula at length len (m, or m + 1 for a periodic window). */
export function build(m: number, sym: boolean, f: (len: number) => Float64Array): Tensor {
  if (!Number.isInteger(m) || m < 0) throw new OmaiError('value', `windows: M must be a non-negative integer, got ${m}`)
  if (m <= 1) return constant(new Float64Array(m).fill(1))
  const len = sym ? m : m + 1
  const w = f(len)
  return constant(sym ? w : w.slice(0, m))
}
