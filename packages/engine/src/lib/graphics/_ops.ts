// Private tensor helpers shared by the shading and tone-mapping terms.
import { greater, maximum, pow, type Value, where } from '../tensor/index.ts'

/** `x^s` for `x > 0` and 0 otherwise, without NaN derivatives at `x <= 0` (the unused branch runs at a safe point). */
export const positivePow = (x: Value, s: Value): Value => where(greater(x, 0), pow(maximum(x, 1e-300), s), 0)
