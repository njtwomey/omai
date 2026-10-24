// Evaluation of rational functions at points of the complex plane, shared by the frequency-response files.
import {
  add,
  type ComplexValue,
  cDiv,
  cMul,
  concat,
  fromStorage,
  reshape,
  sliceAxis,
  type Tensor,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** Points of the complex plane as constant parts. */
export interface Points {
  readonly re: Tensor
  readonly im: Tensor
}

export const constant = (xs: readonly number[]): Tensor => fromStorage(Float64Array.from(xs), [xs.length], 'float64')

/** Horner's rule for a highest-first coefficient value at complex points (differentiable in the coefficients). */
export function horner(c: Value, z: Points): ComplexValue {
  const k = c.shape[0] ?? 0
  let acc: ComplexValue = { re: zeros(z.re.shape), im: zeros(z.re.shape) }
  for (let j = 0; j < k; j++) {
    const cj = reshape(sliceAxis(c, 0, j, j + 1), [])
    const scaled = cMul(acc, z)
    acc = { re: add(scaled.re, cj), im: scaled.im }
  }
  return acc
}

export function tfAt(num: Value, den: Value, z: Points): ComplexValue {
  const d = horner(den, z)
  if (num.shape.length === 1) return cDiv(horner(num, z), d)
  const p = num.shape[0] ?? 0
  const rowsOf = Array.from({ length: p }, (_, i) => cDiv(horner(reshape(sliceAxis(num, 0, i, i + 1), [-1]), z), d))
  const nw = z.re.shape[0] ?? 0
  const pack = (part: 're' | 'im'): Value =>
    reshape(
      concat(
        rowsOf.map((r) => reshape(r[part], [nw, 1])),
        1,
      ),
      [nw, p, 1],
    )
  return { re: pack('re'), im: pack('im') }
}
