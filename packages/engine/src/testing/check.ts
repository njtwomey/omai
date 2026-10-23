// The derivative conformance check as a test assertion. Node only; not published.
import { expect } from 'vitest'
import { type CheckOptions, checkPrimitive } from '../lib/autodiff/index.ts'
import type { TensorLike, Value } from '../lib/tensor/index.ts'

/** Assert that `f` passes {@link checkPrimitive} at `inputs` (default tolerances rtol 1e-5, atol 1e-8). */
export function expectConformant(
  f: (...xs: Value[]) => Value,
  inputs: readonly TensorLike[],
  options: CheckOptions = {},
): void {
  expect(checkPrimitive(f, inputs, { rtol: 1e-5, atol: 1e-8, ...options }).failures).toEqual([])
}
