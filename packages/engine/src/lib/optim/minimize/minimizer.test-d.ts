import { expectTypeOf, test } from 'vitest'
import { sum, type Value } from '../../tensor/index.ts'
import { minimize } from './minimizer.ts'
import type { MinimizeResult, Objective, Parameters, VectorMethodOptions } from './objective.ts'

declare function fake(
  f: Objective,
  x0: Parameters,
  options?: VectorMethodOptions & { readonly memory?: number },
): MinimizeResult

test('inference', () => {
  const r = minimize((x) => sum(x as Value), [1, 2], { method: fake, memory: 3, maxIterations: 4 })
  expectTypeOf(r.fun).toEqualTypeOf<number>()
  // @ts-expect-error unknown option
  minimize((x) => sum(x as Value), [1, 2], { method: fake, bogus: 3 })
})
