import { describe, expectTypeOf, it } from 'vitest'
import type { SdeSolver } from '../nn/blocks/index.ts'
import type { Paths } from './paths.ts'
import { type Sde, sdeint, sdeSolver } from './sde.ts'

describe('sde types', () => {
  it('sdeSolver satisfies the nn SdeSolver contract', () => {
    expectTypeOf(sdeSolver('milstein')).toExtend<SdeSolver>()
  })

  it('sdeint returns paths', () => {
    expectTypeOf(sdeint).returns.toEqualTypeOf<Paths>()
    expectTypeOf<Sde['noise']>().toEqualTypeOf<'diagonal' | 'scalar' | 'general' | undefined>()
  })
})
