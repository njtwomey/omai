import { describe, expectTypeOf, it } from 'vitest'
import { dict } from '../space/index.ts'
import type { Algorithm, StepContext } from './algorithm.ts'
import {
  type Concrete,
  type Contract,
  defineContract,
  event,
  Log,
  On,
  type PathsOf,
  type Prefixed,
  prefixContract,
  type Subscription,
  scalar,
} from './contract.ts'
import { mountContext } from './nesting.ts'
import { type Config, defineMethod, type Method, session } from './session.ts'

const base: Contract<unknown, 'solver/loss' | 'solver/step'> = defineContract({
  'solver/loss': scalar(),
  'solver/step': scalar(),
})
const own = defineContract({ ...base, 'model/accuracy': scalar(), 'epoch/end': event() })

const typed = defineMethod({
  name: 'typed',
  config: dict({}),
  make: () => ({
    name: 'typed',
    config: {},
    init: () => 0,
    step: (s: number, ctx) => {
      ctx.log('solver/loss', s)
      ctx.emit('epoch/end')
      if (ctx.wants('model/accuracy')) ctx.log('model/accuracy', 1)
      return s + 1
    },
  }),
  contract: own,
  defaults: { 'solver/loss': Log() },
})

describe('typed paths', () => {
  it('a contract keeps its keys, and a fragment composes', () => {
    expectTypeOf<PathsOf<typeof own>>().toEqualTypeOf<'solver/loss' | 'solver/step' | 'model/accuracy' | 'epoch/end'>()
    expectTypeOf<PathsOf<typeof base>>().toEqualTypeOf<'solver/loss' | 'solver/step'>()
  })

  it('defineMethod infers the paths from the contract', () => {
    type PathsOfMethod<M> = M extends Method<infer _S, infer _C, infer K> ? K : never
    expectTypeOf<PathsOfMethod<typeof typed>>().toEqualTypeOf<PathsOf<typeof own>>()
  })

  it('a typo in ctx.log, ctx.emit or ctx.wants is a type error', () => {
    defineMethod({
      name: 'typo',
      config: dict({}),
      make: () => ({
        name: 'typo',
        config: {},
        init: () => 0,
        step: (s: number, ctx) => {
          // @ts-expect-error not a path of the contract
          ctx.log('solver/objectve', s)
          // @ts-expect-error not a path of the contract
          ctx.emit('epoch/finish')
          // @ts-expect-error not a path of the contract
          ctx.wants('model/accuracy2')
          return s
        },
      }),
      contract: own,
    })
  })

  it('a typo in defaults or in a session subscription is a type error', () => {
    defineMethod({
      name: 'typo',
      config: dict({}),
      make: () => ({ name: 'typo', config: {}, init: () => 0, step: (s: number) => s }),
      contract: own,
      // @ts-expect-error not a path of the contract
      defaults: { 'solver/lossy': Log() },
    })
    session(typed, { logs: { 'solver/loss': Log(), 'model/*': Log(), '**/step': Log(), 'epoch/end': On(() => {}) } })
    session(typed, { logs: 'all' })
    // @ts-expect-error not a path of the contract
    session(typed, { logs: { 'solver/lossy': Log() } })
  })

  it('a method without typed paths still accepts any string', () => {
    const plain: Method<number> = {
      name: 'plain',
      config: dict({}),
      make: () => ({
        name: 'plain',
        config: {},
        init: () => 0,
        step: (s, ctx) => {
          ctx.log('any/thing', s)
          return s
        },
      }),
      contract: { 'any/thing': scalar() },
    }
    session(plain, { logs: { 'whatever/you/like': Log() } })
    expectTypeOf<StepContext['log']>().parameter(0).toEqualTypeOf<string>()
    expectTypeOf<Subscription>().toEqualTypeOf<Subscription<string>>()
  })

  it('a typed algorithm is accepted where an untyped one is wanted', () => {
    expectTypeOf<Algorithm<number, 'a' | 'b'>>().toExtend<Algorithm<number>>()
    const widened: Method<number> = null as unknown as Method<number, Config, 'a'>
    expectTypeOf(widened).not.toBeAny()
  })

  it('computed parts are template literals; patterns are logged by any path under them', () => {
    type Layered = `layer${number}/weights` | 'loss'
    const ctx = null as unknown as StepContext<Layered>
    ctx.log('layer3/weights', 1)
    ctx.log('loss', 1)
    // @ts-expect-error not a path of the contract
    ctx.log('layer3/bias', 1)
    // @ts-expect-error not a path of the contract
    ctx.log('layerx/weights', 1)
    expectTypeOf<Concrete<'taps/*' | 'loss'>>().toEqualTypeOf<`taps/${string}` | 'loss'>()
    const tapped = null as unknown as StepContext<'taps/*'>
    tapped.log('taps/fc1', 1)
    // @ts-expect-error not under the pattern
    tapped.log('other/fc1', 1)
  })

  it('mounting a contract under a prefix prefixes the types', () => {
    expectTypeOf<Prefixed<'inner', 'a' | 'b'>>().toEqualTypeOf<'inner/a' | 'inner/b'>()
    const mounted = prefixContract('inner', base)
    expectTypeOf<PathsOf<typeof mounted>>().toEqualTypeOf<'inner/solver/loss' | 'inner/solver/step'>()
    const outer = null as unknown as StepContext<'inner/solver/loss' | 'inner/solver/step'>
    const inner = mountContext<'inner', 'solver/loss' | 'solver/step'>(outer, 'inner')
    inner.log('solver/loss', 1)
    // @ts-expect-error not a path of the inner contract
    inner.log('solver/lss', 1)
  })
})
