import { describe, expect, it } from 'vitest'
import { keyFromPath } from '../random/index.ts'
import { recorder } from '../trace/index.ts'
import { createContext, defaultContext, derive, resolveKey, resolveTracer, setDefaultContext } from './index.ts'

describe('contexts', () => {
  it('derive keys by name, reproducibly and independently', () => {
    const ctx = createContext({ seed: 9 })
    expect(ctx.key('a', 1).words).toEqual(keyFromPath(9, 'a', 1).words)
    expect(derive(ctx, 'sub').key('a').words).toEqual(keyFromPath(9, 'sub', 'a').words)
    expect(ctx.key('a').words).not.toEqual(ctx.key('b').words)
  })

  it('resolve explicit keys first, then the context, then the default', () => {
    const ctx = createContext({ seed: 4 })
    const explicit = keyFromPath(1)
    expect(resolveKey({ key: explicit, ctx }, 'f')).toBe(explicit)
    expect(resolveKey({ ctx }, 'f').words).toEqual(ctx.key('f').words)
    const previous = setDefaultContext(createContext({ seed: 11 }))
    try {
      expect(resolveKey({}, 'f').words).toEqual(keyFromPath(11, 'f').words)
    } finally {
      setDefaultContext(previous)
    }
    expect(defaultContext().seed).toBe(previous.seed)
  })

  it('carry the tracer', () => {
    const rec = recorder()
    expect(resolveTracer({ ctx: createContext({ tracer: rec }) })).toBe(rec)
  })
})
