import { dict, int } from 'omai-engine/space'
import type { Algorithm } from 'omai-engine/trace'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import { type ReactElement, useState } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountElement, resetGlobals } from './_harness.tsx'
import { ActiveProvider, useActive } from './active.tsx'
import { type LiveSessions, useSession } from './session/useSession.ts'
import { type LiveRun, useLiveRun, usePlayback } from './usePlayback.ts'

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

function Flag(props: { readonly explicit?: boolean }): ReactElement {
  return <span>{String(useActive(props.explicit))}</span>
}
const flag = (el: ReactElement): string => renderToString(el).replace(/<[^>]+>/g, '')

describe('ActiveProvider and useActive', () => {
  it('is true with no provider', () => {
    expect(flag(<Flag />)).toBe('true')
  })

  it('follows its prop, and true is the default', () => {
    expect(
      flag(
        <ActiveProvider active={false}>
          <Flag />
        </ActiveProvider>,
      ),
    ).toBe('false')
    expect(
      flag(
        <ActiveProvider>
          <Flag />
        </ActiveProvider>,
      ),
    ).toBe('true')
  })

  it('is the AND of every provider above: an inner false wins over an outer true, an inner true stays false inside a false', () => {
    const nest = (outer: boolean, inner: boolean): string =>
      flag(
        <ActiveProvider active={outer}>
          <ActiveProvider active={inner}>
            <Flag />
          </ActiveProvider>
        </ActiveProvider>,
      )
    expect([nest(true, true), nest(true, false), nest(false, true), nest(false, false)]).toEqual([
      'true',
      'false',
      'false',
      'false',
    ])
    // A provider that says nothing between two others changes nothing.
    expect(
      flag(
        <ActiveProvider active={false}>
          <ActiveProvider>
            <Flag />
          </ActiveProvider>
        </ActiveProvider>,
      ),
    ).toBe('false')
  })

  it('is replaced by an explicit option, either way, so a host can override', () => {
    expect(
      flag(
        <ActiveProvider active={false}>
          <Flag explicit />
        </ActiveProvider>,
      ),
    ).toBe('true')
    expect(
      flag(
        <ActiveProvider active>
          <Flag explicit={false} />
        </ActiveProvider>,
      ),
    ).toBe('false')
  })
})

const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }

interface S {
  readonly k: number
}
const count: Method<S, Config> = defineMethod<S, Config>({
  name: 'count',
  config: dict({ len: int(1, 1000, { default: 200 }) }),
  make: (c) => ({
    name: 'count',
    config: c,
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('n', s.k + 1)
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= (c['len'] as number) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as S,
  }),
  contract: { n: scalar() },
  defaults: { n: Log() },
})

// What the hooks returned, and the provider's switches, which a test flips before it renders again.
const seen: {
  run?: LiveRun<number>
  session?: LiveSessions<S, Config>
  playback?: ReturnType<typeof usePlayback>
} = {}
const switches = { outer: true, inner: true, explicit: undefined as boolean | undefined, poke: (): void => undefined }

function Hooks(): null {
  seen.run = useLiveRun(counter, {
    maxSteps: 300,
    start: 'first',
    ...(switches.explicit === undefined ? {} : { active: switches.explicit }),
  })
  seen.session = useSession(count, { steps: 200 }, [], {
    start: 'first',
    ...(switches.explicit === undefined ? {} : { active: switches.explicit }),
  }) as never
  seen.playback = usePlayback(50, {
    fps: 60,
    autoplay: true,
    ...(switches.explicit === undefined ? {} : { active: switches.explicit }),
  })
  return null
}

function Host(): ReactElement {
  const [, set] = useState(0)
  switches.poke = () => set((n) => n + 1)
  return (
    <ActiveProvider active={switches.outer}>
      <ActiveProvider active={switches.inner}>
        <Hooks />
      </ActiveProvider>
    </ActiveProvider>
  )
}

describe('the hooks read the ambient active', () => {
  beforeEach(() => {
    switches.outer = true
    switches.inner = true
    switches.explicit = undefined
  })

  it('compute nothing and keep no clock under an inactive provider, and carry on when it flips', async () => {
    switches.outer = false
    const m = await mountElement(<Host />)
    await frames.frames(4)
    // Nothing computed: the run, the session, and the clock of the playback.
    expect([seen.run?.controller.count, seen.session?.head, seen.playback?.frame]).toEqual([1, 0, 0])
    expect(frames.pending).toBe(0)
    expect(seen.session?.active).toBe(false)

    switches.outer = true
    await inAct(switches.poke)
    await frames.frames(4)
    expect(seen.session?.active).toBe(true)
    expect(seen.run?.controller.count).toBeGreaterThan(1)
    expect(seen.session?.head).toBeGreaterThan(0)
    expect(seen.playback?.frame).toBeGreaterThan(0)

    // And stops again when an inner provider says so, keeping what it has.
    switches.inner = false
    await inAct(switches.poke)
    await frames.frame()
    const [run, head, frame] = [seen.run?.controller.count, seen.session?.head, seen.playback?.frame]
    await frames.frames(4)
    expect([seen.run?.controller.count, seen.session?.head, seen.playback?.frame]).toEqual([run, head, frame])
    expect(frames.pending).toBe(0)
    await m.unmount()
  })

  it('obey an explicit option over the provider', async () => {
    switches.outer = false
    switches.explicit = true
    const m = await mountElement(<Host />)
    await frames.frames(4)
    expect(seen.session?.active).toBe(true)
    expect(seen.run?.controller.count).toBeGreaterThan(1)
    expect(seen.playback?.frame).toBeGreaterThan(0)

    switches.outer = true
    switches.explicit = false
    await inAct(switches.poke)
    await frames.frame()
    const [run, head, frame] = [seen.run?.controller.count, seen.session?.head, seen.playback?.frame]
    await frames.frames(4)
    expect(seen.session?.active).toBe(false)
    expect([seen.run?.controller.count, seen.session?.head, seen.playback?.frame]).toEqual([run, head, frame])
    await m.unmount()
  })
})
