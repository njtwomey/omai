import { dict, int } from 'mdml-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'mdml-engine/trace'
import { type ReactElement, useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { methodRegistry } from '../../producer/registry.ts'
import { type TapeFile, tapeOf } from '../../producer/tape.ts'
import { type FakeFrames, fakeFrames, inAct, invoke, type Mounted, mountElement, resetGlobals } from '../_harness.tsx'
import { SessionExport } from './SessionExport.tsx'
import { type LiveSession, useSession, variantsOf } from './useSession.ts'

interface S {
  readonly k: number
}
const count: Method<S, Config> = defineMethod<S, Config>({
  name: 'count',
  config: dict({ len: int(1, 1000, { default: 30 }) }),
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

const seen: { live?: LiveSession<S, Config> } = {}
const loaded: TapeFile[] = []
function Page(props: { readonly withLoad: boolean }): ReactElement {
  const live = useSession(count, { steps: 12 }, [], { start: 'end' })
  const [at, setAt] = useState<number | null>(null)
  const root = useRef<HTMLDivElement>(null)
  seen.live = live
  return (
    <SessionExport
      live={live}
      at={at}
      onScrub={setAt}
      root={root}
      title="Counting"
      filename="count"
      registry={methodRegistry([count])}
      {...(props.withLoad ? { onLoadTape: (f: TapeFile) => loaded.push(f) } : {})}
    />
  )
}

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
  loaded.length = 0
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

const find = (m: Mounted<void>, pick: (props: Readonly<Record<string, unknown>>) => boolean) =>
  m.elements().find((e) => pick(e.props))
const open = (m: Mounted<void>) =>
  invoke(
    find(m, (p) => p['aria-label'] === 'Export the page'),
    'onClick',
  )
const note = (m: Mounted<void>): string =>
  m.elements().find((e) => e.tag === 'span' && /danger/.test(String(e.props['className'])))?.text ?? ''
const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 5))

describe('SessionExport', () => {
  it('offers the page as stills, a recording and a tape, and loads a tape file only for a method the page has', async () => {
    const m = await mountElement(<Page withLoad />, { browser: true })
    for (let n = 0; n < 200 && seen.live?.done !== true; n++) await frames.frame()
    expect(seen.live?.done).toBe(true)

    // Closed: one labelled button. Opened: the menu says which step the page is at, and offers the range to record.
    const button = (): ReturnType<typeof find> => find(m, (p) => p['aria-label'] === 'Export the page')
    expect([button()?.props['aria-expanded'], find(m, (p) => p['data-testid'] === 'export-menu')]).toEqual([
      false,
      undefined,
    ])
    await open(m)
    expect(button()?.props['aria-expanded']).toBe(true)
    const menu = find(m, (p) => p['data-testid'] === 'export-menu')
    expect(menu).toBeDefined()
    expect(menu?.node && m.elements().find((e) => e.node === menu.node)?.text).toContain('the page at step 12')
    const buttons = m
      .elements()
      .filter((e) => e.tag === 'button' || e.tag === 'label')
      .map((e) => e.text)
    expect(buttons).toEqual(expect.arrayContaining(['PNG', 'SVG', 'GIF', 'save JSON', 'load JSON']))
    const range = (label: string): unknown => find(m, (p) => p['aria-label'] === label)?.props['value']
    expect([range('from step'), range('to step')]).toEqual([0, 12])
    // The range is whole steps and is shown inside the run.
    await invoke(
      find(m, (p) => p['aria-label'] === 'from step'),
      'onChange',
      { target: { value: '3.6' } },
    )
    await invoke(
      find(m, (p) => p['aria-label'] === 'to step'),
      'onChange',
      { target: { value: '99' } },
    )
    expect([range('from step'), range('to step')]).toEqual([4, 12])

    // Loading: text that is not a tape, a tape of a method the page does not have, and a tape it can replay.
    const file = async (text: string): Promise<void> => {
      const input = find(m, (p) => p['data-testid'] === 'load-tape')
      const target = { files: [new File([text], 'x.json')], value: 'x.json' }
      await invoke(input, 'onChange', { target })
      await settle()
      await inAct(() => undefined)
      expect(target.value, 'the input is cleared so the same file can be chosen again').toBe('')
    }
    const session = variantsOf(seen.live as LiveSession<S, Config>)[0]?.live.session
    const tape = await tapeOf('count', [{ label: 'a', session: session as never }], { requests: [], at: 12 })
    await file('this is not json')
    expect([note(m), loaded.length]).toEqual([expect.stringContaining('not'), 0])
    await open(m)
    await file(JSON.stringify({ ...tape, method: 'other' }))
    expect([note(m), loaded.length]).toEqual([expect.stringContaining("no method called 'other' (known: count)"), 0])
    await open(m)
    await file(JSON.stringify(tape))
    expect([note(m), loaded.map((f) => [f.method, f.variants[0]?.head])]).toEqual(['', [['count', 12]]])
    await m.unmount()

    // Without a handler there is no load item (the tape can still be saved).
    const bare = await mountElement(<Page withLoad={false} />, { browser: true })
    await open(bare)
    expect(find(bare, (p) => p['data-testid'] === 'load-tape')).toBeUndefined()
    expect(bare.elements().some((e) => e.text === 'save JSON')).toBe(true)
    await bare.unmount()
  })
})
