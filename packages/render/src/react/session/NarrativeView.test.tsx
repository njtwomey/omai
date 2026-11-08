import { dict, int } from 'omai-engine/space'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import { type ReactElement, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, invoke, type Mounted, mountElement, resetGlobals } from '../_harness.tsx'
import type { Narrative } from './layout.ts'
import { NarrativeView } from './NarrativeView.tsx'
import { type LiveSession, useSession } from './useSession.ts'

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

const narrative: Narrative = {
  title: 'A story',
  pages: [
    {
      id: 'a',
      title: 'Setup',
      text: 'The **first** page, with `code`.\n\nA second paragraph.',
      layout: [],
      moments: [
        { at: 5, label: 'five' },
        { at: 10, label: 'ten' },
      ],
    },
    { id: 'b', title: 'Middle', text: 'The middle page.', layout: [], moments: [{ at: 20 }] },
    { id: 'c', title: 'Aside', text: 'No moments here.', layout: [] },
  ],
}

const seen: { at: number | null; live?: LiveSession<S, Config>; scrubs: (number | null)[] } = { at: null, scrubs: [] }
function Page(): ReactElement {
  const [at, setAt] = useState<number | null>(null)
  const live = useSession(count, { steps: 30 }, [], { start: 'end' })
  seen.at = at
  seen.live = live
  return (
    <NarrativeView
      id="story"
      narrative={narrative}
      live={live}
      at={at}
      onScrub={(t) => {
        seen.scrubs.push(t)
        setAt(t)
      }}
    />
  )
}

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
  seen.at = null
  seen.scrubs = []
  // The key handler skips typing in an input: a browser has the class, Node does not.
  vi.stubGlobal('HTMLInputElement', class {})
})
afterEach(() => {
  frames.restore()
  vi.unstubAllGlobals()
  resetGlobals()
})

const el = (m: Mounted<void>, pick: (props: Readonly<Record<string, unknown>>, tag: string) => boolean) =>
  m.elements().find((e) => pick(e.props, e.tag))
const press = async (m: Mounted<void>, label: string): Promise<void> => {
  const b = el(m, (p) => p['aria-label'] === label)
  expect(b, label).toBeDefined()
  await invoke(b, 'onClick')
}
const moment = (m: Mounted<void>): string => el(m, (p) => p['data-testid'] === 'moment')?.text ?? ''
const tabs = (m: Mounted<void>): string[] =>
  m
    .elements()
    .filter((e) => e.props['role'] === 'tab')
    .map((e) => `${e.props['aria-selected'] === true ? '*' : ''}${e.text}`)

describe('NarrativeView', () => {
  it('tells a session in pages: a switcher, the page text, and a step-through that moves the cursor and the page', async () => {
    const m = await mountElement(<Page />)
    for (let n = 0; n < 400 && seen.live?.done !== true; n++) await frames.frame()
    expect(seen.live?.done).toBe(true)

    // The title, the switcher with the first page chosen, and the page's text with its inline code and bold.
    expect(m.elements().find((e) => e.tag === 'h3')?.text).toBe('A story')
    expect(tabs(m)).toEqual(['*1. Setup', '2. Middle', '3. Aside'])
    expect(m.elements().find((e) => e.tag === 'strong')?.text).toBe('first')
    expect(m.elements().find((e) => e.tag === 'code')?.text).toBe('code')
    expect(m.elements().filter((e) => e.tag === 'p').length).toBe(2)
    // At the end of the run (step 30) the viewer is past both of the page's moments, and the next is on the next page.
    expect(moment(m)).toBe('moment 2 of 2: ten')

    // Previous moves the cursor back through the page's marked steps, and is disabled at the first of the narrative.
    await press(m, 'Previous moment')
    expect([seen.at, moment(m)]).toEqual([10, 'moment 2 of 2: ten'])
    await press(m, 'Previous moment')
    expect([seen.at, moment(m)]).toEqual([5, 'moment 1 of 2: five'])
    expect(el(m, (p) => p['aria-label'] === 'Previous moment')?.props['disabled']).toBe(true)

    // Next goes on through the page, then into the next page's first moment, switching the page and keeping the cursor.
    await press(m, 'Next moment')
    expect(seen.at).toBe(10)
    await press(m, 'Next moment')
    expect([seen.at, tabs(m)[1], moment(m)]).toEqual([20, '*2. Middle', 'moment 1 of 1'])
    expect(m.text()).toContain('The middle page.')

    const preventDefault = vi.fn()
    // Alt+Arrow does the same from the keyboard on the narrative; Arrow alone, or typing in an input, does not.
    const key = (event: object): Promise<void> =>
      invoke(
        el(m, (p, tag) => tag === 'section' && p['aria-label'] === 'A story'),
        'onKeyDown',
        event,
      )
    await key({ key: 'ArrowLeft', altKey: false, target: {}, preventDefault })
    expect(seen.at).toBe(20)
    await key({
      key: 'ArrowLeft',
      altKey: true,
      target: new (globalThis as never as { HTMLInputElement: new () => object }).HTMLInputElement(),
      preventDefault,
    })
    expect(seen.at).toBe(20)
    expect(preventDefault).not.toHaveBeenCalled()
    await key({ key: 'ArrowLeft', altKey: true, target: {}, preventDefault })
    expect([seen.at, tabs(m)[0]]).toEqual([10, '*1. Setup'])
    // The browser is told not to take Alt+Arrow as history navigation.
    expect(preventDefault).toHaveBeenCalledTimes(1)

    // The tabs are one tab stop (the shown page's) and the arrows, Home and End move focus among them; the page is their panel.
    const tabEls = (): ReturnType<typeof m.elements> => m.elements().filter((e) => e.props['role'] === 'tab')
    expect(tabEls().map((e) => e.props['tabIndex'])).toEqual([0, -1, -1])
    const list = el(m, (p) => p['role'] === 'tablist')
    const arrow = async (key: string, from: number): Promise<string | undefined> => {
      const target = tabEls()[from]?.node
      await invoke(list, 'onKeyDown', { key, target, currentTarget: list?.node, preventDefault: vi.fn() })
      return m.focused()?.text
    }
    expect([
      await arrow('ArrowRight', 0),
      await arrow('ArrowLeft', 0),
      await arrow('End', 0),
      await arrow('Home', 2),
    ]).toEqual(['2. Middle', '3. Aside', '3. Aside', '1. Setup'])
    const panel = el(m, (p) => p['role'] === 'tabpanel')
    expect(panel?.props['aria-labelledby']).toBe(tabEls()[0]?.props['id'])
    expect(tabEls()[0]?.props['aria-controls']).toBe(panel?.props['id'])

    // A page switch from the tabs keeps the cursor, and a page with no moments says so.
    await invoke(
      el(m, (p) => p['role'] === 'tab' && p['aria-selected'] === false),
      'onClick',
    )
    expect([seen.at, tabs(m)[1]]).toEqual([10, '*2. Middle'])
    const aside = m.elements().filter((e) => e.props['role'] === 'tab')[2]
    await invoke(aside, 'onClick')
    expect(tabs(m)[2]).toBe('*3. Aside')
    expect(moment(m)).toBe('no marked steps on this page')
    expect(m.text()).toContain('No moments here.')
    await m.unmount()
  })
})
