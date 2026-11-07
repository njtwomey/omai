import { act, type RefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { SessionExport } from './SessionExport.tsx'
import type { LiveSession } from './useSession.ts'

// The page is not drawn here: a capture is a stub, and the moment the page has drawn is one the test lets pass.
const paint = vi.hoisted(() => ({ release: [] as (() => void)[] }))
const saved = vi.hoisted(() => ({ blobs: [] as string[] }))
vi.mock('../_download.ts', () => ({
  nextPaint: () => new Promise<void>((resolve) => paint.release.push(resolve)),
  saveBlob: (_content: unknown, _type: string, name: string) => saved.blobs.push(name),
  saveUrl: () => undefined,
}))
vi.mock('../FigureFrame.tsx', () => ({ figureSvg: () => '<svg xmlns="http://www.w3.org/2000/svg"/>' }))
vi.mock('../../svg/export.ts', async (original) => ({
  ...(await original<object>()),
  svgToPixels: async () => ({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) }),
}))

afterEach(() => {
  resetGlobals()
  paint.release.length = 0
  saved.blobs.length = 0
})

// Lets the recorder run until it waits for the page to draw (or ends).
const letRun = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 5))
const draw = (): Promise<void> =>
  act(async () => {
    paint.release.shift()?.()
    await letRun()
  })

function setup(playing: boolean) {
  const order: string[] = []
  const live = {
    head: 2,
    playing,
    pause: () => order.push('pause'),
    play: () => order.push('play'),
    session: {},
  } as unknown as LiveSession<never, never>
  const el = { offsetWidth: 10, offsetHeight: 10 } as unknown as HTMLElement
  const root = { current: el } as RefObject<HTMLElement | null>
  const onScrub = (t: number | null): number => order.push(`scrub ${t}`)
  const element = <SessionExport live={live} at={1} onScrub={onScrub} root={root} filename="p" />
  return { order, element }
}

const record = (m: Awaited<ReturnType<typeof mountElement>>) =>
  invoke(
    m.elements().find((e) => e.tag === 'button' && e.text === 'GIF'),
    'onClick',
  )

describe('SessionExport recording', () => {
  it('pauses playback while it records, scrubs through the steps, then restores the cursor and playing', async () => {
    const { order, element } = setup(true)
    const m = await mountElement(element, { browser: true })
    await invoke(
      m.elements().find((e) => e.props['aria-label'] === 'Export the page'),
      'onClick',
    )
    await record(m)
    for (let i = 0; i < 3; i++) await draw()
    await act(letRun)
    expect(order).toEqual(['pause', 'scrub 0', 'scrub 1', 'scrub 2', 'scrub 1', 'play'])
    expect(saved.blobs).toEqual(['p.gif'])
    await m.unmount()
  })

  it('stops when the menu unmounts: no more scrubbing, no cursor restore, no file', async () => {
    const { order, element } = setup(false)
    const m = await mountElement(element, { browser: true })
    await invoke(
      m.elements().find((e) => e.props['aria-label'] === 'Export the page'),
      'onClick',
    )
    await record(m)
    await draw()
    await m.unmount()
    for (let i = 0; i < 3; i++) await draw()
    expect(order).toEqual(['pause', 'scrub 0', 'scrub 1'])
    expect(saved.blobs).toEqual([])
  })
})
