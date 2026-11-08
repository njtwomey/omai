import type { Algorithm } from 'omai-engine/trace'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { figure } from '../figure/figure.ts'
import { mountElement, mountHook, resetGlobals } from './_harness.tsx'
import { useLiveRun, usePlayback } from './usePlayback.ts'
import { FramesPlayer } from './views/_frame.tsx'

vi.mock('./Chart.tsx', () => ({ Chart: (): null => null }))

afterEach(() => {
  resetGlobals()
  vi.unstubAllGlobals()
})

const counter: Algorithm<number> = { name: 'counter', config: {}, init: () => 0, step: (s) => s + 1 }
const frames = (): { count: number; option: () => object } => ({ count: 5, option: () => ({}) })

// What starts by itself: a playback asked to autoplay, a live run asked to autoplay, and a cycle view (which loops).
async function started(): Promise<[boolean, boolean, string | undefined]> {
  const a = await mountHook(() => usePlayback(10, { autoplay: true }))
  const playback = a.current.playing
  await a.unmount()
  const b = await mountHook(() => useLiveRun(counter, { maxSteps: 10, autoplay: true }))
  const live = b.current.controller.playing
  await b.unmount()
  const c = await mountElement(<FramesPlayer figure={figure()} frames={frames as never} filename="f" cycle />)
  const button = c.elements().find((e) => e.tag === 'button' && /^(Play|Pause)$/.test(String(e.props['aria-label'])))
  const cycle = button?.props['aria-label'] as string | undefined
  await c.unmount()
  return [playback, live, cycle]
}

describe('reduced motion', () => {
  it('stops anything from starting by itself when the reader prefers less motion, and starts it otherwise', async () => {
    expect(await started()).toEqual([true, true, 'Pause'])
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === '(prefers-reduced-motion: reduce)' }))
    expect(await started()).toEqual([false, false, 'Play'])
  })
})
