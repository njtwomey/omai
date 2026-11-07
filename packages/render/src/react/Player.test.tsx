import { afterEach, describe, expect, it, vi } from 'vitest'
import { invoke, mountElement, resetGlobals } from './_harness.tsx'
import { Player } from './Player.tsx'
import type { PlaybackController } from './usePlayback.ts'

afterEach(resetGlobals)

const controller = (over: Partial<PlaybackController> = {}): PlaybackController => ({
  frame: 3,
  count: 10,
  playing: false,
  fps: 10,
  open: false,
  play: vi.fn(),
  pause: vi.fn(),
  toggle: vi.fn(),
  step: vi.fn(),
  seek: vi.fn(),
  setFps: vi.fn(),
  reset: vi.fn(),
  ...over,
})

// What a key event's target is, as far as the player asks: what it is (a selector it matches), or nothing (the player).
const target = (what: 'button' | 'input' | 'option' | 'slider' | 'tab' | 'player'): object => ({
  closest: (selector: string) =>
    what !== 'player' &&
    selector.split(', ').includes(what === 'button' || what === 'input' ? what : `[role="${what}"]`)
      ? {}
      : null,
})
const press = (key: string, what: Parameters<typeof target>[0]): object => ({
  key,
  target: target(what),
  preventDefault: vi.fn(),
})

describe('Player keys', () => {
  it('Space, the arrows and Home act from the scrubber and the player, and not from a button, an option or a field', async () => {
    for (const [key, verb] of [
      [' ', 'toggle'],
      ['ArrowRight', 'step'],
      ['Home', 'seek'],
    ] as const) {
      const c = controller()
      const m = await mountElement(<Player controller={c} />)
      const root = m.elements()[0]
      for (const what of ['button', 'option', 'input'] as const) {
        const e = press(key, what)
        await invoke(root, 'onKeyDown', e)
        expect(c[verb]).not.toHaveBeenCalled()
        expect((e as { preventDefault: () => void }).preventDefault).not.toHaveBeenCalled()
      }
      for (const what of ['slider', 'player'] as const) await invoke(root, 'onKeyDown', press(key, what))
      expect(c[verb]).toHaveBeenCalledTimes(2)
      await m.unmount()
    }
  })

  it('steps on the arrow keys from anywhere for a while after playing, except from a slider, tab or option', async () => {
    const c = controller({ playing: true })
    const m = await mountElement(<Player controller={c} />)
    expect(m.window.count('keydown')).toBe(1)
    for (const what of ['slider', 'tab', 'option', 'input'] as const)
      await m.window.fire('keydown', press('ArrowRight', what))
    expect(c.step).not.toHaveBeenCalled()
    // The play button keeps the focus after a click, and the arrows still step.
    await m.window.fire('keydown', press('ArrowLeft', 'button'))
    expect(c.step).toHaveBeenCalledWith(-1)
    await m.unmount()
    expect(m.window.count('keydown')).toBe(0)
  })
})
