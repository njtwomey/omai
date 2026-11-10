import { type ReactElement, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { inAct, invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { Player } from '../Player.tsx'
import { ThemeProvider } from '../theme.tsx'
import type { PlaybackController } from '../usePlayback.ts'
import {
  controlSizes,
  iconButtonClass,
  iconButtonClassFor,
  inputBoxClass,
  labelClass,
  menuClass,
  menuItemClass,
  text11Class,
} from './classes.ts'
import { ChevronDownIcon } from './icons.tsx'
import { NumberField } from './NumberField.tsx'
import { Select } from './Select.tsx'
import { CONTROL_SCALE_VAR, ControlScaleProvider, useControlScale } from './scale.tsx'

afterEach(() => {
  vi.unstubAllGlobals()
  resetGlobals()
})

const VAR = 'var(--omai-control-scale,1)'
const noop = (): void => undefined

/** The sizes in the `calc(<n><unit>*var(--omai-control-scale,1))` pieces of a class string, as written. */
const sizesIn = (classes: string): string[] =>
  [...classes.matchAll(/calc\(([\d.]+(?:rem|px))\*var\(--omai-control-scale,1\)\)/g)].map((m) => m[1] as string)

describe('the class tokens', () => {
  it('multiply every size by the variable, with 1 as the fallback, and are the standard sizes at 1', () => {
    expect(CONTROL_SCALE_VAR).toBe('--omai-control-scale')
    // Each token is the standard size of 0.1.10 (h-8 = 2rem and so on) times the variable.
    expect(sizesIn(controlSizes.sm.height)).toEqual(['1.75rem'])
    expect(sizesIn(controlSizes.md.height)).toEqual(['2rem'])
    expect(sizesIn(controlSizes.lg.height)).toEqual(['2.5rem'])
    expect(sizesIn(controlSizes.md.icon)).toEqual(['2rem'])
    expect(sizesIn(controlSizes.md.text)).toEqual(['0.75rem', '1rem'])
    expect(sizesIn(controlSizes.lg.text)).toEqual(['0.875rem', '1.25rem'])
    expect(sizesIn(controlSizes.md.track)).toEqual(['18px', '2rem'])
    expect(sizesIn(controlSizes.md.knob)).toEqual(['0.875rem'])
    expect(sizesIn(controlSizes.md.knobOn)).toEqual(['15px'])
    expect(sizesIn(menuItemClass)).toEqual(['0.625rem', '0.25rem', '0.75rem', '1rem'])
    expect(sizesIn(menuClass)).toEqual(['18rem', '6rem', '0.25rem'])
    expect(sizesIn(text11Class)).toEqual(['11px'])
    for (const size of ['sm', 'md', 'lg'] as const)
      for (const value of Object.values(controlSizes[size])) if (value.includes('calc')) expect(value).toContain(VAR)
    expect(sizesIn(iconButtonClass)).toEqual(['2rem'])
    expect(sizesIn(inputBoxClass)).toEqual(['2rem'])
    expect(sizesIn(labelClass)).toEqual(['0.75rem', '1rem'])
  })

  it('keep the size swap working and the default the same as md', () => {
    expect(iconButtonClassFor('md')).toBe(iconButtonClass)
    expect(sizesIn(iconButtonClassFor('lg'))).toEqual(['2.5rem'])
  })

  it('scale the icons too, from their own pixel size', async () => {
    const m = await mountElement(<ChevronDownIcon size={14} />)
    expect(m.elements()[0]?.props['style']).toEqual({ width: `calc(14px*${VAR})`, height: `calc(14px*${VAR})` })
    expect(m.elements()[0]?.props['width']).toBe(14)
    await m.unmount()
  })
})

describe('ControlScaleProvider', () => {
  it('sets the variable on one wrapper that does not disturb the layout, and tells the hook the factor', async () => {
    const m = await mountElement(
      <ControlScaleProvider scale={1.3} className="deck">
        <Grab />
      </ControlScaleProvider>,
    )
    const wrapper = m.elements()[0]
    expect(wrapper?.props['style']).toEqual({ display: 'contents', '--omai-control-scale': '1.3' })
    expect(wrapper?.props['className']).toBe('deck')
    expect(scale.seen).toBe(1.3)
    await m.unmount()
    const outside = await mountElement(<Grab />)
    expect(scale.seen).toBeUndefined()
    await outside.unmount()
  })
})

// The factor the nearest provider gave, as read by the latest render of `Grab`.
const scale: { seen: number | undefined } = { seen: undefined }
function Grab(): null {
  scale.seen = useControlScale()
  return null
}

const host: { open: () => void } = { open: () => undefined }
function Menu(): ReactElement {
  const [, set] = useState(0)
  host.open = () => set((n) => n + 1)
  return <Select label="loss" value="a" options={['a', 'b']} onChange={noop} appearance="menu" />
}

/** The popup of the select once its trigger is pressed. */
async function opened(el: ReactElement): Promise<{
  readonly popup: { readonly props: Readonly<Record<string, unknown>> } | undefined
  readonly close: () => Promise<void>
}> {
  const m = await mountElement(el, { browser: true })
  const trigger = m.elements().find((e) => e.props['aria-haspopup'] === 'listbox')
  await invoke(trigger, 'onClick')
  return { popup: m.elements().find((e) => e.props['role'] === 'listbox'), close: () => m.unmount() }
}

describe('popups', () => {
  it('carry the scale of the provider they were opened from, and nothing without one', async () => {
    const scaled = await opened(
      <ControlScaleProvider scale={1.3}>
        <Menu />
      </ControlScaleProvider>,
    )
    expect(scaled.popup?.props['style']).toMatchObject({ '--omai-control-scale': '1.3' })
    await scaled.close()
    const plain = await opened(<Menu />)
    expect(plain.popup).toBeDefined()
    expect(Object.keys(plain.popup?.props['style'] as object)).not.toContain('--omai-control-scale')
    await plain.close()
  })

  it('read the scale the host’s CSS gave the element they hang from', async () => {
    vi.stubGlobal('getComputedStyle', () => ({
      getPropertyValue: (name: string) => (name === '--omai-control-scale' ? ' 1.5 ' : ''),
    }))
    const hosted = await opened(<Menu />)
    expect(hosted.popup?.props['style']).toMatchObject({ '--omai-control-scale': '1.5' })
    await hosted.close()
  })

  it('draw their items in the scaled text size', async () => {
    const m = await mountElement(
      <ControlScaleProvider scale={1.3}>
        <Menu />
      </ControlScaleProvider>,
      { browser: true },
    )
    await invoke(
      m.elements().find((e) => e.props['aria-haspopup'] === 'listbox'),
      'onClick',
    )
    const item = m.elements().find((e) => e.props['role'] === 'option')
    expect(String(item?.props['className'])).toContain(`text-[length:calc(0.75rem*${VAR})]`)
    await m.unmount()
  })

  it('also carry the theme of a pinned mode, as the page’s own variables are not the ones the subtree sees', async () => {
    const themed = await opened(
      <ThemeProvider mode="light" tokens={{ accent: '#7c3aed' }}>
        <Menu />
      </ThemeProvider>,
    )
    expect(themed.popup?.props['data-theme']).toBe('light')
    expect(themed.popup?.props['style']).toMatchObject({ '--wb-accent': '#7c3aed', '--wb-bg': '#ffffff' })
    await themed.close()
  })
})

const controller = (): PlaybackController => ({
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
})

describe('Player size', () => {
  const classes = async (size?: 'sm' | 'md' | 'lg'): Promise<string[]> => {
    const m = await mountElement(<Player controller={controller()} stamp {...(size === undefined ? {} : { size })} />)
    const out = m.elements().map((e) => String(e.props['className'] ?? ''))
    await m.unmount()
    return out
  }

  it('is md without one, and sizes the buttons, the scrubber and the stamp otherwise', async () => {
    expect(await classes('md')).toEqual(await classes())
    const buttons = (list: string[]): string[] =>
      list.filter((c) => c.includes('inline-flex')).map((c) => sizesIn(c)[0] as string)
    expect(new Set(buttons(await classes()))).toEqual(new Set(['2rem']))
    expect(new Set(buttons(await classes('sm')))).toEqual(new Set(['1.75rem']))
    expect(new Set(buttons(await classes('lg')))).toEqual(new Set(['2.5rem']))
    // The scrubber's height follows, and the stamp's text.
    expect((await classes('lg')).some((c) => c.includes('h-[calc(2.5rem*') && c.includes('cursor-pointer'))).toBe(true)
    expect((await classes('lg')).some((c) => c.includes('calc(0.875rem*') && c.includes('font-mono'))).toBe(true)
    expect((await classes()).some((c) => c.includes('calc(0.75rem*') && c.includes('font-mono'))).toBe(true)
  })

  it('draws every size of its own with the variable, so the host’s scale reaches it', async () => {
    const list = await classes()
    const wrapper = list.find((c) => c.includes('items-center') && c.includes('min-w-[calc(16rem'))
    expect(wrapper).toContain(`min-w-[calc(16rem*${VAR})]`)
    expect(list.some((c) => c.includes(`size-[calc(0.75rem*${VAR})]`))).toBe(true)
    expect(list.some((c) => c.includes(`h-[calc(0.25rem*${VAR})]`))).toBe(true)
  })
})

describe('NumberField under a scale', () => {
  it('uses the same scaled classes as without one: the variable does the work, nothing is rewritten', async () => {
    const field = (): ReactElement => <NumberField label="k" value={3} onChange={noop} min={1} max={9} />
    const plain = await mountElement(field())
    const before = plain.elements().map((e) => String(e.props['className'] ?? ''))
    await plain.unmount()
    const scaled = await mountElement(<ControlScaleProvider scale={2}>{field()}</ControlScaleProvider>)
    const after = scaled
      .elements()
      .slice(1)
      .map((e) => String(e.props['className'] ?? ''))
    await scaled.unmount()
    expect(after).toEqual(before)
    await inAct(noop)
  })
})
