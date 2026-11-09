import { afterEach, describe, expect, it } from 'vitest'
import { mountElement, resetGlobals } from '../_harness.tsx'
import { Choice } from './Choice.tsx'
import { controlWidth, iconButtonClass, iconButtonClassFor, inputBoxClass, inputBoxClassFor } from './classes.ts'
import { NumberField } from './NumberField.tsx'
import { Select } from './Select.tsx'
import { Toggle } from './Toggle.tsx'

afterEach(() => {
  resetGlobals()
})

const noop = (): void => undefined

describe('controlWidth', () => {
  it('keeps the default, and maps fill, auto, pixels and CSS lengths', () => {
    expect(controlWidth(undefined, 'w-full max-w-xs')).toEqual({ className: 'w-full max-w-xs' })
    expect(controlWidth('fill', 'w-full max-w-xs')).toEqual({ className: 'w-full' })
    expect(controlWidth('auto', 'w-full max-w-xs').className).toContain('w-fit')
    expect(controlWidth(176, 'x')).toEqual({ className: 'shrink-0', style: { width: 176 } })
    expect(controlWidth('12rem', 'x').style).toEqual({ width: '12rem' })
  })

  it('swaps the md height of the shared classes for the size asked', () => {
    expect(iconButtonClassFor('md')).toBe(iconButtonClass)
    expect(inputBoxClassFor('md')).toBe(inputBoxClass)
    expect(iconButtonClassFor('lg')).toContain('size-[calc(2.5rem*')
    expect(iconButtonClassFor('lg')).not.toContain('size-[calc(2rem*')
    expect(inputBoxClassFor('sm')).toContain('h-[calc(1.75rem*')
  })
})

/** The first element, the control's root, and its class and style. */
async function root(element: React.ReactElement): Promise<{ className: string; style: unknown; text: string }> {
  const m = await mountElement(element)
  const first = m.elements()[0]
  const out = { className: String(first?.props['className'] ?? ''), style: first?.props['style'], text: m.text() }
  await m.unmount()
  return out
}

describe('size and width on the controls', () => {
  const field = (extra: object = {}): React.ReactElement => (
    <NumberField label="k" value={3} onChange={noop} min={1} max={9} {...extra} />
  )
  const select = (extra: object = {}): React.ReactElement => (
    <Select label="loss" value="a" options={['a', 'b']} onChange={noop} appearance="menu" {...extra} />
  )
  const choice = (extra: object = {}): React.ReactElement => (
    <Choice label="mode" value="a" options={['a', 'b']} onChange={noop} {...extra} />
  )
  const toggle = (extra: object = {}): React.ReactElement => (
    <Toggle label="log" value={false} onChange={noop} {...extra} />
  )

  it('keeps the default widths when no width is given', async () => {
    expect((await root(field())).className).toContain('w-full min-w-36 max-w-xs')
    expect((await root(select())).className).toContain('w-full min-w-36 max-w-xs')
    expect((await root(choice())).className).toContain('w-full min-w-36 max-w-sm')
    expect((await root(toggle())).className).toContain('min-w-36 max-w-xs')
  })

  it('takes a pixel width as an inline style with no max-width fighting it, and fill and auto as classes', async () => {
    for (const make of [field, select, choice, toggle]) {
      const px = await root(make({ width: 176 }))
      expect(px.style).toEqual({ width: 176 })
      expect(px.className).not.toContain('max-w')
      expect(px.className).not.toContain('min-w')
      expect((await root(make({ width: '14rem' }))).style).toEqual({ width: '14rem' })
      expect((await root(make({ width: 'fill' }))).className).toContain('w-full')
      expect((await root(make({ width: 'fill' }))).className).not.toContain('max-w')
      expect((await root(make({ width: 'auto' }))).className).toContain('w-fit')
    }
  })

  it('keeps the caller’s className beside the width', async () => {
    expect((await root(field({ className: 'mt-2' }))).className).toContain('mt-2')
  })

  it('sizes the height and text: sm, md (the old look), lg', async () => {
    const html = async (make: (extra: object) => React.ReactElement, size?: string): Promise<string> => {
      const m = await mountElement(make(size === undefined ? {} : { size }))
      const all = m
        .elements()
        .map((e) => String(e.props['className'] ?? ''))
        .join(' ')
      await m.unmount()
      return all
    }
    for (const make of [field, select, choice]) {
      const dflt = await html(make)
      expect(await html(make, 'md')).toBe(dflt)
      expect(dflt).toMatch(/\b(?:h|size)-\[calc\(2rem\*/)
      expect(await html(make, 'sm')).toMatch(/\b(?:h|size)-\[calc\(1\.75rem\*/)
      const lg = await html(make, 'lg')
      expect(lg).toMatch(/\b(?:h|size)-\[calc\(2\.5rem\*/)
      expect(lg).toContain('calc(0.875rem*')
      expect(lg).not.toMatch(/\b(?:h|size)-\[calc\(2rem\*/)
    }
    expect(await html(toggle, 'md')).toBe(await html(toggle))
    expect(await html(toggle, 'sm')).toContain('w-[calc(26px*')
    expect(await html(toggle, 'lg')).toContain('w-[calc(2.5rem*')
  })

  it('wraps a label by default and truncates only when asked', async () => {
    const labelOf = async (element: React.ReactElement, tag: string): Promise<string> => {
      const m = await mountElement(element)
      const el = m
        .elements()
        .find((e) => e.tag === tag && typeof e.props['className'] === 'string' && e.text === 'a long label')
      await m.unmount()
      return String(el?.props['className'])
    }
    const long = 'a long label'
    for (const [make, tag] of [
      [(x: object) => <NumberField label={long} value={3} onChange={noop} {...x} />, 'label'],
      [
        (x: object) => <Select label={long} value="a" options={['a']} onChange={noop} appearance="menu" {...x} />,
        'label',
      ],
      [(x: object) => <Choice label={long} value="a" options={['a']} onChange={noop} {...x} />, 'span'],
      [(x: object) => <Toggle label={long} value={false} onChange={noop} {...x} />, 'label'],
    ] as const) {
      expect(await labelOf(make({}), tag)).not.toContain('truncate')
      expect(await labelOf(make({}), tag)).toContain('break-words')
      expect(await labelOf(make({ truncate: true }), tag)).toContain('truncate')
    }
  })
})
