import { afterEach, describe, expect, it, vi } from 'vitest'
import { invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { menuClass, menuItemClass } from './classes.ts'
import { NumberField, type NumberFieldProps } from './NumberField.tsx'
import { numberLadder } from './number.ts'

afterEach(() => {
  resetGlobals()
})

// A strength shown as words: 0 is "none", 1 is "weak", 2 is "strong".
const WORDS = ['none', 'weak', 'strong']
const props: Omit<NumberFieldProps, 'onChange'> = {
  label: 'strength',
  value: 1,
  min: 0,
  max: 2,
  integer: true,
  format: (v) => WORDS[v] ?? String(v),
  parse: (t) => {
    const at = WORDS.indexOf(t.trim().toLowerCase())
    return at >= 0 ? at : /^\d+$/.test(t.trim()) ? Number(t) : null
  },
}

async function field(onChange = vi.fn()) {
  const m = await mountElement(<NumberField {...props} onChange={onChange} />)
  const input = () => m.elements().find((e) => e.tag === 'input')
  const shown = () => String(input()?.props['value'])
  const focus = () => invoke(input(), 'onFocus', { currentTarget: { select: () => undefined } })
  const type = (text: string) => invoke(input(), 'onChange', { target: { value: text } })
  return { m, input, shown, focus, type, onChange }
}

describe('NumberField format and parse', () => {
  it('shows the committed value through format, and the slider readout and the dropdown too', async () => {
    const { m, shown } = await field()
    expect(shown()).toBe('weak')
    expect(m.elements().some((e) => e.props['aria-valuetext'] === 'weak')).toBe(true)
    await m.unmount()
  })

  it('shows a draft exactly as typed while focused (never reformatted under the caret), and parses it as it is typed', async () => {
    const { m, shown, focus, type, input } = await field()
    await focus()
    expect(shown()).toBe('1')
    // Half a word is not a value yet: the draft stays as typed and the field is flagged, nothing is applied.
    await type('str')
    expect(shown()).toBe('str')
    expect(input()?.props['aria-invalid']).toBe(true)
    await type('Strong')
    expect(shown()).toBe('Strong')
    expect(input()?.props['aria-invalid']).toBeUndefined()
    await m.unmount()
  })

  it('commits the parsed draft on blur and shows it formatted again', async () => {
    const onChange = vi.fn()
    const { m, shown, focus, type, input } = await field(onChange)
    await focus()
    await type('strong')
    await invoke(input(), 'onBlur')
    expect(onChange).toHaveBeenCalledWith(2)
    // The parent has not changed `value` in this test, so the field shows the value it was given, formatted.
    expect(shown()).toBe('weak')
    await m.unmount()
  })

  it('reverts an unreadable draft on blur without calling onChange, and refuses it on Enter with its message', async () => {
    const onChange = vi.fn()
    const { m, shown, focus, type, input } = await field(onChange)
    await focus()
    await type('loud')
    await invoke(input(), 'onKeyDown', { key: 'Enter', preventDefault: () => undefined })
    expect(onChange).not.toHaveBeenCalled()
    expect(shown()).toBe('loud')
    await invoke(input(), 'onBlur')
    expect(onChange).not.toHaveBeenCalled()
    expect(shown()).toBe('weak')
    await m.unmount()
  })
})

describe('NumberField ladder popup', () => {
  // The rem of the first `calc(<n>rem*...)` after a prefix in a class string.
  const rem = (classes: string, prefix: string): number => {
    const found = new RegExp(`${prefix}\\[calc\\(([\\d.]+)rem`).exec(classes)
    if (found === null) throw new Error(`no ${prefix} in ${classes}`)
    return Number(found[1])
  }

  it('offers the bound itself as the last value: 0 to 2 ends ... 1.8, 2', async () => {
    const ladder = numberLadder({ min: 0, max: 2 })
    expect(ladder).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
    const picked = vi.fn()
    const m = await mountElement(<NumberField label="amplitude" value={1} min={0} max={2} onChange={picked} />, {
      browser: true,
    })
    const trigger = m.elements().find((e) => e.props['aria-label'] === 'Pick a value')
    await invoke(trigger, 'onClick')
    const options = m.elements().filter((e) => e.props['role'] === 'option')
    expect(options.map((o) => o.text)).toEqual(ladder.map(String))
    // The last value is there and picking it sets the bound.
    await invoke(options[10], 'onClick')
    expect(picked).toHaveBeenCalledWith(2)
    await m.unmount()
  })

  it('draws the popup tall enough for all 11 rows, so the last one is not clipped below a scroll', () => {
    // A row is its two vertical paddings and its line height; the panel adds its own padding and a border (about 2px).
    const row = 2 * rem(menuItemClass, 'py-') + rem(menuItemClass, '/')
    const panel = 2 * rem(menuClass, 'py-') + 2 / 16
    expect(rem(menuClass, 'max-h-')).toBeGreaterThanOrEqual(11 * row + panel)
  })
})
