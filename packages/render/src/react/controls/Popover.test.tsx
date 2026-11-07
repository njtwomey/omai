import { type ReactElement, useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { inAct, mountElement, resetGlobals } from '../_harness.tsx'
import { Popover } from './Popover.tsx'

afterEach(() => {
  resetGlobals()
})

const onClose = vi.fn()
const host: { toggle: () => void } = { toggle: () => undefined }
function Host(): ReactElement {
  const anchor = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  host.toggle = () => setOpen((o) => !o)
  return (
    <>
      <button type="button" ref={anchor}>
        anchor
      </button>
      <Popover anchor={anchor} open={open} onClose={onClose} role="menu" label="Actions">
        <span>inside</span>
      </Popover>
    </>
  )
}

describe('Popover', () => {
  it('draws its content in the body only while open, closes on Escape and on a press outside, and leaves no listener behind', async () => {
    const m = await mountElement(<Host />, { browser: true })
    const listening = (): number[] => [
      m.document.count('pointerdown'),
      m.document.count('keydown'),
      m.window.count('scroll'),
      m.window.count('resize'),
    ]
    const menu = (): ReturnType<typeof m.elements>[number] | undefined =>
      m.elements().find((e) => e.props['role'] === 'menu')
    expect(menu()).toBeUndefined()
    expect(listening()).toEqual([0, 0, 0, 0])

    // Open: the labelled menu is in the body, hidden for its first measurement then placed, and four listeners are on.
    await inAct(host.toggle)
    expect(menu()?.props['aria-label']).toBe('Actions')
    expect(menu()?.text).toBe('inside')
    expect(listening()).toEqual([1, 1, 1, 1])

    // A press inside the menu or on the anchor, and any other key, leave it open; a press elsewhere and Escape close it.
    const anchor = m.elements().find((e) => e.text === 'anchor')
    await m.document.fire('pointerdown', { target: menu()?.node })
    await m.document.fire('pointerdown', { target: anchor?.node })
    await m.document.fire('keydown', { key: 'a' })
    expect(onClose).not.toHaveBeenCalled()
    await m.document.fire('pointerdown', { target: {} })
    expect(onClose).toHaveBeenCalledTimes(1)
    await m.document.fire('keydown', { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(2)

    // Closed again: the content leaves the body and all four listeners are removed.
    await inAct(host.toggle)
    expect(menu()).toBeUndefined()
    expect(listening()).toEqual([0, 0, 0, 0])
    await m.unmount()
  })
})
