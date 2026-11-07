import { afterEach, describe, expect, it, vi } from 'vitest'
import { invoke, type Mounted, mountElement, resetGlobals } from './_harness.tsx'
import { FigureFrame, type FrameSize } from './FigureFrame.tsx'

afterEach(() => {
  resetGlobals()
})

/** Press the element with this aria-label: call its `onClick` the way React would. */
async function press(m: Mounted<void>, label: string): Promise<void> {
  await invoke(
    m.elements().find((e) => e.props['aria-label'] === label),
    'onClick',
  )
}
const labels = (m: Mounted<void>): string[] => m.elements().flatMap((e) => (e.props['aria-label'] as string) ?? [])

describe('FigureFrame', () => {
  it('shows its title and caption, offers export and expand, and while expanded draws a larger modal that Escape closes', async () => {
    const sizes: FrameSize[] = []
    const onExport = vi.fn()
    const m = await mountElement(
      <FigureFrame
        title="Loss"
        caption="over time"
        width={200}
        height={100}
        exportFormats={['png', 'gif']}
        onExport={onExport}
      >
        {(size) => {
          sizes.push(size)
          return <span>drawn</span>
        }}
      </FigureFrame>,
    )
    expect(m.text()).toContain('Loss')
    expect(m.text()).toContain('over time')
    expect(labels(m)).toEqual(expect.arrayContaining(['Export', 'Expand']))
    expect(m.window.count('keydown')).toBe(0)

    // The export menu lists the formats; a choice goes to the handler with the frame element, and closes the menu.
    await press(m, 'Export')
    const items = m.elements().filter((e) => e.props['role'] === 'menuitem')
    expect(items.map((e) => e.text)).toEqual(['PNG', 'GIF'])
    await invoke(items[1], 'onClick')
    expect(onExport).toHaveBeenCalledWith('gif', expect.anything())
    expect(m.elements().some((e) => e.props['role'] === 'menu')).toBe(false)

    // Expanded: a dialog whose content is the modal's size (92% of the viewport, the figure's aspect), listeners on.
    await press(m, 'Expand')
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(true)
    expect(labels(m)).toContain('Close')
    expect(sizes.at(-1)).toEqual({ width: 1104, height: 552, expanded: true })
    expect([m.window.count('keydown'), m.window.count('resize')]).toEqual([1, 1])
    await m.window.fire('keydown', { key: 'a' })
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(true)
    await m.window.fire('keydown', { key: 'Escape' })
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(false)
    // Closing removes both listeners, and the page content is back at its own size.
    expect([m.window.count('keydown'), m.window.count('resize')]).toEqual([0, 0])
    expect(sizes.at(-1)).toEqual({ width: 200, height: 100, expanded: false })
    await m.unmount()
  })
})
