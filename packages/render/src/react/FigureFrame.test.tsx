import type { ReactElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { inAct, invoke, type Mounted, mountElement, resetGlobals } from './_harness.tsx'
import { FigureChromeProvider, FigureFrame, type FrameSize } from './FigureFrame.tsx'
import { Controls, Readouts } from './FigureParts.tsx'

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
      { browser: true },
    )
    expect(m.text()).toContain('Loss')
    expect(m.text()).toContain('over time')
    expect(labels(m)).toEqual(expect.arrayContaining(['Export', 'Expand']))
    expect(m.window.count('keydown')).toBe(0)

    // The export menu lists the formats; a choice goes to the handler with the frame element, and closes the menu.
    await press(m, 'Export')
    const items = m.elements().filter((e) => e.props['role'] === 'menuitem')
    expect(items.map((e) => e.text)).toEqual(['PNG', 'GIF'])
    // Focus moves into the menu; Escape closes it and gives focus back to the export button.
    expect(m.focused()?.text).toBe('PNG')
    await m.document.fire('keydown', { key: 'Escape', stopPropagation: () => undefined })
    expect(m.elements().some((e) => e.props['role'] === 'menu')).toBe(false)
    expect(m.focused()?.props['aria-label']).toBe('Export')
    await press(m, 'Export')
    await invoke(
      m.elements().find((e) => e.props['role'] === 'menuitem' && e.text === 'GIF'),
      'onClick',
    )
    expect(onExport).toHaveBeenCalledWith('gif', expect.anything())
    expect(m.elements().some((e) => e.props['role'] === 'menu')).toBe(false)

    // Expanded: a dialog whose content is the modal's size (92% of the viewport, the figure's aspect), listeners on.
    await press(m, 'Expand')
    const dialog = m.elements().find((e) => e.props['role'] === 'dialog')
    expect(dialog?.props['aria-labelledby']).toBeDefined()
    // Focus is in the dialog.
    expect(m.focused()?.props['role']).toBe('dialog')
    expect(labels(m)).toContain('Close')
    expect(sizes.at(-1)).toEqual({ width: 1104, height: 552, expanded: true })
    expect([m.window.count('keydown'), m.window.count('resize')]).toEqual([1, 1])
    await m.window.fire('keydown', { key: 'a' })
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(true)
    await m.window.fire('keydown', { key: 'Escape' })
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(false)
    // Focus is back on the Expand button.
    expect(m.focused()?.props['aria-label']).toBe('Expand')
    // Closing removes both listeners, and the page content is back at its own size.
    expect([m.window.count('keydown'), m.window.count('resize')]).toEqual([0, 0])
    expect(sizes.at(-1)).toEqual({ width: 200, height: 100, expanded: false })
    await m.unmount()
  })

  it('says why an export gave nothing, beside the export button, instead of doing nothing, and keeps the figure alive', async () => {
    const playback = { count: 0, fps: 10, frame: 0, seek: () => undefined, pause: () => undefined }
    const m = await mountElement(
      <FigureFrame title="Loss" width={200} height={100} exportFormats={['png', 'gif', 'mp4']} playback={playback}>
        <span>drawn</span>
      </FigureFrame>,
      { browser: true },
    )
    const choose = async (format: string): Promise<void> => {
      await press(m, 'Export')
      await invoke(
        m.elements().find((e) => e.props['role'] === 'menuitem' && e.text === format),
        'onClick',
      )
      // The recording runs as a chain of promises: let it settle.
      await inAct(() => undefined)
      await inAct(() => undefined)
    }
    const alerts = (): string[] => m.elements().flatMap((e) => (e.props['role'] === 'alert' ? [e.text] : []))
    expect(alerts()).toEqual([])
    // A figure with nothing to draw (no chart, scene or overlay in the stand-in tree) used to do nothing.
    await choose('PNG')
    expect(alerts()).toEqual(['export failed: the figure holds nothing to export'])
    // A recording that produced no frame fails in the encoder; its message is shown, and the readout is not left busy.
    await choose('GIF')
    expect(alerts()).toEqual(['export failed: encodeGif: need at least one frame'])
    expect(m.text()).not.toContain('encoding')
    // A format the frame cannot make without a handler says so; the next choice clears the old note first.
    await choose('MP4')
    expect(alerts()).toEqual(['export failed: mp4 needs an export handler or a playback'])
    expect(labels(m)).toContain('Export')
    await m.unmount()
  })
})

describe('FigureChromeProvider', () => {
  const frame = (chrome?: boolean): ReactElement => (
    <FigureFrame title="T" width={200} height={100} {...(chrome === undefined ? {} : { chrome })}>
      <span>x</span>
    </FigureFrame>
  )

  it('sets the default of the frames below it; the frame’s own prop wins, and the innermost provider', async () => {
    const hidden = await mountElement(<FigureChromeProvider chrome={false}>{frame()}</FigureChromeProvider>)
    expect(labels(hidden)).toEqual([])
    expect(hidden.text()).toContain('T')
    await hidden.unmount()
    const own = await mountElement(<FigureChromeProvider chrome={false}>{frame(true)}</FigureChromeProvider>)
    expect(labels(own)).toEqual(expect.arrayContaining(['Export', 'Expand']))
    await own.unmount()
    const nested = await mountElement(
      <FigureChromeProvider chrome={false}>
        <FigureChromeProvider chrome={true}>{frame()}</FigureChromeProvider>
      </FigureChromeProvider>,
    )
    expect(labels(nested)).toContain('Expand')
    await nested.unmount()
    // An undefined setting leaves the one above.
    const through = await mountElement(
      <FigureChromeProvider chrome={false}>
        <FigureChromeProvider chrome={undefined}>{frame()}</FigureChromeProvider>
      </FigureChromeProvider>,
    )
    expect(labels(through)).toEqual([])
    await through.unmount()
    const plain = await mountElement(frame())
    expect(labels(plain)).toContain('Expand')
    await plain.unmount()
  })
})

describe('FigureFrame slots', () => {
  it('draws the purpose, controls, readouts and caption in order around the figure, and describes the figure by its purpose', async () => {
    const m = await mountElement(
      <FigureFrame
        title="Loss"
        purpose="How the loss falls"
        controls={<button type="button">control</button>}
        readouts={[
          { label: 'epoch', value: 12 },
          { label: 'loss', value: '0.31' },
        ]}
        caption="the end"
        width={200}
        height={100}
      >
        <span>drawn</span>
      </FigureFrame>,
    )
    const text = m.text()
    const order = ['Loss', 'How the loss falls', 'control', 'drawn', 'epoch', '0.31', 'the end'].map((s) =>
      text.indexOf(s),
    )
    expect(order.every((i) => i >= 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    const figure = m.elements().find((e) => e.tag === 'figure')
    const purpose = m.elements().find((e) => e.text === 'How the loss falls')
    expect(figure?.props['aria-describedby']).toBe(purpose?.props['id'])
    expect(m.elements().some((e) => e.tag === 'dl')).toBe(true)
    await m.unmount()
  })

  it('draws nothing for slots not given, and no readouts list for an empty one', async () => {
    const m = await mountElement(
      <FigureFrame title="T" readouts={[]} width={200} height={100}>
        <span>x</span>
      </FigureFrame>,
    )
    expect(m.elements().some((e) => e.tag === 'dl' || e.tag === 'p')).toBe(false)
    expect(m.elements().find((e) => e.tag === 'figure')?.props['aria-describedby']).toBeUndefined()
    await m.unmount()
  })
})

describe('Readouts and Controls', () => {
  it('Readouts is a description list of label and value pairs with tabular numerals, nothing when empty', async () => {
    const m = await mountElement(
      <Readouts
        items={[
          { label: 'a', value: 1 },
          { label: 'b', value: 2 },
        ]}
      />,
    )
    expect(m.elements().map((e) => e.tag)).toEqual(['dl', 'div', 'dt', 'dd', 'div', 'dt', 'dd'])
    expect(m.elements()[0]?.props['style']).toMatchObject({ fontVariantNumeric: 'tabular-nums' })
    expect(
      m
        .elements()
        .filter((e) => e.tag === 'dd')
        .map((e) => e.text),
    ).toEqual(['1', '2'])
    await m.unmount()
    const empty = await mountElement(<Readouts items={[]} />)
    expect(empty.elements()).toEqual([])
    await empty.unmount()
  })

  it('Controls is a wrapping flex row, a named group when given a label', async () => {
    const m = await mountElement(
      <Controls label="Model">
        <span>one</span>
      </Controls>,
    )
    const row = m.elements()[0]
    expect(row?.props['style']).toMatchObject({ display: 'flex', flexWrap: 'wrap', gap: 12 })
    expect(row?.props['role']).toBe('group')
    expect(row?.props['aria-label']).toBe('Model')
    await m.unmount()
    const plain = await mountElement(<Controls>x</Controls>)
    expect(plain.elements()[0]?.props['role']).toBeUndefined()
    await plain.unmount()
  })
})

describe('FigureFrame permalink', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })
  const frame = (extra: object = {}): ReactElement => (
    <FigureFrame title="Loss over time!" width={200} height={100} {...extra}>
      <span>drawn</span>
    </FigureFrame>
  )
  const anchor = (m: Mounted<void>): ReturnType<Mounted<void>['elements']>[number] | undefined =>
    m.elements().find((e) => e.tag === 'a')

  it('links the title to #id (a slug of the title, or the given id) and gives the frame that id', async () => {
    const m = await mountElement(frame(), { browser: true })
    expect(anchor(m)?.props['href']).toBe('#loss-over-time')
    expect(m.elements().find((e) => e.tag === 'figure')?.props['id']).toBe('loss-over-time')
    expect(m.elements().find((e) => e.tag === 'figure')?.props['style']).toMatchObject({ scrollMarginTop: 16 })
    expect(m.text()).toContain('Loss over time!')
    await m.unmount()
    const given = await mountElement(frame({ id: 'fig-loss' }), { browser: true })
    expect(anchor(given)?.props['href']).toBe('#fig-loss')
    expect(given.elements().find((e) => e.tag === 'figure')?.props['id']).toBe('fig-loss')
    await given.unmount()
    // A title that is not a string has no slug: React's id, made fragment-safe.
    const node = await mountElement(frame({ title: <em>Loss</em> }), { browser: true })
    expect(String(anchor(node)?.props['href'])).toMatch(/^#figure-[A-Za-z0-9_-]+$/)
    await node.unmount()
  })

  it('permalink={false} leaves a plain title and no id, and the expand and export buttons stay', async () => {
    const m = await mountElement(frame({ permalink: false }), { browser: true })
    expect(anchor(m)).toBeUndefined()
    expect(m.elements().find((e) => e.tag === 'figure')?.props['id']).toBeUndefined()
    expect(m.elements().some((e) => e.props['role'] === 'status')).toBe(false)
    expect(labels(m)).toEqual(expect.arrayContaining(['Export', 'Expand']))
    await m.unmount()
    // Without a title there is nothing to link.
    const bare = await mountElement(
      <FigureFrame width={200} height={100}>
        <span>x</span>
      </FigureFrame>,
      { browser: true },
    )
    expect(anchor(bare)).toBeUndefined()
    await bare.unmount()
  })

  it('on a click sets the hash, scrolls, copies the full address and announces "Copied" politely for a moment', async () => {
    vi.useFakeTimers()
    const writeText = vi.fn(() => Promise.resolve())
    const location = { hash: '', origin: 'https://notes.test', pathname: '/a/b', search: '?q=1' }
    vi.stubGlobal('location', location)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const m = await mountElement(frame(), { browser: true })
    const status = (): string => m.elements().find((e) => e.props['role'] === 'status')?.text ?? 'missing'
    expect(m.elements().find((e) => e.props['role'] === 'status')?.props['aria-live']).toBe('polite')
    expect(status()).toBe('')
    const preventDefault = vi.fn()
    await invoke(anchor(m), 'onClick', { preventDefault })
    await inAct(() => undefined)
    expect(preventDefault).toHaveBeenCalled()
    expect(location.hash).toBe('loss-over-time')
    expect(writeText).toHaveBeenCalledWith('https://notes.test/a/b?q=1#loss-over-time')
    expect(status()).toBe('Copied')
    await inAct(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(status()).toBe('')
    await m.unmount()
  })

  it('still sets the hash where there is no clipboard, and does not throw when the copy is refused', async () => {
    const location = { hash: '', origin: 'https://notes.test', pathname: '/', search: '' }
    vi.stubGlobal('location', location)
    vi.stubGlobal('navigator', {})
    const m = await mountElement(frame(), { browser: true })
    await invoke(anchor(m), 'onClick', { preventDefault: () => undefined })
    expect(location.hash).toBe('loss-over-time')
    expect(m.elements().find((e) => e.props['role'] === 'status')?.text).toBe('')
    await m.unmount()
    vi.stubGlobal('navigator', { clipboard: { writeText: () => Promise.reject(new Error('denied')) } })
    const refused = await mountElement(frame(), { browser: true })
    await invoke(anchor(refused), 'onClick', { preventDefault: () => undefined })
    await inAct(() => undefined)
    expect(refused.elements().find((e) => e.props['role'] === 'status')?.text).toBe('')
    await refused.unmount()
  })

  it('shows the link icon on hover and focus only', async () => {
    const m = await mountElement(frame(), { browser: true })
    const iconOpacity = (): unknown => {
      const icon = m.elements().find((e) => e.props['aria-hidden'] === 'true' && e.tag === 'span')
      return (icon?.props['style'] as { opacity: number } | undefined)?.opacity
    }
    expect(iconOpacity()).toBe(0)
    await invoke(anchor(m), 'onFocus')
    expect(iconOpacity()).toBeGreaterThan(0)
    await invoke(anchor(m), 'onBlur')
    expect(iconOpacity()).toBe(0)
    await invoke(anchor(m), 'onPointerEnter')
    expect(iconOpacity()).toBeGreaterThan(0)
    await m.unmount()
  })

  it('scrolls to the frame and outlines it when the page opens at its hash, then lets go; other hashes do nothing', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('location', { hash: '#loss-over-time', origin: 'https://n.test', pathname: '/', search: '' })
    const outline = (m: Mounted<void>): string =>
      String((m.elements().find((e) => e.tag === 'figure')?.props['style'] as { outline: string } | undefined)?.outline)
    const m = await mountElement(frame(), { browser: true })
    expect(outline(m)).toContain('var(--wb-accent')
    expect(m.window.count('hashchange')).toBe(1)
    await inAct(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(outline(m)).toContain('transparent')
    await m.unmount()
    expect(m.window.count('hashchange')).toBe(0)
    vi.stubGlobal('location', { hash: '#elsewhere', origin: 'https://n.test', pathname: '/', search: '' })
    const other = await mountElement(frame(), { browser: true })
    expect(outline(other)).toContain('transparent')
    await other.unmount()
  })

  it('leaves the expand modal working: the title link is outside it and the dialog is labelled by the title', async () => {
    const m = await mountElement(frame(), { browser: true })
    await press(m, 'Expand')
    const dialog = m.elements().find((e) => e.props['role'] === 'dialog')
    const title = m.elements().find((e) => e.props['id'] === dialog?.props['aria-labelledby'])
    expect(title?.text).toBe('Loss over time!')
    await m.window.fire('keydown', { key: 'Escape' })
    expect(m.elements().some((e) => e.props['role'] === 'dialog')).toBe(false)
    await m.unmount()
  })
})
