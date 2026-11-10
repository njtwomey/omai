import { renderToString } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'
import { type FoundElement, invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { DraggableLine, type DraggableLineProps } from './DraggableLine.tsx'
import { InteractionOverlay } from './InteractionOverlay.tsx'
import { linearProjection } from './projection.ts'

afterEach(resetGlobals)

const projection = linearProjection({ x: [0, 10], y: [0, 1], width: 200, height: 100 })
const html = (props: Partial<DraggableLineProps> = {}): string =>
  renderToString(
    <InteractionOverlay projection={projection} width={200} height={100}>
      <DraggableLine axis="x" value={5} onChange={() => undefined} {...props} />
    </InteractionOverlay>,
  )

describe('DraggableLine handle', () => {
  it('is a bar across the line, 10 by 20, unless asked otherwise', () => {
    const out = html()
    expect(out).toMatch(/<rect[^>]*x="-5"[^>]*y="-10"[^>]*width="10"[^>]*height="20"/)
    expect(out).not.toContain('<polygon')
    expect(out).not.toContain('<circle')
  })

  it('takes a shape and a size: a circle of the diameter, a square of the side, a diamond of the width and height', () => {
    expect(html({ handleShape: 'circle', handleSize: 16 })).toContain('r="8"')
    expect(html({ handleShape: 'square', handleSize: 12 })).toMatch(/<rect[^>]*width="12"[^>]*height="12"/)
    expect(html({ handleShape: 'diamond', handleSize: 16 })).toContain('points="0,-8 8,0 0,8 -8,0"')
    // A diamond without a size is 10 across.
    expect(html({ handleShape: 'diamond' })).toContain('points="0,-5 5,0 0,5 -5,0"')
    // A bar's size is its width, and its length twice that.
    expect(html({ handleSize: 6 })).toMatch(/<rect[^>]*width="6"[^>]*height="12"/)
  })

  it('reads the handle object too, and the flat props win over it', () => {
    expect(html({ handle: { shape: 'diamond', size: 20 } })).toContain('points="0,-10 10,0 0,10 -10,0"')
    expect(html({ handle: { shape: 'diamond', size: 20 }, handleShape: 'circle', handleSize: 4 })).toContain('r="2"')
  })
})

describe('DraggableLine span', () => {
  const boxed = linearProjection({
    x: [0, 10],
    y: [0, 1],
    width: 200,
    height: 100,
    margins: { top: 10, right: 20, bottom: 30, left: 40 },
  })
  const lineOf = (projection: typeof boxed, props: Partial<DraggableLineProps>): string =>
    renderToString(
      <InteractionOverlay projection={projection} width={200} height={100}>
        <DraggableLine axis="x" value={5} onChange={() => undefined} {...props} />
      </InteractionOverlay>,
    )

  it('runs the whole overlay when the projection does not know its plotting box', () => {
    const { rect: _rect, ...bare } = boxed
    const out = lineOf(bare as typeof boxed, {})
    expect(out).toMatch(/y1="0"[^>]*y2="100"|y2="100"[^>]*y1="0"/)
  })

  it('stops at the plotting box when the projection knows it, so the line does not cross the axis labels', () => {
    const out = lineOf(boxed, {})
    // The box is y from 10 to 70 (the margins are 10 above and 30 below).
    expect(out).toMatch(/y1="10"[^>]*y2="70"|y2="70"[^>]*y1="10"/)
    expect(out).not.toMatch(/y2="100"/)
  })

  it('keeps an explicit span as given', () => {
    const out = lineOf(boxed, { span: [0, 1] })
    expect(out).toMatch(/y1="70"[^>]*y2="10"|y2="10"[^>]*y1="70"/)
  })
})

describe('DraggableLine label', () => {
  it('sits right of a vertical line, and flips to its left when it would run off the overlay', () => {
    const near = renderToString(
      <InteractionOverlay projection={projection} width={200} height={100}>
        <DraggableLine axis="x" value={9.8} onChange={() => undefined} label="mean μ" />
      </InteractionOverlay>,
    )
    expect(near).toMatch(/<text[^>]*text-anchor="end"/)
    const mid = renderToString(
      <InteractionOverlay projection={projection} width={200} height={100}>
        <DraggableLine axis="x" value={2} onChange={() => undefined} label="mean μ" />
      </InteractionOverlay>,
    )
    expect(mid).toMatch(/<text[^>]*text-anchor="start"/)
  })
})

describe('DraggableLine handleFill', () => {
  const handleFill = (props: Partial<DraggableLineProps>): string | undefined =>
    /<rect[^>]*x="-5"[^>]*fill="([^"]*)"|<rect[^>]*fill="([^"]*)"[^>]*x="-5"/.exec(
      html({ colour: 'red', ...props }),
    )?.[1]

  it('is the page colour by default and with `bg`, the colour with `solid`, or any CSS colour', () => {
    expect(handleFill({})).toBe('var(--wb-bg)')
    expect(handleFill({ handleFill: 'bg' })).toBe('var(--wb-bg)')
    expect(handleFill({ handleFill: 'solid' })).toBe('red')
    expect(handleFill({ handleFill: '#fc0' })).toBe('#fc0')
  })

  it('keeps `handle.fill` and lets the flat prop win over it', () => {
    expect(handleFill({ handle: { fill: 'solid' } })).toBe('red')
    expect(handleFill({ handle: { fill: 'hollow' } })).toBe('var(--wb-bg)')
    expect(handleFill({ handle: { fill: 'solid' }, handleFill: 'bg' })).toBe('var(--wb-bg)')
    expect(handleFill({ handle: { fill: 'hollow' }, handleFill: '#fc0' })).toBe('#fc0')
  })
})

describe('DraggableLine focus ring and keyStep', () => {
  const rings = (m: Awaited<ReturnType<typeof mountElement>>): FoundElement[] =>
    m.elements().filter((e) => e.props['data-focus-ring'] !== undefined)
  const slider = (m: Awaited<ReturnType<typeof mountElement>>): FoundElement => {
    const found = m.elements().find((e) => e.props['role'] === 'slider')
    if (found === undefined) throw new Error('no slider')
    return found
  }
  const key = (k: string, shiftKey = false): object => ({ key: k, shiftKey, preventDefault: () => undefined })
  const at = { clientX: 0, clientY: 0, pointerId: 1, currentTarget: { setPointerCapture: () => undefined } }
  const host = (value: number, props: Partial<DraggableLineProps>, got: number[]): ReturnType<typeof mountElement> =>
    mountElement(
      <InteractionOverlay projection={projection} width={200} height={100}>
        <DraggableLine
          axis="x"
          value={value}
          onChange={(v) => {
            got.push(v)
          }}
          {...props}
        />
      </InteractionOverlay>,
    )

  it('shows a ring on the line and its handle when focus comes from the keyboard, and not after a pointer press', async () => {
    const m = await host(5, {}, [])
    expect(rings(m)).toHaveLength(0)
    await invoke(slider(m), 'onPointerDown', at)
    await invoke(slider(m), 'onFocus')
    await invoke(slider(m), 'onPointerUp', at)
    expect(rings(m)).toHaveLength(0)
    await invoke(slider(m), 'onBlur')
    await invoke(slider(m), 'onFocus')
    // The line's own ring and the handle's.
    expect(rings(m)).toHaveLength(2)
    expect(rings(m).every((e) => String(e.props['stroke']).includes('--wb-accent'))).toBe(true)
    await invoke(slider(m), 'onBlur')
    expect(rings(m)).toHaveLength(0)
    await m.unmount()
  })

  it('shows the ring when an arrow key is pressed on a line the pointer focused', async () => {
    const m = await host(5, {}, [])
    await invoke(slider(m), 'onPointerDown', at)
    await invoke(slider(m), 'onFocus')
    await invoke(slider(m), 'onKeyDown', key('ArrowRight'))
    expect(rings(m)).toHaveLength(2)
    await m.unmount()
  })

  it('nudges by keyStep, apart from the snap step, Shift ten times', async () => {
    const got: number[] = []
    const m = await host(5, { step: 1, keyStep: 0.1 }, got)
    await invoke(slider(m), 'onKeyDown', key('ArrowRight'))
    await invoke(slider(m), 'onKeyDown', key('ArrowLeft', true))
    expect(got).toEqual([5.1, 4])
    await m.unmount()
  })

  it('nudges by step when keyStep is not given, and respects the bounds', async () => {
    const got: number[] = []
    const m = await host(5.3, { step: 0.5, max: 6 }, got)
    await invoke(slider(m), 'onKeyDown', key('ArrowRight'))
    await invoke(slider(m), 'onKeyDown', key('ArrowRight', true))
    expect(got).toEqual([6, 6])
    await m.unmount()
    const keyed: number[] = []
    const k = await host(5, { keyStep: 0.4, max: 5.5 }, keyed)
    await invoke(slider(k), 'onKeyDown', key('ArrowRight', true))
    expect(keyed).toEqual([5.5])
    await k.unmount()
  })
})
