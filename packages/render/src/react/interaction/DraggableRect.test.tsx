import { type ReactElement, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { type FoundElement, invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { DraggableRect, type DraggableRectProps } from './DraggableRect.tsx'
import { InteractionOverlay } from './InteractionOverlay.tsx'
import { linearProjection } from './projection.ts'
import type { DataRect } from './selection.ts'

let restore: (() => void) | undefined
afterEach(() => {
  restore?.()
  restore = undefined
  resetGlobals()
})

// A 100 by 100 pixel figure over 0..10 on both axes: ten pixels a unit, y up.
const projection = linearProjection({ x: [0, 10], y: [0, 10], width: 100, height: 100 })
const flippedX = linearProjection({ x: [10, 0], y: [0, 10], width: 100, height: 100 })
const invertedY = linearProjection({ x: [0, 10], y: [10, 0], width: 100, height: 100 })

const log: DataRect[] = []

function Host(props: {
  readonly start?: DataRect
  readonly projection?: typeof projection
  readonly rest?: Partial<DraggableRectProps>
}): ReactElement {
  const [rect, setRect] = useState<DataRect>(props.start ?? { x: [2, 6], y: [2, 8] })
  return (
    <InteractionOverlay projection={props.projection ?? projection} width={100} height={100}>
      <DraggableRect
        x={rect.x}
        y={rect.y}
        onChange={(r) => {
          log.push(r)
          setRect(r)
        }}
        {...props.rest}
      />
    </InteractionOverlay>
  )
}

type Mount = Awaited<ReturnType<typeof mountElement>>
const grips = (m: Mount): FoundElement[] => m.elements().filter((e) => e.props['data-rect-grip'] !== undefined)
const grip = (m: Mount, name: string): FoundElement => {
  const found = grips(m).find((e) => e.props['data-rect-grip'] === name)
  if (found === undefined) throw new Error(`no grip ${name}`)
  return found
}
const last = (): DataRect => log[log.length - 1] as DataRect

// The pointer at pixel (x, y) of the overlay: its stand-in has a 100 by 100 box at the origin, so pixels are client coordinates.
const at = (x: number, y: number): object => ({
  clientX: x,
  clientY: y,
  pointerId: 1,
  currentTarget: { setPointerCapture: () => undefined },
})

async function mount(element: ReactElement): Promise<Mount> {
  log.length = 0
  const m = await mountElement(element)
  // Give the overlay's svg a size, so the pointer maps to data.
  const proto = Object.getPrototypeOf(m.elements()[0]?.node) as { getBoundingClientRect: () => object }
  const original = proto.getBoundingClientRect
  restore = () => {
    proto.getBoundingClientRect = original
  }
  proto.getBoundingClientRect = () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 })
  return m
}

// Drag a grip from one pixel to another and release.
async function drag(m: Mount, name: string, from: [number, number], to: [number, number]): Promise<void> {
  await invoke(grip(m, name), 'onPointerDown', at(...from))
  await invoke(grip(m, name), 'onPointerMove', at(...to))
  await invoke(grip(m, name), 'onPointerUp', at(...to))
}

describe('DraggableRect grips', () => {
  it('shows eight grips and the body by default, four with `corners` or `edges`, and no body grip when it does not move', async () => {
    const all = await mount(<Host />)
    expect(grips(all).map((e) => e.props['data-rect-grip'])).toEqual([
      'body',
      'bottom left corner',
      'bottom right corner',
      'top left corner',
      'top right corner',
      'left edge',
      'right edge',
      'bottom edge',
      'top edge',
    ])
    await all.unmount()
    const corners = await mount(<Host rest={{ handles: 'corners' }} />)
    expect(grips(corners)).toHaveLength(5)
    await corners.unmount()
    const edges = await mount(<Host rest={{ handles: 'edges', moveable: false }} />)
    expect(grips(edges).map((e) => e.props['data-rect-grip'])).toEqual([
      'left edge',
      'right edge',
      'bottom edge',
      'top edge',
    ])
    await edges.unmount()
  })

  it('draws the grips at the rectangle, the corners and the middle of the edges', async () => {
    const m = await mount(<Host />)
    expect(grip(m, 'top right corner').props['transform']).toBe('translate(60, 20)')
    expect(grip(m, 'bottom left corner').props['transform']).toBe('translate(20, 80)')
    expect(grip(m, 'left edge').props['transform']).toBe('translate(20, 50)')
    expect(grip(m, 'top edge').props['transform']).toBe('translate(40, 20)')
    await m.unmount()
  })

  it('draws the grips in the shape and size asked: a bar lies along its edge and is a square on a corner', async () => {
    const m = await mount(<Host rest={{ handleShape: 'bar', handleSize: 6 }} />)
    const size = (name: string): [unknown, unknown] => {
      const [, shape] = (grip(m, name).node as { childNodes: { props: Record<string, unknown> }[] }).childNodes
      return [shape?.props['width'], shape?.props['height']]
    }
    expect(size('top right corner')).toEqual([6, 6])
    expect(size('left edge')).toEqual([6, 12])
    expect(size('top edge')).toEqual([12, 6])
    await m.unmount()
  })
})

describe('DraggableRect dragging', () => {
  it('a corner changes both ranges', async () => {
    const m = await mount(<Host />)
    await drag(m, 'top right corner', [60, 20], [80, 10])
    expect(last()).toEqual({ x: [2, 8], y: [2, 9] })
    await m.unmount()
  })

  it('an edge changes one range and ignores the other direction', async () => {
    const m = await mount(<Host />)
    await drag(m, 'right edge', [60, 50], [70, 90])
    expect(last()).toEqual({ x: [2, 7], y: [2, 8] })
    await m.unmount()
    const top = await mount(<Host />)
    await drag(top, 'top edge', [40, 20], [90, 10])
    expect(last()).toEqual({ x: [2, 6], y: [2, 9] })
    await top.unmount()
  })

  it('keeps the grab offset: the edge moves by what the pointer moved, not to where it is', async () => {
    const m = await mount(<Host />)
    await drag(m, 'right edge', [64, 50], [74, 50])
    expect(last().x).toEqual([2, 7])
    await m.unmount()
  })

  it('the body moves the whole rectangle', async () => {
    const m = await mount(<Host />)
    await drag(m, 'body', [40, 50], [50, 60])
    expect(last()).toEqual({ x: [3, 7], y: [1, 7] })
    await m.unmount()
  })

  it('does not move when it is not moveable', async () => {
    const m = await mount(<Host rest={{ moveable: false }} />)
    expect(grips(m).some((e) => e.props['data-rect-grip'] === 'body')).toBe(false)
    await m.unmount()
  })

  it('stays inside min and max, moving the body with its size kept', async () => {
    const rest = { min: { x: 0, y: 0 }, max: { x: 10, y: 10 } }
    const m = await mount(<Host rest={rest} />)
    await drag(m, 'top right corner', [60, 20], [200, -50])
    expect(last()).toEqual({ x: [2, 10], y: [2, 10] })
    await drag(m, 'bottom left corner', [20, 80], [-100, 300])
    expect(last()).toEqual({ x: [0, 10], y: [0, 10] })
    await m.unmount()
    const body = await mount(<Host rest={rest} />)
    await drag(body, 'body', [40, 50], [-300, 50])
    expect(last()).toEqual({ x: [0, 4], y: [2, 8] })
    await drag(body, 'body', [20, 50], [900, -900])
    expect(last()).toEqual({ x: [6, 10], y: [4, 10] })
    await body.unmount()
  })

  it('keeps the rectangle at least minSize, and normalised, when an edge is dragged past the other', async () => {
    const m = await mount(<Host rest={{ minSize: { x: 1, y: 0.5 } }} />)
    await drag(m, 'left edge', [20, 50], [90, 50])
    expect(last().x).toEqual([5, 6])
    await drag(m, 'top edge', [55, 20], [55, 99])
    expect(last().y).toEqual([2, 2.5])
    await m.unmount()
    const both = await mount(<Host rest={{ minSize: 2 }} />)
    await drag(both, 'right edge', [60, 50], [0, 50])
    expect(last().x).toEqual([2, 4])
    await both.unmount()
  })

  it('snaps the moved edges to a step, a number or one per axis', async () => {
    const m = await mount(<Host rest={{ step: 0.5 }} />)
    await drag(m, 'top right corner', [60, 20], [63, 17])
    expect(last()).toEqual({ x: [2, 6.5], y: [2, 8.5] })
    await m.unmount()
    const per = await mount(<Host rest={{ step: { x: 1 } }} />)
    await drag(per, 'right edge', [60, 50], [67, 50])
    expect(last().x).toEqual([2, 7])
    await drag(per, 'top edge', [40, 20], [40, 17])
    expect(last().y).toEqual([2, 8.3])
    await per.unmount()
  })

  it('does not call onChange when nothing changed', async () => {
    const m = await mount(<Host />)
    await drag(m, 'right edge', [60, 50], [60, 50])
    expect(log).toEqual([])
    await m.unmount()
  })

  it('Escape puts the rectangle back as the drag found it, and the drag is over', async () => {
    const m = await mount(<Host />)
    await invoke(grip(m, 'top right corner'), 'onPointerDown', at(60, 20))
    await invoke(grip(m, 'top right corner'), 'onPointerMove', at(80, 10))
    expect(last()).toEqual({ x: [2, 8], y: [2, 9] })
    await m.window.fire('keydown', { key: 'Escape', preventDefault: () => undefined })
    expect(last()).toEqual({ x: [2, 6], y: [2, 8] })
    const count = log.length
    await invoke(grip(m, 'top right corner'), 'onPointerMove', at(90, 0))
    expect(log).toHaveLength(count)
    expect(m.window.count('keydown')).toBe(0)
    await m.unmount()
  })
})

describe('DraggableRect axes', () => {
  it('stays normalised when given its ranges high end first', async () => {
    const m = await mount(<Host start={{ x: [6, 2], y: [8, 2] }} />)
    expect(grip(m, 'top right corner').props['transform']).toBe('translate(60, 20)')
    await drag(m, 'right edge', [60, 50], [70, 50])
    expect(last()).toEqual({ x: [2, 7], y: [2, 8] })
    await m.unmount()
  })

  it('follows a flipped x axis: the grips are where the data is, and the low end stays low', async () => {
    const m = await mount(<Host projection={flippedX} />)
    // x runs 10 to 0 across the figure, so the data range 2 to 6 is at pixels 80 down to 40.
    expect(grip(m, 'left edge').props['transform']).toBe('translate(40, 50)')
    expect(grip(m, 'right edge').props['transform']).toBe('translate(80, 50)')
    await drag(m, 'right edge', [80, 50], [90, 50])
    expect(last().x).toEqual([1, 6])
    await drag(m, 'left edge', [40, 50], [60, 50])
    expect(last().x).toEqual([1, 4])
    await m.unmount()
  })

  it('follows an inverted y axis', async () => {
    const m = await mount(<Host projection={invertedY} />)
    // y runs 10 at the bottom to 0 at the top: the range 2 to 8 is at pixels 20 to 80, so its low end is the top.
    expect(grip(m, 'top edge').props['transform']).toBe('translate(40, 20)')
    await drag(m, 'top edge', [40, 20], [40, 10])
    expect(last().y).toEqual([1, 8])
    await m.unmount()
  })
})

describe('DraggableRect keyboard', () => {
  const key = (k: string, shiftKey = false): object => ({ key: k, shiftKey, preventDefault: () => undefined })

  it('the arrows nudge the focused grip by step, Shift by ten steps', async () => {
    const m = await mount(<Host rest={{ step: 0.5 }} />)
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    expect(last()).toEqual({ x: [2, 6.5], y: [2, 8] })
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowLeft'))
    expect(last().x).toEqual([2, 6])
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight', true))
    expect(last().x).toEqual([2, 11])
    await m.unmount()
  })

  it('moves both edges of a corner and the whole rectangle from the body', async () => {
    const m = await mount(<Host rest={{ step: 0.5 }} />)
    await invoke(grip(m, 'top left corner'), 'onKeyDown', key('ArrowRight'))
    await invoke(grip(m, 'top left corner'), 'onKeyDown', key('ArrowUp'))
    expect(last()).toEqual({ x: [2.5, 6], y: [2, 8.5] })
    await invoke(grip(m, 'body'), 'onKeyDown', key('ArrowDown'))
    expect(last()).toEqual({ x: [2.5, 6], y: [1.5, 8] })
    await m.unmount()
  })

  it('ignores other keys, respects the bounds and the minimum size', async () => {
    const m = await mount(<Host rest={{ step: 1, max: { x: 7 }, minSize: 1 }} />)
    const before = log.length
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('a'))
    expect(log).toHaveLength(before)
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight', true))
    expect(last().x).toEqual([2, 7])
    await invoke(grip(m, 'left edge'), 'onKeyDown', key('ArrowRight', true))
    expect(last().x).toEqual([6, 7])
    await m.unmount()
  })

  it('moves one pixel of data without a step, the way the arrows point on a flipped axis', async () => {
    const m = await mount(<Host projection={flippedX} />)
    // The right edge on screen is the low end of the data; right moves it right, towards lower x, by 0.1 (one pixel).
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    expect(last().x[0]).toBeCloseTo(1.9, 9)
    await m.unmount()
    const up = await mount(<Host projection={invertedY} />)
    // y grows downwards: the top edge is the low end, and Up moves it up, towards lower y.
    await invoke(grip(up, 'top edge'), 'onKeyDown', key('ArrowUp'))
    expect(last().y[0]).toBeCloseTo(1.9, 9)
    await up.unmount()
  })
})

describe('DraggableRect handleFill', () => {
  // The grip's shape is the second child of its group (after the transparent hit area).
  const fillOf = (m: Mount, name: string): unknown => {
    const [, shape] = (grip(m, name).node as { childNodes: { props: Record<string, unknown> }[] }).childNodes
    return shape?.props['fill']
  }

  it('fills the grips with the page colour unless asked, and with the colour while one is dragged', async () => {
    const m = await mount(<Host rest={{ colour: 'red' }} />)
    expect(fillOf(m, 'left edge')).toBe('var(--wb-bg)')
    await invoke(grip(m, 'left edge'), 'onPointerDown', at(20, 50))
    expect(fillOf(m, 'left edge')).toBe('red')
    await m.unmount()
  })

  it('`solid` fills with the colour, a CSS colour with that colour, `bg` as the default', async () => {
    const solid = await mount(<Host rest={{ colour: 'red', handleFill: 'solid' }} />)
    expect(fillOf(solid, 'top edge')).toBe('red')
    expect(fillOf(solid, 'top right corner')).toBe('red')
    await solid.unmount()
    const custom = await mount(<Host rest={{ colour: 'red', handleFill: '#fc0' }} />)
    expect(fillOf(custom, 'top edge')).toBe('#fc0')
    await invoke(grip(custom, 'top edge'), 'onPointerDown', at(40, 20))
    expect(fillOf(custom, 'top edge')).toBe('#fc0')
    await custom.unmount()
    const bg = await mount(<Host rest={{ colour: 'red', handleFill: 'bg' }} />)
    expect(fillOf(bg, 'top edge')).toBe('var(--wb-bg)')
    await bg.unmount()
  })
})

describe('DraggableRect focus ring', () => {
  const rings = (m: Mount): FoundElement[] => m.elements().filter((e) => e.props['data-focus-ring'] !== undefined)
  const key = (k: string): object => ({ key: k, shiftKey: false, preventDefault: () => undefined })

  it('draws no ring until something has focus, and none on a pointer press', async () => {
    const m = await mount(<Host />)
    expect(rings(m)).toHaveLength(0)
    // A pointer press focuses the grip right after it: no ring.
    await invoke(grip(m, 'right edge'), 'onPointerDown', at(60, 50))
    await invoke(grip(m, 'right edge'), 'onFocus')
    await invoke(grip(m, 'right edge'), 'onPointerUp', at(60, 50))
    expect(rings(m)).toHaveLength(0)
    await m.unmount()
  })

  it('draws a ring in the accent colour on a grip focused from the keyboard, and takes it off on blur', async () => {
    const m = await mount(<Host />)
    await invoke(grip(m, 'top left corner'), 'onFocus')
    expect(rings(m)).toHaveLength(1)
    expect(rings(m)[0]?.props['stroke']).toContain('--wb-accent')
    // The ring is part of that grip's group.
    expect((grip(m, 'top left corner').node as { childNodes: unknown[] }).childNodes).toContain(rings(m)[0]?.node)
    await invoke(grip(m, 'top left corner'), 'onBlur')
    expect(rings(m)).toHaveLength(0)
    await m.unmount()
  })

  it('moves the ring with the focus, one at a time', async () => {
    const m = await mount(<Host />)
    await invoke(grip(m, 'top left corner'), 'onFocus')
    await invoke(grip(m, 'top left corner'), 'onBlur')
    await invoke(grip(m, 'left edge'), 'onFocus')
    expect(rings(m)).toHaveLength(1)
    expect((grip(m, 'left edge').node as { childNodes: unknown[] }).childNodes).toContain(rings(m)[0]?.node)
    await m.unmount()
  })

  it('a focus after a finished press is the keyboard again, and a key press on a pointer-focused grip shows the ring', async () => {
    const m = await mount(<Host />)
    await invoke(grip(m, 'right edge'), 'onPointerDown', at(60, 50))
    await invoke(grip(m, 'right edge'), 'onPointerUp', at(60, 50))
    // The press took no focus; a later Tab shows the ring.
    await invoke(grip(m, 'right edge'), 'onFocus')
    expect(rings(m)).toHaveLength(1)
    await invoke(grip(m, 'right edge'), 'onBlur')
    await invoke(grip(m, 'right edge'), 'onPointerDown', at(60, 50))
    await invoke(grip(m, 'right edge'), 'onFocus')
    expect(rings(m)).toHaveLength(0)
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    expect(rings(m)).toHaveLength(1)
    await m.unmount()
  })

  it('draws a thicker accent outline around the rectangle when the body has keyboard focus', async () => {
    const m = await mount(<Host />)
    await invoke(grip(m, 'body'), 'onFocus')
    expect(rings(m)).toHaveLength(1)
    expect(rings(m)[0]?.props).toMatchObject({ x: 17, y: 17, width: 46, height: 66 })
    await invoke(grip(m, 'body'), 'onBlur')
    expect(rings(m)).toHaveLength(0)
    await invoke(grip(m, 'body'), 'onPointerDown', at(40, 50))
    await invoke(grip(m, 'body'), 'onFocus')
    expect(rings(m)).toHaveLength(0)
    await m.unmount()
  })

  it('does not hide the focus with an inline outline of none and draws the ring itself', async () => {
    const m = await mount(<Host />)
    // The ring is drawn in SVG, so the browser's own outline is switched off on the grips.
    expect((grip(m, 'body').props['style'] as { outline: string }).outline).toBe('none')
    await m.unmount()
  })
})

describe('DraggableRect keyStep', () => {
  const key = (k: string, shiftKey = false): object => ({ key: k, shiftKey, preventDefault: () => undefined })

  it('a drag snaps to the coarse step and the keys nudge by the finer keyStep, Shift by ten', async () => {
    const m = await mount(<Host rest={{ step: 1, keyStep: 0.1 }} />)
    await drag(m, 'right edge', [60, 50], [73, 50])
    expect(last().x).toEqual([2, 7])
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    expect(last().x).toEqual([2, 7.1])
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    expect(last().x).toEqual([2, 7.2])
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowLeft', true))
    expect(last().x).toEqual([2, 6.2])
    await m.unmount()
  })

  it('takes one step per axis, the axis left out falling back to step', async () => {
    const m = await mount(<Host rest={{ step: 0.5, keyStep: { x: 0.1 } }} />)
    await invoke(grip(m, 'top right corner'), 'onKeyDown', key('ArrowRight'))
    await invoke(grip(m, 'top right corner'), 'onKeyDown', key('ArrowUp'))
    expect(last()).toEqual({ x: [2, 6.1], y: [2, 8.5] })
    await m.unmount()
  })

  it('works without a step, and moves the body whole', async () => {
    const m = await mount(<Host rest={{ keyStep: 0.25 }} />)
    await invoke(grip(m, 'body'), 'onKeyDown', key('ArrowLeft'))
    expect(last()).toEqual({ x: [1.75, 5.75], y: [2, 8] })
    await m.unmount()
  })

  it('still stops at the bounds and the minimum size', async () => {
    const m = await mount(<Host rest={{ keyStep: 0.3, max: { x: 6.2 }, minSize: 3.9 }} />)
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight', true))
    expect(last().x).toEqual([2, 6.2])
    await invoke(grip(m, 'left edge'), 'onKeyDown', key('ArrowRight', true))
    expect(last().x[1] - last().x[0]).toBeCloseTo(3.9, 9)
    await m.unmount()
  })

  it('leaves the keys on step when keyStep is not given', async () => {
    const m = await mount(<Host start={{ x: [2, 6.3], y: [2, 8] }} rest={{ step: 0.5 }} />)
    await invoke(grip(m, 'right edge'), 'onKeyDown', key('ArrowRight'))
    // 6.3 + 0.5 = 6.8 rounds to the grid, as it always did.
    expect(last().x).toEqual([2, 7])
    await m.unmount()
  })
})

describe('DraggableRect drawable', () => {
  const layer = (m: Mount): FoundElement => {
    const found = m.elements().find((e) => e.props['data-rect-draw'] !== undefined)
    if (found === undefined) throw new Error('no draw layer')
    return found
  }
  const press = (m: Mount, p: [number, number]): Promise<void> => invoke(layer(m), 'onPointerDown', at(...p))
  const move = (m: Mount, p: [number, number]): Promise<void> => invoke(layer(m), 'onPointerMove', at(...p))
  const release = (m: Mount, p: [number, number]): Promise<void> => invoke(layer(m), 'onPointerUp', at(...p))
  const sweep = async (m: Mount, from: [number, number], to: [number, number]): Promise<void> => {
    await press(m, from)
    await move(m, to)
    await release(m, to)
  }

  it('renders no layer unless drawable, so it swallows no clicks', async () => {
    const m = await mount(<Host />)
    expect(m.elements().some((e) => e.props['data-rect-draw'] !== undefined)).toBe(false)
    await m.unmount()
    const on = await mount(<Host rest={{ drawable: false }} />)
    expect(on.elements().some((e) => e.props['data-rect-draw'] !== undefined)).toBe(false)
    await on.unmount()
  })

  it('puts the layer under the body and the grips, covering the overlay', async () => {
    const m = await mount(<Host rest={{ drawable: true }} />)
    const order = m.elements().map((e) => e.node)
    const at0 = order.indexOf(layer(m).node)
    expect(at0).toBeGreaterThan(-1)
    expect(at0).toBeLessThan(order.indexOf(grip(m, 'body').node))
    expect(at0).toBeLessThan(order.indexOf(grip(m, 'left edge').node))
    expect(layer(m).props).toMatchObject({ x: 0, y: 0, width: 100, height: 100 })
    await m.unmount()
  })

  it('draws a new rectangle from a press outside the old one, following the pointer, and calls onDrawEnd on release', async () => {
    const ends: DataRect[] = []
    const m = await mount(<Host rest={{ drawable: true, onDrawEnd: (r) => ends.push(r) }} />)
    await press(m, [80, 90])
    expect(log).toEqual([])
    await move(m, [90, 60])
    expect(last()).toEqual({ x: [8, 9], y: [1, 4] })
    await move(m, [95, 40])
    expect(last()).toEqual({ x: [8, 9.5], y: [1, 6] })
    expect(ends).toEqual([])
    await release(m, [95, 40])
    expect(ends).toEqual([{ x: [8, 9.5], y: [1, 6] }])
    // The rectangle is the new one now: its grips moved.
    expect(grip(m, 'top right corner').props['transform']).toBe('translate(95, 40)')
    await m.unmount()
  })

  it('normalises a rectangle drawn up and to the left, low end first', async () => {
    const m = await mount(<Host rest={{ drawable: true }} />)
    await sweep(m, [80, 90], [30, 40])
    expect(last()).toEqual({ x: [3, 8], y: [1, 6] })
    await m.unmount()
  })

  it('normalises on a flipped x axis and an inverted y axis', async () => {
    const flipped = await mount(<Host projection={flippedX} rest={{ drawable: true }} />)
    // x runs 10 to 0: pixel 20 is x = 8 and pixel 10 is x = 9, and the pointer going left grows x.
    await sweep(flipped, [20, 90], [10, 60])
    expect(last()).toEqual({ x: [8, 9], y: [1, 4] })
    await flipped.unmount()
    const inverted = await mount(<Host projection={invertedY} rest={{ drawable: true }} />)
    // y runs 0 at the top: pixel 90 is y = 9 and pixel 70 is y = 7.
    await sweep(inverted, [80, 90], [90, 70])
    expect(last()).toEqual({ x: [8, 9], y: [7, 9] })
    await inverted.unmount()
  })

  it('stays inside min and max, from the press and from the pointer', async () => {
    const rest = { drawable: true, min: { x: 0, y: 0 }, max: { x: 9, y: 9 } }
    const high = await mount(<Host rest={rest} />)
    await sweep(high, [80, 90], [150, -50])
    expect(last()).toEqual({ x: [8, 9], y: [1, 9] })
    await high.unmount()
    const low = await mount(<Host rest={rest} />)
    await sweep(low, [80, 90], [-40, 130])
    expect(last()).toEqual({ x: [0, 8], y: [0, 1] })
    await low.unmount()
    // A press beyond the bound starts at the bound.
    const beyond = await mount(<Host rest={rest} />)
    await sweep(beyond, [95, 20], [75, 40])
    expect(last()).toEqual({ x: [7.5, 9], y: [6, 8] })
    await beyond.unmount()
  })

  it('snaps both corners to step, one per axis too', async () => {
    const m = await mount(<Host rest={{ drawable: true, step: 0.5 }} />)
    await sweep(m, [83, 87], [93, 57])
    expect(last()).toEqual({ x: [8.5, 9.5], y: [1.5, 4.5] })
    await m.unmount()
    const per = await mount(<Host rest={{ drawable: true, step: { x: 1 } }} />)
    await sweep(per, [83, 90], [93, 57])
    expect(last()).toEqual({ x: [8, 9], y: [1, 4.3] })
    await per.unmount()
  })

  it('keeps the rectangle at least minSize while it is drawn, turning the other way at a bound', async () => {
    const m = await mount(<Host rest={{ drawable: true, minSize: 2 }} />)
    await press(m, [80, 90])
    await move(m, [85, 85])
    expect(last()).toEqual({ x: [8, 10], y: [1, 3] })
    await release(m, [85, 85])
    await m.unmount()
    const bounded = await mount(<Host rest={{ drawable: true, minSize: 2, max: { x: 10 } }} />)
    await press(bounded, [90, 90])
    await move(bounded, [96, 90])
    expect(last().x).toEqual([7, 9])
    await release(bounded, [96, 90])
    await bounded.unmount()
  })

  it('a plain click, or a wobble under three pixels, draws nothing and keeps the rectangle', async () => {
    const ends: DataRect[] = []
    const m = await mount(<Host rest={{ drawable: true, onDrawEnd: (r) => ends.push(r) }} />)
    await press(m, [80, 90])
    await release(m, [80, 90])
    await press(m, [80, 90])
    await move(m, [81, 91])
    await release(m, [81, 91])
    expect(log).toEqual([])
    expect(ends).toEqual([])
    expect(grip(m, 'top right corner').props['transform']).toBe('translate(60, 20)')
    await m.unmount()
  })

  it('Escape puts the starting rectangle back and the draw is over', async () => {
    const ends: DataRect[] = []
    const m = await mount(<Host rest={{ drawable: true, onDrawEnd: (r) => ends.push(r) }} />)
    await press(m, [80, 90])
    await move(m, [95, 40])
    expect(last()).toEqual({ x: [8, 9.5], y: [1, 6] })
    await m.window.fire('keydown', { key: 'Escape', preventDefault: () => undefined })
    expect(last()).toEqual({ x: [2, 6], y: [2, 8] })
    const count = log.length
    await move(m, [99, 10])
    await release(m, [99, 10])
    expect(log).toHaveLength(count)
    expect(ends).toEqual([])
    expect(m.window.count('keydown')).toBe(0)
    await m.unmount()
  })

  it('Escape before the pointer moved far enough changes nothing', async () => {
    const m = await mount(<Host rest={{ drawable: true }} />)
    await press(m, [80, 90])
    await m.window.fire('keydown', { key: 'Escape', preventDefault: () => undefined })
    expect(log).toEqual([])
    await m.unmount()
  })

  it('ignores a press on the rectangle (when it cannot move) and leaves the body to move it otherwise', async () => {
    const m = await mount(<Host rest={{ drawable: true, moveable: false }} />)
    await press(m, [40, 50])
    await move(m, [70, 80])
    expect(log).toEqual([])
    await m.unmount()
    const movable = await mount(<Host rest={{ drawable: true }} />)
    await drag(movable, 'body', [40, 50], [50, 60])
    expect(last()).toEqual({ x: [3, 7], y: [1, 7] })
    await movable.unmount()
  })

  it('the grips still resize while drawable', async () => {
    const m = await mount(<Host rest={{ drawable: true }} />)
    await drag(m, 'top right corner', [60, 20], [80, 10])
    expect(last()).toEqual({ x: [2, 8], y: [2, 9] })
    await m.unmount()
  })
})
