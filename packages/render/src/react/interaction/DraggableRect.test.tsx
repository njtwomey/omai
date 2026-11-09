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
