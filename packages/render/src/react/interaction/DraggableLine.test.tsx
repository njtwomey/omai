import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DraggableLine, type DraggableLineProps } from './DraggableLine.tsx'
import { InteractionOverlay } from './InteractionOverlay.tsx'
import { linearProjection } from './projection.ts'

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
