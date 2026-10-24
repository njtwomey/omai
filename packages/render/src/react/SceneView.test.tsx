import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { arrow, circle, group, path, rect, scene, text } from 'whitebox-engine/scene'
import { tokens } from '../theme/tokens.ts'
import { arrowHeadPoints, resolvePaint, SceneView, smoothPathData } from './SceneView.tsx'

describe('SceneView', () => {
  it('resolves theme tokens and passes literal colours through', () => {
    expect(resolvePaint('tab0', tokens('light'))).toBe(tokens('light').tab0)
    expect(resolvePaint('#123456', tokens('dark'))).toBe('#123456')
    expect(resolvePaint('none', tokens('dark'))).toBe('none')
    expect(resolvePaint(undefined, tokens('dark'))).toBeUndefined()
  })

  it('makes smooth paths through every point and arrow heads at the tip', () => {
    const d = smoothPathData([
      [0, 0],
      [10, 10],
      [20, 0],
    ])
    expect(d.startsWith('M0,0')).toBe(true)
    expect(d.endsWith('20,0')).toBe(true)
    expect(arrowHeadPoints([0, 0], [10, 0], 4).startsWith('10,0')).toBe(true)
  })

  it('draws a scene as SVG with node ids', () => {
    const s = scene(100, 50, [
      rect(0, 0, 10, 10, { id: 'r', fill: 'tab1' }),
      circle(20, 20, 5),
      arrow(0, 40, 50, 40),
      path(
        [
          [0, 0],
          [5, 5],
          [10, 0],
        ],
        { smooth: true },
      ),
      group([text(5, 5, 'hi')], { translate: [10, 10] }),
    ])
    const html = renderToString(<SceneView scene={s} mode="light" highlight={['r']} />)
    expect(html).toContain('viewBox="0 0 100 50"')
    expect(html).toContain('data-node-id="r"')
    expect(html).toContain(tokens('light').accent)
    expect(html).toContain('>hi</text>')
  })
})
