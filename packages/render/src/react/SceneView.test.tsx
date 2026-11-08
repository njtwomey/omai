import { arrow, axes3d, circle, group, path, rect, scene, scene3d, text } from 'omai-engine/scene'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { tokens } from '../theme/tokens.ts'
import { Scene3DView } from './Scene3DView.tsx'
import { SceneView } from './SceneView.tsx'

describe('SceneView', () => {
  it('draws named nodes in the paints of `tones`, leaving the scene as it was', () => {
    const s = scene(100, 50, [
      rect(0, 0, 10, 10, { id: 'pole', fill: 'tab1' }),
      rect(20, 0, 10, 10, { id: 'cart', fill: 'tab0' }),
    ])
    const plain = renderToString(<SceneView scene={s} mode="light" />)
    const toned = renderToString(<SceneView scene={s} mode="light" tones={{ pole: 'tab3' }} />)
    expect(plain).toContain(tokens('light').tab1)
    expect(toned).not.toContain(tokens('light').tab1)
    expect(toned).toContain(tokens('light').tab3)
    expect(toned).toContain(tokens('light').tab0)
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

  it('projects a 3-D scene for orbiting', () => {
    const html = renderToString(<Scene3DView scene={scene3d(axes3d(1))} width={200} height={150} />)
    expect(html).toContain('role="application"')
    expect(html).toContain('viewBox="0 0 200 150"')
  })
})
