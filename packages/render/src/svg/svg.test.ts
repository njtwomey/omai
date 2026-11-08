import {
  camera,
  circle,
  ellipse,
  group,
  image,
  layer3d,
  line,
  path,
  point3d,
  rect,
  scene,
  scene3d,
  tex,
  text,
} from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { lineChart } from '../charts/lines.ts'
import { figure } from '../figure/figure.ts'
import { chartToSvg, composeSvg, exportSize, svgDataUrl, svgToPng } from './export.ts'
import { arrowHeadPoints, roundedPathData, smoothPathData, straightPathData } from './paths.ts'
import { escapeXml, sceneToSvg } from './svg.ts'

describe('sceneToSvg', () => {
  it('draws every node kind with the default paint rules', () => {
    const s = scene(
      100,
      50,
      [
        rect(0, 0, 10, 10, { radius: 2, title: 'r<' }),
        circle(5, 5, 2, { stroke: 'accent' }),
        ellipse(5, 5, 2, 1, { fill: 'none', stroke: '#123456', dash: [2, 1] }),
        { ...line(0, 0, 10, 10, { id: 'l' }), arrow: true },
        path(
          [
            [0, 0],
            [5, 5],
            [10, 0],
          ],
          { smooth: true, arrow: true },
        ),
        path(
          [
            [0, 0],
            [5, 5],
            [10, 0],
          ],
          { closed: true },
        ),
        text(1, 2, 'a & b', { baseline: 'middle', anchor: 'middle', weight: 'bold' }),
        tex(1, 2, '\\xvec', { anchor: 'end' }),
        image(
          0,
          0,
          4,
          4,
          [
            [0, 1],
            [1, 0],
          ],
          { smoothing: false },
        ),
        group([circle(0, 0, 1)], { translate: [3, 4], rotate: 10, scale: 2, opacity: 0.5 }),
        layer3d(scene3d([point3d([0, 0, 0])]), camera(), { x: 0, y: 0, width: 20, height: 20 }),
      ],
      { title: 'all', background: 'bg' },
    )
    const svg = sceneToSvg(s, { mode: 'dark', width: 200 })
    expect(svg).toContain('height="100"')
    expect(svg).toContain('<title>all</title>')
    expect(svg).toContain('fill="#0a0a0a"')
    expect(svg).toContain('<title>r&lt;</title>')
    expect(svg).toContain('stroke="#e5e5e5"')
    expect(svg).toContain('stroke-dasharray="2 1"')
    expect(svg).toContain('data-node-id="l"')
    expect(svg.match(/<polygon/g)?.length).toBe(2)
    expect(svg).toContain('a &amp; b')
    expect(svg).toContain('<math')
    expect(svg).toContain('data:image/png;base64,')
    expect(svg).toContain('image-rendering:pixelated')
    expect(svg).toContain('translate(3 4) rotate(10) scale(2)')
    expect(sceneToSvg(scene(1, 1, []), { background: 'none' })).not.toContain('<rect')
  })
  it('writes path data and arrow heads', () => {
    expect(straightPathData([])).toBe('')
    expect(
      smoothPathData([
        [0, 0],
        [1, 1],
      ]),
    ).toBe('M0,0L1,1')
    expect(
      smoothPathData(
        [
          [0, 0],
          [1, 1],
          [2, 0],
        ],
        true,
      ).endsWith('Z'),
    ).toBe(true)
    expect(arrowHeadPoints([0, 0], [0, 0], 2)).toBe('0,0 0,0 0,0')
    expect(escapeXml('<a>')).toBe('&lt;a&gt;')
  })
  it('writes rounded corners: a quarter circle at each right angle, limited to half the shorter segment', () => {
    const corner: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
    ]
    expect(roundedPathData(corner, 4)).toBe('M0,0L6,0C8.2092,0 10,1.7908 10,4L10,10')
    // Half of the shorter segment at most: 10 long segments give a radius of 5 for a request of 50.
    expect(roundedPathData(corner, 50)).toBe(roundedPathData(corner, 5))
    expect(roundedPathData(corner, 0)).toBe(straightPathData(corner))
    expect(roundedPathData(corner.slice(0, 2), 4)).toBe(straightPathData(corner.slice(0, 2)))
    expect(roundedPathData([...corner, [0, 10]], 3, true).endsWith('Z')).toBe(true)
    const svg = sceneToSvg(scene(20, 20, [path(corner, { stroke: 'fg', cornerRadius: 4, arrow: true })]))
    expect(svg).toContain('d="M0,0L6,0C8.2092,0 10,1.7908 10,4L10,10"')
    expect(svg.match(/<polygon/g)?.length).toBe(1)
  })
})

describe('figure export', () => {
  it('renders charts to SVG and composes figures', () => {
    const chart = chartToSvg(lineChart([{ y: [1, 2, 3] }]), { width: 300, height: 200, mode: 'dark' })
    expect(chart).toContain('<svg')
    const svg = composeSvg(400, 200, [
      { x: 0, y: 0, width: 300, height: 200, svg: `<?xml version="1.0"?>${chart}` },
      { x: 300, y: 0, width: 100, height: 200, html: '<b>x</b>' },
      { x: 0, y: 0, width: 1, height: 1 },
    ])
    expect(svg).not.toContain('<?xml')
    expect(svg).toContain('<svg x="0" y="0" width="300" height="200"')
    expect(svgDataUrl('<svg/>')).toContain('%3Csvg')
    expect(exportSize(figure(), 'page').scale).toBe(1)
    expect(exportSize(figure({ aspect: 1 }), 'slide')).toEqual({ width: 1080, height: 1080, scale: 1.688 })
    expect(exportSize(figure(), 'blog').width).toBe(1440)
  })
  it('refuses to rasterise outside a browser', async () => {
    await expect(svgToPng('<svg/>', 1, 1)).rejects.toThrow(/browser/)
  })
})
