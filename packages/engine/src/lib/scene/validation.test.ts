import { describe, expect, it } from 'vitest'
import {
  arrow3d,
  camera,
  circle,
  ellipse,
  face3d,
  fitCamera,
  grid,
  group,
  image,
  label3d,
  layer3d,
  line,
  mergeStructures,
  path,
  point3d,
  polygon,
  polyline3d,
  project,
  projectPoint,
  rect,
  scene,
  scene3d,
  segment3d,
  structure,
  tex,
  text,
  timeline,
  triangle3d,
  validateScene,
} from './index.ts'

const node = (id: string): { id: string; label: string; role: 'other' } => ({ id, label: id, role: 'other' })

describe('the scene builders check their arguments where the drawing is made', () => {
  it('rejects non-finite coordinates, negative sizes, duplicate ids, a merged dangling edge and a bad grid or timeline', () => {
    const bad: [string, () => unknown][] = [
      ['NaN width', () => scene(Number.NaN, 5, [])],
      ['negative height', () => scene(5, -1, [])],
      ['infinite width', () => scene(Number.POSITIVE_INFINITY, 5, [])],
      ['negative rect width', () => rect(0, 0, -10, 5)],
      ['NaN rect corner', () => rect(Number.NaN, 0, 1, 1)],
      ['negative radius', () => circle(0, 0, -1)],
      ['NaN centre', () => circle(0, Number.NaN, 1)],
      ['negative ellipse radius', () => ellipse(0, 0, 3, -1)],
      ['infinite line end', () => line(0, 0, Number.POSITIVE_INFINITY, 1)],
      ['no columns', () => grid([scene(1, 1, [])], 0, [10, 10])],
      ['fractional columns', () => grid([scene(1, 1, [])], 1.5, [10, 10])],
      ['empty cell', () => grid([scene(1, 1, [])], 1, [0, 10])],
      ['duplicate node ids', () => structure('g', [node('a'), node('a')], [])],
      [
        'merged edge to a missing node',
        () => mergeStructures([structure('g', [node('a')], [{ from: 'a', to: 'zzz' }])]),
      ],
      [
        'path with a NaN vertex',
        () =>
          path([
            [0, 0],
            [Number.NaN, 1],
          ]),
      ],
      ['path with a three-number vertex', () => path([[0, 0, 0] as unknown as [number, number]])],
      [
        'polygon with an infinite vertex',
        () =>
          polygon([
            [0, 0],
            [1, Number.POSITIVE_INFINITY],
            [1, 1],
          ]),
      ],
      ['text at NaN', () => text(Number.NaN, 0, 'a')],
      ['text of negative size', () => text(0, 0, 'a', { size: -1 })],
      ['tex at an infinite point', () => tex(0, Number.POSITIVE_INFINITY, 'x')],
      [
        'image with a negative width',
        () => image(0, 0, -1, 1, { width: 1, height: 1, rgba: new Uint8ClampedArray(4) }),
      ],
      ['group with scale 0', () => group([], { scale: 0 })],
      ['group with a NaN translation', () => group([], { translate: [Number.NaN, 0] })],
      ['group with an infinite rotation', () => group([], { rotate: Number.POSITIVE_INFINITY })],
      ['3-D point at NaN', () => point3d([0, Number.NaN, 0])],
      ['3-D segment to infinity', () => segment3d([0, 0, 0], [0, 0, Number.POSITIVE_INFINITY])],
      ['3-D arrow with a NaN tail', () => arrow3d([Number.NaN, 0, 0], [1, 0, 0])],
      [
        '3-D polyline with a NaN point',
        () =>
          polyline3d([
            [0, 0, 0],
            [Number.NaN, 0, 0],
          ]),
      ],
      ['3-D triangle with a NaN vertex', () => triangle3d([0, 0, 0], [1, 0, 0], [0, Number.NaN, 0])],
      [
        '3-D face with a NaN vertex',
        () =>
          face3d([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, Number.NaN],
          ]),
      ],
      ['3-D label at NaN', () => label3d([Number.NaN, 0, 0], 'x')],
      ['3-D layer with a negative box', () => layer3d(scene3d([]), camera(), { x: 0, y: 0, width: -1, height: 1 })],
      ['camera elevation 200', () => camera({ elevation: 200 })],
      ['camera azimuth NaN', () => camera({ azimuth: Number.NaN })],
      ['camera target at infinity', () => camera({ target: [0, Number.POSITIVE_INFINITY, 0] })],
      ['camera at infinite distance', () => camera({ distance: Number.POSITIVE_INFINITY })],
      ['negative fps', () => timeline([], { fps: -1 })],
      ['NaN step', () => timeline([{ kind: 'scene', step: Number.NaN, scene: scene(1, 1, []) }])],
    ]
    for (const [name, make] of bad) expect(make, name).toThrow(expect.objectContaining({ code: 'value' }))
    // Zero sizes are drawings of nothing, not errors; a valid structure and timeline are built as before.
    expect(scene(0, 0, []).width).toBe(0)
    expect(rect(0, 0, 0, 0).width).toBe(0)
    expect(circle(0, 0, 0).r).toBe(0)
    expect(structure('g', [node('a'), node('b')], [{ from: 'a', to: 'b' }]).edges).toHaveLength(1)
    expect(structure('part', [node('a')], [{ from: 'a', to: 'elsewhere' }]).edges).toHaveLength(1)
    expect(timeline([{ kind: 'scene', step: 2, scene: scene(1, 1, []) }]).fps).toBe(10)
  })

  it('validateScene walks a scene assembled by hand and names the path of the first fault', () => {
    const ok = scene(10, 10, [
      group(
        [
          circle(5, 5, 2, { id: 'a' }),
          path([
            [0, 0],
            [1, 1],
          ]),
        ],
        { translate: [1, 1] },
      ),
    ])
    expect(validateScene(ok)).toBe(ok)
    const bad = (children: readonly unknown[], pattern: RegExp, width = 10): void =>
      expect(() => validateScene({ kind: 'scene', width, height: 10, children } as never)).toThrow(
        expect.objectContaining({ code: 'value', message: expect.stringMatching(pattern) }),
      )
    bad(
      [{ kind: 'group', children: [{ kind: 'circle', cx: 0, cy: 0, r: -1 }] }],
      /children\[0\]\.children\[0\]: r is -1/,
    )
    bad([{ kind: 'rect', x: 0, y: 0, width: Number.NaN, height: 1 }], /children\[0\]: width is NaN/)
    bad([{ kind: 'path', points: [[0, 0], [1]] }], /points\[1\] is \[1\]/)
    bad([{ kind: 'group', children: [], scale: 0 }], /scale is 0/)
    bad([{ kind: 'circle', cx: 0, cy: 0, r: 1, fill: '#zzz' }], /fill is "#zzz", expected a colour/)
    bad([{ kind: 'circle', cx: 0, cy: 0, r: 1, opacity: 2 }], /opacity is 2/)
    bad([circle(0, 0, 1, { id: 'x' }), circle(1, 1, 1, { id: 'x' })], /duplicate id 'x'/)
    bad([{ kind: 'blob' }], /unknown node kind "blob"/)
    bad(
      [{ kind: 'image', x: 0, y: 0, width: 1, height: 1, pixels: { width: 2, height: 1, rgba: {} } }],
      /pixels.rgba has undefined/,
    )
    bad([], /width is -1/, -1)
    expect(() => validateScene({ kind: 'structure' } as never)).toThrow(/not a scene/)
    const layer = layer3d(scene3d([point3d([0, 0, 0])]), camera(), { x: 0, y: 0, width: 5, height: 5 })
    expect(validateScene(scene(10, 10, [layer])).children).toHaveLength(1)
    bad([{ ...layer, camera: { ...layer.camera, elevation: 120 } }], /camera elevation is 120/)
  })

  it('degenerate cameras: one point keeps the distance, a segment behind the eye is cut, a NaN point is dropped', () => {
    const cam = camera({ azimuth: -90, elevation: 0, distance: 5 })
    const viewport = { width: 100, height: 100 }
    // fitCamera of one point (or coincident ones) aims at it and keeps its own distance and extent.
    const fitted = fitCamera(cam, scene3d([point3d([1, 1, 1]), point3d([1, 1, 1])]))
    expect([fitted.target, fitted.distance, fitted.extent]).toEqual([[1, 1, 1], cam.distance, cam.extent])
    expect(projectPoint(fitted, [1, 1, 1], viewport)).toEqual({ x: 50, y: 50, depth: 5 })
    expect(fitCamera(cam, scene3d([])).target).toEqual([0, 0, 0])
    // A segment from behind the eye to the target: the part in front is drawn, an arrow loses its head at the cut.
    const items = [segment3d([0, -6, 0], [0, 0, 0]), arrow3d([0, -6, 0], [0, 0, 0]), arrow3d([0, 0, 0], [0, -6, 0])]
    const long = project(scene3d(items), cam, viewport)
    expect(long.children.map((n) => n.kind)).toEqual(['line', 'line', 'line'])
    // Only the arrow whose head is still the original end keeps it.
    expect(long.children.filter((n) => (n as { arrow?: boolean }).arrow === true)).toHaveLength(1)
    // A segment wholly behind the eye is gone; a face with one vertex behind is clipped to a polygon, not dropped.
    expect(project(scene3d([segment3d([0, -7, 0], [0, -6, 0])]), cam, viewport).children).toHaveLength(0)
    const clipped = project(scene3d([triangle3d([0, -6, 0], [1, 0, 0], [-1, 0, 0])]), cam, viewport).children
    expect(clipped.map((n) => n.kind)).toEqual(['path'])
    expect((clipped[0] as unknown as { points: unknown[] }).points.length).toBe(4)
    expect(project(scene3d([triangle3d([0, -6, 0], [1, -7, 0], [-1, -8, 0])]), cam, viewport).children).toHaveLength(0)
    // A non-finite coordinate in a hand-made item has no image.
    const nan = { kind: 'point' as const, at: [Number.NaN, 0, 0] as const }
    expect(project(scene3d([nan]), cam, viewport).children).toHaveLength(0)
    expect(projectPoint(cam, [0, 0, Number.POSITIVE_INFINITY], viewport)).toBeUndefined()
  })
})
