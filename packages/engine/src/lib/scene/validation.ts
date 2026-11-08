/**
 * Checking a whole scene: `validateScene` walks a scene assembled by hand or read back from JSON and says where it
 * is not a drawing (a non-finite coordinate, a negative size, a scale of zero, a colour that is not one, a
 * duplicate id), the check the builders make one node at a time.
 */

import { OmaiError } from '../base/index.ts'
import { parsePaint } from './paint.ts'
import type { Camera } from './projection.ts'
import type { Item3D } from './three-d.ts'
import type { Paint, Scene, SceneNode } from './types.ts'

type Fields = Readonly<Record<string, unknown>>

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** The checks of one validation, which names the path of the node it fails at and remembers the ids it has seen. */
class Walk {
  readonly ids = new Set<string>()

  fail(at: string, message: string): never {
    throw new OmaiError('value', `validateScene: ${at}: ${message}`)
  }

  finite(at: string, name: string, v: unknown): void {
    if (!isNum(v)) this.fail(at, `${name} is ${String(v)}, expected a finite number`)
  }

  size(at: string, name: string, v: unknown): void {
    if (!(isNum(v) && v >= 0)) this.fail(at, `${name} is ${String(v)}, expected a size >= 0`)
  }

  paint(at: string, name: string, v: unknown): void {
    if (v === undefined) return
    try {
      parsePaint(v as Paint)
    } catch {
      this.fail(at, `${name} is ${JSON.stringify(v)}, expected a colour`)
    }
  }

  points(at: string, points: unknown, dims: number): void {
    if (!Array.isArray(points)) this.fail(at, 'points is not an array')
    for (const [i, p] of (points as unknown[]).entries())
      if (!(Array.isArray(p) && p.length === dims && p.every(isNum)))
        this.fail(at, `points[${i}] is ${JSON.stringify(p)}, expected ${dims} finite numbers`)
  }

  style(at: string, node: Fields): void {
    this.paint(at, 'fill', node['fill'])
    this.paint(at, 'stroke', node['stroke'])
    if (node['strokeWidth'] !== undefined) this.size(at, 'strokeWidth', node['strokeWidth'])
    for (const key of ['opacity', 'fillOpacity', 'strokeOpacity']) {
      const opacity = node[key]
      if (opacity !== undefined && !(isNum(opacity) && opacity >= 0 && opacity <= 1))
        this.fail(at, `${key} is ${String(opacity)}, expected a number in [0, 1]`)
    }
    const dash = node['dash']
    if (dash !== undefined && !(Array.isArray(dash) && dash.every((d) => isNum(d) && d >= 0)))
      this.fail(at, `dash is ${JSON.stringify(dash)}, expected lengths >= 0`)
    const id = node['id']
    if (id === undefined) return
    if (typeof id !== 'string') this.fail(at, `id is ${JSON.stringify(id)}, expected a string`)
    if (this.ids.has(id as string)) this.fail(at, `duplicate id '${String(id)}'`)
    this.ids.add(id as string)
  }

  camera(at: string, cam: Camera): void {
    const c = cam as unknown as Fields
    if (c['projection'] !== 'perspective' && c['projection'] !== 'orthographic')
      this.fail(at, `camera projection is ${JSON.stringify(c['projection'])}`)
    this.finite(at, 'camera azimuth', c['azimuth'])
    const elevation = c['elevation']
    if (!(isNum(elevation) && elevation >= -90 && elevation <= 90))
      this.fail(at, `camera elevation is ${String(elevation)}, expected a number in [-90, 90]`)
    for (const k of ['distance', 'extent'] as const)
      if (!(isNum(c[k]) && c[k] > 0)) this.fail(at, `camera ${k} is ${String(c[k])}, expected a number > 0`)
    const fov = c['fov']
    if (!(isNum(fov) && fov > 0 && fov < 180)) this.fail(at, `camera fov is ${String(fov)}, expected (0, 180)`)
    this.points(at, [c['target']], 3)
  }

  item3d(at: string, item: Item3D): void {
    const it = item as unknown as Fields
    switch (item.kind) {
      case 'point':
      case 'label':
        this.points(at, [it['at']], 3)
        break
      case 'segment':
        this.points(at, [it['from'], it['to']], 3)
        break
      case 'polyline':
      case 'face':
        this.points(at, it['points'], 3)
        break
      default:
        this.fail(at, `unknown 3-D item kind ${JSON.stringify(it['kind'])}`)
    }
    this.style(at, it)
  }

  image(at: string, n: Fields): void {
    const pixels = n['pixels'] as { width?: unknown; height?: unknown; rgba?: { length?: unknown } } | undefined
    const { width: w, height: h } = pixels ?? {}
    if (!(Number.isInteger(w) && Number.isInteger(h) && (w as number) >= 0 && (h as number) >= 0))
      this.fail(at, 'pixels have no integer width and height')
    const bytes = (w as number) * (h as number) * 4
    if (pixels?.rgba?.length !== bytes)
      this.fail(at, `pixels.rgba has ${String(pixels?.rgba?.length)} entries, expected ${bytes}`)
  }

  node(at: string, node: SceneNode): void {
    const n = node as unknown as Fields
    switch (node.kind) {
      case 'rect':
        for (const k of ['x', 'y']) this.finite(at, k, n[k])
        for (const k of ['width', 'height']) this.size(at, k, n[k])
        if (n['radius'] !== undefined) this.size(at, 'radius', n['radius'])
        break
      case 'circle':
        for (const k of ['cx', 'cy']) this.finite(at, k, n[k])
        this.size(at, 'r', n['r'])
        break
      case 'ellipse':
        for (const k of ['cx', 'cy']) this.finite(at, k, n[k])
        for (const k of ['rx', 'ry']) this.size(at, k, n[k])
        break
      case 'line':
        for (const k of ['x1', 'y1', 'x2', 'y2']) this.finite(at, k, n[k])
        break
      case 'path':
        this.points(at, n['points'], 2)
        break
      case 'text':
      case 'tex':
        for (const k of ['x', 'y']) this.finite(at, k, n[k])
        if (n['size'] !== undefined) this.size(at, 'size', n['size'])
        this.paint(at, 'fill', n['fill'])
        break
      case 'image':
        for (const k of ['x', 'y']) this.finite(at, k, n[k])
        for (const k of ['width', 'height']) this.size(at, k, n[k])
        this.image(at, n)
        break
      case 'group': {
        if (n['translate'] !== undefined) this.points(at, [n['translate']], 2)
        if (n['rotate'] !== undefined) this.finite(at, 'rotate', n['rotate'])
        const scale = n['scale']
        if (scale !== undefined && !(isNum(scale) && scale !== 0))
          this.fail(at, `scale is ${String(scale)}, expected a finite number other than 0`)
        this.style(at, n)
        for (const [i, child] of node.children.entries()) this.node(`${at}.children[${i}]`, child)
        return
      }
      case 'layer3d':
        for (const k of ['x', 'y']) this.finite(at, k, n[k])
        for (const k of ['width', 'height']) this.size(at, k, n[k])
        this.camera(at, node.camera)
        for (const [i, item] of node.scene.items.entries()) this.item3d(`${at}.scene.items[${i}]`, item)
        break
      default:
        this.fail(at, `unknown node kind ${JSON.stringify(n['kind'])}`)
    }
    this.style(at, n)
  }
}

/**
 * Check that a scene is a drawing: every size is a finite number >= 0, every coordinate and vertex finite, a group's
 * scale non-zero, every paint a colour, opacities in [0, 1], ids unique, images with as many bytes as their size
 * says, and 3-D layers with a valid camera and finite points. The builders check one node when it is made; this walks
 * a scene assembled by hand or deserialised, and renderers can call it at their entry points.
 *
 * @param s - The scene.
 * @returns The same scene.
 * @throws OmaiError with code `value` naming the path of the first node that is not valid (`validateScene:
 *   children[1].children[0]: r is -1, expected a size >= 0`).
 * @example
 * ```ts
 * import { circle, scene, validateScene } from 'omai-engine/scene'
 * validateScene(scene(10, 10, [circle(5, 5, 2)])).width // => 10
 * ```
 */
export function validateScene(s: Scene): Scene {
  const walk = new Walk()
  if ((s as { kind?: unknown }).kind !== 'scene') walk.fail('scene', 'not a scene (its kind is not "scene")')
  walk.size('scene', 'width', s.width)
  walk.size('scene', 'height', s.height)
  walk.paint('scene', 'background', s.background)
  if (!Array.isArray(s.children)) walk.fail('scene', 'children is not an array')
  for (const [i, child] of s.children.entries()) walk.node(`children[${i}]`, child)
  return s
}
