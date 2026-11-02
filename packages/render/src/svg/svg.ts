/**
 * A scene as a standalone SVG document: `sceneToSvg` resolves theme tokens for the colour mode, projects 3-D layers,
 * sets TeX labels as MathML in foreign objects and embeds images as PNG data URLs, so a scene renders without a
 * browser.
 */

import { type Paint, projectLayers, type Scene, type SceneNode, type Style } from 'mdml-engine/scene'
import { texToHtml } from '../maths/tex.ts'
import { type ColourMode, resolvePaint } from '../theme/tokens.ts'
import { arrowHeadPoints, smoothPathData, straightPathData } from './paths.ts'
import { pngDataUrl } from './png.ts'

/**
 * Escape text for SVG and XML content and attribute values.
 *
 * @param text - The text.
 * @returns The escaped text.
 * @example
 * ```ts
 * import { escapeXml } from 'mdml-render/svg'
 * escapeXml('a < b & "c"') // => 'a &lt; b &amp; &quot;c&quot;'
 * ```
 */
export function escapeXml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function attrs(a: Readonly<Record<string, string | number | undefined>>): string {
  return Object.entries(a)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => ` ${k}="${escapeXml(String(v))}"`)
    .join('')
}

function paint(p: Paint | undefined, mode: ColourMode): string | undefined {
  return p === undefined ? undefined : resolvePaint(p, mode)
}

function stroke(s: Style, mode: ColourMode, fallback?: Paint): Record<string, string | number | undefined> {
  return {
    stroke: paint(s.stroke ?? fallback, mode) ?? 'none',
    'stroke-width': s.strokeWidth ?? 1,
    'stroke-dasharray': s.dash?.join(' '),
    opacity: s.opacity,
  }
}

function closed(s: Style, mode: ColourMode): Record<string, string | number | undefined> {
  const fill = s.fill === undefined && s.stroke === undefined ? 'fg' : s.fill
  return { fill: paint(fill, mode) ?? 'none', ...stroke(s, mode) }
}

function title(node: SceneNode): string {
  return node.title === undefined ? '' : `<title>${escapeXml(node.title)}</title>`
}

function nodeSvg(node: SceneNode, mode: ColourMode, font: string): string {
  const id = node.id === undefined ? {} : { 'data-node-id': node.id }
  switch (node.kind) {
    case 'rect':
      return `<rect${attrs({ ...id, x: node.x, y: node.y, width: node.width, height: node.height, rx: node.radius, ...closed(node, mode) })}>${title(node)}</rect>`
    case 'circle':
      return `<circle${attrs({ ...id, cx: node.cx, cy: node.cy, r: node.r, ...closed(node, mode) })}>${title(node)}</circle>`
    case 'ellipse':
      return `<ellipse${attrs({ ...id, cx: node.cx, cy: node.cy, rx: node.rx, ry: node.ry, ...closed(node, mode) })}>${title(node)}</ellipse>`
    case 'line': {
      const s = stroke(node, mode, 'fg')
      const head =
        node.arrow === true
          ? `<polygon${attrs({ points: arrowHeadPoints([node.x1, node.y1], [node.x2, node.y2], 4 + 2 * (node.strokeWidth ?? 1)), fill: s['stroke'], opacity: node.opacity })}/>`
          : ''
      return `<g${attrs(id)}>${title(node)}<line${attrs({ x1: node.x1, y1: node.y1, x2: node.x2, y2: node.y2, ...s })}/>${head}</g>`
    }
    case 'path': {
      const isClosed = node.closed === true
      const d = node.smooth === true ? smoothPathData(node.points, isClosed) : straightPathData(node.points, isClosed)
      const p = isClosed ? closed(node, mode) : { fill: paint(node.fill, mode) ?? 'none', ...stroke(node, mode, 'fg') }
      const last = node.points[node.points.length - 1]
      const before = node.points[node.points.length - 2]
      const head =
        node.arrow === true && last !== undefined && before !== undefined
          ? `<polygon${attrs({ points: arrowHeadPoints(before, last, 4 + 2 * (node.strokeWidth ?? 1)), fill: p['stroke'], opacity: node.opacity })}/>`
          : ''
      return `<g${attrs(id)}>${title(node)}<path${attrs({ d, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...p })}/>${head}</g>`
    }
    case 'text': {
      const baseline =
        node.baseline === 'top'
          ? 'hanging'
          : node.baseline === 'middle'
            ? 'central'
            : node.baseline === 'bottom'
              ? 'text-after-edge'
              : 'alphabetic'
      return `<text${attrs({ ...id, x: node.x, y: node.y, 'font-size': node.size ?? 12, 'font-family': font, 'font-weight': node.weight, fill: paint(node.fill ?? 'fg', mode), 'text-anchor': node.anchor ?? 'start', 'dominant-baseline': baseline })}>${escapeXml(node.text)}</text>`
    }
    case 'tex': {
      const shift = node.anchor === 'middle' ? '-50%' : node.anchor === 'end' ? '-100%' : '0'
      const html = texToHtml(node.tex, { output: 'mathml' })
      const colour = paint(node.fill ?? 'fg', mode) ?? 'currentColor'
      return `<foreignObject${attrs({ ...id, x: node.x, y: node.y, width: 1, height: 1, overflow: 'visible' })}><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;white-space:nowrap;transform:translate(${shift},-50%);font-size:${node.size ?? 12}px;color:${colour}">${html}</div></foreignObject>`
    }
    case 'image':
      return `<image${attrs({ ...id, x: node.x, y: node.y, width: node.width, height: node.height, preserveAspectRatio: 'none', href: pngDataUrl(node.pixels), style: node.smoothing === false ? 'image-rendering:pixelated' : undefined })}/>`
    case 'group': {
      const [tx, ty] = node.translate ?? [0, 0]
      const transform = `translate(${tx} ${ty}) rotate(${node.rotate ?? 0}) scale(${node.scale ?? 1})`
      return `<g${attrs({ ...id, transform, opacity: node.opacity })}>${title(node)}${node.children.map((c) => nodeSvg(c, mode, font)).join('')}</g>`
    }
    case 'layer3d':
      return ''
  }
}

/** Options of {@link sceneToSvg}. */
export interface SceneSvgOptions {
  readonly mode?: ColourMode
  /** Drawn width (default the scene's); the height keeps the scene's aspect unless given. */
  readonly width?: number
  readonly height?: number
  /** Fill the background (default the scene's background, else none). */
  readonly background?: Paint
  readonly font?: string
}

/**
 * A scene as a standalone SVG document: theme tokens resolved for the mode, 3-D layers projected, TeX labels set as
 * MathML in foreign objects, images embedded as PNG data URLs. The same drawing rules as `SceneView`.
 *
 * @param scene - The scene.
 * @param options - Colour mode, size, background and font.
 * @returns The SVG text.
 * @index function A Scene to standalone SVG text: `sceneToSvg(scene, { mode: 'dark' })`.
 * @example
 * ```ts
 * import { sceneToSvg } from 'mdml-render/svg'
 * import { circle, scene } from 'mdml-engine/scene'
 * sceneToSvg(scene(10, 10, [circle(5, 5, 2, { fill: 'tab1' })])).includes('<circle cx="5" cy="5" r="2" fill="#ff7f0e"') // => true
 * ```
 */
export function sceneToSvg(scene: Scene, options: SceneSvgOptions = {}): string {
  const mode = options.mode ?? 'light'
  const s = projectLayers(scene)
  const width = options.width ?? s.width
  const height = options.height ?? (options.width === undefined ? s.height : (s.height * width) / s.width)
  const bg = options.background ?? s.background
  const font = options.font ?? "system-ui, -apple-system, 'Segoe UI', sans-serif"
  const back =
    bg === undefined || bg === 'none'
      ? ''
      : `<rect width="100%" height="100%" fill="${escapeXml(resolvePaint(bg, mode))}"/>`
  const body = s.children.map((c) => nodeSvg(c, mode, font)).join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${s.width} ${s.height}">${s.title === undefined ? '' : `<title>${escapeXml(s.title)}</title>`}${back}${body}</svg>`
}
