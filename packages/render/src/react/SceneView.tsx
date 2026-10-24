import { type CSSProperties, type PointerEvent, type ReactElement, type ReactNode, useMemo } from 'react'
import { type Paint, projectLayers, type Scene, type SceneNode, type Style } from 'whitebox-engine/scene'
import { arrowHeadPoints, smoothPathData, straightPathData } from '../svg/paths.ts'
import { pngDataUrl } from '../svg/png.ts'
import { type ColourMode, type Tokens, resolvePaint as themePaint, tokens as themeTokens } from '../theme/tokens.ts'
import { Tex } from './MathView.tsx'
import { useColourMode } from './theme.tsx'

/** Props of {@link SceneView}. */
export interface SceneViewProps {
  readonly scene: Scene
  /** Drawn width in pixels (default the scene's width); the height follows the scene's aspect unless given. */
  readonly width?: number
  readonly height?: number
  /** Colour mode for theme tokens (default the page's). */
  readonly mode?: ColourMode
  /** Called with the id of the innermost node with an id under a click, and the node. */
  readonly onNodeClick?: (id: string, node: SceneNode) => void
  /** Called with the id of the innermost node with an id under the pointer, or `null` as it leaves them. */
  readonly onNodeHover?: (id: string | null, node: SceneNode | null) => void
  /** Ids of nodes to draw emphasised (an accent outline). */
  readonly highlight?: readonly string[]
  /** The part of the scene to show, `[x, y, width, height]` in scene coordinates (default all of it): pan and zoom. */
  readonly viewBox?: readonly [number, number, number, number]
  /** Content drawn over the scene, in scene coordinates (overlays, handles). */
  readonly children?: ReactNode
  readonly className?: string
  readonly style?: CSSProperties
}

function Image(props: { readonly node: Extract<SceneNode, { kind: 'image' }> }): ReactElement | null {
  const { node } = props
  const url = useMemo(() => pngDataUrl(node.pixels), [node.pixels])
  return (
    <image
      href={url}
      x={node.x}
      y={node.y}
      width={node.width}
      height={node.height}
      preserveAspectRatio="none"
      style={{ imageRendering: node.smoothing === false ? 'pixelated' : 'auto' }}
      data-node-id={node.id}
    />
  )
}

/** A scene paint for the theme, or `undefined` for none given. */
function resolvePaint(paint: Paint | undefined, t: Tokens): string | undefined {
  return paint === undefined ? undefined : themePaint(paint, t.mode)
}

interface Ctx {
  readonly t: Tokens
  readonly highlight: ReadonlySet<string>
}

function strokeProps(s: Style, ctx: Ctx, defaultStroke?: Paint): Record<string, unknown> {
  const stroke = resolvePaint(s.stroke ?? defaultStroke, ctx.t)
  return {
    stroke: stroke ?? 'none',
    strokeWidth: s.strokeWidth ?? 1,
    ...(s.dash === undefined ? {} : { strokeDasharray: s.dash.join(' ') }),
    ...(s.opacity === undefined ? {} : { opacity: s.opacity }),
  }
}

function closedProps(s: Style, ctx: Ctx): Record<string, unknown> {
  const filled = s.fill === undefined && s.stroke === undefined ? 'fg' : s.fill
  return { fill: resolvePaint(filled, ctx.t) ?? 'none', ...strokeProps(s, ctx) }
}

function emphasis(node: SceneNode, ctx: Ctx): Record<string, unknown> {
  return node.id !== undefined && ctx.highlight.has(node.id)
    ? { stroke: ctx.t.accent, strokeWidth: Math.max(2, ('strokeWidth' in node ? node.strokeWidth : undefined) ?? 2) }
    : {}
}

function Node(props: { readonly node: SceneNode; readonly ctx: Ctx }): ReactNode {
  const { node, ctx } = props
  const common = { 'data-node-id': node.id }
  const title = node.title === undefined ? null : <title>{node.title}</title>
  switch (node.kind) {
    case 'rect':
      return (
        <rect
          {...common}
          x={node.x}
          y={node.y}
          width={node.width}
          height={node.height}
          rx={node.radius}
          {...closedProps(node, ctx)}
          {...emphasis(node, ctx)}
        >
          {title}
        </rect>
      )
    case 'circle':
      return (
        <circle {...common} cx={node.cx} cy={node.cy} r={node.r} {...closedProps(node, ctx)} {...emphasis(node, ctx)}>
          {title}
        </circle>
      )
    case 'ellipse':
      return (
        <ellipse
          {...common}
          cx={node.cx}
          cy={node.cy}
          rx={node.rx}
          ry={node.ry}
          {...closedProps(node, ctx)}
          {...emphasis(node, ctx)}
        >
          {title}
        </ellipse>
      )
    case 'line': {
      const s = strokeProps(node, ctx, 'fg')
      const size = 4 + 2 * (node.strokeWidth ?? 1)
      return (
        <g {...common}>
          {title}
          <line x1={node.x1} y1={node.y1} x2={node.x2} y2={node.y2} {...s} {...emphasis(node, ctx)} />
          {node.arrow === true ? (
            <polygon
              points={arrowHeadPoints([node.x1, node.y1], [node.x2, node.y2], size)}
              fill={String(s['stroke'])}
              opacity={node.opacity}
            />
          ) : null}
        </g>
      )
    }
    case 'path': {
      const closed = node.closed === true
      const d = node.smooth === true ? smoothPathData(node.points, closed) : straightPathData(node.points, closed)
      const paint = closed
        ? closedProps(node, ctx)
        : { fill: resolvePaint(node.fill, ctx.t) ?? 'none', ...strokeProps(node, ctx, 'fg') }
      const last = node.points[node.points.length - 1]
      const before = node.points[node.points.length - 2]
      const size = 4 + 2 * (node.strokeWidth ?? 1)
      return (
        <g {...common}>
          {title}
          <path d={d} strokeLinejoin="round" strokeLinecap="round" {...paint} {...emphasis(node, ctx)} />
          {node.arrow === true && last !== undefined && before !== undefined ? (
            <polygon
              points={arrowHeadPoints(before, last, size)}
              fill={String(paint['stroke'])}
              opacity={node.opacity}
            />
          ) : null}
        </g>
      )
    }
    case 'text':
      return (
        <text
          {...common}
          x={node.x}
          y={node.y}
          fontSize={node.size ?? 12}
          fill={resolvePaint(node.fill ?? 'fg', ctx.t)}
          textAnchor={node.anchor ?? 'start'}
          dominantBaseline={
            node.baseline === 'top'
              ? 'hanging'
              : node.baseline === 'middle'
                ? 'central'
                : node.baseline === 'bottom'
                  ? 'text-after-edge'
                  : 'alphabetic'
          }
          fontWeight={node.weight}
          style={{ fontFamily: ctx.t.font }}
        >
          {node.text}
        </text>
      )
    case 'tex': {
      const shift = node.anchor === 'middle' ? '-50%' : node.anchor === 'end' ? '-100%' : '0'
      return (
        <foreignObject {...common} x={node.x} y={node.y} width={1} height={1} overflow="visible">
          <div
            style={{
              display: 'inline-block',
              whiteSpace: 'nowrap',
              transform: `translate(${shift}, -50%)`,
              fontSize: node.size ?? 12,
              color: resolvePaint(node.fill ?? 'fg', ctx.t),
            }}
          >
            <Tex>{node.tex}</Tex>
          </div>
        </foreignObject>
      )
    }
    case 'image':
      return <Image node={node} />
    case 'group': {
      const [tx, ty] = node.translate ?? [0, 0]
      const transform = `translate(${tx} ${ty}) rotate(${node.rotate ?? 0}) scale(${node.scale ?? 1})`
      return (
        <g {...common} transform={transform} opacity={node.opacity}>
          {title}
          {node.children.map((c, i) => (
            <Node key={c.id ?? i} node={c} ctx={ctx} />
          ))}
        </g>
      )
    }
    case 'layer3d':
      // Projected away by projectLayers before drawing.
      return null
  }
}

function findNode(nodes: readonly SceneNode[], id: string): SceneNode | null {
  for (const n of nodes) {
    if (n.id === id) return n
    if (n.kind === 'group') {
      const found = findNode(n.children, id)
      if (found !== null) return found
    }
  }
  return null
}

/**
 * A `Scene` (plain data from whitebox-engine: environments, diagrams and models produce them) drawn as SVG elements,
 * with theme tokens resolved for the colour mode, 3-D layers projected, TeX labels typeset, and hover and click
 * reported by node id. The scene keeps its own coordinates (a `viewBox`), so it scales to any drawn size.
 *
 * @param props - The scene, its drawn size, the colour mode, event handlers, highlighted ids and overlay children.
 * @returns The SVG.
 */
export function SceneView(props: SceneViewProps): ReactElement {
  const { scene, onNodeClick, onNodeHover, highlight = [], children, className, style } = props
  const pageMode = useColourMode()
  const mode = props.mode ?? pageMode
  const flat = useMemo(() => projectLayers(scene), [scene])
  const ctx: Ctx = useMemo(() => ({ t: themeTokens(mode), highlight: new Set(highlight) }), [mode, highlight])
  const width = props.width ?? scene.width
  const height = props.height ?? (props.width === undefined ? scene.height : (width * scene.height) / scene.width)
  const idAt = (e: PointerEvent<SVGSVGElement>): string | null =>
    (e.target as Element).closest?.('[data-node-id]')?.getAttribute('data-node-id') ?? null
  const interactive = onNodeClick !== undefined || onNodeHover !== undefined
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: clicks report nodes to the page; keyboard access belongs to its controls
    <svg
      width={width}
      height={height}
      viewBox={props.viewBox === undefined ? `0 0 ${scene.width} ${scene.height}` : props.viewBox.join(' ')}
      className={className}
      style={{ display: 'block', maxWidth: '100%', height: 'auto', ...style }}
      role="img"
      aria-label={scene.title ?? 'scene'}
      onPointerMove={
        onNodeHover === undefined
          ? undefined
          : (e) => {
              const id = idAt(e)
              onNodeHover(id, id === null ? null : findNode(flat.children, id))
            }
      }
      onPointerLeave={onNodeHover === undefined ? undefined : () => onNodeHover(null, null)}
      onClick={
        onNodeClick === undefined
          ? undefined
          : (e) => {
              const id = (e.target as Element).closest?.('[data-node-id]')?.getAttribute('data-node-id')
              const node = id == null ? null : findNode(flat.children, id)
              if (id != null && node !== null) onNodeClick(id, node)
            }
      }
      cursor={interactive ? 'pointer' : undefined}
    >
      {scene.title === undefined ? null : <title>{scene.title}</title>}
      <rect
        x={0}
        y={0}
        width={scene.width}
        height={scene.height}
        fill={resolvePaint(scene.background ?? 'none', ctx.t)}
      />
      {flat.children.map((c, i) => (
        <Node key={c.id ?? i} node={c} ctx={ctx} />
      ))}
      {children}
    </svg>
  )
}
