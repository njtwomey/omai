/**
 * Building and timing `Timeline`s: from a parameter sweep, from the `frame` events of a run, from overlays on one
 * structure (message passing, search frontiers, forward passes), plus easings, interpolation between corresponding
 * scenes and in-between frames (`tweenTimeline`), and `frameAt`, the frame shown at a time.
 */

import { OmaiError } from 'omai-engine/base'
import {
  type Frame,
  isScene,
  type Scene,
  type SceneNode,
  type Structure,
  type StructureOverlay,
  type Timeline,
  timeline,
} from 'omai-engine/scene'
import { frames as frameEvents, type Run } from 'omai-engine/trace'

/** An easing curve: progress in [0, 1] to eased progress in [0, 1]. */
export type Easing = (t: number) => number

/**
 * The standard easings: `linear`, `easeIn`, `easeOut` and `easeInOut` (cubic).
 *
 * @index table The easing curves of a timeline's tweens, by name: `easings.easeInOut`
 */
export const easings: Readonly<Record<'linear' | 'easeIn' | 'easeOut' | 'easeInOut', Easing>> = {
  linear: (t) => t,
  easeIn: (t) => t * t * t,
  easeOut: (t) => 1 - (1 - t) ** 3,
  easeInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
}

const NUMERIC = [
  'x',
  'y',
  'width',
  'height',
  'cx',
  'cy',
  'r',
  'rx',
  'ry',
  'x1',
  'y1',
  'x2',
  'y2',
  'opacity',
  'strokeWidth',
  'size',
  'rotate',
  'scale',
] as const

function lerpNode(a: SceneNode, b: SceneNode, t: number): SceneNode {
  if (a.kind !== b.kind) return t < 0.5 ? a : b
  const out: Record<string, unknown> = { ...(t < 0.5 ? a : b) }
  const ra = a as unknown as Readonly<Record<string, unknown>>
  const rb = b as unknown as Readonly<Record<string, unknown>>
  for (const k of NUMERIC) {
    const u = ra[k]
    const v = rb[k]
    if (typeof u === 'number' && typeof v === 'number') out[k] = u + (v - u) * t
  }
  const ta = ra['translate']
  const tb = rb['translate']
  if (Array.isArray(ta) && Array.isArray(tb) && ta.length === 2 && tb.length === 2)
    out['translate'] = [0, 1].map((i) => (ta[i] as number) + ((tb[i] as number) - (ta[i] as number)) * t)
  if (a.kind === 'path' && b.kind === 'path' && a.points.length === b.points.length)
    out['points'] = a.points.map((p, i) => {
      const q = b.points[i] ?? p
      return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]
    })
  if (a.kind === 'group' && b.kind === 'group') out['children'] = lerpChildren(a.children, b.children, t)
  return out as unknown as SceneNode
}

function lerpChildren(a: readonly SceneNode[], b: readonly SceneNode[], t: number): readonly SceneNode[] {
  if (a.length !== b.length) return t < 0.5 ? a : b
  return a.map((n, i) => lerpNode(n, b[i] ?? n, t))
}

/**
 * Interpolate between two scenes: nodes that correspond (same position in the tree and the same kind) have their
 * coordinates, sizes, opacities, path points and group transforms blended; anything else switches halfway.
 *
 * @param a - The scene at t = 0.
 * @param b - The scene at t = 1.
 * @param t - The progress, 0 to 1.
 * @returns The blended scene.
 * @example
 * ```ts
 * import { interpolateScene } from 'omai-render/animation'
 * import { circle, scene } from 'omai-engine/scene'
 * const s = interpolateScene(scene(10, 10, [circle(0, 0, 1)]), scene(10, 10, [circle(4, 0, 3)]), 0.5)
 * s.children[0] // => { kind: 'circle', cx: 2, cy: 0, r: 2 }
 * ```
 */
export function interpolateScene(a: Scene, b: Scene, t: number): Scene {
  const u = Math.min(1, Math.max(0, t))
  return {
    ...(u < 0.5 ? a : b),
    width: a.width + (b.width - a.width) * u,
    height: a.height + (b.height - a.height) * u,
    children: lerpChildren(a.children, b.children, u),
  }
}

/**
 * A timeline with `k - 1` eased in-between frames inserted between consecutive scene frames (structure frames are kept
 * as they are), and its frame rate multiplied by `k`, so it plays at the same speed but smoothly.
 *
 * @param tl - The timeline.
 * @param k - Frames per original frame (at least 1).
 * @param easing - The easing between frames (default `easeInOut`).
 * @returns The tweened timeline.
 * @throws OmaiError with code `value` for k below 1.
 * @index function Add in-between frames to a timeline: `tweenTimeline(tl, 4)`.
 * @example
 * ```ts
 * import { tweenTimeline } from 'omai-render/animation'
 * import { scene, sceneTimeline } from 'omai-engine/scene'
 * tweenTimeline(sceneTimeline([scene(1, 1, []), scene(1, 1, [])]), 4).frames.length // => 5
 * ```
 */
export function tweenTimeline(tl: Timeline, k: number, easing: Easing = easings.easeInOut): Timeline {
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('value', 'tweenTimeline: k must be an integer >= 1')
  const out: Frame[] = []
  tl.frames.forEach((f, i) => {
    const next = tl.frames[i + 1]
    out.push({ ...f, step: f.step * k })
    if (next === undefined || f.kind !== 'scene' || next.kind !== 'scene') return
    for (let j = 1; j < k; j++)
      out.push({
        kind: 'scene',
        step: f.step * k + ((next.step - f.step) * k * j) / k,
        scene: interpolateScene(f.scene, next.scene, easing(j / k)),
      })
  })
  return timeline(out, { fps: tl.fps * k, ...(tl.title === undefined ? {} : { title: tl.title }) })
}

/**
 * A timeline from a parameter sweep: one scene per value.
 *
 * @param values - The parameter values, in order.
 * @param build - The scene for a value.
 * @param options - Frames per second (default 10), title and captions per value.
 * @returns The timeline.
 * @index function A timeline from a parameter sweep: `sweepTimeline(values, (v) => scene)`.
 * @example
 * ```ts
 * import { sweepTimeline } from 'omai-render/animation'
 * import { circle, scene } from 'omai-engine/scene'
 * sweepTimeline([1, 2, 3], (r) => scene(10, 10, [circle(5, 5, r)])).frames.length // => 3
 * ```
 */
export function sweepTimeline<V>(
  values: readonly V[],
  build: (value: V, index: number) => Scene,
  options: { readonly fps?: number; readonly title?: string; readonly caption?: (value: V) => string } = {},
): Timeline {
  return timeline(
    values.map((v, i) => ({
      kind: 'scene' as const,
      step: i,
      scene: build(v, i),
      ...(options.caption === undefined ? {} : { caption: options.caption(v) }),
    })),
    { fps: options.fps ?? 10, ...(options.title === undefined ? {} : { title: options.title }) },
  )
}

/**
 * A timeline from the `frame` events of a run (environments and algorithms that record scenes), optionally decoding
 * frame payloads that are not scenes.
 *
 * @param run - The trace.
 * @param options - A decoder for non-scene payloads, the run id, frames per second.
 * @returns The timeline (frames whose payload is neither a scene nor decodable are skipped).
 * @index function A timeline from the frame events of a run: `runTimeline(run)`.
 * @example
 * ```ts
 * import { runTimeline } from 'omai-render/animation'
 * import { scene } from 'omai-engine/scene'
 * const run = { schema: 'omai.run/1' as const, events: [{ type: 'frame' as const, runId: 'r', step: 0, frame: scene(2, 2, []) as never }] }
 * runTimeline(run).frames.length // => 1
 * ```
 */
export function runTimeline(
  run: Run,
  options: {
    readonly decode?: (payload: unknown, step: number) => Scene | undefined
    readonly runId?: string
    readonly fps?: number
  } = {},
): Timeline {
  const out: Frame[] = []
  for (const f of frameEvents(run, options.runId)) {
    const s = isScene(f.frame) ? (f.frame as unknown as Scene) : options.decode?.(f.frame, f.step)
    if (s !== undefined) out.push({ kind: 'scene', step: f.step, scene: s })
  }
  return timeline(out, { fps: options.fps ?? 10 })
}

/**
 * A timeline of overlays on one structure (active nodes and edges, badges): the frames behind animated message passing,
 * search frontiers and forward passes.
 *
 * @param structure - The structure.
 * @param overlays - One overlay per frame.
 * @param options - Frames per second (default 4) and captions.
 * @returns The timeline.
 * @index function A timeline of overlays on one structure: `structureTimeline(structure, overlays)`.
 * @example
 * ```ts
 * import { structureTimeline } from 'omai-render/animation'
 * import { structure } from 'omai-engine/scene'
 * structureTimeline(structure('s', [], []), [{ active: ['a'] }, { active: ['b'] }]).frames[1].overlay.active // => ['b']
 * ```
 */
export function structureTimeline(
  structure: Structure,
  overlays: readonly StructureOverlay[],
  options: { readonly fps?: number; readonly captions?: readonly string[] } = {},
): Timeline & { readonly frames: readonly (Frame & { readonly kind: 'structure' })[] } {
  const frames = overlays.map((overlay, step) => {
    const caption = options.captions?.[step]
    return { kind: 'structure' as const, step, structure, overlay, ...(caption === undefined ? {} : { caption }) }
  })
  return timeline(frames, { fps: options.fps ?? 4 }) as Timeline & {
    readonly frames: readonly (Frame & { readonly kind: 'structure' })[]
  }
}

/**
 * The index of the frame shown at a time since playback started.
 *
 * @param tl - The timeline.
 * @param seconds - Elapsed time.
 * @param loop - Start over after the last frame (default false: hold it).
 * @returns The frame index (0 for an empty timeline).
 * @example
 * ```ts
 * import { frameAt } from 'omai-render/animation'
 * import { scene, sceneTimeline } from 'omai-engine/scene'
 * const tl = sceneTimeline([scene(1, 1, []), scene(1, 1, []), scene(1, 1, [])], { fps: 2 })
 * frameAt(tl, 0.6) // => 1
 * frameAt(tl, 9) // => 2
 * frameAt(tl, 1.6, true) // => 0
 * ```
 */
export function frameAt(tl: Timeline, seconds: number, loop = false): number {
  const n = tl.frames.length
  if (n === 0) return 0
  const i = Math.floor(Math.max(0, seconds) * tl.fps)
  return loop ? i % n : Math.min(n - 1, i)
}
