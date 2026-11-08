/**
 * Turning a timeline into pictures: every frame as a scene (structure frames drawn on one steady layout, nodes sized
 * for every frame's badges), a scene rasterised to RGBA pixels (through SVG and a canvas in the browser, the engine's
 * software rasteriser in Node, which leaves out text), and `timelineFrames`, the pixel frames GIF and WebM encoding
 * take.
 */

import { OmaiError } from 'omai-engine/base'
import { type Frame, type Pixels, rasterise, type Scene, type StructureOverlay, type Timeline } from 'omai-engine/scene'
import { diagram } from '../structures/diagram.ts'
import { svgToPixels } from '../svg/export.ts'
import { sceneToSvg } from '../svg/svg.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'

/**
 * Every frame of a timeline as a scene: scene frames as they are, structure frames drawn with `diagram` and their
 * overlay, every node sized for the badges of all the timeline's frames so the layout holds still from frame to frame.
 *
 * @param tl - The timeline.
 * @returns One scene per frame, in order.
 * @example
 * ```ts
 * import { scene, timeline } from 'omai-engine/scene'
 * import { timelineScenes } from 'omai-render/animation'
 * timelineScenes(timeline([{ kind: 'scene', step: 0, scene: scene(4, 3, []) }]))[0].width // => 4
 * ```
 */
export function timelineScenes(tl: Timeline): Scene[] {
  const overlays: StructureOverlay[] = tl.frames.flatMap((f) => (f.kind === 'structure' ? [f.overlay] : []))
  return tl.frames.map((f: Frame) =>
    f.kind === 'scene' ? f.scene : diagram(f.structure, { overlay: f.overlay, frames: overlays }),
  )
}

/** Options of {@link rasteriseScene} and {@link timelineFrames}. */
export interface FrameOptions {
  readonly mode?: ColourMode
  /** Output width in pixels (default the scene's; the height keeps the aspect). */
  readonly width?: number
  /** Fill colour behind the scene (default the theme's background). */
  readonly background?: string
}

/**
 * A scene as RGBA pixels: in a browser through its SVG (`sceneToSvg`) drawn on a canvas, so text and TeX come out as
 * on the page; elsewhere (Node) with omai-engine's software rasteriser, which leaves out text.
 *
 * @param s - The scene.
 * @param options - Colour mode, output width and background.
 * @returns A promise of the pixels.
 * @example
 * ```ts
 * import { rect, scene } from 'omai-engine/scene'
 * import { rasteriseScene } from 'omai-render/animation'
 * const px = await rasteriseScene(scene(4, 2, [rect(0, 0, 4, 2, { fill: '#ff0000' })]), { width: 8 })
 * [px.width, px.height, px.rgba[0]] // => [8, 4, 255]
 * ```
 */
export async function rasteriseScene(s: Scene, options: FrameOptions = {}): Promise<Pixels> {
  const mode = options.mode ?? 'light'
  const width = Math.max(1, Math.round(options.width ?? s.width))
  const height = Math.max(1, Math.round((width * s.height) / Math.max(1e-9, s.width)))
  const background = options.background ?? tokens(mode).bg
  if (typeof document !== 'undefined' && typeof Image !== 'undefined') {
    const svg = sceneToSvg(s, { mode, width, height })
    return svgToPixels(svg, width, height, { background })
  }
  return rasterise(s, { width, height, mode, background: background as `#${string}`, samples: 2 })
}

/**
 * Every frame of a timeline as RGBA pixels, all the size of the first (see {@link timelineScenes} and
 * {@link rasteriseScene}): the input of `encodeGif` and `encodeWebm`.
 *
 * @param tl - The timeline.
 * @param options - Colour mode, output width and background.
 * @returns A promise of the frames.
 * @throws OmaiError with code `value` for an empty timeline.
 * @index function Pixel frames of a timeline for GIF or video: `await timelineFrames(tl)`.
 */
export async function timelineFrames(tl: Timeline, options: FrameOptions = {}): Promise<Pixels[]> {
  const scenes = timelineScenes(tl)
  const first = scenes[0]
  if (first === undefined) throw new OmaiError('value', 'timelineFrames: the timeline has no frames')
  const width = Math.round(options.width ?? first.width)
  const out: Pixels[] = []
  // Later scenes are drawn at the first one's size, so every frame matches.
  for (const s of scenes)
    out.push(await rasteriseScene({ ...s, width: first.width, height: first.height }, { ...options, width }))
  return out
}
