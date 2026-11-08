/**
 * Self-playing animated SVG: `animatedSvg` draws every frame of a timeline into one document and shows them in turn
 * with CSS keyframes, so the file plays on its own in a browser, a slide or a README, with no script.
 */

import { OmaiError } from 'omai-engine/base'
import { isTimeline, type Scene, type Timeline } from 'omai-engine/scene'
import { escapeXml, sceneToSvg } from '../svg/svg.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { timelineScenes } from './frames.ts'

/** Options of {@link animatedSvg}. */
export interface AnimatedSvgOptions {
  readonly mode?: ColourMode
  /** Frames per second (default the timeline's, else 10). */
  readonly fps?: number
  /** Drawn width (default the first scene's; the height keeps its aspect). */
  readonly width?: number
  /** Fill the background with the theme's (default true). */
  readonly background?: boolean
  /** Play once and hold the last frame instead of looping (default false). */
  readonly once?: boolean
}

/**
 * An animated SVG of a timeline (or a list of scenes): every frame drawn into one document and shown in turn by CSS
 * keyframes, so the file plays by itself in a browser, a slide or a README, with no script. Each frame is visible for
 * `1 / fps` seconds of the cycle; the whole loops (or plays once and holds the last frame).
 *
 * @param input - The timeline or scenes.
 * @param options - Colour mode, frames per second, width, background and looping.
 * @returns The SVG text.
 * @throws OmaiError with code `value` when there are no frames.
 * @index function A self-playing animated SVG from a timeline: `animatedSvg(timeline, { fps: 10 })`.
 * @example
 * ```ts
 * import { circle, scene } from 'omai-engine/scene'
 * import { animatedSvg } from 'omai-render/animation'
 * const svg = animatedSvg([scene(10, 10, [circle(3, 5, 2)]), scene(10, 10, [circle(7, 5, 2)])], { fps: 2 })
 * svg.includes('@keyframes') // => true
 * svg.match(/class="wb-f wb-f\d+"/g).length // => 2
 * ```
 */
export function animatedSvg(input: Timeline | readonly Scene[], options: AnimatedSvgOptions = {}): string {
  const scenes = isTimeline(input) ? timelineScenes(input) : [...input]
  const first = scenes[0]
  if (first === undefined) throw new OmaiError('value', 'animatedSvg: there are no frames')
  const mode = options.mode ?? 'light'
  const fps = options.fps ?? (isTimeline(input) ? input.fps : 10)
  const n = scenes.length
  const period = n / fps
  const width = options.width ?? first.width
  const height = (first.height * width) / first.width
  const share = 100 / n
  const fmt = (v: number): string => String(Math.round(v * 1e4) / 1e4)
  const once = options.once === true
  // Every frame shows for its share of the cycle, starting at its own delay; played once, the last one stays.
  const css = [
    `.wb-f{visibility:hidden;animation:wb-show ${fmt(period)}s step-end ${once ? '1' : 'infinite'}}`,
    `@keyframes wb-show{0%{visibility:visible}${fmt(share)}%{visibility:hidden}100%{visibility:hidden}}`,
    ...scenes.map((_, i) => `.wb-f${i}{animation-delay:${fmt(i / fps)}s}`),
    ...(once
      ? [
          `.wb-f${n - 1}{animation:wb-last 0s ${fmt((n - 1) / fps)}s forwards}`,
          '@keyframes wb-last{to{visibility:visible}}',
        ]
      : []),
  ].join('')
  const back =
    options.background === false ? '' : `<rect width="100%" height="100%" fill="${escapeXml(tokens(mode).bg)}"/>`
  const frames = scenes
    .map((s, i) => {
      const inner = sceneToSvg(s, { mode, width: first.width, height: first.height }).replace(
        /^<svg /,
        '<svg x="0" y="0" ',
      )
      return `<g class="wb-f wb-f${i}">${inner}</g>`
    })
    .join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(width)}" height="${fmt(height)}" viewBox="0 0 ${first.width} ${first.height}"><style>${css}</style>${back}${frames}</svg>`
}
