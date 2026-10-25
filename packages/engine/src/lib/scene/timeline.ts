/**
 * Timelines: frames keyed by step, each a scene or a structure with highlights laid over it, the data behind every
 * animation, playback control and GIF or video export.
 */

import type { Structure } from './structure.ts'
import type { Scene } from './types.ts'

/** Highlights laid over a structure in one frame of an animation. */
export interface StructureOverlay {
  readonly active?: readonly string[]
  readonly edges?: readonly { readonly from: string; readonly to: string }[]
  readonly badges?: Readonly<Record<string, Readonly<Record<string, string | number>>>>
}

/** One frame of a timeline: a scene, or a structure with overlays, at a step. */
export type Frame =
  | { readonly kind: 'scene'; readonly step: number; readonly scene: Scene; readonly caption?: string }
  | {
      readonly kind: 'structure'
      readonly step: number
      readonly structure: Structure
      readonly overlay: StructureOverlay
      readonly caption?: string
    }

/**
 * Frames keyed by step: the data behind every animation, playback control and GIF or video export.
 *
 * @concept animation
 */
export interface Timeline {
  readonly kind: 'timeline'
  readonly frames: readonly Frame[]
  readonly fps: number
  readonly title?: string
}

/**
 * Build a timeline from frames (sorted by step).
 *
 * @param frames - The frames.
 * @param options - Frames per second for playback (default 10) and a title.
 * @returns The timeline.
 * @example
 * ```ts
 * import { timeline, scene } from 'mdml-engine/scene'
 * timeline([{ kind: 'scene', step: 1, scene: scene(1, 1, []) }]).frames.length // => 1
 * ```
 */
export function timeline(
  frames: readonly Frame[],
  options: { readonly fps?: number; readonly title?: string } = {},
): Timeline {
  const { fps = 10, ...rest } = options
  return { kind: 'timeline', frames: [...frames].sort((a, b) => a.step - b.step), fps, ...rest }
}

/**
 * Timeline from a sequence of scenes, one per step.
 *
 * @param scenes - The scenes in order.
 * @param options - Frames per second and a title.
 * @returns The timeline.
 * @example
 * ```ts
 * import { sceneTimeline, scene } from 'mdml-engine/scene'
 * sceneTimeline([scene(1, 1, []), scene(1, 1, [])]).frames[1].step // => 1
 * ```
 */
export function sceneTimeline(
  scenes: readonly Scene[],
  options: { readonly fps?: number; readonly title?: string } = {},
): Timeline {
  return timeline(
    scenes.map((s, step) => ({ kind: 'scene' as const, step, scene: s })),
    options,
  )
}

/**
 * Whether a value is a {@link Timeline}.
 *
 * @param value - Any value.
 * @returns `true` for timelines.
 * @example
 * ```ts
 * import { isTimeline, timeline } from 'mdml-engine/scene'
 * isTimeline(timeline([])) // => true
 * ```
 */
export function isTimeline(value: unknown): value is Timeline {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'timeline'
}
