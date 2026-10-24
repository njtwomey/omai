import type { Json } from 'whitebox-engine/base'
import type { Episode } from 'whitebox-engine/rl'
import { grid, type Scene, type Timeline, timeline } from 'whitebox-engine/scene'

/** What happened at one step of an episode, as a player shows it under the frame. */
export interface EpisodeStep {
  /** The step (0 is the reset). */
  readonly t: number
  /** The action taken to reach this frame (none at the reset). */
  readonly action?: Json
  readonly reward?: number
  /** The return so far. */
  readonly return: number
  /** Whether this is the last frame, and how the episode ended. */
  readonly end?: 'terminated' | 'truncated'
}

/**
 * The steps of an episode, one per rendered frame: the reset, then each action with its reward and the running return.
 *
 * @param episode - An episode from `runEpisode` (with frames or not).
 * @returns `length + 1` steps.
 * @example
 * ```ts
 * import { episodeSteps } from 'whitebox-render/gym'
 * const ep = { observations: [0, 1], states: [0, 1], actions: [1], rewards: [2], infos: [{}], return: 2, length: 1, terminated: true, truncated: false }
 * episodeSteps(ep).map((s) => s.return) // => [0, 2]
 * ```
 */
export function episodeSteps<S, O, A>(episode: Episode<S, O, A>): EpisodeStep[] {
  const out: EpisodeStep[] = [{ t: 0, return: 0 }]
  let ret = 0
  for (let t = 0; t < episode.length; t++) {
    ret += episode.rewards[t] ?? 0
    const last = t === episode.length - 1
    out.push({
      t: t + 1,
      action: (episode.actions[t] ?? null) as unknown as Json,
      reward: episode.rewards[t] ?? 0,
      return: ret,
      ...(last && episode.terminated ? { end: 'terminated' as const } : {}),
      ...(last && !episode.terminated && episode.truncated ? { end: 'truncated' as const } : {}),
    })
  }
  return out
}

/**
 * An episode's rendered frames as a `Timeline`, each frame captioned with its step, action, reward and return, for the
 * player and for GIF or video export.
 *
 * @param episode - An episode rendered with `frames: true`.
 * @param options - Frames per second (default 10) and a title.
 * @returns The timeline (empty when the episode has no frames).
 * @example
 * ```ts
 * import { scene } from 'whitebox-engine/scene'
 * import { episodeTimeline } from 'whitebox-render/gym'
 * const ep = { observations: [0, 1], states: [0, 1], actions: [1], rewards: [2], infos: [{}], return: 2, length: 1, terminated: true, truncated: false, frames: [scene(1, 1, []), scene(1, 1, [])] }
 * episodeTimeline(ep).frames[1].caption // => 't = 1 · action 1 · reward 2 · return 2 (terminated)'
 * ```
 */
export function episodeTimeline<S, O, A>(
  episode: Episode<S, O, A>,
  options: { readonly fps?: number; readonly title?: string } = {},
): Timeline {
  const steps = episodeSteps(episode)
  return timeline(
    (episode.frames ?? []).map((s, i) => ({
      kind: 'scene' as const,
      step: i,
      scene: s,
      caption: stepCaption(steps[i]),
    })),
    { fps: options.fps ?? 10, ...(options.title === undefined ? {} : { title: options.title }) },
  )
}

/**
 * A step as one line: `t = 3 · action 1 · reward 1 · return 3`, with how the episode ended on its last frame.
 *
 * @param step - The step.
 * @returns The caption.
 */
export function stepCaption(step: EpisodeStep | undefined): string {
  if (step === undefined) return ''
  if (step.t === 0) return 't = 0 · reset'
  const num = (v: number): string => String(Number(v.toPrecision(4)))
  const parts = [
    `t = ${step.t}`,
    `action ${JSON.stringify(step.action)}`,
    `reward ${num(step.reward ?? 0)}`,
    `return ${num(step.return)}`,
  ]
  return `${parts.join(' · ')}${step.end === undefined ? '' : ` (${step.end})`}`
}

/**
 * A filmstrip: every `every`-th frame (and the last) side by side in rows of `columns`, each in a cell of the given size.
 *
 * @param frames - The scenes.
 * @param options - Columns (default 6), the cell size (default 120 by 90) and the stride (default 1).
 * @returns One scene holding the strip.
 */
export function filmstrip(
  frames: readonly Scene[],
  options: { readonly columns?: number; readonly cell?: readonly [number, number]; readonly every?: number } = {},
): Scene {
  const every = Math.max(1, options.every ?? 1)
  const picked = frames.filter((_, i) => i % every === 0 || i === frames.length - 1)
  return grid(picked, options.columns ?? 6, options.cell ?? [120, 90])
}
