import { type ReactElement, type ReactNode, useMemo } from 'react'
import { key as makeKey } from 'whitebox-engine/random'
import { type Env, type Episode, type Policy, runEpisode } from 'whitebox-engine/rl'
import type { Scene } from 'whitebox-engine/scene'
import { episodeSteps, stepCaption } from '../gym/frames.ts'
import { cx } from './controls/classes.ts'
import { Player } from './Player.tsx'
import { SceneView } from './SceneView.tsx'
import { usePlayback } from './usePlayback.ts'

/** Props of {@link EpisodePlayer}. */
export interface EpisodePlayerProps<S, O, A> {
  readonly env: Env<S, O, A>
  readonly policy: Policy<O, A>
  /** The episode's seed (default 0): the same seed replays the same episode. */
  readonly seed?: number
  /** A step cap (default 500). */
  readonly maxSteps?: number
  /** Playback frames per second (default 15). */
  readonly fps?: number
  /** Drawn width of the frames (default their own width). */
  readonly width?: number
  /** Draw something over a frame (a value overlay, the agent's view), given the state and the step. */
  readonly decorate?: (frame: Scene, state: S, t: number) => Scene
  /** Called with the episode once it has run. */
  readonly onEpisode?: (episode: Episode<S, O, A>) => void
  readonly className?: string
}

/**
 * Play an episode of an environment under a policy: the episode is run once up front (`runEpisode`, reproducible from
 * its seed) with every state rendered, then played with the {@link Player}, each frame captioned with its step, action,
 * reward and return.
 *
 * @param props - The environment, policy, seed, step cap, rate, width and an optional decoration per frame.
 * @returns The player.
 */
export function EpisodePlayer<S, O, A>(props: EpisodePlayerProps<S, O, A>): ReactElement {
  const { env, policy, seed = 0, maxSteps = 500, fps = 15, width, decorate, onEpisode, className } = props
  const episode = useMemo(() => {
    const ep = runEpisode(env, policy, makeKey(seed), { maxSteps, frames: true })
    onEpisode?.(ep)
    return ep
  }, [env, policy, seed, maxSteps, onEpisode])
  const steps = useMemo(() => episodeSteps(episode), [episode])
  const frames = episode.frames ?? []
  const playback = usePlayback(frames.length, { fps })
  const t = playback.frame
  const base = frames[t]
  const shown =
    base === undefined ? undefined : decorate === undefined ? base : decorate(base, episode.states[t] as S, t)
  return (
    <div className={cx('flex w-fit max-w-full flex-col gap-2', className)}>
      {shown === undefined ? null : <SceneView scene={shown} {...(width === undefined ? {} : { width })} />}
      <Player controller={playback} stamp={(f) => stepCaption(steps[f])} />
    </div>
  )
}

/** Props of {@link EnvView}. */
export interface EnvViewProps<S, O, A> {
  readonly env: Env<S, O, A>
  /** The current state (the page keeps it, so other figures can read it). */
  readonly state: S
  /** Drawn width (default the frame's own). */
  readonly width?: number
  /** Draw something over the frame. */
  readonly decorate?: (frame: Scene, state: S) => Scene
  /** Content beside the frame (controls, readouts). */
  readonly children?: ReactNode
  readonly className?: string
}

/**
 * One consistent view of any environment's state: its `render` as a scene, optionally decorated, with room for
 * controls. Drive it with {@link useEnvStepper} to reset and step by hand.
 *
 * @param props - The environment, its state, width, decoration and extra content.
 * @returns The view.
 */
export function EnvView<S, O, A>(props: EnvViewProps<S, O, A>): ReactElement {
  const { env, state, width, decorate, children, className } = props
  const frame = useMemo(() => {
    const f = env.render(state)
    return decorate === undefined ? f : decorate(f, state)
  }, [env, state, decorate])
  return (
    <div className={cx('flex flex-wrap items-start gap-4', className)}>
      <SceneView scene={frame} {...(width === undefined ? {} : { width })} />
      {children}
    </div>
  )
}
