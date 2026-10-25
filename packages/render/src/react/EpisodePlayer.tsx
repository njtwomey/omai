/**
 * An episode player: an environment run once under a policy from a seed, every state rendered, then played back with
 * the `Player`, each frame captioned with its step, action, reward and return.
 */

import { key as makeKey } from 'mdml-engine/random'
import { type Env, type Episode, type Policy, runEpisode } from 'mdml-engine/rl'
import type { Scene } from 'mdml-engine/scene'
import { type ReactElement, useMemo } from 'react'
import { episodeSteps, stepCaption } from '../gym/frames.ts'
import { cx } from './controls/classes.ts'
import { FigureFrame } from './FigureFrame.tsx'
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
 * reward and return. The frames sit in a `FigureFrame`, so the episode expands and exports (as a GIF or WebM too).
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
      {shown === undefined ? null : (
        // One figure: its frame expands and exports the scene, and records the episode as a GIF or WebM.
        <FigureFrame
          width={width ?? shown.width}
          height={Math.round(((width ?? shown.width) * shown.height) / Math.max(1, shown.width))}
          filename="episode"
          playback={playback}
        >
          {(size) => <SceneView scene={shown} width={size.width} height={size.height} />}
        </FigureFrame>
      )}
      <Player controller={playback} stamp={(f) => stepCaption(steps[f])} />
    </div>
  )
}
