/**
 * The React view of an audio clip: its waveform above its spectrogram with a playhead, which follows the viewed step
 * and is moved by clicking the picture, and a button that plays the clip through WebAudio from the playhead, moving
 * the step cursor along as it sounds. Playback needs a browser's `AudioContext`; without one the button says so.
 */

import { type AudioClip, audioDuration, playheadAt, stepAt } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { audioLayout, audioSceneAt, spectrogramOf } from '../../views/audio.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { SceneView } from '../SceneView.tsx'
import { ToolButton } from '../session/parts.tsx'
import { useColourMode } from '../theme.tsx'
import type { RecordViewProps } from './_frame.tsx'

/** Props of {@link AudioView}: the record, the tape step the playhead stands at and what moves the step. */
export interface AudioViewProps extends RecordViewProps<AudioClip> {
  /** The tape step (default 0). */
  readonly step?: number
  /** Called with the step to move the cursor to: by a click on the picture or by the sound playing. */
  readonly onStep?: (step: number) => void
}

const W = 440
const H = 300

/**
 * Plays a clip from a time through WebAudio and reports the time as it sounds; one playback at a time.
 *
 * @param clip - The clip.
 * @param onTime - Called with the time in seconds as the clip plays (and at its end).
 * @returns Whether it plays, whether the browser can, and the play and stop functions.
 */
export function useClipPlayback(
  clip: AudioClip,
  onTime: (seconds: number) => void,
): { readonly playing: boolean; readonly available: boolean; play: (from: number) => void; stop: () => void } {
  const [playing, setPlaying] = useState(false)
  const available = typeof AudioContext !== 'undefined'
  const state = useRef<{ ctx: AudioContext; source: AudioBufferSourceNode; started: number; from: number } | null>(null)
  const raf = useRef(0)
  const handler = useRef(onTime)
  handler.current = onTime
  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current)
    try {
      state.current?.source.stop()
      void state.current?.ctx.close()
    } catch {
      // Already stopped.
    }
    state.current = null
    setPlaying(false)
  }, [])
  const play = useCallback(
    (from: number) => {
      if (typeof AudioContext === 'undefined') return
      stop()
      const ctx = new AudioContext()
      const buffer = ctx.createBuffer(1, clip.samples.length, clip.sampleRate)
      buffer.copyToChannel(Float32Array.from(clip.samples), 0)
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)
      const start = from >= audioDuration(clip) - 0.01 ? 0 : from
      source.start(0, start)
      state.current = { ctx, source, started: ctx.currentTime, from: start }
      setPlaying(true)
      const tick = (): void => {
        const s = state.current
        if (s === null) return
        const t = s.from + (s.ctx.currentTime - s.started)
        if (t >= audioDuration(clip)) {
          handler.current(audioDuration(clip))
          stop()
          return
        }
        handler.current(t)
        raf.current = requestAnimationFrame(tick)
      }
      raf.current = requestAnimationFrame(tick)
    },
    [clip, stop],
  )
  useEffect(() => stop, [stop])
  return { playing, available, play, stop }
}

/**
 * An audio clip with its playhead. Clicking the picture moves the playhead (the step cursor, through `onStep`); the
 * button plays the clip from the playhead and the cursor follows the sound.
 *
 * @param props - The record, the step, the callback and the frame settings.
 * @returns The figure.
 */
export function AudioView(props: AudioViewProps): ReactElement {
  const { record, step = 0, onStep } = props
  const mode = useColourMode()
  const spectrogram = useMemo(() => record.spectrogram ?? spectrogramOf(record.samples, record.sampleRate), [record])
  const scene = useMemo(
    () => audioSceneAt(record, step, { width: W, height: H, mode, spectrogram }),
    [record, step, mode, spectrogram],
  )
  const last = useRef(step)
  const follow = useCallback(
    (seconds: number) => {
      const s = stepAt(record, seconds)
      if (s !== last.current) {
        last.current = s
        onStep?.(s)
      }
    },
    [record, onStep],
  )
  const player = useClipPlayback(record, follow)
  const dur = audioDuration(record)
  const time = playheadAt(record, step)
  const seek = (e: React.MouseEvent<HTMLButtonElement>): void => {
    const r = e.currentTarget.getBoundingClientRect()
    const L = audioLayout(W, H)
    const px = ((e.clientX - r.left) / r.width) * W
    const t = Math.min(dur, Math.max(0, ((px - L.left) / L.plot) * dur))
    last.current = stepAt(record, t)
    onStep?.(last.current)
    if (player.playing) player.play(t)
  }
  return (
    <FigureFrame
      width={W}
      height={H}
      filename={props.filename ?? 'audio'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <ToolButton
              label={player.playing ? '■ stop' : '▶ play sound'}
              title={player.playing ? 'Stop the sound' : 'Play the clip from the playhead'}
              pressed={player.playing}
              disabled={!player.available}
              onClick={() => (player.playing ? player.stop() : player.play(time))}
            />
            <span
              className="text-xs text-muted tabular-nums"
              aria-live="off"
            >{`playhead ${time.toFixed(2)} s of ${dur.toFixed(2)} s`}</span>
          </div>
          <span className="text-xs text-muted">
            {props.caption ?? record.caption ?? 'Click the picture to move the playhead.'}
            {player.available ? '' : ' Audio playback is not available in this browser.'}
          </span>
        </div>
      }
    >
      <button
        type="button"
        onClick={seek}
        onKeyDown={(e) => {
          const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
          if (d === 0) return
          e.preventDefault()
          last.current = Math.max(0, step + d)
          onStep?.(last.current)
        }}
        aria-label="The clip: click to move the playhead, or use the left and right arrow keys to step it"
        className="block cursor-crosshair border-0 bg-transparent p-0"
        data-mdml-svg=""
        style={{ width: W, height: H }}
      >
        <SceneView scene={scene} width={W} height={H} mode={mode} />
      </button>
    </FigureFrame>
  )
}
