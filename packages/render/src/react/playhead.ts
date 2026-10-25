/**
 * The pure timing behind the player: the frame rates on offer, how many frames are due after some elapsed time
 * (carrying the remainder so playback keeps its rate on any display), stepping with or without looping, and time
 * stamps.
 */

/** The frame rates a player's menu offers. */
export const FPS_CHOICES: readonly number[] = [1, 2, 5, 10, 15, 24, 30, 60]

/**
 * How many frames are due after `elapsed` milliseconds at `fps` frames per second, with the time left over carried to
 * the next tick (so playback keeps its rate whatever the display's refresh rate). At most `cap` frames are due at once,
 * so a tab returning from the background does not race ahead.
 *
 * @param elapsed - Milliseconds since the last frame was due, including any carried remainder.
 * @param fps - The target frame rate.
 * @param cap - The most frames per tick (default 4).
 * @returns The frames due and the remainder in milliseconds.
 */
export function framesDue(elapsed: number, fps: number, cap = 4): { readonly frames: number; readonly carry: number } {
  if (!(fps > 0) || !(elapsed > 0)) return { frames: 0, carry: Math.max(elapsed, 0) }
  const interval = 1000 / fps
  const frames = Math.floor(elapsed / interval)
  if (frames > cap) return { frames: cap, carry: 0 }
  return { frames, carry: elapsed - frames * interval }
}

/**
 * The frame `delta` frames on from `frame` in a sequence of `count`: wrapping round with `loop`, else stopping at the
 * ends.
 *
 * @param frame - The current frame.
 * @param delta - How far to move (negative to go back).
 * @param count - The number of frames.
 * @param loop - Whether to wrap.
 * @returns The new frame, and whether it reached the last frame without wrapping.
 */
export function advanceFrame(
  frame: number,
  delta: number,
  count: number,
  loop: boolean,
): { readonly frame: number; readonly ended: boolean } {
  if (count <= 0) return { frame: 0, ended: true }
  const next = frame + delta
  if (loop) return { frame: ((next % count) + count) % count, ended: false }
  const clamped = Math.min(Math.max(next, 0), count - 1)
  return { frame: clamped, ended: delta > 0 && clamped === count - 1 }
}

/**
 * A time stamp for a frame: minutes, seconds and hundredths at the given rate (`0:01.25`).
 *
 * @param frame - The frame.
 * @param fps - Frames per second of the timeline's own clock.
 * @returns The stamp.
 */
export function timeStamp(frame: number, fps: number): string {
  const seconds = fps > 0 ? frame / fps : 0
  const m = Math.floor(seconds / 60)
  const s = seconds - m * 60
  return `${m}:${s.toFixed(2).padStart(5, '0')}`
}
