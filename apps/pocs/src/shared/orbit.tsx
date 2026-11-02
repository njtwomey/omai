import { orbitCamera } from 'mdml-engine/datasets'
import type { PinholeCamera } from 'mdml-engine/graphics'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { PixelCanvas } from './PixelCanvas.tsx'

/** Where a camera on the orbit is: degrees about the vertical axis and above the horizontal plane. */
export interface Pose {
  readonly azimuth: number
  readonly elevation: number
}

/** The view the page opens on: between two training cameras, above the scene. */
export const START_POSE: Pose = { azimuth: 35, elevation: 30 }

/** The camera of a pose, on the orbit of the given radius around the origin, looking at it. */
export function poseCamera(pose: Pose, radius: number, fov: number): PinholeCamera {
  return orbitCamera(radius, pose.elevation, pose.azimuth, fov, 0.1, 100)
}

/** The drawn width for an image of `side` pixels: the largest whole multiple of `side` within `limit` CSS pixels. */
export function drawnSize(side: number, limit: number): number {
  return side * Math.max(1, Math.floor(limit / side))
}

/** What {@link useProgressiveImage} has drawn so far. */
export interface Progressive {
  readonly rgb: readonly number[] | undefined
  /** The side in pixels of the image drawn and the stage it is (0 is the first, coarsest). */
  readonly side: number
  readonly stage: number
  /** How long the last render of each stage took, in milliseconds. */
  readonly ms: readonly (number | undefined)[]
}

/**
 * Renders a picture in stages of growing resolution without holding the page: the first stage (small, quick enough to
 * follow a drag) is drawn in the next animation frame after anything in `deps` changes, and each later one a moment
 * later if nothing has changed since, so a drag shows the preview and letting go sharpens it. A change cancels the
 * stages still to come.
 *
 * @param sides - The side in pixels of each stage, smallest first.
 * @param render - Renders the picture at a side (read at the moment it runs).
 * @param deps - What the picture depends on.
 * @param settleMs - The pause before each later stage (default 120).
 * @returns The latest image and how long each stage took.
 */
export function useProgressiveImage(
  sides: readonly number[],
  render: (side: number) => readonly number[],
  deps: readonly unknown[],
  settleMs = 120,
): Progressive {
  const latest = useRef(render)
  latest.current = render
  const [state, setState] = useState<Progressive>({ rgb: undefined, side: sides[0] ?? 1, stage: 0, ms: [] })
  const times = useRef<(number | undefined)[]>([])
  // biome-ignore lint/correctness/useExhaustiveDependencies: the caller names what the picture depends on
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    let live = true
    const run = (stage: number): void => {
      if (!live) return
      const side = sides[stage] as number
      const started = performance.now()
      const rgb = latest.current(side)
      times.current[stage] = performance.now() - started
      setState({ rgb, side, stage, ms: [...times.current] })
      if (stage + 1 < sides.length) timer = setTimeout(() => run(stage + 1), settleMs)
    }
    const frame = requestAnimationFrame(() => run(0))
    return () => {
      live = false
      cancelAnimationFrame(frame)
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [...deps, sides.join(','), settleMs])
  return state
}

/**
 * A picture the reader drags to turn the camera: horizontal drags move the azimuth, vertical ones the elevation, and a
 * double click returns to the start. The picture is drawn pixelated, at a whole multiple of its resolution.
 *
 * @param props - The pose and its setter, the image, the drawn size and the caption.
 * @returns The draggable picture.
 */
export function OrbitImage(props: {
  readonly pose: Pose
  readonly onPose: (pose: Pose) => void
  readonly image: Progressive
  /** The drawn width in CSS pixels, at most (the largest whole multiple of the image's side that fits). */
  readonly limit: number
  readonly caption: string
}): ReactNode {
  const { pose, onPose, image, limit, caption } = props
  const drag = useRef<{ x: number; y: number; pose: Pose } | null>(null)
  return (
    <PixelCanvas
      width={image.side}
      height={image.side}
      rgb={image.rgb}
      size={drawnSize(image.side, limit)}
      caption={caption}
    >
      <div
        role="application"
        aria-label="drag to turn the camera around the scene"
        className="absolute inset-0 cursor-grab touch-none active:cursor-grabbing"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, pose }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (d === null || !e.currentTarget.hasPointerCapture(e.pointerId)) return
          onPose({
            azimuth: d.pose.azimuth - (e.clientX - d.x) * 0.7,
            elevation: Math.max(-80, Math.min(85, d.pose.elevation + (e.clientY - d.y) * 0.5)),
          })
        }}
        onPointerUp={() => {
          drag.current = null
        }}
        onDoubleClick={() => onPose(START_POSE)}
      />
    </PixelCanvas>
  )
}
