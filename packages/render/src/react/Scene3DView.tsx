/**
 * A 3-D scene (points, segments, polylines, faces, labels and surfaces from the engine's `scene3d`) with a
 * drag-to-orbit camera: dragging turns the azimuth and elevation and the scene is projected again.
 */

import { type Camera, fitCamera, layer3d, camera as makeCamera, orbit, type Scene3D, scene } from 'omai-engine/scene'
import { type ReactElement, useMemo, useRef, useState } from 'react'
import { cx } from './controls/classes.ts'
import { SceneView } from './SceneView.tsx'

/** Props of {@link Scene3DView}. */
export interface Scene3DViewProps {
  readonly scene: Scene3D
  /** The starting camera (default one fitted to the scene); the view orbits from it. */
  readonly camera?: Camera
  /** Drawn size in pixels (default 480 by 360). */
  readonly width?: number
  readonly height?: number
  /** Degrees of orbit per pixel dragged (default 0.5). */
  readonly sensitivity?: number
  /** Called with the camera as the reader orbits. */
  readonly onCamera?: (camera: Camera) => void
  readonly className?: string
}

/**
 * A 3-D scene (`scene3d` from omai-engine: points, segments, polylines, faces, labels, surfaces) with a
 * drag-to-orbit camera: dragging turns the azimuth and elevation and the scene is projected again; double-click returns
 * to the starting view. The arrow keys orbit too while the view has focus.
 *
 * @param props - The scene, the starting camera, the size, the drag sensitivity and a camera callback.
 * @returns The view.
 * @index function React component: a 3-D scene you orbit by dragging: `<Scene3DView scene={scene3d} />`
 */
export function Scene3DView(props: Scene3DViewProps): ReactElement {
  const { scene: s3, width = 480, height = 360, sensitivity = 0.5, onCamera, className } = props
  const start = useMemo(() => props.camera ?? fitCamera(makeCamera(), s3), [props.camera, s3])
  const [cam, setCam] = useState<Camera | null>(null)
  const current = cam ?? start
  const drag = useRef<{ x: number; y: number; cam: Camera } | null>(null)
  const move = (next: Camera): void => {
    setCam(next)
    onCamera?.(next)
  }
  const drawing = useMemo(
    () =>
      scene(
        width,
        height,
        [layer3d(s3, current, { x: 0, y: 0, width, height })],
        s3.title === undefined ? {} : { title: s3.title },
      ),
    [s3, current, width, height],
  )
  return (
    <div
      role="application"
      // biome-ignore lint/a11y/noNoninteractiveTabindex: an application region takes focus so the arrow keys orbit
      tabIndex={0}
      aria-label="3-D view: drag or use the arrow keys to orbit"
      className={cx(
        'w-fit max-w-full cursor-grab touch-none rounded-lg outline-none select-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/60 active:cursor-grabbing',
        className,
      )}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { x: e.clientX, y: e.clientY, cam: current }
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (d === null || !e.currentTarget.hasPointerCapture(e.pointerId)) return
        move(orbit(d.cam, -(e.clientX - d.x) * sensitivity, (e.clientY - d.y) * sensitivity))
      }}
      onPointerUp={() => {
        drag.current = null
      }}
      onDoubleClick={() => move(start)}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 15 : 5
        const turn: Record<string, [number, number]> = {
          ArrowLeft: [step, 0],
          ArrowRight: [-step, 0],
          ArrowUp: [0, step],
          ArrowDown: [0, -step],
        }
        const t = turn[e.key]
        if (t === undefined) return
        e.preventDefault()
        move(orbit(current, t[0], t[1]))
      }}
    >
      <SceneView scene={drawing} />
    </div>
  )
}
