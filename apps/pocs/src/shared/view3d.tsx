import type { PinholeCamera } from 'mdml-engine/graphics'
import {
  type Camera,
  camera as makeCamera,
  fitCamera,
  type Item3D,
  point3d,
  type Scene3D,
  scene3d,
  segment3d,
  type Vec3,
} from 'mdml-engine/scene'
import { Scene3DView } from 'mdml-render/react'
import { type ReactNode, useMemo } from 'react'

/** The scene layer is z up and the library's cameras are y up: a point of the scene's world as the layer takes it. */
export const up = (p: readonly number[]): Vec3 => [p[0] as number, -(p[2] as number), p[1] as number]

/** A colour with channels in [0, 1] as a paint of the scene layer. */
export function rgba(rgb: readonly number[], alpha = 1): `rgba(${string})` {
  const c = (v: number): number => Math.round(255 * Math.min(1, Math.max(0, v)))
  return `rgba(${c(rgb[0] as number)}, ${c(rgb[1] as number)}, ${c(rgb[2] as number)}, ${alpha.toFixed(3)})`
}

/** Where a camera at the orbit's radius is drawn: nearer the scene, so the 3-D view need not hold the whole orbit. */
export const DRAWN_RADIUS = 1.25

/** A camera drawn (nearer than it is) as the pyramid from its eye to the picture plane a little way along its view. */
export function frustum(
  full: PinholeCamera,
  depth = 0.5,
  style: { stroke: 'accent' | 'muted' } = { stroke: 'accent' },
): Item3D[] {
  const k = DRAWN_RADIUS / Math.hypot(...full.eye)
  const cam = { ...full, eye: full.eye.map((v) => v * k) as unknown as Vec3 }
  const f = unit(cam.target.map((v, i) => v - (cam.eye[i] as number)))
  const r = unit(cross(f, cam.up))
  const u = cross(r, f)
  const h = depth * Math.tan((cam.fov * Math.PI) / 360)
  const at = (a: number, b: number): Vec3 =>
    up(cam.eye.map((v, i) => v + depth * (f[i] as number) + a * h * (r[i] as number) + b * h * (u[i] as number)))
  const corners = [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)]
  const eye = up(cam.eye)
  const s = { stroke: style.stroke, strokeWidth: 1.2 }
  return [
    ...corners.map((c) => segment3d(eye, c, s)),
    ...corners.map((c, i) => segment3d(c, corners[(i + 1) % 4] as Vec3, s)),
  ]
}

const cross = (a: readonly number[], b: readonly number[]): number[] => [
  (a[1] as number) * (b[2] as number) - (a[2] as number) * (b[1] as number),
  (a[2] as number) * (b[0] as number) - (a[0] as number) * (b[2] as number),
  (a[0] as number) * (b[1] as number) - (a[1] as number) * (b[0] as number),
]
const unit = (a: readonly number[]): number[] => {
  const n = Math.hypot(...a)
  return a.map((v) => v / n)
}

/** Dots at the eyes of the training cameras. */
export function cameraDots(cameras: readonly PinholeCamera[]): Item3D[] {
  return cameras.map((c) => {
    const k = DRAWN_RADIUS / Math.hypot(...c.eye)
    return point3d(up(c.eye.map((v) => v * k)), { fill: 'muted', r: 2.2, opacity: 0.8 })
  })
}

/** The thin cube that bounds the scene, so a sparse 3-D view still has a frame to orient by. */
export function frame(half: number): Item3D[] {
  const corners: Vec3[] = []
  for (const x of [-half, half])
    for (const y of [-half, half]) for (const z of [-half, half]) corners.push(up([x, y, z]))
  const edges: Item3D[] = []
  for (const [i, a] of corners.entries())
    for (const [j, b] of corners.entries()) {
      const differing = [0, 1, 2].filter((k) => a[k] !== b[k]).length
      if (j > i && differing === 1) edges.push(segment3d(a, b, { stroke: 'grid', strokeWidth: 0.8 }))
    }
  return edges
}

/**
 * The 3-D scene of items in an orbitable view, fitted once to a cube around the scene so it does not rescale as the
 * items change (a fitted scene would zoom as the Gaussians or the density move).
 *
 * @param props - The items and the size.
 * @returns The view.
 */
export function Orbitable(props: {
  readonly items: readonly Item3D[]
  readonly width: number
  readonly height: number
  /** Half the width of the cube the starting view fits (default 0.95, the scene's cube is 1). */
  readonly fit?: number
}): ReactNode {
  const { items, width, height, fit = 0.95 } = props
  const start = useMemo<Camera>(() => {
    const box: Scene3D = scene3d(frame(fit))
    return fitCamera(makeCamera({ azimuth: -55, elevation: 22 }), box, 1.0)
  }, [fit])
  const scene = useMemo(() => scene3d([...frame(1), ...items]), [items])
  return <Scene3DView scene={scene} camera={start} width={width} height={height} />
}
