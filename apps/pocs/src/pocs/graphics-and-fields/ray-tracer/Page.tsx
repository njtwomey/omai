import type { Vec3 } from 'omai-engine/base'
import {
  checkerboard,
  imageToPixels,
  type Material,
  material,
  pinholeCamera,
  plane,
  pointLight,
  rayScene,
  renderWhitted,
  sphere,
} from 'omai-engine/graphics'
import { bool, choice, dict, real } from 'omai-engine/space'
import { ControlPanel, SpaceControls } from 'omai-render/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { useGameLoop } from '../../../shared/useGameLoop.ts'

// A tiny Whitted ray tracer from the library's `graphics`: the scene is data, `renderWhitted` traces it, and this page
// only moves the camera, picks the resolution and puts the pixels on a canvas. The camera sits at `eye`, looks along
// `yaw` (0 faces -z, positive turns right) and `pitch`; y is up.

const SPACE = dict({
  width: choice([240, 360, 480, 640], { default: 480, label: 'full resolution (width in pixels)' }),
  reflections: bool({ default: true, label: 'mirror reflections' }),
  lightX: real(-8, 8, { default: 4, label: 'light x' }),
  lightY: real(1, 10, { default: 6, label: 'light height' }),
  lightZ: real(-8, 8, { default: 3, label: 'light z' }),
})
type Controls = { width: number; reflections: boolean; lightX: number; lightY: number; lightZ: number }
const START: Controls = { width: 480, reflections: true, lightX: 4, lightY: 6, lightZ: 3 }

const ASPECT = 16 / 9
const COARSE = 4 // while keys are held, the image is rendered at 1/4 of the full width
const SETTLE_MS = 160 // idle time before the full-resolution render
const SPEED = 3 // units per second
const TURN = 1.6 // radians per second

const shiny = (albedo: [number, number, number], mirror: number): Material =>
  material({ model: 'blinn-phong', albedo, specular: [0.7, 0.7, 0.7], shininess: 80, mirror })
const floorAt = (u: number, v: number): Vec3 =>
  checkerboard(u, v, { scale: 1, a: [0.75, 0.75, 0.72], b: [0.18, 0.18, 0.2] })

function buildScene(c: Controls): ReturnType<typeof rayScene> {
  const mirror = c.reflections ? 1 : 0
  return rayScene(
    [
      { shape: plane([0, 0, 0], [0, 1, 0]), material: shiny([1, 1, 1], 0.12 * mirror), albedoMap: floorAt },
      { shape: sphere([0, 1, 0], 1), material: shiny([0.9, 0.2, 0.15], 0.25 * mirror) },
      { shape: sphere([2.4, 0.7, 1], 0.7), material: shiny([0.2, 0.75, 0.3], 0.1 * mirror) },
      { shape: sphere([-2.3, 0.9, 0.6], 0.9), material: shiny([0.9, 0.9, 0.95], 0.7 * mirror) },
      { shape: sphere([0.9, 0.35, 2.4], 0.35), material: shiny([0.25, 0.4, 0.95], 0.15 * mirror) },
    ],
    {
      lights: [pointLight([c.lightX, c.lightY, c.lightZ], [70, 70, 70])],
      background: [0.35, 0.55, 0.9],
      ambient: [0.06, 0.06, 0.07],
    },
  )
}
interface Cam {
  x: number
  y: number
  z: number
  yaw: number
  pitch: number
}

export default function RayTracerPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const small = useRef<HTMLCanvasElement | null>(null)
  const cam = useRef<Cam>({ x: 0, y: 2.4, z: 6.5, yaw: 0, pitch: -0.3 })
  const scene = useMemo(() => buildScene(controls), [controls])
  const state = useRef({ scene, depth: 1, width: controls.width, lastInput: 0, dirty: true, refine: true })
  const info = useRef<HTMLSpanElement | null>(null)

  // A new scene, depth or size means the picture is stale: render coarse at once, refine when settled.
  useEffect(() => {
    const s = state.current
    s.scene = scene
    s.depth = controls.reflections ? 1 : 0
    s.width = controls.width
    s.dirty = true
    s.refine = true
    s.lastInput = 0
  }, [scene, controls.reflections, controls.width])

  const height = Math.round(controls.width / ASPECT)

  const draw = (div: number): string => {
    const s = state.current
    const w = Math.max(8, Math.round(s.width / div))
    const h = Math.max(8, Math.round(w / ASPECT))
    const c = cam.current
    const cp = Math.cos(c.pitch)
    const eye: [number, number, number] = [c.x, c.y, c.z]
    const target: [number, number, number] = [
      c.x + cp * Math.sin(c.yaw),
      c.y + Math.sin(c.pitch),
      c.z - cp * Math.cos(c.yaw),
    ]
    const t0 = performance.now()
    const r = renderWhitted(s.scene, pinholeCamera({ eye, target, fov: 55 }), w, h, { maxDepth: s.depth })
    const px = imageToPixels(r.image)
    const ms = performance.now() - t0
    const main = canvas.current
    const ctx = main?.getContext('2d')
    if (main === null || ctx === null || ctx === undefined) return ''
    if (div === 1) {
      const img = ctx.createImageData(w, h)
      img.data.set(px.rgba)
      ctx.putImageData(img, 0, 0)
    } else {
      const tmp = small.current ?? document.createElement('canvas')
      small.current = tmp
      tmp.width = w
      tmp.height = h
      const tctx = tmp.getContext('2d')
      if (tctx === null) return ''
      const img = tctx.createImageData(w, h)
      img.data.set(px.rgba)
      tctx.putImageData(img, 0, 0)
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(tmp, 0, 0, main.width, main.height)
    }
    return `${w} x ${h}, ${r.stats.rays} rays (${r.stats.shadowRays} shadow), ${Math.round(ms)} ms`
  }

  const show = (text: string): void => {
    if (info.current !== null) info.current.textContent = text === '' ? '' : ` Last frame: ${text}.`
  }

  useGameLoop((dt, held, now) => {
    const s = state.current
    const c = cam.current
    const key = (code: string): number => (held.has(code) ? 1 : 0)
    const fwd = key('KeyW') - key('KeyS')
    const side = key('KeyD') - key('KeyA')
    const yaw = key('ArrowRight') - key('ArrowLeft')
    const pitch = key('ArrowUp') - key('ArrowDown')
    if (fwd !== 0 || side !== 0 || yaw !== 0 || pitch !== 0) {
      c.yaw += yaw * TURN * dt
      c.pitch = Math.max(-1.3, Math.min(1.3, c.pitch + pitch * TURN * 0.6 * dt))
      c.x += (Math.sin(c.yaw) * fwd + Math.cos(c.yaw) * side) * SPEED * dt
      c.z += (-Math.cos(c.yaw) * fwd + Math.sin(c.yaw) * side) * SPEED * dt
      s.lastInput = now
      s.dirty = true
      s.refine = true
    }
    if (s.dirty) {
      s.dirty = false
      show(draw(COARSE))
    } else if (s.refine && held.size === 0 && now - s.lastInput > SETTLE_MS) {
      s.refine = false
      show(draw(1))
    }
  })

  return (
    <div className="flex flex-col gap-4">
      <ControlPanel title="resolution, reflections and the light" columns={3}>
        <SpaceControls space={SPACE} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <canvas
        ref={canvas}
        width={controls.width}
        height={height}
        aria-label="ray traced scene"
        style={{
          width: '100%',
          maxWidth: 960,
          aspectRatio: '16 / 9',
          background: '#000',
          borderRadius: 6,
          imageRendering: 'pixelated',
        }}
      />
      <p className="text-xs text-muted">
        W A S D move (forward, left, back, right); the arrow keys turn and pitch. It renders at a quarter of the
        resolution while a key is held and at full resolution once you stop.
        <span ref={info} />
      </p>
    </div>
  )
}
