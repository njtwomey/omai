import { type ReactNode, useMemo, useState } from 'react'
import { bakeTexture, bricks } from 'whitebox-engine/graphics'
import { fromAscii, type Pose } from 'whitebox-engine/worlds'
import { raycastViews } from 'whitebox-render/gym'
import {
  ControlPanel,
  DraggablePoints,
  InteractionOverlay,
  linearProjection,
  NumberField,
  Overlaid,
  SceneView,
  Toggle,
  useHeldKeys,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Ray casting',
  question: 'How do I show a first-person view of a grid world, and where its rays go?',
  explain:
    '`raycastViews(world, pose)` renders the first-person picture by column ray casting (the engine’s `renderRaycast`), one ray per column, and a map with the view cone. Click the figure and walk: W or ↑ forward, S or ↓ back, A and D to step sideways, ← and → to turn (`useHeldKeys` captures them while the figure has focus); or drag the eye on the map. Fewer rays make a coarser picture and a sparser fan; switch off the fish-eye correction to see the distortion it removes.',
}

const MAP = [
  '############',
  '#....#.....#',
  '#.##.#.###.#',
  '#.#......#.#',
  '#.#.####.#.#',
  '#...#..#...#',
  '###.#..###.#',
  '#..........#',
  '############',
]
const WORLD = fromAscii(MAP)
const CELL = 30
const SURFACES = {
  walls: { wall: bakeTexture(64, 64, (u, v) => bricks(u, v)) },
  floor: [0.32, 0.3, 0.28] as const,
  ceiling: [0.55, 0.68, 0.85] as const,
}
// Data coordinates are cells, y growing downwards as in the world (the y range runs from the bottom row up to 0).
const projection = linearProjection({
  x: [0, MAP[0]?.length ?? 0],
  y: [MAP.length, 0],
  width: (MAP[0]?.length ?? 0) * CELL,
  height: MAP.length * CELL,
})
const KEYS = ['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']
const SPEED = 2.5 // cells per second
const TURN = Math.PI * 0.75 // radians per second

const open = (x: number, y: number): boolean => MAP[Math.floor(y)]?.[Math.floor(x)] === '.'

export default function RayCasting(): ReactNode {
  const [pose, setPose] = useState<Pose>({ x: 1.5, y: 1.5, heading: 0.3 })
  const [rays, setRays] = useState(64)
  const [fisheye, setFisheye] = useState(false)
  const keys = useHeldKeys(KEYS, {
    onFrame: (held, dt) =>
      setPose((p) => {
        const has = (...k: string[]): boolean => k.some((x) => held.has(x))
        const heading = p.heading + ((has('ArrowRight') ? 1 : 0) - (has('ArrowLeft') ? 1 : 0)) * TURN * dt
        const forward = (has('w', 'ArrowUp') ? 1 : 0) - (has('s', 'ArrowDown') ? 1 : 0)
        const side = (has('d') ? 1 : 0) - (has('a') ? 1 : 0)
        const dx = (Math.cos(heading) * forward - Math.sin(heading) * side) * SPEED * dt
        const dy = (Math.sin(heading) * forward + Math.cos(heading) * side) * SPEED * dt
        // Slide along walls: take each axis of the move only where it stays on open floor.
        const x = open(p.x + dx, p.y) ? p.x + dx : p.x
        const y = open(x, p.y + dy) ? p.y + dy : p.y
        return { x, y, heading }
      }),
  })
  const views = useMemo(
    () =>
      raycastViews(WORLD, pose, {
        width: rays,
        height: Math.max(4, Math.round(rays * 0.625)),
        cell: CELL,
        fisheye,
        viewWidth: 400,
        rayEvery: 1,
        ...SURFACES,
      }),
    [pose, rays, fisheye],
  )
  const size = { width: views.topDown.width, height: views.topDown.height }
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="rays (columns)" value={rays} onChange={setRays} min={8} max={320} integer log />
        <NumberField
          label="heading (degrees)"
          value={Math.round((((pose.heading * 180) / Math.PI) % 360) + 360) % 360}
          onChange={(deg) => setPose((p) => ({ ...p, heading: (deg * Math.PI) / 180 }))}
          min={0}
          max={359}
          step={5}
        />
        <Toggle label="fish-eye (no correction)" value={fisheye} onChange={setFisheye} />
      </ControlPanel>
      <div
        {...keys.props}
        className="flex flex-wrap items-start gap-4 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
      >
        <Overlaid
          {...size}
          overlay={
            <InteractionOverlay projection={projection} {...size}>
              <DraggablePoints
                points={[[pose.x, pose.y]]}
                onChange={([p]) => {
                  if (p !== undefined && open(p[0], MAP.length - p[1]))
                    setPose((q) => ({ ...q, x: p[0], y: MAP.length - p[1] }))
                }}
                labels={['eye']}
                radius={7}
              />
            </InteractionOverlay>
          }
        >
          <SceneView scene={views.topDown} />
        </Overlaid>
        <div className="flex flex-col gap-1">
          <SceneView scene={views.firstPerson} style={{ imageRendering: 'pixelated' }} />
          <span className="text-xs text-muted">
            {keys.held.size > 0 ? 'moving…' : 'Click here, then W A S D or the arrow keys to move.'}
          </span>
        </div>
      </div>
    </div>
  )
}
