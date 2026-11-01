import {
  bakeTexture,
  bricks,
  ddaCast,
  imageToPixels,
  renderRaycast,
  type Surface,
  viewConeScene,
} from 'mdml-engine/graphics'
import { key } from 'mdml-engine/random'
import { choice, dict, int, real } from 'mdml-engine/space'
import {
  cellCentre,
  generateMaze,
  type MazeAlgorithmName,
  type Pose,
  stepPose,
  worldGeometry,
} from 'mdml-engine/worlds'
import { ControlPanel, SceneView, SpaceControls } from 'mdml-render/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { useGameLoop } from '../../shared/useGameLoop.ts'

// A Wolfenstein-style maze from the library: `generateMaze` makes the walls, `renderRaycast` draws the first-person
// view by casting one DDA ray per screen column, `viewConeScene` draws the minimap with the same rays, and `stepPose`
// moves the player with wall collision. This page only reads the keys and puts the pixels on a canvas.

const ALGORITHMS: MazeAlgorithmName[] = ['backtracker', 'prim', 'kruskal', 'wilson', 'binaryTree', 'sidewinder']
const SPACE = dict({
  rows: int(4, 20, { default: 8, label: 'rows' }),
  cols: int(4, 20, { default: 8, label: 'columns' }),
  algorithm: choice(ALGORITHMS, { default: 'backtracker', label: 'maze algorithm' }),
  seed: int(0, 999, { default: 7, label: 'seed' }),
  fog: real(0, 0.6, { default: 0.18, label: 'fog (distance shading)' }),
})
type Controls = { rows: number; cols: number; algorithm: MazeAlgorithmName; seed: number; fog: number }
const START: Controls = { rows: 8, cols: 8, algorithm: 'backtracker', seed: 7, fog: 0.18 }

const W = 320
const H = 200
const FOV = Math.PI / 3
const SPEED = 2.4 // cells per second
const TURN = 2.2 // radians per second
const RADIUS = 0.2
const BRICK = bakeTexture(64, 64, (u, v) =>
  bricks(u, v, { rows: 6, cols: 3, brick: [0.62, 0.3, 0.2], mortarColour: [0.55, 0.52, 0.48] }),
)
const FOG_COLOUR: [number, number, number] = [0.01, 0.01, 0.02]

/** The player's starting pose: in the start cell, facing the direction with the longest view. */
function startPose(w: ReturnType<typeof generateMaze>): Pose {
  const [x, y] = cellCentre(w, w.starts[0] ?? 0)
  let best = 0
  let far = -1
  for (let k = 0; k < 4; k++) {
    const d = ddaCast(w, [x, y], (k * Math.PI) / 2).hit?.distance ?? 0
    if (d > far) {
      far = d
      best = k
    }
  }
  return { x, y, heading: (best * Math.PI) / 2 }
}

export default function RaycasterMazePage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const maze = useMemo(
    () => generateMaze(key(controls.seed), { rows: controls.rows, cols: controls.cols, algorithm: controls.algorithm }),
    [controls.seed, controls.rows, controls.cols, controls.algorithm],
  )
  const geometry = useMemo(() => worldGeometry(maze), [maze])
  const goal = useMemo(() => cellCentre(maze, maze.goals[0] ?? maze.n - 1), [maze])
  const walls = useMemo<Record<string, Surface>>(
    () => Object.fromEntries(maze.terrains.map((t) => [t.name, BRICK])),
    [maze],
  )

  const canvas = useRef<HTMLCanvasElement | null>(null)
  const pose = useRef<Pose>(startPose(maze))
  const dirty = useRef(true)
  const minimap = useRef<((p: Pose) => void) | null>(null)
  const won = useRef<HTMLSpanElement | null>(null)

  // A new maze or fog: back to the start, redraw.
  useEffect(() => {
    pose.current = startPose(maze)
    dirty.current = true
  }, [maze])

  useGameLoop((dt, held) => {
    const key = (code: string): number => (held.has(code) ? 1 : 0)
    const forward = key('KeyW') + key('ArrowUp') - key('KeyS') - key('ArrowDown')
    const strafe = key('KeyD') - key('KeyA')
    const turn = key('ArrowRight') - key('ArrowLeft')
    if (forward !== 0 || strafe !== 0 || turn !== 0) {
      pose.current = stepPose(
        geometry,
        pose.current,
        { forward: forward * SPEED * dt, strafe: strafe * SPEED * dt, turn: turn * TURN * dt },
        RADIUS,
      ).pose
      dirty.current = true
    }
    if (!dirty.current) return
    dirty.current = false
    const p = pose.current
    const ctx = canvas.current?.getContext('2d')
    if (ctx !== null && ctx !== undefined) {
      const frame = renderRaycast(maze, p, {
        width: W,
        height: H,
        fov: FOV,
        walls,
        edge: BRICK,
        floor: [0.16, 0.14, 0.12],
        ceiling: [0.08, 0.09, 0.14],
        fog: { colour: FOG_COLOUR, density: controls.fog },
        sprites: [{ x: goal[0], y: goal[1], colour: [1, 0.85, 0.2], size: 0.35 }],
      })
      const img = ctx.createImageData(W, H)
      img.data.set(imageToPixels(frame.image).rgba)
      ctx.putImageData(img, 0, 0)
    }
    minimap.current?.(p)
    if (won.current !== null) {
      const near = Math.hypot(p.x - goal[0], p.y - goal[1]) < 0.5
      won.current.textContent = near ? ' You reached the gold marker. Pick another seed for a new maze.' : ''
    }
  })

  return (
    <div className="flex flex-col gap-4">
      <ControlPanel title="the maze and the fog" columns={3}>
        <SpaceControls
          space={SPACE}
          value={controls}
          onChange={(v) => {
            dirty.current = true
            setControls(v as unknown as Controls)
          }}
        />
      </ControlPanel>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
        <canvas
          ref={canvas}
          width={W}
          height={H}
          aria-label="first-person view"
          style={{
            flex: '1 1 480px',
            maxWidth: 800,
            minWidth: 0,
            aspectRatio: `${W} / ${H}`,
            background: '#000',
            borderRadius: 6,
            imageRendering: 'pixelated',
          }}
        />
        <Minimap maze={maze} bus={minimap} />
      </div>
      <p className="text-xs text-muted">
        W / S (or Up / Down) walk, A / D strafe, Left / Right turn. Find the gold marker in the far corner.
        <span ref={won} />
      </p>
    </div>
  )
}

/** The top-down map: the world, the player and the rays the first-person view casts, redrawn as the player moves. */
function Minimap(props: {
  readonly maze: ReturnType<typeof generateMaze>
  readonly bus: { current: ((p: Pose) => void) | null }
}): ReactNode {
  const { maze, bus } = props
  const [pose, setPose] = useState<Pose>(() => startPose(maze))
  useEffect(() => {
    bus.current = setPose
    return () => {
      bus.current = null
    }
  }, [bus])
  const cell = Math.max(14, Math.min(32, Math.floor(360 / Math.max(maze.n ** 0.5, 1))))
  const scene = useMemo(
    () => viewConeScene(maze, pose, { rays: 13, fov: FOV, cell, highlight: -1 }),
    [maze, pose, cell],
  )
  return (
    <div style={{ flex: '0 0 auto', maxWidth: '100%' }}>
      <SceneView scene={scene} />
    </div>
  )
}
