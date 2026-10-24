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
  type Point2,
  SceneView,
  Toggle,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Ray casting',
  question: 'How do I show a first-person view of a grid world, and where its rays go?',
  explain:
    '`raycastViews(world, pose)` renders the first-person picture by column ray casting (the engine’s `renderRaycast`) and a map with the view cone: every few columns’ ray out to the wall it hits. Drag the eye on the map, turn it with the heading; switch off the fish-eye correction to see the distortion it removes.',
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
const [ROWS, COLS] = [9, 12]
const projection = linearProjection({
  x: [0, COLS * CELL],
  y: [ROWS * CELL, 0],
  width: COLS * CELL,
  height: ROWS * CELL,
})

export default function RayCasting(): ReactNode {
  const [eye, setEye] = useState<Point2>([1.5 * CELL, 1.5 * CELL])
  const [heading, setHeading] = useState(20)
  const [fisheye, setFisheye] = useState(false)
  const views = useMemo(() => {
    const pose: Pose = { x: eye[0] / CELL, y: eye[1] / CELL, heading: (heading * Math.PI) / 180 }
    return raycastViews(WORLD, pose, { width: 160, height: 100, cell: CELL, fisheye, viewWidth: 400, ...SURFACES })
  }, [eye, heading, fisheye])
  // Keep the eye on open floor: snap back if dragged into a wall.
  const onFloor = (p: Point2): Point2 => {
    const r = Math.floor(p[1] / CELL)
    const c = Math.floor(p[0] / CELL)
    const row = [
      '############',
      '#....#.....#',
      '#.##.#.###.#',
      '#.#......#.#',
      '#.#.####.#.#',
      '#...#..#...#',
      '###.#..###.#',
      '#..........#',
      '############',
    ][r]
    return row?.[c] === '.' ? p : eye
  }
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="heading (degrees)" value={heading} onChange={setHeading} min={-180} max={180} step={5} />
        <Toggle label="fish-eye (no correction)" value={fisheye} onChange={setFisheye} />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-4">
        <Overlaid
          width={views.topDown.width}
          height={views.topDown.height}
          overlay={
            <InteractionOverlay projection={projection} width={views.topDown.width} height={views.topDown.height}>
              <DraggablePoints
                points={[eye]}
                onChange={([p]) => p !== undefined && setEye(onFloor(p))}
                labels={['eye']}
                radius={7}
              />
            </InteractionOverlay>
          }
        >
          <SceneView scene={views.topDown} />
        </Overlaid>
        <SceneView scene={views.firstPerson} />
      </div>
    </div>
  )
}
