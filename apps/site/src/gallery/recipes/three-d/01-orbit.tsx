import { type ReactNode, useMemo, useState } from 'react'
import { axes3d, label3d, point3d, polyline3d, scene3d, surface3d } from 'whitebox-engine/scene'
import { Choice, ControlPanel, Scene3DView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Orbit a 3-D scene',
  question: 'How do I show a surface, a curve or points in three dimensions and let the reader turn them?',
  explain:
    'A `scene3d` holds points, segments, polylines, faces and labels in 3-D (`surface3d` makes a mesh from a function or a grid of heights); `Scene3DView` projects it through a camera and orbits on drag (or the arrow keys once focused); double-click returns to the start.',
}

const grid = Array.from({ length: 25 }, (_, i) => -2 + i / 6)

export default function Orbit(): ReactNode {
  const [shape, setShape] = useState<'surface' | 'helix' | 'cloud'>('surface')
  const s = useMemo(() => {
    if (shape === 'surface')
      return scene3d(
        [
          ...surface3d(grid, grid, (x, y) => Math.sin(x * 1.5) * Math.cos(y * 1.5) * 0.8, {
            fill: (z) => (z > 0 ? 'tab1' : 'tab0'),
            stroke: 'grid',
          }),
          ...axes3d(2.4),
        ],
        { title: 'sin(1.5x) cos(1.5y)' },
      )
    if (shape === 'helix')
      return scene3d([
        polyline3d(
          Array.from({ length: 300 }, (_, i): [number, number, number] => {
            const t = (i / 299) * 6 * Math.PI
            return [Math.cos(t), Math.sin(t), t / (3 * Math.PI) - 1]
          }),
          { stroke: 'tab2', strokeWidth: 2 },
        ),
        ...axes3d(1.6),
        label3d([0, 0, 1.2], 'helix'),
      ])
    // A deterministic cloud on a sphere (golden-angle spiral).
    return scene3d([
      ...Array.from({ length: 200 }, (_, i) => {
        const z = 1 - (2 * (i + 0.5)) / 200
        const r = Math.sqrt(1 - z * z)
        const a = i * Math.PI * (3 - Math.sqrt(5))
        return point3d([r * Math.cos(a), r * Math.sin(a), z], { fill: z > 0 ? 'tab4' : 'tab9', r: 3 })
      }),
      ...axes3d(1.4),
    ])
  }, [shape])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <Choice label="scene" value={shape} onChange={setShape} options={['surface', 'helix', 'cloud']} />
      </ControlPanel>
      <Scene3DView scene={s} key={shape} />
    </div>
  )
}
