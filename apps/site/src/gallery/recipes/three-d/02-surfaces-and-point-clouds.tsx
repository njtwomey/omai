import { key, normal } from 'mdml-engine/random'
import { axes3d, point3d, scene3d, surface3d } from 'mdml-engine/scene'
import { toArray } from 'mdml-engine/tensor'
import { Choice, ControlPanel, FigureFrame, NumberField, Scene3DView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Surfaces and point clouds',
  question: 'How do I draw a function of two variables as a surface, or a 3-D scatter of points?',
  explain:
    '`surface3d(x, y, f, { fill })` meshes a function over a grid, here coloured by height; `point3d` draws each point of a cloud. Both orbit in `Scene3DView`: drag to turn, double-click to come back. The cloud is three Gaussian clusters; pull them apart with the separation.',
}

const grid = (n: number): number[] => Array.from({ length: n }, (_, i) => -2 + (4 * i) / (n - 1))
type SurfaceShape = 'saddle' | 'ripple' | 'clusters'

export default function SurfacesAndPointClouds(): ReactNode {
  const [shape, setShape] = useState<SurfaceShape>('ripple')
  const [detail, setDetail] = useState(25)
  const [separation, setSeparation] = useState(1.2)
  const scene = useMemo(() => {
    if (shape !== 'clusters') {
      const f =
        shape === 'saddle'
          ? (x: number, y: number): number => 0.25 * (x * x - y * y)
          : (x: number, y: number): number => 0.6 * Math.cos(2 * Math.hypot(x, y)) * Math.exp(-0.25 * (x * x + y * y))
      return scene3d([
        ...surface3d(grid(detail), grid(detail), f, {
          fill: (z) => (z > 0.2 ? 'tab3' : z > 0 ? 'tab1' : z > -0.2 ? 'tab0' : 'tab9'),
          stroke: 'grid',
        }),
        ...axes3d(2.4),
      ])
    }
    const z = toArray(normal(key(1), [3 * 3 * 80])) as number[]
    const centres: [number, number, number][] = [
      [separation, 0, 0],
      [-separation / 2, separation * 0.87, 0],
      [-separation / 2, -separation * 0.87, separation * 0.5],
    ]
    return scene3d([
      ...centres.flatMap((c, k) =>
        Array.from({ length: 80 }, (_, i) => {
          const b = 3 * (k * 80 + i)
          return point3d([c[0] + 0.35 * (z[b] ?? 0), c[1] + 0.35 * (z[b + 1] ?? 0), c[2] + 0.35 * (z[b + 2] ?? 0)], {
            fill: (['tab0', 'tab1', 'tab2'] as const)[k] ?? 'tab0',
            r: 2.5,
          })
        }),
      ),
      ...axes3d(2),
    ])
  }, [shape, detail, separation])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="scene"
          value={shape}
          onChange={setShape}
          options={['ripple', 'saddle', 'clusters'] as SurfaceShape[]}
        />
        <NumberField label="grid points" value={detail} onChange={setDetail} min={6} max={60} integer />
        <NumberField label="separation" value={separation} onChange={setSeparation} min={0} max={2.5} step={0.1} />
      </ControlPanel>
      <FigureFrame
        width={520}
        height={400}
        filename="surfaces-and-point-clouds"
        caption="Drag to orbit; double-click to return to the start."
      >
        <Scene3DView scene={scene} key={shape} width={520} height={400} />
      </FigureFrame>
    </div>
  )
}
