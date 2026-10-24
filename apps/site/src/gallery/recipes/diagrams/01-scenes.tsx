import { type ReactNode, useMemo, useState } from 'react'
import {
  arrow,
  axes3d,
  camera,
  circle,
  ellipse,
  group,
  image,
  layer3d,
  path,
  rect,
  scene,
  scene3d,
  surface3d,
  tex,
  text,
} from 'whitebox-engine/scene'
import { ControlPanel, NumberField, SceneFrame, SceneView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Scenes',
  question: 'How do I draw a scene: shapes, arrows, curves, TeX, images and a 3-D layer?',
  explain:
    'A `Scene` is plain data from whitebox-engine (`scene`, `rect`, `circle`, `path`, `arrow`, `text`, `tex`, `image`, `group`, `layer3d`); colours are theme tokens (`fg`, `accent`, `tab0` to `tab9`) or literal colours. `SceneView` draws it as SVG in the page’s colour mode and reports hover and clicks by node id: hover a shape, click to highlight it.',
}

const checker = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => ((i + j) % 2 === 0 ? 1 : 0)))
const xs = Array.from({ length: 13 }, (_, i) => -1.5 + i / 4)

export default function Scenes(): ReactNode {
  const [azimuth, setAzimuth] = useState(35)
  const [hover, setHover] = useState<string | null>(null)
  const [picked, setPicked] = useState<string[]>([])
  const s = useMemo(
    () =>
      scene(
        640,
        360,
        [
          rect(20, 20, 140, 90, { id: 'rect', fill: 'tab0', radius: 8, title: 'a rounded rectangle' }),
          circle(240, 65, 45, { id: 'circle', fill: 'tab1', opacity: 0.85 }),
          ellipse(370, 65, 70, 35, { id: 'ellipse', fill: 'none', stroke: 'tab2', strokeWidth: 3, dash: [6, 4] }),
          arrow(30, 160, 190, 200, { id: 'arrow', stroke: 'accent', strokeWidth: 2 }),
          path(
            [
              [220, 210],
              [260, 140],
              [320, 230],
              [370, 150],
              [430, 200],
            ],
            { id: 'curve', smooth: true, stroke: 'tab3', strokeWidth: 3, arrow: true },
          ),
          text(30, 250, 'Plain text, in the theme’s colours', { size: 14 }),
          tex(30, 300, String.raw`p(\thetavec \mid \mathcal{D}) \propto p(\mathcal{D} \mid \thetavec)\, p(\thetavec)`, {
            size: 16,
          }),
          group([image(0, 0, 80, 80, checker, { id: 'image', smoothing: false })], {
            translate: [470, 20],
            rotate: 10,
          }),
          layer3d(
            scene3d([...surface3d(xs, xs, (x, y) => Math.exp(-(x * x + y * y)), { stroke: 'grid' }), ...axes3d(1.6)]),
            camera({ azimuth, elevation: 30 }),
            { x: 430, y: 135, width: 180, height: 210 },
            { id: 'surface' },
          ),
        ],
        { title: 'shapes, curves, text and a 3-D layer' },
      ),
    [azimuth],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField
          label="camera azimuth (degrees)"
          value={azimuth}
          onChange={setAzimuth}
          min={0}
          max={360}
          step={5}
        />
        <div className="self-center text-xs text-muted">
          hovering: <span className="font-mono text-fg">{hover ?? '—'}</span>
        </div>
      </ControlPanel>
      <SceneFrame
        scene={s}
        filename="scene"
        highlight={picked}
        onNodeHover={(id) => setHover(id)}
        onNodeClick={(id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]))}
      />
    </div>
  )
}
