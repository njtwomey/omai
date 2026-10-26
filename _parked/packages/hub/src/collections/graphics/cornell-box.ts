/**
 * Hub entry for the Cornell box path-traced in the browser: a mirror and a glass sphere (or two blocks) in a room with
 * a red and a green wall, the image sharpening as the samples per pixel double.
 */

import { cornellBox as cornellScene } from 'mdml-engine/graphics'
import { progressiveRender } from 'mdml-engine/methods/graphics'
import { choice, dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** The Cornell box path-traced live: the noise falls away as the samples per pixel double. */
export const cornellBox: HubEntry = defineEntry({
  id: 'graphics/cornell-box',
  title: 'Path tracing a Cornell box',
  summary:
    'Light followed backwards from the camera, bounce by bounce: soft shadows, colour bleeding off the walls, a mirror and a caustic under the glass sphere, sharpening as samples accumulate.',
  collection: 'graphics',
  tags: ['graphics', 'path tracing', 'Monte Carlo', 'rendering'],
  datasets: [],
  params: dict({
    contents: choice(['spheres', 'blocks'] as const, { default: 'spheres' }),
    size: int(32, 192, { default: 96, label: 'image size (pixels)' }),
    samples: int(1, 512, { default: 128, label: 'samples per pixel' }),
    bounces: int(1, 8, { default: 5, label: 'most bounces' }),
    lamp: real(2, 40, { default: 12, label: 'lamp radiance' }),
    exposure: real(-3, 3, { default: 0, label: 'exposure (stops)' }),
  }),
  run: ({ params, key, progress }) => {
    const { scene, camera } = cornellScene({
      contents: (params['contents'] ?? 'spheres') as 'spheres' | 'blocks',
      lamp: num(params['lamp'], 12),
    })
    const size = num(params['size'], 96)
    const samples = num(params['samples'], 128)
    return progressiveRender(scene, camera, size, size, {
      key,
      samples,
      maxBounces: num(params['bounces'], 5),
      exposure: num(params['exposure'], 0),
      ...(progress === undefined
        ? {}
        : {
            onProgress: (fraction, record) =>
              progress({
                fraction,
                label: `${record.steps[record.steps.length - 1]?.iteration ?? 0} of ${samples} samples per pixel`,
                exposition: [record],
              }),
          }),
    })
  },
  views: [{ record: 'imageSteps', title: 'The render, as the samples per pixel double' }],
  methods: [
    { file: 'methods/graphics/rendering.ts', name: 'progressiveRender' },
    { file: 'lib/graphics/scenes.ts', name: 'cornellBox' },
  ],
  references: [
    'Goral, Torrance, Greenberg and Battaile (1984), Modeling the interaction of light between diffuse surfaces',
    'Kajiya (1986), The rendering equation',
  ],
})
