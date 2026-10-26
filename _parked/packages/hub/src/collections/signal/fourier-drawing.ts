/**
 * Hub entry for drawing with rotating circles: an icon, a picture, a classic curve or a capital letter in one of 66
 * fonts, redrawn by its complex Fourier series, the largest terms chained as rotating arrows, with a zoom on the pen and
 * the spectrum.
 */

import { fontGlyphs, fontInstances, glyphOutline, outlineShapes, shapeOutline } from 'mdml-engine/datasets/outlines'
import { fourierDrawing } from 'mdml-engine/methods/signal'
import { bool, choice, dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)
const str = (v: SpaceValue | undefined, fallback: string): string => (typeof v === 'string' ? v : fallback)

// Points round each outline: the DFT length, so up to N - 1 rotating arrows.
const N = 1024
const ids = (group: 'icons' | 'drawings' | 'curves'): string[] => outlineShapes(group).map((s) => s.id)
const FONT_NAMES = fontInstances().map((f) => `${f.family} ${f.style}`)
const LETTERS = fontGlyphs().map((g) => g.char)

/** A closed outline redrawn by rotating circles: its Fourier series, the largest terms first. */
export const fourierDrawingEntry: HubEntry = defineEntry({
  id: 'signal/fourier-drawing',
  title: 'Drawing with rotating circles',
  summary:
    'Any closed outline is a sum of circles turning at whole-number speeds. Keep the largest and watch them redraw an icon, a curve or a letter; fewer circles round it off.',
  collection: 'signal',
  tags: ['Fourier series', 'epicycles', 'FFT', 'Parseval'],
  datasets: [],
  params: dict({
    source: choice(['icons', 'drawings', 'curves', 'letters'] as const, { default: 'icons', label: 'shape' }),
    icon: choice(ids('icons'), { default: 'clef-treble', when: { key: 'source', in: ['icons'] } }),
    drawing: choice(ids('drawings'), { default: 'brontosaurus', when: { key: 'source', in: ['drawings'] } }),
    curve: choice(ids('curves'), { default: 'trefoil', when: { key: 'source', in: ['curves'] } }),
    font: choice(FONT_NAMES, { default: 'Bodoni Moda Bold', when: { key: 'source', in: ['letters'] } }),
    letter: choice(LETTERS, { default: 'G', when: { key: 'source', in: ['letters'] } }),
    keep: choice(['terms', 'energy'] as const, { default: 'terms', label: 'keep' }),
    terms: int(1, N - 1, { log: true, default: 60, label: 'terms K', when: { key: 'keep', in: ['terms'] } }),
    energy: real(50, 99.99, { default: 99.5, label: 'energy kept, %', when: { key: 'keep', in: ['energy'] } }),
    order: choice(['magnitude', 'speed', 'frequency', 'random'] as const, {
      default: 'magnitude',
      label: 'arrow order',
    }),
    zoomed: bool({ default: true, label: 'zoom on the pen' }),
    zoom: real(10, 1000, { log: true, default: 100, label: 'magnification', when: { key: 'zoomed', in: [true] } }),
    steps: int(90, 4096, { log: true, default: 360, label: 'frames per cycle' }),
  }),
  run: ({ params, key }) => {
    const source = str(params['source'], 'icons')
    let points: [number, number][]
    let name: string
    if (source === 'letters') {
      const font = str(params['font'], 'Bodoni Moda Bold')
      const letter = str(params['letter'], 'G')
      points = glyphOutline(Math.max(0, FONT_NAMES.indexOf(font)), letter, N).map((p) => [p[0], p[1]])
      name = `${letter} in ${font}`
    } else {
      const field = source === 'icons' ? 'icon' : source === 'drawings' ? 'drawing' : 'curve'
      const id = str(params[field], ids(source as 'icons' | 'drawings' | 'curves')[0] ?? 'circle')
      points = shapeOutline(id, N).map((p) => [p[0], p[1]])
      name = outlineShapes().find((s) => s.id === id)?.label ?? id
    }
    return fourierDrawing(points, {
      keep:
        str(params['keep'], 'terms') === 'energy'
          ? { energy: num(params['energy'], 99.5) / 100 }
          : { count: num(params['terms'], 60) },
      order: str(params['order'], 'magnitude') as 'magnitude' | 'speed' | 'frequency' | 'random',
      key,
      steps: num(params['steps'], 360),
      ...(params['zoomed'] === false ? {} : { zoom: num(params['zoom'], 100) }),
      source: name,
    })
  },
  views: [{ record: 'epicycles', title: 'The outline redrawn by rotating arrows' }],
  methods: [
    { file: 'methods/signal/epicycles.ts', name: 'fourierDrawing' },
    { file: 'lib/signal/epicycles.ts', name: 'curveSpectrum' },
  ],
  references: [
    'Icons from Lucide (lucide.dev), ISC licence; fonts from Google Fonts (SIL OFL 1.1 and Apache 2.0)',
    'Campbell and Kautz (2014), Learning a Manifold of Fonts (the font outlines in correspondence)',
  ],
})
