/**
 * Closed outlines to draw with: Lucide icons (animals, plants, objects), simple pictures (a brontosaurus, an oak, an
 * elephant, a whale, a teapot) and classic closed curves (a heart, a lemniscate, a trefoil, Lissajous figures, an
 * epitrochoid, the butterfly curve, a Koch snowflake, a Moore curve), each resampled as one closed path of any number of
 * points, centred, of unit radius and y up, for Fourier series of curves, shape descriptors and morphs.
 *
 * Provenance: the shapes come from the AI Field Notes epicycles note (ml-concepts). The icons are Lucide's
 * (https://lucide.dev), copied from lucide-react 1.48.0 as SVG elements on a 24 x 24 canvas, under the ISC licence:
 *
 *   ISC License. Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All
 *   other copyright (c) for Lucide are held by Lucide Contributors 2022. Permission to use, copy, modify, and/or
 *   distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright
 *   notice and this permission notice appear in all copies. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS
 *   ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 *   EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 *   TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * The pictures are SVG paths on a 100 x 100 canvas drawn for the note.
 */

import { MdmlError } from '../../base/index.ts'
import { normaliseOutline, resampleClosed, type SvgElement, svgOutline } from '../../graphics/index.ts'
import type { Point2 } from '../../numerics/geometry/index.ts'

const LUCIDE: readonly (readonly [string, string, readonly SvgElement[]])[] = [
  [
    'cat',
    'Cat',
    [
      [
        'path',
        {
          d: 'M12 5c.67 0 1.35.09 2 .26 1.78-2 5.03-2.84 6.42-2.26 1.4.58-.42 7-.42 7 .57 1.07 1 2.24 1 3.44C21 17.9 16.97 21 12 21s-9-3-9-7.56c0-1.25.5-2.4 1-3.44 0 0-1.89-6.42-.5-7 1.39-.58 4.72.23 6.5 2.23A9.04 9.04 0 0 1 12 5Z',
        },
      ],
      ['path', { d: 'M8 14v.5' }],
      ['path', { d: 'M16 14v.5' }],
      ['path', { d: 'M11.25 16.25h1.5L12 17l-.75-.75Z' }],
    ],
  ],
  [
    'dog',
    'Dog',
    [
      ['path', { d: 'M11.25 16.25h1.5L12 17z' }],
      ['path', { d: 'M16 14v.5' }],
      [
        'path',
        {
          d: 'M4.42 11.247A13.152 13.152 0 0 0 4 14.556C4 18.728 7.582 21 12 21s8-2.272 8-6.444a11.702 11.702 0 0 0-.493-3.309',
        },
      ],
      ['path', { d: 'M8 14v.5' }],
      [
        'path',
        {
          d: 'M8.5 8.5c-.384 1.05-1.083 2.028-2.344 2.5-1.931.722-3.576-.297-3.656-1-.113-.994 1.177-6.53 4-7 1.923-.321 3.651.845 3.651 2.235A7.497 7.497 0 0 1 14 5.277c0-1.39 1.844-2.598 3.767-2.277 2.823.47 4.113 6.006 4 7-.08.703-1.725 1.722-3.656 1-1.261-.472-1.855-1.45-2.239-2.5',
        },
      ],
    ],
  ],
  [
    'bird',
    'Bird',
    [
      ['path', { d: 'M16 7h.01' }],
      ['path', { d: 'M3.4 18H12a8 8 0 0 0 8-8V7a4 4 0 0 0-7.28-2.3L2 20' }],
      ['path', { d: 'm20 7 2 .5-2 .5' }],
      ['path', { d: 'M10 18v3' }],
      ['path', { d: 'M14 17.75V21' }],
      ['path', { d: 'M7 18a6 6 0 0 0 3.84-10.61' }],
    ],
  ],
  [
    'rabbit',
    'Rabbit',
    [
      ['path', { d: 'M13 16a3 3 0 0 1 2.24 5' }],
      ['path', { d: 'M18 12h.01' }],
      [
        'path',
        {
          d: 'M18 21h-8a4 4 0 0 1-4-4 7 7 0 0 1 7-7h.2L9.6 6.4a1 1 0 1 1 2.8-2.8L15.8 7h.2c3.3 0 6 2.7 6 6v1a2 2 0 0 1-2 2h-1a3 3 0 0 0-3 3',
        },
      ],
      ['path', { d: 'M20 8.54V4a2 2 0 1 0-4 0v3' }],
      ['path', { d: 'M7.612 12.524a3 3 0 1 0-1.6 4.3' }],
    ],
  ],
  [
    'snail',
    'Snail',
    [
      ['path', { d: 'M2 13a6 6 0 1 0 12 0 4 4 0 1 0-8 0 2 2 0 0 0 4 0' }],
      ['circle', { cx: '10', cy: '13', r: '8' }],
      ['path', { d: 'M2 21h12c4.4 0 8-3.6 8-8V7a2 2 0 1 0-4 0v6' }],
      ['path', { d: 'M18 3 19.1 5.2' }],
      ['path', { d: 'M22 3 20.9 5.2' }],
    ],
  ],
  [
    'turtle',
    'Turtle',
    [
      [
        'path',
        { d: 'm12 10 2 4v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3a8 8 0 1 0-16 0v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3l2-4h4Z' },
      ],
      ['path', { d: 'M4.82 7.9 8 10' }],
      ['path', { d: 'M15.18 7.9 12 10' }],
      ['path', { d: 'M16.93 10H20a2 2 0 0 1 0 4H2' }],
    ],
  ],
  [
    'squirrel',
    'Squirrel',
    [
      ['path', { d: 'M15.236 22a3 3 0 0 0-2.2-5' }],
      ['path', { d: 'M16 20a3 3 0 0 1 3-3h1a2 2 0 0 0 2-2v-2a4 4 0 0 0-4-4V4' }],
      ['path', { d: 'M18 13h.01' }],
      [
        'path',
        {
          d: 'M18 6a4 4 0 0 0-4 4 7 7 0 0 0-7 7c0-5 4-5 4-10.5a4.5 4.5 0 1 0-9 0 2.5 2.5 0 0 0 5 0C7 10 3 11 3 17c0 2.8 2.2 5 5 5h10',
        },
      ],
    ],
  ],
  [
    'fish',
    'Fish',
    [
      ['path', { d: 'M6.5 12c.94-3.46 4.94-6 8.5-6 3.56 0 6.06 2.54 7 6-.94 3.47-3.44 6-7 6s-7.56-2.53-8.5-6Z' }],
      ['path', { d: 'M18 12v.5' }],
      ['path', { d: 'M16 17.93a9.77 9.77 0 0 1 0-11.86' }],
      [
        'path',
        { d: 'M7 10.67C7 8 5.58 5.97 2.73 5.5c-1 1.5-1 5 .23 6.5-1.24 1.5-1.24 5-.23 6.5C5.58 18.03 7 16 7 13.33' },
      ],
      ['path', { d: 'M10.46 7.26C10.2 5.88 9.17 4.24 8 3h5.8a2 2 0 0 1 1.98 1.67l.23 1.4' }],
      ['path', { d: 'm16.01 17.93-.23 1.4A2 2 0 0 1 13.8 21H9.5a5.96 5.96 0 0 0 1.49-3.98' }],
    ],
  ],
  [
    'rat',
    'Rat',
    [
      ['path', { d: 'M13 22H4a2 2 0 0 1 0-4h12' }],
      ['path', { d: 'M13.236 18a3 3 0 0 0-2.2-5' }],
      ['path', { d: 'M16 9h.01' }],
      [
        'path',
        { d: 'M16.82 3.94a3 3 0 1 1 3.237 4.868l1.815 2.587a1.5 1.5 0 0 1-1.5 2.1l-2.872-.453a3 3 0 0 0-3.5 3' },
      ],
      ['path', { d: 'M17 4.988a3 3 0 1 0-5.2 2.052A7 7 0 0 0 4 14.015 4 4 0 0 0 8 18' }],
    ],
  ],
  [
    'panda',
    'Panda',
    [
      ['path', { d: 'M11.25 17.25h1.5L12 18z' }],
      ['path', { d: 'm15 12 2 2' }],
      ['path', { d: 'M17.902 6.599a8 8 0 0 0-.5-.5' }],
      [
        'path',
        {
          d: 'M2 14.5C2 19.47 6.48 22 12 22s10-2.53 10-7.5a10 10 0 0 0-1.3-4.83 4.5 4.5 0 1 0-7.05-5.5 8 8 0 0 0-3.3 0 4.5 4.5 0 1 0-7.04 5.5A10 10 0 0 0 2 14.5',
        },
      ],
      ['path', { d: 'M6.099 6.599a8 8 0 0 1 .5-.5' }],
      ['path', { d: 'm9 12-2 2' }],
    ],
  ],
  [
    'shell',
    'Shell',
    [
      [
        'path',
        {
          d: 'M14 11a2 2 0 1 1-4 0 4 4 0 0 1 8 0 6 6 0 0 1-12 0 8 8 0 0 1 16 0 10 10 0 1 1-20 0 11.93 11.93 0 0 1 2.42-7.22 2 2 0 1 1 3.16 2.44',
        },
      ],
    ],
  ],
  [
    'bug',
    'Bug',
    [
      ['path', { d: 'M12 20v-9' }],
      ['path', { d: 'M14 7a4 4 0 0 1 4 4v3a6 6 0 0 1-12 0v-3a4 4 0 0 1 4-4z' }],
      ['path', { d: 'M14.12 3.88 16 2' }],
      ['path', { d: 'M21 21a4 4 0 0 0-3.81-4' }],
      ['path', { d: 'M21 5a4 4 0 0 1-3.55 3.97' }],
      ['path', { d: 'M22 13h-4' }],
      ['path', { d: 'M3 21a4 4 0 0 1 3.81-4' }],
      ['path', { d: 'M3 5a4 4 0 0 0 3.55 3.97' }],
      ['path', { d: 'M6 13H2' }],
      ['path', { d: 'm8 2 1.88 1.88' }],
      ['path', { d: 'M9 7.13V6a3 3 0 1 1 6 0v1.13' }],
    ],
  ],
  [
    'paw-print',
    'Paw print',
    [
      ['circle', { cx: '11', cy: '4', r: '2' }],
      ['circle', { cx: '18', cy: '8', r: '2' }],
      ['circle', { cx: '20', cy: '16', r: '2' }],
      ['path', { d: 'M9 10a5 5 0 0 1 5 5v3.5a3.5 3.5 0 0 1-6.84 1.045Q6.52 17.48 4.46 16.84A3.5 3.5 0 0 1 5.5 10Z' }],
    ],
  ],
  [
    'tree-pine',
    'Pine tree',
    [
      [
        'path',
        {
          d: 'm17 14 3 3.3a1 1 0 0 1-.7 1.7H4.7a1 1 0 0 1-.7-1.7L7 14h-.3a1 1 0 0 1-.7-1.7L9 9h-.2A1 1 0 0 1 8 7.3L12 3l4 4.3a1 1 0 0 1-.8 1.7H15l3 3.3a1 1 0 0 1-.7 1.7H17Z',
        },
      ],
      ['path', { d: 'M12 22v-3' }],
    ],
  ],
  [
    'tree-deciduous',
    'Tree',
    [
      [
        'path',
        {
          d: 'M8 19a4 4 0 0 1-2.24-7.32A3.5 3.5 0 0 1 9 6.03V6a3 3 0 1 1 6 0v.04a3.5 3.5 0 0 1 3.24 5.65A4 4 0 0 1 16 19Z',
        },
      ],
      ['path', { d: 'M12 19v3' }],
    ],
  ],
  [
    'tree-palm',
    'Palm tree',
    [
      ['path', { d: 'M13 8c0-2.76-2.46-5-5.5-5S2 5.24 2 8h2l1-1 1 1h4' }],
      ['path', { d: 'M13 7.14A5.82 5.82 0 0 1 16.5 6c3.04 0 5.5 2.24 5.5 5h-3l-1-1-1 1h-3' }],
      [
        'path',
        { d: 'M5.89 9.71c-2.15 2.15-2.3 5.47-.35 7.43l4.24-4.25.7-.7.71-.71 2.12-2.12c-1.95-1.96-5.27-1.8-7.42.35' },
      ],
      ['path', { d: 'M11 15.5c.5 2.5-.17 4.5-1 6.5h4c2-5.5-.5-12-1-14' }],
    ],
  ],
  [
    'leaf',
    'Leaf',
    [
      [
        'path',
        {
          d: 'M11 20a10 10 0 0010-10 25.9 25.9 0 00-1.04-7.281 1 1 0 00-1.755-.325C15.833 5.5 13 5.5 9.8 6.1A7 7 0 0011 20',
        },
      ],
      ['path', { d: 'M2 21a5 5 0 012.911-4.544C7.613 15.212 8.351 15.24 11 13' }],
    ],
  ],
  [
    'flower',
    'Flower',
    [
      ['circle', { cx: '12', cy: '12', r: '3' }],
      ['path', { d: 'M12 16.5A4.5 4.5 0 1 1 7.5 12 4.5 4.5 0 1 1 12 7.5a4.5 4.5 0 1 1 4.5 4.5 4.5 4.5 0 1 1-4.5 4.5' }],
      ['path', { d: 'M12 7.5V9' }],
      ['path', { d: 'M7.5 12H9' }],
      ['path', { d: 'M16.5 12H15' }],
      ['path', { d: 'M12 16.5V15' }],
      ['path', { d: 'm8 8 1.88 1.88' }],
      ['path', { d: 'M14.12 9.88 16 8' }],
      ['path', { d: 'm8 16 1.88-1.88' }],
      ['path', { d: 'M14.12 14.12 16 16' }],
    ],
  ],
  [
    'flower-2',
    'Tulip',
    [
      [
        'path',
        { d: 'M12 5a3 3 0 1 1 3 3m-3-3a3 3 0 1 0-3 3m3-3v1M9 8a3 3 0 1 0 3 3M9 8h1m5 0a3 3 0 1 1-3 3m3-3h-1m-2 3v-1' },
      ],
      ['circle', { cx: '12', cy: '8', r: '2' }],
      ['path', { d: 'M12 10v12' }],
      ['path', { d: 'M12 22c4.2 0 7-1.667 7-5-4.2 0-7 1.667-7 5Z' }],
      ['path', { d: 'M12 22c-4.2 0-7-1.667-7-5 4.2 0 7 1.667 7 5Z' }],
    ],
  ],
  [
    'feather',
    'Feather',
    [
      ['path', { d: 'M14.086 18.412A2 2 0 0112.67 19H5v-7.672a2 2 0 01.586-1.414L11.75 3.75a6 6 0 118.49 8.49z' }],
      ['path', { d: 'M16 8 2 22' }],
      ['path', { d: 'M17.488 15H9' }],
    ],
  ],
  [
    'rocket',
    'Rocket',
    [
      ['path', { d: 'M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5' }],
      ['path', { d: 'M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09' }],
      ['path', { d: 'M9 12a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.4 22.4 0 0 1-4 2z' }],
      ['path', { d: 'M9 12H4s.55-3.03 2-4c1.62-1.08 5 .05 5 .05' }],
    ],
  ],
  [
    'sailboat',
    'Sailboat',
    [
      ['path', { d: 'M10 2v15' }],
      ['path', { d: 'M7 22a4 4 0 0 1-4-4 1 1 0 0 1 1-1h16a1 1 0 0 1 1 1 4 4 0 0 1-4 4z' }],
      ['path', { d: 'M9.159 2.46a1 1 0 0 1 1.521-.193l9.977 8.98A1 1 0 0 1 20 13H4a1 1 0 0 1-.824-1.567z' }],
    ],
  ],
  [
    'anchor',
    'Anchor',
    [
      ['path', { d: 'M12 6v16' }],
      ['path', { d: 'm19 13 2-1a9 9 0 0 1-18 0l2 1' }],
      ['path', { d: 'M9 11h6' }],
      ['circle', { cx: '12', cy: '4', r: '2' }],
    ],
  ],
  [
    'crown',
    'Crown',
    [
      [
        'path',
        {
          d: 'M11.562 3.266a.5.5 0 0 1 .876 0L15.39 8.87a1 1 0 0 0 1.516.294L21.183 5.5a.5.5 0 0 1 .798.519l-2.834 10.246a1 1 0 0 1-.956.734H5.81a1 1 0 0 1-.957-.734L2.02 6.02a.5.5 0 0 1 .798-.519l4.276 3.664a1 1 0 0 0 1.516-.294z',
        },
      ],
      ['path', { d: 'M5 21h14' }],
    ],
  ],
  [
    'ghost',
    'Ghost',
    [
      ['path', { d: 'M15 10v1' }],
      [
        'path',
        {
          d: 'M7.528 20.472a1.6 1.6 0 012.277 0l1.057 1.056a1.6 1.6 0 002.276 0l1.057-1.056a1.6 1.6 0 012.277 0l1.114 1.114a1.4 1.4 0 002.414-1V10a8 8 0 00-16 0v10.586a1.4 1.4 0 002.414 1z',
        },
      ],
      ['path', { d: 'M9 10v1' }],
    ],
  ],
  [
    'key',
    'Key',
    [
      ['path', { d: 'm2 21 9.6-9.6' }],
      ['path', { d: 'm7.5 15.5 2.3 2.3a1 1 0 0 1 0 1.4l-2.1 2.1a1 1 0 0 1-1.4 0L4 19' }],
      ['circle', { cx: '15.5', cy: '7.5', r: '5.5' }],
    ],
  ],
  [
    'guitar',
    'Guitar',
    [
      ['path', { d: 'm11.9 12.1 4.514-4.514' }],
      [
        'path',
        {
          d: 'M20.1 2.3a1 1 0 0 0-1.4 0l-1.114 1.114A2 2 0 0 0 17 4.828v1.344a2 2 0 0 1-.586 1.414A2 2 0 0 1 17.828 7h1.344a2 2 0 0 0 1.414-.586L21.7 5.3a1 1 0 0 0 0-1.4z',
        },
      ],
      ['path', { d: 'm6 16 2 2' }],
      [
        'path',
        {
          d: 'M8.23 9.85A3 3 0 0 1 11 8a5 5 0 0 1 5 5 3 3 0 0 1-1.85 2.77l-.92.38A2 2 0 0 0 12 18a4 4 0 0 1-4 4 6 6 0 0 1-6-6 4 4 0 0 1 4-4 2 2 0 0 0 1.85-1.23z',
        },
      ],
    ],
  ],
  [
    'clef-treble',
    'Treble clef',
    [
      [
        'path',
        {
          d: 'M10.586 21.414a2 2 0 0 0 3.378-1.791L11.036 4.377a2 2 0 1 1 3.378 1.037C12.414 7.414 7 8 7 13a5 5 0 0 0 5 5 5 4 0 0 0 5-4 3 3 0 0 0-3-3 3 2 0 0 0-3 2',
        },
      ],
    ],
  ],
  [
    'music',
    'Music',
    [
      ['path', { d: 'M9 18V5l12-2v13' }],
      ['circle', { cx: '6', cy: '18', r: '3' }],
      ['circle', { cx: '18', cy: '16', r: '3' }],
    ],
  ],
  [
    'heart',
    'Heart',
    [
      [
        'path',
        {
          d: 'M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5',
        },
      ],
    ],
  ],
  [
    'star',
    'Star',
    [
      [
        'path',
        {
          d: 'M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z',
        },
      ],
    ],
  ],
  ['cloud', 'Cloud', [['path', { d: 'M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z' }]]],
  [
    'bike',
    'Bike',
    [
      ['circle', { cx: '18.5', cy: '17.5', r: '3.5' }],
      ['circle', { cx: '5.5', cy: '17.5', r: '3.5' }],
      ['circle', { cx: '15', cy: '5', r: '1' }],
      ['path', { d: 'M12 17.5V14l-3-3 4-3 2 3h2' }],
    ],
  ],
  [
    'apple',
    'Apple',
    [
      ['path', { d: 'M12 6.528V3a1 1 0 0 1 1-1h0' }],
      [
        'path',
        {
          d: 'M18.237 21A15 15 0 0 0 22 11a6 6 0 0 0-10-4.472A6 6 0 0 0 2 11a15.1 15.1 0 0 0 3.763 10 3 3 0 0 0 3.648.648 5.5 5.5 0 0 1 5.178 0A3 3 0 0 0 18.237 21',
        },
      ],
    ],
  ],
  [
    'cherry',
    'Cherry',
    [
      ['path', { d: 'M2 17a5 5 0 0 0 10 0c0-2.76-2.5-5-5-3-2.5-2-5 .24-5 3Z' }],
      ['path', { d: 'M12 17a5 5 0 0 0 10 0c0-2.76-2.5-5-5-3-2.5-2-5 .24-5 3Z' }],
      ['path', { d: 'M7 14c3.22-2.91 4.29-8.75 5-12 1.66 2.38 4.94 9 5 12' }],
      ['path', { d: 'M22 9c-4.29 0-7.14-2.33-10-7 5.71 0 10 4.67 10 7Z' }],
    ],
  ],
  [
    'carrot',
    'Carrot',
    [
      ['path', { d: 'M15 16a1 1 0 0 0-7-7q-4 4-5.987 12.385a.5.5 0 0 0 .602.602Q11 20 15 16l-3-3' }],
      ['path', { d: 'M15 9q4 4 7 0-3-4-7 0 4-4 0-7-4 3 0 7' }],
      ['path', { d: 'm8 15-2.58-2.58' }],
    ],
  ],
  [
    'banana',
    'Banana',
    [
      ['path', { d: 'M4 13c3.5-2 8-2 10 2a5.5 5.5 0 0 1 8 5' }],
      [
        'path',
        {
          d: 'M5.15 17.89c5.52-1.52 8.65-6.89 7-12C11.55 4 11.5 2 13 2c3.22 0 5 5.5 5 8 0 6.5-4.2 12-10.49 12C5.11 22 2 22 2 20c0-1.5 1.14-1.55 3.15-2.11Z',
        },
      ],
    ],
  ],
  ['mountain', 'Mountain', [['path', { d: 'm8 3 4 8 5-5 5 15H2L8 3z' }]]],
  [
    'origami',
    'Origami',
    [
      ['path', { d: 'M12 12V4a1 1 0 0 1 1-1h6.297a1 1 0 0 1 .651 1.759l-4.696 4.025' }],
      ['path', { d: 'm12 21-7.414-7.414A2 2 0 0 1 4 12.172V6.415a1.002 1.002 0 0 1 1.707-.707L20 20.009' }],
      [
        'path',
        {
          d: 'm12.214 3.381 8.414 14.966a1 1 0 0 1-.167 1.199l-1.168 1.163a1 1 0 0 1-.706.291H6.351a1 1 0 0 1-.625-.219L3.25 18.8a1 1 0 0 1 .631-1.781l4.165.027',
        },
      ],
    ],
  ],
  [
    'bone',
    'Bone',
    [
      [
        'path',
        {
          d: 'M17 10c.7-.7 1.69 0 2.5 0a2.5 2.5 0 1 0 0-5 .5.5 0 0 1-.5-.5 2.5 2.5 0 1 0-5 0c0 .81.7 1.8 0 2.5l-7 7c-.7.7-1.69 0-2.5 0a2.5 2.5 0 0 0 0 5c.28 0 .5.22.5.5a2.5 2.5 0 1 0 5 0c0-.81-.7-1.8 0-2.5Z',
        },
      ],
    ],
  ],
  [
    'plane',
    'Plane',
    [
      [
        'path',
        {
          d: 'M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z',
        },
      ],
    ],
  ],
]

const path = (d: string): SvgElement => ['path', { d }]
const circle = (cx: number, cy: number, r: number): SvgElement => ['circle', { cx: `${cx}`, cy: `${cy}`, r: `${r}` }]

const DRAWINGS: readonly (readonly [string, string, readonly SvgElement[]])[] = [
  [
    'brontosaurus',
    'Brontosaurus',
    [
      path(
        'M2 62C15 58 25 52 33 46C40 36 58 34 68 42C72 36 76 22 80 14C82 10 90 9 94 13C96 16 93 19 88 18' +
          'C84 20 82 30 80 40C79 48 77 52 75 56L76 84L70 84L69 66L66 66L66 84L60 84L60 64C55 66 48 66 44 64' +
          'L44 84L38 84L37 66L35 66L35 84L29 84L29 62C22 64 12 65 2 62Z',
      ),
      circle(89, 13.5, 0.8),
    ],
  ],
  [
    'oak',
    'Oak tree',
    [
      path(
        'M44 92C46 80 46 70 45 62C40 64 30 64 26 58A12 12 0 0 1 20 40A14 14 0 0 1 32 22A16 16 0 0 1 54 12' +
          'A15 15 0 0 1 74 22A13 13 0 0 1 82 42A12 12 0 0 1 72 58C68 62 60 64 55 62C54 70 54 80 56 92' +
          'C60 94 64 95 68 96L32 96C36 95 40 94 44 92Z',
      ),
    ],
  ],
  [
    'elephant',
    'Elephant',
    [
      path(
        'M20 40C22 26 36 18 50 20C62 16 80 20 86 34C90 44 90 56 86 64L86 86L78 86L77 68C72 70 66 70 62 68' +
          'L62 86L54 86L54 66C50 64 46 60 44 56C40 62 36 70 34 78C33 84 30 88 26 86C24 84 26 80 28 78' +
          'C30 70 30 62 28 56C24 54 20 48 20 40Z',
      ),
      path('M44 30C50 34 50 46 44 52'),
      circle(32, 36, 1.5),
    ],
  ],
  [
    'whale',
    'Whale',
    [
      path(
        'M8 30C12 38 16 44 22 48C30 40 52 34 70 36C86 38 96 50 94 62C92 72 80 76 64 76C44 76 30 66 22 54' +
          'C18 60 10 66 4 66C8 58 12 54 18 51C12 46 8 40 8 30Z',
      ),
      circle(80, 58, 1.5),
      path('M74 34C72 26 70 22 66 18'),
      path('M74 34C76 26 80 22 84 18'),
    ],
  ],
  [
    'teapot',
    'Teapot',
    [
      path('M28 80C20 70 20 52 30 44C40 36 60 36 70 44C80 52 80 70 72 80Z'),
      path('M38 40C42 32 58 32 62 40'),
      circle(50, 30, 3),
      path('M24 62C16 60 11 50 6 38'),
      path('M75 50C88 46 92 66 75 72'),
    ],
  ],
]

/** Points of a curve (x(t), y(t)) for t over `turns` full turns. */
function param(x: (t: number) => number, y: (t: number) => number, turns = 1, m = 4000): Point2[] {
  return Array.from({ length: m }, (_, i) => {
    const t = (2 * Math.PI * turns * i) / m
    return [x(t), y(t)] as Point2
  })
}

const polar = (r: (t: number) => number, turns = 1): Point2[] =>
  param(
    (t) => r(t) * Math.cos(t),
    (t) => r(t) * Math.sin(t),
    turns,
  )

/** A regular `k`-gon, or a `k`-pointed star when `inner` < 1, with a vertex at the top. */
function polygon(k: number, inner = 1): Point2[] {
  const m = inner < 1 ? 2 * k : k
  return Array.from({ length: m }, (_, i) => {
    const a = Math.PI / 2 + (2 * Math.PI * i) / m
    const r = inner < 1 && i % 2 ? inner : 1
    return [r * Math.cos(a), r * Math.sin(a)] as Point2
  })
}

/** Vertices of an L-system drawn by a turtle: F steps forward; + and − turn by `angle` degrees. */
function turtle(axiom: string, rules: Readonly<Record<string, string>>, depth: number, angle: number): Point2[] {
  let s = axiom
  for (let i = 0; i < depth; i++) s = [...s].map((ch) => rules[ch] ?? ch).join('')
  let [x, y, h] = [0, 0, 0]
  const pts: Point2[] = [[0, 0]]
  for (const ch of s) {
    if (ch === 'F') {
      x += Math.cos(h)
      y += Math.sin(h)
      pts.push([x, y])
    } else if (ch === '+') h += (angle * Math.PI) / 180
    else if (ch === '-') h -= (angle * Math.PI) / 180
  }
  // A closed curve returns to its start; drop the repeated point.
  return Math.hypot(x, y) < 1e-9 ? pts.slice(0, -1) : pts
}

const CURVES: readonly (readonly [string, string, () => Point2[]])[] = [
  ['circle', 'Circle', () => polygon(400)],
  ['ellipse', 'Ellipse', () => param((t) => 1.6 * Math.cos(t), Math.sin)],
  [
    'square',
    'Square',
    () => [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ],
  ],
  ['triangle', 'Triangle', () => polygon(3)],
  ['star-polygon', 'Five-pointed star', () => polygon(5, 0.38)],
  [
    'heart-curve',
    'Heart',
    () =>
      param(
        (t) => 16 * Math.sin(t) ** 3,
        (t) => 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t),
      ),
  ],
  ['cardioid', 'Cardioid', () => polar((t) => 1 - Math.cos(t))],
  [
    'lemniscate',
    'Lemniscate of Bernoulli',
    () =>
      param(
        (t) => Math.cos(t) / (1 + Math.sin(t) ** 2),
        (t) => (Math.sin(t) * Math.cos(t)) / (1 + Math.sin(t) ** 2),
      ),
  ],
  [
    'trefoil',
    'Trefoil',
    () =>
      param(
        (t) => Math.sin(t) + 2 * Math.sin(2 * t),
        (t) => Math.cos(t) - 2 * Math.cos(2 * t),
      ),
  ],
  [
    'lissajous',
    'Lissajous figure 3 : 2',
    () =>
      param(
        (t) => Math.cos(3 * t),
        (t) => Math.sin(2 * t),
      ),
  ],
  ['rose', 'Five-petal rose', () => polar((t) => Math.cos(5 * t), 0.5)],
  [
    'epitrochoid',
    'Epitrochoid',
    () =>
      param(
        (t) => 8 * Math.cos(t) - 5 * Math.cos((8 / 3) * t),
        (t) => 8 * Math.sin(t) - 5 * Math.sin((8 / 3) * t),
        3,
      ),
  ],
  [
    'butterfly',
    'Butterfly curve',
    () => polar((t) => Math.exp(Math.cos(t)) - 2 * Math.cos(4 * t) - Math.sin(t / 12) ** 5, 6),
  ],
  ['koch', 'Koch snowflake', () => turtle('F--F--F', { F: 'F+F--F+F' }, 4, 60)],
  ['moore', 'Moore curve', () => turtle('LFL+F+LFL', { L: '-RF+LFL+FR-', R: '+LF-RFR-FL+' }, 3, 90)],
]

/** The groups of outline shapes. */
export type OutlineGroup = 'icons' | 'drawings' | 'curves'

/** An outline shape of the catalogue: its id, its name, its group and where it comes from. */
export interface OutlineShape {
  readonly id: string
  readonly label: string
  readonly group: OutlineGroup
  readonly credit: string
}

interface Entry extends OutlineShape {
  readonly make: (n: number) => Point2[]
}

const ENTRIES: readonly Entry[] = [
  ...LUCIDE.map(
    ([id, label, els]): Entry => ({
      id,
      label,
      group: 'icons',
      credit: 'Lucide, ISC licence',
      make: (n) => svgOutline(els, n),
    }),
  ),
  ...DRAWINGS.map(
    ([id, label, els]): Entry => ({
      id,
      label,
      group: 'drawings',
      credit: 'drawn for the AI Field Notes',
      make: (n) => svgOutline(els, n),
    }),
  ),
  ...CURVES.map(
    ([id, label, f]): Entry => ({
      id,
      label,
      group: 'curves',
      credit: 'parametric curve',
      make: (n) => normaliseOutline(resampleClosed(f(), n), false),
    }),
  ),
]

const BY_ID = new Map(ENTRIES.map((e) => [e.id, e]))

/**
 * The outline shapes available, optionally of one group.
 *
 * @param group - The group (default every group).
 * @returns The shapes: id, name, group and credit.
 * @example
 * ```ts
 * import { outlineShapes } from 'mdml-engine/datasets/outlines'
 * outlineShapes('curves').slice(0, 3).map((s) => s.id) // => ['circle', 'ellipse', 'square']
 * ```
 */
export function outlineShapes(group?: OutlineGroup): OutlineShape[] {
  return ENTRIES.filter((e) => group === undefined || e.group === group).map(({ id, label, group: g, credit }) => ({
    id,
    label,
    group: g,
    credit,
  }))
}

/**
 * An outline shape as `n` points spaced uniformly along its closed path, centred, of unit maximum radius and y up.
 *
 * @param id - The shape's id (see {@link outlineShapes}).
 * @param n - How many points.
 * @returns The points.
 * @throws MdmlError with code `value` for an unknown id.
 * @example
 * ```ts
 * import { shapeOutline } from 'mdml-engine/datasets/outlines'
 * const p = shapeOutline('circle', 100)
 * [p.length, Math.round(Math.max(...p.map(([x, y]) => Math.hypot(x, y))) * 1e6) / 1e6] // => [100, 1]
 * ```
 */
export function shapeOutline(id: string, n: number): Point2[] {
  const e = BY_ID.get(id)
  if (e === undefined) throw new MdmlError('value', `shapeOutline: no shape '${id}'`)
  return e.make(n)
}
