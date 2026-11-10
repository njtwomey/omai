import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Raycaster maze',
  summary:
    'A first-person maze in the style of Wolfenstein 3D: the library generates a maze, a DDA ray caster draws it column by column with distance fog, and a minimap shows your field of view. W S move, A D strafe, the arrow keys turn.',
  tags: ['graphics', 'ray casting', 'DDA', 'maze', 'game', 'first person', 'wolfenstein', 'worlds', 'keyboard', 'toy'],
  added: '2026-11-01',
  group: 'graphics-and-fields',
}
