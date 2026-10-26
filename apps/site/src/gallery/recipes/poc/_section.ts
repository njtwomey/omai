import type { Section } from '../../recipe.ts'

/** A proof of concept of the methods design: methods with log contracts, sessions, eager against live. */
export const section: Partial<Section> = {
  title: 'Proof of concept: sessions',
  blurb:
    'Methods declare a log contract; a session runs them eagerly or live on a tape you can scrub, seek, probe and fork. Four worked cases to judge the design by.',
}
