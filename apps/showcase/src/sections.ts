/**
 * The sections of the splash page, in the order they appear. A story names its section in its metadata (`section: 'uncertainty'`);
 * a section with no story is not shown. This is the one place the order lives: add a section here, then give a story its id.
 */
export const SECTIONS = [
  {
    id: 'uncertainty',
    title: 'Knowing how sure we are',
    blurb: 'Posteriors, bands and intervals: an answer with its doubt attached.',
  },
  { id: 'models', title: 'Models that learn', blurb: 'Fitting a function to data, from lines to networks.' },
  {
    id: 'inference',
    title: 'Expectation propagation',
    blurb: 'Message passing: how it works, and what it is used for.',
  },
  { id: 'structure', title: 'Structure', blurb: 'Sequences, graphs and hierarchies.' },
  { id: 'worlds', title: 'Small worlds', blurb: 'Simulations, games and agents.' },
] as const

/** The id of a section. */
export type SectionId = (typeof SECTIONS)[number]['id']
