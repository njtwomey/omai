/**
 * The names under a run's root key, written once: `init` seeds `Algorithm.init`, `steps` is the key every step key is
 * folded from (step `t` draws from `foldIn(foldIn(root, steps), t)`), and `make` seeds what a session's `Method.make`
 * builds. They are what lets `resume`, `seek` and `fork` reproduce a run exactly, so the drivers and the session read
 * them from here and nowhere else. Internal.
 */

/** The tags folded into a run's root key. */
export const keyTags = { init: 0, steps: 1, make: 7 } as const
