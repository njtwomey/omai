# Methods

The methods layer: named algorithms and estimators written on the log contract and sessions (`docs/methods.md`,
`docs/methods_port/`), each a light wrapper over the library that returns what an explanation needs. A method of
nothing but its configuration (a process, a sampler, a search) runs as a session like any estimator.

Every area has a folder here whose `index.ts` opens with what the area is for: the areas with methods, and
placeholders for the areas whose methods are not yet written. This folder is the index of methods: what exists and
what is planned. The previous methods, parked on 2026-10-26, are in git history at 933f0dd8 under
`_parked/packages/engine/src/methods/`; their algorithms move home to `lib/` as they are needed
(`docs/methods_port/inventory.md`).

A method is a light wrapper over the library. If something a method needs is not in `lib/`, that is the thing to fix.
