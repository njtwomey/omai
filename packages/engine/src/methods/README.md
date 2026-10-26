# Methods

Empty on purpose. The methods layer is being rebuilt on the log contract and sessions (`docs/methods.md`), with one
base per way of running and particulars of a few lines each (`docs/methods_port/`). The previous methods are parked,
untouched, at `_parked/packages/engine/src/methods/`; their algorithms move home to `lib/` as they are needed
(`docs/methods_port/inventory.md`), and the constructables already have: environments and agents in
`lib/environments` and `lib/agents`, search problems in `lib/problems`, the progressive renderer in `lib/graphics`.

A method is a light wrapper over the library. If something a method needs is not in `lib/`, that is the thing to fix.
