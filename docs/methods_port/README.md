# Porting the methods: the mental model, and what is in this folder

This folder is the design for moving the methods layer onto the log contract and sessions. It exists so that we agree
on what we are building before anything is migrated. `methods.md` (one level up) is the design of the mechanism; this
folder is about how the _existing_ methods become small on top of it, and how new things stay trivial to add.

Read in this order: this file (the mental model, alignment questions), `bases.md` (the generic bases and what a
particular method is left to write), `extensibility.md` (what adding something costs), `leverage.md` (what is there to
build on, and where it is complicated).

## The mental model, in plain terms

1. **The library does everything.** A method does not implement anything numerical; it packages the library for an
   application: fit this, train that, learn a policy, solve a puzzle, simulate a flock.
2. **A method is also a complete account of how it got its answer.** Not the answer alone. The parameters as they
   changed, the decisions and the candidates not chosen, the signals of learning, the internal activity, the data that
   went in, the randomness. A method that returns only a fitted model has done half its job.
3. **Every run is a tape.** Anything worth seeing later is either on the tape or computable from a state on the tape.
   Nothing is chosen in advance: no snapshot lists, no "record these ten epochs", no side histories inside states.
4. **The method says what it can give; the caller says what it wants; the system reconciles.** The method publishes a
   contract of paths (`training/loss`, `smo/working-set`, `agent/q`). The caller subscribes with `Log` and `On`. With
   nothing subscribed the method is the raw algorithm at full speed. Turning everything off is free.
5. **Eager or live is the session's business.** A method does not know, and must not care, whether it ran up front in
   a worker or step by step as someone pressed play. Same tape, same numbers.
6. **Bases do the hand-holding.** There is one base per _way of running_ (one-shot, iterative solver, trainer, agent in
   an environment, search or simulation). A base owns the loop, the standard events, serialisation, the standard
   probes and the record plumbing. A particular method is a few declarations: what to build, what it can log, what
   can be derived from a state.
7. **Adding a new thing to see is one line.** If it can be computed from a state, one `derived(...)` line in the
   contract. If it only exists mid-step, one `ctx.log` and one contract line. No change to the base, the records, the
   views or anything that consumes the method.
8. **The hub is irrelevant to this.** The gallery proof of concept is a window onto sessions, not the design.

## What the port is not

- Not a rewrite of any algorithm. Solvers, trainers and environments in `lib/` stay as they are; the port renames their
  channels and removes private histories.
- Not a new record system. Structural records (a tree, a coefficient table, a margin) stay; the step-like ones become
  projections of the tape.
- Not a hub change.

## Where I think we agree, and where I want confirmation

1. **One base per way of running, with the estimator _kind_ as a small add-on.** Not one base per kind. A classifier
   and a clusterer solved iteratively share the iterative base; what "classifier" adds is its verbs (`predict`,
   `predictProba`), its standard probes (a decision surface, probabilities on a grid) and how its records carry the
   training rows. Five bases, seven kind add-ons, not thirty-five things. `bases.md` section 2 makes the case; if you
   meant one base per kind, say so and I will show that version too.
2. **`fit()` stays the scikit-learn-shaped surface.** `lasso({ alpha }).fit(data)` keeps working and returns a fitted
   model with records. Underneath it is a session run to completion with the method's `defaults` subscription. A
   caller who wants the tape, a schedule, a probe or a hook passes a subscription to `fit` or holds the session.
3. **Sweeps are studies over runs, not one run.** A regularisation path, a cross-validation curve, a learning curve,
   restarts of k-means: each point is a fit of its own. They ride the tape as child runs of a study, and their records
   are projections over those children. This is the one place the port adds a base rather than removing code.
4. **Contracts are plain strings, checked when a session is made.** Typed `keyof` contracts are possible later and
   cost a generic on every method; not now.
5. **Records persist for the structural things; generic views cover the rest.** A tape of scalars, tensors, messages
   and frames draws itself. We keep writing bespoke records only where the generic view is not the explanation.

If any of these is not what you meant, that is the conversation to have before a line of the port is written.
