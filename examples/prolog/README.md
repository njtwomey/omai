# A Prolog program and the search that answers one question

Prolog answers a question by searching. A program is a list of facts and rules. A question (a query) is a goal, and the
search tries to prove it from the program. This example runs a small family database and draws the search.

## The program

```
parent(tom, bob).   parent(tom, liz).   parent(tom, sam).
parent(bob, ann).   parent(bob, pat).   parent(liz, joe).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

The first six lines are facts (who is whose parent). The last is a rule: X is a grandparent of Z if X is a parent of some
Y and Y is a parent of Z.

## What the program does

- It loads the program with `consult` and asks `?- grandparent(tom, W).` ("who are Tom's grandchildren?") with `solve`,
  which uses SLD resolution, Prolog's search, and records every step on a tape.
- It turns the recorded search into a tree with `sldTree`, and the render package draws the tree.

## How to read the figure

- The program is listed on the left with each clause numbered, in the same way as the arrows are named.
- The top box is the query. The first arrow uses the rule (`grandparent/2 #1`), which turns the query into two smaller goals,
  `parent(tom, _2), parent(_2, W)`.
- That goal can use any of the first three facts about Tom (`parent/2 #1`, `#2`, `#3`), giving three branches: Bob, Liz and
  Sam are tried one at a time. Under each, the second goal is solved with the facts about that child.
- A box with `[ ]` is an empty goal. Everything has been proved, and the value of `W` along that path is an answer:
  `ann` and `pat` through Bob, `joe` through Liz.
- The box marked `fail` is the one branch that went nowhere: Sam has no children in the program, so no clause matches and
  the search backs up (backtracks).
- 20 steps and 7 inferences were needed in all.

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="prolog"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
