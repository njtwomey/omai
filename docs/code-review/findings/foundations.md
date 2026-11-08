# Foundations review: base, context, tensor, random, space, trace, autodiff

Read-only review of `packages/engine/src/lib/{base,context,tensor,random,space,trace,autodiff}` (about 17,700 lines of non-test
source). All paths below are relative to `packages/engine/src/lib/`. Probes cited as "probe" are throwaway node scripts run
from the scratchpad (not the repo) against the source; nothing in the repository was changed.

## Summary

1. Healthy: the shape of the core is sound. Per-op dispatch on tiny tensors costs 0.24 us, `grad` of a small MLP is 3x its
   forward pass, reverse mode is an iterative topological sweep, Philox keys pass known-answer tests, and `checkPrimitive`
   (value, vjp, jvp, dot-product, vmap and mutation checks) is run property-style over the elementwise, reduction, shape and
   linear primitives. No `Math.random`, clock or `console` use in scope and no `// allow:` opt-outs; TSDoc is complete (488
   exported functions scanned: the only public gaps are `optionTable`, `logContext`, `contractOf`, all missing an `@example`).
2. Not healthy, 1: **degenerate inputs**. Every recent flaky-test class has a live twin in the core: `matmul` skips zero
   entries (so `0 * Infinity` is 1), `tril`/`triu` multiply by a mask (so `Infinity` above the diagonal becomes NaN),
   `all([])` is false, `cumsum` of a bool wraps at 256, `pow(1, Infinity)` is NaN, and the gradients of `prod`, `norm`, `std`,
   `pow(x, 0)` are NaN at zero (F3-F7). The property generators avoid exactly these points (`awayFromZero`, float64 only).
3. Not healthy, 2: **two accidental-quadratic or wasteful paths with one-line fixes**: `scheduler.fired` copies its whole
   history on every event (20,000 firings take 1.9 s, 40,000 take 6.9 s; F1) and `gammaVariates` builds a 256-double block per
   variate (236 ms against 6 ms for 20,000 draws; F2). `einsum` materialises the full outer product (22x slower than `matmul`;
   F11) and reverse mode computes the cotangent of constants (F12).
4. Not healthy, 3: **duplicated machinery inside `trace`**: three Algorithm drivers that already disagree on the final status
   (F8), `serialise.ts` re-implementing `base/json.ts` with a different number encoding (F18), `modes()` re-implementing
   `optionTable()` (F19), and every event written twice as interface and zod schema behind a cast (F21).
5. Layering: no file in scope imports against a sound order. `context` is simply above `trace` (nothing below imports it) and
   `scene` imports only `base`, so `autodiff/graph.ts -> scene` is correct; both are rank mistakes in the first-guess order,
   not code mistakes (F17). The real structural debt is import cycles inside `tensor` that no tool reports (F24).
6. The backend abstraction is declared but not used: `Context.backend` and `.dtype` have no reader, 37 `cpu.wrap` sites
   hard-wire the backend, `capabilities` is never consulted, four backend kernels are dead (F22, F23).
7. Tests: statement coverage of the scope is good (base 85%, tensor 90%, random 98%, space 97%, trace 95%, autodiff 95%) but
   branch coverage of `tensor/_cpu` is 60-65%: the dtype-specific branches (float32, int32, bool) and the degenerate-input
   branches have no direct test (F31). No test checks second derivatives of primitives (F32).
8. The three highest-value fixes: (a) F1 + F2 + F11 (measured 10x-250x wins, each a small change); (b) F3-F7 with the test
   table of F31 (corrects results on NaN/Inf/zero/empty inputs, small kernel changes); (c) F8 + F18 + F19 (one stepper, one
   JSON codec, one option table).
9. Hot paths that are fine: tensor-tensor `add` of 1e6 doubles is 5 ms (raw loop 1.8 ms), `sum` 6.7 ms, `matmul` 512 is 155 ms
   (about 1.1 ns per multiply-add, adequate for a reference backend), Philox bits 15 ns per word, `uniform` 1e6 in 41 ms.
10. Not read or not verified: comments and TSDoc prose (read through a comment-stripped view, plus a script for tags);
    `sobol-directions.data.ts` (generated data); `_jet-rules.ts` lines 330-358; no tests were run, so every behavioural claim
    is from reading plus the probes named in the entry; coverage figures are Phase 0's (commit b09e46da), line numbers there
    may be off by a few lines.

Entries by kind: hot-path 7, consistency 14, duplicate 7, layering 2, rule 2, test-gap 4, native-array 1 (37 in all).

---

### F1: `scheduler.fired` copies its full history on every event (quadratic)

kind: hot-path
evidence: trace/logging.ts:98 (`steps.set(ev, [...(steps.get(ev) ?? []), t])`); trace/logging.ts:94-101 also loops over every subscribed log on each firing; probe: 10,000 firings 71 ms, 20,000 firings 1,891 ms, 40,000 firings 6,948 ms.
what: every `ctx.emit(path)` in a session copies the whole list of earlier firing steps, so an event that fires each step of a long run is O(n^2) in time and garbage. `stepsOf` (logging.ts:107, `Session.stepsOf`, session.ts) is the only reader.
fix: push onto the stored array (`const a = steps.get(ev); if (a) a.push(t) else steps.set(ev, [t])`) and have `stepsOf` return it (it is typed `readonly number[]`); precompute the list of schedule-subscribed logs once in `scheduler()` instead of scanning `resolved.logs` per firing.
size: S (<1h)
risk: low (readers get a live view of a growing array instead of a snapshot; only `Session.stepsOf` and `eager: [event, n]` mode read it)
preserves behaviour: yes
changes public surface: no

### F2: `gammaVariates` generates a 256-double Philox block for every variate

kind: hot-path
evidence: random/continuous.ts:294-299 (`out[i] = gammaOne(alpha, uniformStream(foldIn(k, i)))`); `uniformStream` default `block = 256` draws `unitDoubles(foldIn(k, b), 256)`, i.e. 512 words, 128 Philox blocks, for a sampler that usually consumes 3-6 uniforms; probe: 20,000 variates 236 ms, `gamma(k, [20000], 2)` (one shared stream) 6 ms. Callers: dists/_sampling.ts:45 (Dirichlet and beta sampling), agents/bandits.ts:318 and :442 (per-step Thompson sampling).
what: 40x slower than the shared-stream sampler for no reason; it sits under `dists` sampling and the bandit agents.
fix: make `uniformStream` produce its block lazily (compute Philox blocks of 4 words, i.e. 2 doubles, on demand within the block). The values are identical to today's, so seeds and goldens keep working; a per-variate stream then costs a few blocks.
size: S (<1h)
risk: low (values must stay bit-identical; `random.golden.test.ts` and `gammaVariates` test at random.test.ts:181 pin it)
preserves behaviour: yes
changes public surface: no

### F3: `matmul` skips zero entries and `tril`/`triu` multiply by a mask: NaN and Infinity are lost or created

kind: consistency
evidence: tensor/_cpu/structure.ts:32 (`if (aip === 0) continue`); tensor/create.ts:320-344 (`tril` is `mul(v, cast(tri(...)))`, `triu` is `mul(v, cast(sub(1, tri(...))))`); probes: `matmul([[0, 1]], [[Infinity], [1]])` gives `[[1]]` and with NaN `[[1]]` (numpy: NaN); `tril([[1, Infinity], [3, 4]])` gives `[[1, NaN], [3, 4]]` and with NaN the same (numpy: 0 above the diagonal).
what: two multiplicative shortcuts that disagree with IEEE and numpy at non-finite inputs, in opposite directions. `matmul` can hide a NaN (a diverged model produces a finite loss); `tril` can create NaN from an Infinity mask (attention masks with -inf are common).
fix: delete the `aip === 0` shortcut (or keep it only behind a one-pass "no non-finite in b" check; the plain loop is the same speed for dense data) and build `tril`/`triu` with `where(mask, v, 0)`.
size: S (<1h)
risk: low (results change only for non-finite inputs; the matmul skip is a speed-up for sparse data, measure before dropping; golden fixtures use finite data)
preserves behaviour: no (NaN/Infinity now propagate as in numpy)
changes public surface: no

### F4: reductions of empty axes return identities that are wrong or invalid

kind: consistency
evidence: tensor/_cpu/reduce.ts:25 (accumulator filled with the identity, `max` gives -Infinity, `min` +Infinity, then stored back into the output dtype at :33); tensor/ops/compare.ts:342 (`all` is `greater(min(cast(x != 0, int32)), 0)`); tensor/_cpu/reduce.ts:43-52 (`argReduce` on an empty axis returns 0); probes: `all(zeros([0]))` is false (numpy: True), `max(zeros([0]))` is -Infinity for floats and 0 for ints (numpy raises), `mean(zeros([0]))` is NaN.
what: empty reductions are silently defined by the accumulator's identity pushed through the storage cast; `all([])` is wrong, `any([])` right by accident, `argmax` of nothing is index 0.
fix: reduction ops raise `OmaiError('shape')` for max/min/argmax/argmin over a zero-length axis (numpy's rule) and `all`/`any` use their own identities (true/false); `sum`/`prod` keep 0/1.
size: S (<1h)
risk: low-medium (code that relied on -Infinity for empty maxima, e.g. a masked max over an empty group, would start to throw; grep the engine for `max(` over possibly-empty axes before merging)
preserves behaviour: no (empty max/min/argmax raise; `all([])` becomes true)
changes public surface: no

### F5: dtype integrity at three boundaries: bool cumulative ops, unnormalised bool storage, `unique` storage class

kind: consistency
evidence: tensor/ops/reduce.ts:363 and :377 (`cumsum`/`cumprod` take bool input as is, while `sum` at :188 casts bool to int32); probe: `cumsum` of 300 `true` has dtype `bool` and last value 44; tensor/tensor.ts:96 (`tensor(values, { dtype: 'bool' })` stores the raw numbers) and :174 (`asTensor(Uint8Array)` becomes a `bool` tensor with any byte); probes: `tensor([2, 256, NaN], { dtype: 'bool' })` holds `[2, 0, 0]`, `cast(bool 2, 'float64')` is 2 and `mean` of a bool tensor holding `[2, 0]` is 1; tensor/ops/indexing.ts:582 (`unique` builds an int32 or float32 tensor around `Float64Array.from(keys) as never`); probe: `unique(int32 tensor).values.data` is a `Float64Array`.
what: the tensor contract is "bool storage is 0/1 and the storage class matches the dtype". `full`, `scalar` and `cast` normalise; `tensor`, `asTensor(Uint8Array)` and `fromStorage` do not, and two ops produce wrong dtypes. The bool wrap at 256 is silent data corruption.
fix: cast bool to int32 in `cumsum`/`cumprod` (as `sum`/`prod` do); normalise to 0/1 in `tensor()`, `asTensor(Uint8Array)` (and document `fromStorage` as trusted); allocate `unique` values with `allocate(t.dtype, n)`.
size: S (<1h)
risk: low (only affects inputs that were already wrong)
preserves behaviour: no (cumulative ops on bool return int32; bool constructors normalise)
changes public surface: no

### F6: `scatterAdd`, `segmentSum` and the scatter kernels never range-check indices

kind: consistency
evidence: tensor/_cpu/structure.ts:180-197 (`takeScatter`: `dst = (o * size + (indices[t] ?? 0)) * inner`, no bound test) and :218-237 (`putAlongAxisAdd`, same); tensor/ops/indexing.ts:376 (`scatterAdd`) and :395 (`segmentSum`) pass user indices straight through, while `take` throws `value` for the same mistake (structure.ts:167); probes: `scatterAdd([2, 2], [[2], [0]], [[1], [1]], 1)` gives `[[0, 0], [2, 0]]` (the update for column 2 landed in the next row), `segmentSum([1, 2, 3], [0, 5, 1], 2)` gives `[1, 3]` (the 2 vanished).
what: out-of-range scatter indices corrupt a neighbouring lane (axis > 0) or are dropped (axis 0); the gather side throws. Negative indices are also not wrapped, unlike `take`.
fix: validate in the two ops (the indices are concrete) with the same error and negative wrapping as `take`; leave the kernels unchecked (they are trusted internals) but document it.
size: S (<1h)
risk: low
preserves behaviour: no (invalid indices now throw; negative indices wrap)
changes public surface: no

### F7: derivative rules that return NaN at points where the derivative exists

kind: consistency
evidence: tensor/ops/reduce.ts:64 and the jvp above it (`prod`: `div(mul(gx, out), x)`, the code comment says "exact away from zeros"); tensor/ops/arithmetic.ts:46-57 (`pow` jvp/vjp: `mul(b, pow(a, sub(b, 1)))`); tensor/ops/reduce.ts:476-487 (`norm` is `sqrt(sum(square))`); tensor/ops/reduce.ts `std` is `sqrt(variance)`; probes: `grad(prod)([0, 2])` is `[NaN, 0]` (should be `[2, 0]`), `grad(prod)([0, 0])` is `[NaN, NaN]` (should be 0), `grad(norm)([0, 0])` is `[NaN, NaN]` (torch gives 0), `grad(std)([1, 1])` is `[NaN, NaN]`, `grad(sum(pow(x, 0)))([0, 1])` is `[NaN, 0]` (should be 0), `grad(sum(sqrt))([0])` is NaN (infinite is correct).
what: gradient-based code that starts at zeros (zero-initialised parameters, a zero residual, a constant batch) gets NaN gradients. This is the class the flaky tests keep hitting. The generators in `elementwise.property.test.ts` use `awayFromZero` and `tensorOf(s, 0.1, 5)`, so none of it is tested.
fix: `prod`: compute the product of the others with an exclusive cumulative product from both ends (no division); `norm` ord 2: give the vjp as `g * x / where(n == 0, 1, n)`; `std`: same safe division; `pow`: guard `b == 0` in the `a` rule (`where(b == 0, 0, ...)`) the way `powLogTerm` already guards `a == 0`.
size: M (1-3h; each rule is small, the test table in F31 is most of the work)
risk: medium (derivative rules; `checkPrimitive` property tests will catch regressions off zero; note `prod` jvp/vjp must stay differentiable for higher order)
preserves behaviour: no (NaN becomes the correct finite value at the degenerate points)
changes public surface: no

### F8: three Algorithm drivers that duplicate each other and already disagree

kind: duplicate
evidence: trace/algorithm.ts:171-231 (`runAlgorithm`), :236-259 (`iterate`), :288-360 (`initialSlice`, `runFor`); each re-derives `foldIn(root, 0)` for init and `foldIn(root, 1)` (algorithm.ts:188, :244, runFor) for steps, and trace/session.ts:208 uses `foldIn(root, 7)`; divergence: `iterate` ends with `t >= maxSteps ? 'completed' : algorithm.done?.(...)` (algorithm.ts:249) while `runAlgorithm` (:219) and `runFor` ask `done` at the limit; probe: an algorithm whose `done` reports `'converged'` at step 3 with `maxSteps: 3` ends `'converged'` in `runAlgorithm` and `'completed'` in `iterate`; `initialSlice`/`runFor` start the run with `seed: null` (algorithm.ts:300, :346) where `runAlgorithm` records the seed, so sliced runs get the id `name:none`.
what: the same step loop (init key, step key per `t`, done check, observe on schedule, checkpoint) is written three times with small differences. The key namespace (0 init, 1 steps, 7 make) is a constant that must agree across five sites for `stateAt`/`resume`/`seek` to reproduce a run; nothing names it.
fix: one internal `stepper(algorithm, root, run)` that owns the key derivation, `done` rule and observe/checkpoint order; `runAlgorithm`, `iterate`, `runFor` become thin loops over it; export the three key tags from one place; record the seed in the sliced runs.
size: M (1-3h)
risk: low-medium (`algorithm.test.ts`, `trace.test.ts`, `session.test.ts` and the property tests pin the behaviour; the `iterate` status change is a (correct) behaviour fix)
preserves behaviour: no (`iterate` final status and sliced-run seed/id change)
changes public surface: no

### F9: random samplers: parameter validation is inconsistent and degenerate parameters give wrong draws

kind: consistency
evidence: random/discrete.ts:219-222 (`geometric`: `Math.log(1 - p)`); probe: `geometric(key, 1e-18, [3])` returns `[0, 0, 0]` (the support starts at 1) because `1 - 1e-18 === 1`, and `p = 1e-10` loses six digits of the log; random/discrete.ts `ints` stores into an `Int32Array`, so `poisson`, `binomial` and `geometric` values beyond 2^31 wrap (probe: `geometric(key, 1e-10, ...)` has mean 7e7 where the true mean is 1e10); probe: `poisson(key, Infinity, [1])` returns `[0]` (`!(lambda >= 0)` accepts it); random/continuous.ts:118-190 (`normal`, `exponential`, `laplace`, `logistic`, `gumbel`, `cauchy`, `uniform`, `vonMises`, `studentT`) accept negative or zero scales, `rate <= 0`, `high < low` while `gamma`, `beta`, `truncatedNormal`, `dirichlet`, `randint`, `geometric` validate; `studentT(df <= 0)` fails with the message of `gamma: alpha must be positive`; `bernoulli` (discrete.ts:35) does not check `p`.
what: two validation conventions in one file, one real numeric bug (`geometric` for small `p`) and a silent int32 wrap in three discrete samplers.
fix: validate every sampler's parameters with one small helper (`finitePositive(name, value)` etc.); use `Math.log1p(-p)` in `geometric` and `binomialWaiting` (`Math.log(1 - q)`, _variates.ts:105); reject results that would not fit int32 (or return a float64 tensor for those large cases).
size: S (<1h)
risk: low (values for ordinary `p` change in the last bits from `log1p`, which could move a golden fixture within tolerance)
preserves behaviour: no (invalid parameters now throw; `geometric` small-`p` fixed)
changes public surface: no

### F10: key path parts alias because `foldIn` truncates silently

kind: consistency
evidence: random/key.ts:87-89 (`philox4x32([data >>> 0, ...])`, no check) and :153-156 (`keyFromPath` passes numbers straight through; the TSDoc says "non-negative integers"); context/context.ts:54 (`ctx.key(...names)` is `keyFromPath`); probe: `keyFromPath(0, 'a', -1)` equals `keyFromPath(0, 'a', 4294967295)` and `ctx.key('a', 1.5)` equals `ctx.key('a', 1)`.
what: the reproducibility guarantee ("independent parts never share randomness") fails for negative, fractional or >= 2^32 indices, and for NaN (`NaN >>> 0` is 0, so `foldIn(k, NaN)` equals `foldIn(k, 0)`). Callers that compute an index (episode, fold, replicate) can alias without any error.
fix: in `foldIn` throw `OmaiError('value')` unless `data` is an integer in `[0, 2^32)`; if some existing caller relies on negative or fractional indices, route it through a named hash instead.
size: S (<1h)
risk: medium (an engine-wide grep for `foldIn(` with computed arguments is needed first, and a legitimate caller passing a float would start to throw; changing the derived keys is not needed)
preserves behaviour: yes for valid use (invalid use throws)
changes public surface: no

### F11: `einsum` materialises the product of all operands before summing

kind: hot-path
evidence: tensor/ops/linear.ts:207-262 (`product = mul(product, v)` over operands aligned to the full letter set at :248, then one `sum`); probes: `einsum('ij,jk->ik')` on 200x200 takes 241 ms against 11 ms for `matmul` (an 8e6-element intermediate, 64 MB), and attention scores `einsum('bqd,bkd->bqk')` on 8x64x32 takes 32 ms against 1.6 ms for the transposed `matmul`. 43 call sites use `einsum` (nn/blocks, nn/additive, nn/kan, kernels/dot-product.ts).
what: cost and memory are the product of every index extent, not the cost of the contraction.
fix: contract pairwise (left to right, or greedy by smallest intermediate): for each pair, put shared-and-summed letters last, `reshape` to 2-D, `matmul`, `reshape` back, summing letters that appear nowhere else first. `tensordot` (linear.ts:132) already does the 2-D reduction.
size: M (1-3h)
risk: medium (results agree to rounding, not bitwise; gradients come for free from `matmul`; `checkPrimitive` over a handful of specs pins it)
preserves behaviour: yes (values to rounding)
changes public surface: no

### F12: reverse mode computes cotangents for inputs that need none

kind: hot-path
evidence: autodiff/reverse.ts:148-160 (`rule(gs, inputs, params, outputs)` is called for every node and the cotangents of untracked inputs are then dropped at :158 `if (g !== null && (x.node !== null || x.leaf))`); tensor/ops/_factories.ts:72-79 (binary vjp computes both `ga`, `gb` and sums both back to shape); tensor/ops/linear.ts `matmulPrimitive.vjp` computes both products; probes: `grad` of `sum(matmul(X, W))` with respect to `W` only costs 483 ms for 512x512, the same as differentiating with respect to both (474 ms), against 155 ms for the forward product; `grad(sum(mul(mul(x, 2), 3)))` costs 51 ms against 16 ms forward (the constants 2 and 3 get a full-size cotangent product and a reduction each).
what: about a third of the backward work of a matmul with one constant operand (the first layer of any model, whose input is data) and one full-size multiply plus reduction per `op(x, constant)` is thrown away.
fix: pass an optional fifth argument to `vjp`, `needs: readonly boolean[]` (true where the input is tracked), computed in `backward`; `binaryPrimitive` and `matmulPrimitive` honour it and return `null` early; other rules ignore it (backward compatible, adopt rule by rule).
size: M (1-3h)
risk: medium (touches the `Primitive.vjp` contract and `checkPrimitive` must keep passing; the extra argument is optional)
preserves behaviour: yes
changes public surface: yes (optional parameter on `Primitive.vjp`)

### F13: softmax, log-softmax, logsumexp and softplus are chains of 8-10 primitives

kind: hot-path
evidence: tensor/ops/reduce.ts:402-460 (`logsumexp` = max, stopGradient, isfinite, where, sub, exp, sum, log, add, reshape; `softmax` and `logSoftmax` add a further sub and exp); tensor/ops/math.ts:373 (`softplus` = greater, where x3, exp, neg, log1p, add); probe: `softmax` of a 1000x1000 matrix takes 127 ms (each full-size op costs 5-20 ms), about 10x a fused kernel; these are the inner loop of every classifier, attention block and loss.
what: the intermediate arrays dominate; a fused primitive reads the row once for max, once for the sum and once for the output.
fix: add `logsumexp` and `softmax` as primitives (kernel over `lanes`, `vjp` written with `softmax * (g - sum(g * softmax))`, `jvp` likewise), keep today's composite as the reference used by the conformance tests; `softplus` as a unary kernel with `sigmoid` as its derivative.
size: M (1-3h)
risk: medium (new primitives need `checkPrimitive`, batch rules, and the existing composite values must match to 1e-12)
preserves behaviour: yes (values to rounding)
changes public surface: no

### F14: the CPU kernels pay for generality on every element

kind: hot-path
evidence: tensor/_cpu/elementwise.ts:95-102 (`map2` always builds two `Int32Array` index maps even when both shapes equal the output, then loads through them), :114-125 (`where` builds three); :104-111 (`binary` and `compare` call a per-element closure, `compare` wraps a second closure); tensor/_cpu/_util.ts:44-64 (`walkStrided` calls `visit(i, offset)` per element and maintains a counter array; used by `reduce` (_cpu/reduce.ts:29), `transpose`, `slice`, `sliceScatter`, `pad`); tensor/_cpu/structure.ts:120-138 (`concat` copies element by element where `TypedArray.set` of a `subarray` is a memcpy); tensor/tensor.ts:93-96 (`tensor()` builds a boxed `number[]` before filling the typed array); tensor/tensor.ts:171-174 (`asTensor` copies a typed array on every call, so `add(Float64Array, ...)` pays a copy per operand); probes: `add(a, b)` of 1e6 doubles 5.0 ms (raw loop 1.8 ms), `mul(a, 2)` 17 ms once other ops have made `fn` megamorphic, `sum(a)` 6.7 ms, `where(greater(a, 0.5), a, 0)` 31 ms.
what: allocation of two 4 MB index maps and an indirect call per element on the hottest path of every operation; the measured cost is 3-10x a specialised loop, not catastrophic.
fix: in `map2`, branch on three cases before building anything: equal shapes (direct loop), one operand of size 1 (scalar loop), general (today's path); in `reduce`, add a path for contiguous trailing-axis and full reductions; make the inner (last-axis) loop of `walkStrided` a plain `for`; `concat` and `take` copy runs with `set`/`copyWithin`.
size: M (1-3h)
risk: low (pure refactor of kernels covered by the property and golden tests; verify float32 and int paths by the dtype table in F31)
preserves behaviour: yes
changes public surface: no

### F15: scalar kernels differ from numpy at the edges

kind: consistency
evidence: tensor/_cpu/elementwise.ts:48 (`pow: a ** b`), :52 (`floorDiv: Math.floor(a / b)`), :54 (`mod: a - b * Math.floor(a / b)`); tensor/ops/math.ts:147-155 (`log10`, `log2` are `log(x) / LN10`, `/ LN2`); probes: `pow(1, Infinity)`, `pow(1, NaN)`, `pow(-1, Infinity)` are NaN (JS `**`; numpy and C give 1), `floorDiv(1, 0.1)` is 10 and `mod(1, 0.1)` is 0 (numpy: 9 and 0.0999...; the exact residual is what `fmod` gives), `log10(1000)` is `2.9999999999999996` (numpy 3).
what: ordinary idioms (`floor(log10(x))` for a decade, `x // step` for a bin index, `1 ** nan`) differ from the reference the project targets ("scipy parity"). `mod` also loses precision when `a / b` is large.
fix: kernels: `pow` special-cases `a === 1` and `b === 0`/`NaN` per C99; `mod` through `r = a % b; if (r !== 0 && (r < 0) !== (b < 0)) r += b`, `floorDiv` as `(a - mod(a, b)) / b`; add `log2`/`log10` unary kernels (with the derivative `1 / (x ln b)`).
size: S (<1h) for pow/mod/floorDiv, M with the new unary kernels
risk: medium (a fixture computed with the old formulas could move; nothing relies on 1 // 0.1 by design but run the golden tests)
preserves behaviour: no (edge values change to the numpy ones)
changes public surface: yes for the new `log2`/`log10` entries of `UnaryKernel`; no otherwise

### F16: the error code `not-differentiable` means two different things and is used as a control signal

kind: consistency
evidence: base/errors.ts:17; thrown for "needs a concrete value" at tensor/traced.ts:95, tensor/tensor.ts:254, tensor/pytree.ts:229, tensor/format.ts:88, autodiff/check.ts:25, and for "no derivative rule" at autodiff/forward.ts:51, reverse.ts:144, taylor.ts:63, _jet-rules.ts:349; consumed as a signal by autodiff/vmap.ts:138-141 (`isConcreteValueError` re-runs the whole function as a per-example loop on any `not-differentiable`) and autodiff/graph.ts:135 (draws the graph without gradients).
what: `vmap` of a function that contains a genuine missing derivative rule (for example `vmap(grad(f))` where `f` meets a primitive without `vjp`) silently falls back to the loop and then fails (or succeeds slowly) instead of reporting the missing rule; the function body also runs twice, so side effects (taps, logging) repeat.
fix: add an `ErrorCode` `'concrete-required'` for the "needs a concrete value" cases and let `vmap` catch only that; keep `not-differentiable` for missing rules.
size: S (<1h)
risk: low-medium (code or tests that branch on the old code for the concrete case)
preserves behaviour: no (vmap no longer swallows missing-rule errors)
changes public surface: yes (new member of `ErrorCode`)

### F17: layering: the proposed order is wrong for `context` and `scene`, the code is right

kind: layering
evidence: imports read in this review: `base` imports nothing from the core (zod only); `tensor` imports `base`; `random` imports `base`, `tensor`; `space` imports `base`, `random`, `tensor`; `trace` imports `base`, `random`, `space`, `tensor`; `autodiff` imports `base`, `random` (check.ts:8), `tensor`, `scene` (graph.ts:8); `context` imports `random`, `tensor`, `trace` (context.ts:9-11, resolve.ts:7-8) and is imported by none of the other six; `scene` imports only `base` (checked over `scene/**`).
what: nothing in scope violates a stack of base < tensor < random < space < trace < context, with autodiff beside trace and above tensor/random. The two "against the order" reports are artefacts of ranking `context` low and `scene` high.
fix: change the first-guess order in `make layers`/the scan to base < scene < tensor < random < space < trace < context and autodiff after tensor/random/scene; no code change. If `autodiff/graph.ts` ever needs to stay free of `scene`, move `computationGraph`/`graphStats` (the only users, plus nn/describe.ts) up to `scene` or `exposition`.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F18: `trace/serialise.ts` re-implements `base/json.ts` with different encodings

kind: duplicate
evidence: trace/serialise.ts:10-41 (a `TYPED` table and `typedName` identical to base/json.ts:68-97); :43-70 (`stateToJson` encodes typed arrays as `{ $typed, data }`, tensors as `{ $tensor }`); :106-112 (`numberToJson` writes `'NaN'`, `'Infinity'`, `'-Infinity'`); base/json.ts:42-65 (`encodeNumber` writes `'nan'`, `'inf'`, `'-inf'`), :125-145 (`toJson` encodes typed arrays as `{ $typed, values }` with `{ $number: ... }` for non-finite elements); tensor/format.ts:89 writes tensors with `encodeNumber`.
what: three spellings of "a non-finite number in JSON" and two shapes of "a typed array in JSON" in one engine; checkpoint states and traced tensors use different ones, so a state cannot be fed to `fromJson` or a tensor JSON to `stateFromJson`.
fix: `stateToJson(x)` = `toJson(x)` after mapping tensors to `TensorJson` (a one-line pre-pass), `stateFromJson` = `fromJson` plus the tensor hook; `numberToJson`/`numberFromJson` delegate to `encodeNumber`/`decodeNumber`. Read the old checkpoint spellings in `stateFromJson` for one release if stored traces exist.
size: M (1-3h)
risk: medium (persisted checkpoints and golden traces change format unless the old spelling is kept on the read side)
preserves behaviour: no (checkpoint JSON format changes)
changes public surface: no

### F19: `modes()` re-implements `optionTable()`

kind: duplicate
evidence: trace/modes.ts:123-160 against space/tables.ts:64-95: the same six-line construction `union(Object.fromEntries(names.map((n) => [n, table[n]?.space])), { tag, ...default, ...label, ...description })`, the same `name in table` fallback to the first or default option (modes.ts:140, tables.ts:83), the same `knobsOf`/`asRecord` record coercion; CLAUDE.md names both as the two slot tables (algorithms, values). Phase 0 also found `knobsOf` copied into five `*-options.ts` files outside this scope.
what: a mode is an option whose properties are `{ needs, extras, params }`; the extra code in `modes` is only `admissible`/`needs` and `make(problem, config, key, others)`.
fix: build `modes` on `optionTable` (an `Option` with `make(knobs)` returning a function of `(problem, key)` and `properties` holding the rest); export `knobsOf` from `space/tables.ts` for the other copies.
size: M (1-3h)
risk: low-medium (type-level tests in trace/contract.test-d.ts and modes.test.ts pin it)
preserves behaviour: yes
changes public surface: no

### F20: `in` on plain objects accepts inherited names (`constructor`, `toString`)

kind: consistency
evidence: space/_union.ts:21 (`name in space.options`), space/tables.ts:83 (`name in table`), trace/modes.ts:140 (`name in table`), space/operations.ts (`name in space.options` in `contains` and `clip`); probe: `contains(union, { kind: 'constructor' })` returns `undefined` (not `false`) because the Object constructor is treated as the option's space and the `switch` falls through.
what: a configuration coming from a URL or a control (the stated use of these spaces) can name `constructor` and reach a function instead of an option; `contains` returns a non-boolean.
fix: `Object.hasOwn(table, name)` at the four sites; give `contains` a `default: return false`.
size: S (<1h)
risk: low
preserves behaviour: yes (except for the invalid names)
changes public surface: no

### F21: every trace event is written twice, as an interface and a zod schema, joined by a cast

kind: duplicate
evidence: trace/events.ts:30-165 (nine interfaces, `TraceEvent` union, `Run`) and :167-217 (`TraceEventSchema` and `RunSchema`, hand-written, ending `]) as z.ZodType<TraceEvent>` at :217); base/json.ts:13-27 does the same for `Json`/`JsonSchema`.
what: the cast makes the compiler accept any schema, so adding a field to an interface without the schema (or the reverse) is invisible until a trace fails to parse at the boundary.
fix: the smallest change is a `test-d.ts` asserting `z.infer<typeof TraceEventSchema>` and `TraceEvent` are mutually assignable (exact optional property handling may need `exactOptionalPropertyTypes` care); the larger one derives the interfaces from the schemas.
size: S (<1h) for the type test, M for derivation
risk: low
preserves behaviour: yes
changes public surface: no (type test); yes if the interfaces become inferred types

### F22: the backend abstraction is declared but nothing uses it

kind: rule
evidence: context/context.ts:52-53, :75-76, :103-104 (`Context.backend` and `.dtype` are stored and overridden, and read by nothing in the engine: no `ctx.backend`/`ctx.dtype` outside context.ts); tensor/backend.ts:71-96 (`BackendCapabilities`, `supportsDType`: no caller outside tests and the example); tensor/backend.ts:38, :52-54 (`'not'`, `'and'`, `'or'`, `'xor'` kernels) are never bound by any primitive (coverage: tensor/_cpu/elementwise.ts:40, :55-57 never run) because `logicalAnd`/`Or`/`Xor`/`Not` are composites of notEqual, cast, add and equal (tensor/ops/compare.ts:237-262, seven dispatches per `logicalAnd`).
what: three pieces of a backend story that do not connect: the context carries a backend nobody consults, the contract lists kernels no op calls, and the capabilities exist for a dispatcher that does not exist. A second backend author would implement dead kernels.
fix: either wire (ops choose `backend` from the first concrete input, constructors accept `options.backend`, logic ops bind the `and/or/xor/not` kernels) or delete the unused parts (`Context.backend`/`dtype`, the four kernels, the capabilities) until a second backend exists.
size: M (1-3h)
risk: medium (removing exported fields is a surface change; wiring touches create.ts and indexing.ts)
preserves behaviour: yes
changes public surface: yes (removed or newly honoured fields and kernel names)

### F23: constructors hard-wire `cpu` and handle `dtype`/`backend` options inconsistently

kind: consistency
evidence: `cpu.wrap` in 37 places inside tensor (tensor/create.ts:143, :164, :183, :211, :230, :273, :304, :364; tensor/ops/indexing.ts, 16 sites; tensor/pytree.ts:245; tensor/format.ts:109; tensor/tensor.ts:171-174); `full`/`zeros`/`ones` accept `{ dtype, backend }` (create.ts:31) but `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `tri`, `meshgrid`, `indices` take no options and always return float64 on `cpu`; `asTensor(x, dtype)` ignores `dtype` for a Value or typed array (tensor/tensor.ts:169-176; probe: `asTensor(float64 tensor, 'int32').dtype` is float64); `ravel`/`unravel` always rebuild float64 (pytree.ts:245) so float32 leaves change dtype on the round trip; `treePaths` skips invalid leaves silently (pytree.ts:167-181) where `treeFlatten` throws (:72-76); `diag` only accepts a 1-D input (create.ts:245) where numpy's `diag` also extracts a diagonal.
what: option handling differs between neighbouring constructors and some ignore the option they appear to take.
fix: give the range/identity constructors the same `CreateOptions<D>` (default float64), make `asTensor` reject or apply a conflicting `dtype`, keep leaf dtypes in `unravel`, make `treePaths` throw like `treeFlatten`.
size: M (1-3h)
risk: medium (changes return types of constructors from `Tensor<'float64'>` to `Tensor<D>` defaulting to float64, compatible for existing callers)
preserves behaviour: yes for existing calls
changes public surface: yes (added optional parameters)

### F24: import cycles inside `tensor` that `make layers` cannot see

kind: layering
evidence: probe (strongly connected components over non-test files of the seven domains): one cycle of 6 files `tensor/cpu.ts` <-> `tensor/_cpu/{_util,elementwise,reduce,sort,structure}.ts` (`_cpu/_util.ts:2` imports `cpu` only to put it in `wrap`'s result); one cycle of 9 files `tensor/batching.ts` + `tensor/ops/{_coerce,_factories,arithmetic,compare,indexing,math,reduce,shape}.ts` (batching.ts:9 imports ops/shape.ts, which imports batching for its batch rules); no cycle in the other six domains. The batching.ts header comment ("every export is a function declaration so operation modules can use them whatever the module evaluation order") is the cost: module-top `const prims = { exp: unaryPrimitive(...) }` (ops/math.ts, compare.ts, reduce.ts) run at import time and work only because of evaluation order.
what: the layer checker catches cycles between domains only, so a file-level cycle can be introduced or reordered by an import change and break at load time rather than at a lint.
fix: smallest: extend `make layers` to report file-level cycles and baseline the two existing ones. Structural: `_cpu` takes the backend as a parameter (or `cpu.ts` defines `wrap` and kernels import it from a leaf), and batching.ts imports the ops it needs lazily through a small leaf module of shape helpers.
size: S for the check, M for removing the cycles
risk: low for the check, medium for untangling
preserves behaviour: yes
changes public surface: no

### F25: Taylor-mode rules find their primitives by running public ops on placeholders

kind: duplicate
evidence: autodiff/_jet-rules.ts:215-238 (`primitiveOf` runs an op once on abstract placeholders to catch the primitive it binds; the comment says "most primitives are not exported"), :240 (`const matmulPrimitive = primitiveOf(matmul, ...)` although `matmulPrimitive` is exported from tensor and imported in autodiff/custom.ts:11), :253-256 (`LINEAR` set of 16 primitives found the same way), :262-296 (`RULES` map of 25 more); the arithmetic primitives are exported with `@internal` at tensor/ops/arithmetic.ts:18-132; unlisted primitives fall back silently to `jvpSeries` (taylor.ts:63-ish), which is correct but slower.
what: a second registry of "all tensor primitives" in autodiff, keyed by discovery at import time; a new tensor primitive needs a matching entry here or silently takes the slow path, and the discovery relies on placeholder tracing working for each op.
fix: import the exported primitives instead of rediscovering them, and give `Primitive` an optional `linear?: boolean` flag (tensor stays free of jet types) so `LINEAR` disappears; keep the `RULES` map keyed by primitive identity but built from the imports.
size: M (1-3h)
risk: low-medium (taylor.test.ts pins values; the flag is additive)
preserves behaviour: yes
changes public surface: yes (optional field on `Primitive`)

### F26: small helpers written twice inside the scope, and copies elsewhere of core helpers

kind: duplicate
evidence: (a) "concrete tensor or throw": tensor/traced.ts:93 `concreteOrThrow` is canonical; copies at tensor/tensor.ts:253 and autodiff/check.ts:24, and outside scope explain/gradients.ts:33, explain/internal.ts:32 (these two identical), data/schema/_coerce.ts:6, kernels/expansion.ts:67; (b) `roundHalfEven` identical at tensor/_cpu/elementwise.ts:8, numerics/approximation/floating-point.ts:128, systems/frequency.ts:105, vision/_image.ts:76; (c) `clipTo` at stochastic/diffusion-samplers.ts:261 against `clip` at tensor/ops/arithmetic.ts:369; (d) `elementBounds` (space/operations.ts:20 and space/encoding.ts:16) and `snap` (operations.ts:24 and encoding.ts:32) copied between two files of one folder, and `sample` re-derives the log/linear interval mapping that `fromUnit` (encoding.ts) already has (the encoding copy clamps the rounded endpoints, `sample` does not); (e) Fisher-Yates in random/permutation.ts:27 and again in random/sequences.ts:165-178 (`latinHypercube`); (f) `reservoir` writes algorithms L and R once for arrays and again for iterables (random/permutation.ts:181-260); (g) the malformed-path check in `defineContract` (trace/contract.ts:178) and `defineMethod` (trace/session.ts:81).
what: each is small; together they show the rule "no local copies of generic helpers" is not enforced below the numeric kernels (`blas.ts`).
fix: (a) call `concreteOrThrow`; (b) one `roundHalfEven` in `base/numbers.ts`; (c) call `clip(x, lo, hi)` (it takes numbers); (d) a `space/_scale.ts` with `elementBounds`, `snap`, `fromUnit`; (e) `permutation` called with a derived key; (f) run the array case through the iterator branch; (g) a shared `assertPath`.
size: S (<1h) each, M in all
risk: low (each is covered by existing tests; (a) changes an error message)
preserves behaviour: yes (except the (a) messages)
changes public surface: no (b adds one export to `base`)

### F27: same names for different things inside the foundations, and a stateful generator that the docs forbid in library code

kind: consistency
evidence: autodiff/custom.ts:98 (`stopGradient` is a pure wrapper of tensor's `stopGradient`, exported from both `omai-engine/autodiff` and `omai-engine/tensor`); trace/contract.ts exports `tensor`, `scalar`, `message`, `frame`, `event`, `derived` (contract constructors) while `tensor` is also tensor/tensor.ts's constructor; space/operations.ts exports `clip` and `flatten`, tensor/ops export `clip` and `flatten`; random/permutation.ts end (`keys` is `split`); random/generator.ts:15 says "Library code takes Key arguments; use this class in scripts and notebooks" but library code uses it: environments/minatar.ts:60, environments/procedural.ts:252 and :574, optim/minimize/particle-swarm.ts:126 and :157.
what: Phase 0's collision list, read for this scope: one real duplicate (`stopGradient`), one alias (`keys`), several same-word-different-meaning exports, and a documented rule the code does not follow.
fix: re-export `stopGradient` from tensor (`export { stopGradient } from '../tensor/index.ts'`), delete `keys`, prefix the contract constructors on import (or rename to `scalarPath`...) only if collisions bite in a barrel, and either update the `Generator` doc or pass keys in the four library uses.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: yes (removing `keys`, changing where `stopGradient` is defined)

### F28: copies forced by the layer order: `normalQuantile`, `logGammaSmall`, and an unvalidated `estimate`

kind: duplicate
evidence: base/results.ts:117-147 (Acklam normal quantile, relative error 1.2e-9) against numerics/special/_kernels.ts:138 `ndtri`; random/_unit.ts:28-39 (Lanczos log-gamma, "about 15 digits") against numerics/special/_kernels.ts:227 `lgamma` and stochastic/diffusions.ts:670 `lgammaOf`; base/results.ts:83-93 (`estimate(value, se, level)` takes any `level`, hard-codes 0.95 at :83 and :113) against base/options.ts:101 `resolveConfidenceLevel` and :86 `defaultConfidenceLevel`; probe: `estimate(1, 0.5, 1.5)` returns an interval of NaN.
what: `base` and `random` sit below `numerics`, so they cannot import the accurate special functions; the result is lower-precision private copies (harmless for interval widths, and the samplers only need log-gamma at integers and half-integers) plus `estimate` ignoring the validator that sits in its own package.
fix: call `resolveConfidenceLevel` and `defaultConfidenceLevel` in `estimate`/`formatResult`; either move the two scalar kernels (`ndtri`, `lgamma`, a few dozen lines each) down into `base/` and import them from `numerics/special` and `random`, or document the copies as deliberate (a comment with the precision and a test pinning agreement with `ndtri`/`lgamma` to 1e-8).
size: S for `estimate`, M for moving the kernels
risk: low
preserves behaviour: yes for valid levels (invalid ones now throw)
changes public surface: no

### F29: two numeric vocabularies (`base` plain numbers, `tensor` ops) with different degenerate conventions

kind: consistency
evidence: base/numbers.ts:22 `isClose(a, b, rtol = 1e-9, atol = 0)` against tensor/ops/compare.ts:360 `allclose(a, b, rtol = 1e-5, atol = 1e-8)` (probe: `isClose(0, 1e-300)` is false, `allclose([0], [1e-300])` is true; `allclose` on a traced value returns false silently because of `isTensor(ok) && ...`); base/numbers.ts:358 `varianceOf([1], 1)` is NaN against tensor/ops/reduce.ts `variance` throwing `value` (probe); base/numbers.ts:277 `softmaxOf([-Infinity, -Infinity])` is NaN, as the tensor `softmax` (consistent, but undocumented in `softmaxOf`); base/numbers.ts:238 `range(0, 5, 0)` returns `[]` where tensor `arange` throws on step 0 (create.ts:139); `quantileOf` clamps `q` silently (:302) and accepts NaN.
what: the base kernels are intentional (plain numbers, used where tensors are overkill, below tensor in the stack) but the two vocabularies disagree on defaults and on whether degenerate input throws, returns NaN or returns a value; the base ones also duplicate tensor semantics by name (`meanOf`, `varianceOf`, `softmaxOf`, `linspaceNumbers`).
fix: align the defaults of `isClose`/`allclose` (state the numpy defaults in both or give them different names), make `allclose` throw on a traced input, pick one rule for `n <= ddof` (NaN for the kernels is defensible: say so in the TSDoc of both), validate `step` and `q` in `range`/`quantileOf`.
size: S (<1h)
risk: low
preserves behaviour: no (defaults and degenerate results of `allclose`, `range` change)
changes public surface: no

### F30: space constructors validate unevenly

kind: consistency
evidence: space/types.ts:166-187 (`real` and `int` check `high >= low`; `int` rounds `low` up and `high` down after the check); :238 (`box` checks nothing: bounds, shape, default); :276 (`multiDiscrete` no check); :295 (`multiBinary` no check); :259 (`discrete` checks `n`); probe: `int(0.2, 0.8)` returns `{ low: 1, high: 0 }`.
what: an empty `IntSpace` is accepted, and sampling from it (`sample`, space/operations.ts int branch) returns `high`, outside the range.
fix: validate after rounding in `int`; check `shape` entries are non-negative integers and bounds broadcast in `box`; check `nvec` entries and `n` are positive integers.
size: S (<1h)
risk: low
preserves behaviour: no (invalid spaces now throw)
changes public surface: no

### F31: test gap: dtype handling and degenerate inputs of the core ops have no direct test

kind: test-gap
evidence: no test file exists for tensor/dtype.ts, shape.ts, tensor.ts, create.ts, format.ts, primitive.ts, backend/cpu or _cpu (tensor has 6 test files, all about ops, complex, pytree); only 4 test files in the engine mention `float32` and none of them is a tensor test; `resultType`, `weakResultType`, `broadcastShapes`, `normaliseAxis` are covered by `@example` only; branch coverage: tensor/_cpu/structure.ts 60%, elementwise.ts 61%, _util.ts 65%, ops/compare.ts 57%, ops/indexing.ts 67%; tensor/ops/elementwise.property.test.ts uses float64 tensors and domain-avoiding generators (`awayFromZero`, `tensorOf(s, 0.1, 5)`), no NaN/Infinity/empty/zero-size input; the bugs of F3-F7, F15 were all found by one-line probes.
what: the invariant "an op preserves or promotes dtype by the documented rules and handles 0, +-Infinity, NaN, empty and size-1 inputs like numpy" is tested nowhere, and it is exactly where the core fails.
fix: two table-driven test files (about 250 lines together, replacing nothing): (1) `dtype.test.ts`: a dtype x op-family matrix (unary, binary, compare, reduce, cumulative, where, concat, take, matmul over float32/int32/bool) asserting result dtype from `resultType`/`weakResultType`, storage class, bool 0/1, and int32 truncation; (2) `degenerate.test.ts`: about 40 rows `(op, input, expected)` with expected values from numpy for 0, -0, +-Infinity, NaN, empty and size-1 inputs (matmul with zero rows, tril with Infinity, `all([])`, `pow(1, inf)`, `mod`, `log10(1000)`, cumsum of 300 bools, `prod` gradient at zero, `norm` gradient at zero, scatter out of range).
size: M (1-3h)
risk: low (tests only; some rows will fail until F3-F7, F15 land: add them as `it.fails` or land together)
preserves behaviour: yes
changes public surface: no

### F32: test gap: second derivatives of primitives are never checked

kind: test-gap
evidence: autodiff/check.ts:154-212 (`checkPrimitive` checks vjp and jvp against finite differences, the dot-product identity, vmap against a loop and input mutation, all first order); tensor/primitive.ts:27-31 documents that rules "are written with operations, so they are themselves differentiable (higher-order derivatives work)"; the only second-order tests are polynomial and Hessian cases in autodiff/transforms.test.ts:21-65 and autodiff/taylor.test.ts.
what: a rule that calls a kernel directly, `primalOf`, or a non-differentiable branch would pass every first-order check and fail under `hessian` or `jacfwd(grad)`. `scatter` modes max/min already use `primalOf` (tensor/ops/indexing.ts:523).
fix: add `order?: 2` to `CheckOptions`: compare `jvp(vjp(f))` and `vjp(jvp(f))` (the Hessian-vector product both ways) with a finite difference of the first-order gradient, and turn it on in the elementwise, reduction and structural property tests for the smooth primitives (one option plus one call site per table).
size: M (1-3h)
risk: low
preserves behaviour: yes
changes public surface: yes (new optional field of `CheckOptions`)

### F33: test gap: samplers are tested by moments, not by distribution; key independence by one pair

kind: test-gap
evidence: random/random.test.ts:70-120 ("samplers have the right moments"), random.golden.test.ts (against scipy.stats; two grep hits for chi/ks/independence in all of random tests); random/random.test.ts:55-62 ("are reproducible and independent": one split pair compared for inequality); no test that `split(k, n)` children are uncorrelated, that `foldIn(k, i)` and `split(k)[i]` never collide, or that a sampler's prefix is stable across shapes (`normal(k, [n])` is a prefix of `normal(k, [m])` for n < m, the property that makes shapes reproducible).
what: a Philox or sampler mistake that keeps the first two moments (a wrong tail, a correlated pair of streams, a Box-Muller pair off by one) passes today's tests.
fix: two tests: `samplers.gof.test.ts` (Kolmogorov-Smirnov statistic of 20,000 draws against the exact CDF of `dists` for uniform, normal, exponential, laplace, logistic, gumbel, cauchy, gamma at 0.5/2/20, beta, studentT, truncatedNormal, vonMises; chi-square on counts for categorical, binomial (both branches), poisson (both branches), geometric) with a fixed key and a generous threshold; `keys.test.ts` (correlation of 64 split children below 5 sigma, no key collisions among `split`/`foldIn`/`keyFromPath` over a grid, prefix stability of every vectorised sampler).
size: M (1-3h)
risk: low (fixed keys make them deterministic)
preserves behaviour: yes
changes public surface: no

### F34: test gap: the three drivers are not compared, and several small exports are never called

kind: test-gap
evidence: trace/trace.test.ts:129 tests `iterate` for its states only, not its status; trace/algorithm.test.ts covers `runFor`; nothing runs one algorithm through `runAlgorithm`, `iterate`, `runFor` and `session` and compares keys, states and statuses (F8); trace/replay.ts 50% (`replay`, `mergeRuns`, `isRun` never run); tracer.ts `filtered` never run; base/errors.ts 36% (`invariant`, `assertFinite`, `assertShape`); base/numbers.ts `isClose`, `allClose`, `range` never run; tensor/format.ts 14%; tensor/create.ts `geomspace`, `fullLike`; tensor/dtype.ts `finfo`; tensor/ops/shape.ts `flatten`, `chunk`; tensor/ops/indexing.ts `nonzero` and its closures (15 of 86 functions); autodiff/transforms.ts `linearize`. (Several have an `@example` run by `scripts/examples.ts`, which the coverage run does not count.)
what: one cheap test would have caught F8; the rest are one-line gaps that matter little.
fix: `drivers.test.ts`: a small algorithm with `done`, observe and checkpoint run through the four entry points, asserting equal state sequences, step keys and final status; plus one `smoke.test.ts` that calls each export above once with a known answer (about 30 lines).
size: S (<1h) for the driver test, S for the smoke file
risk: low
preserves behaviour: yes
changes public surface: no

### F35: dead or unchecked API and incomplete rule enforcement

kind: rule
evidence: no non-test caller in the whole repo: `assertShape`, `assertFinite`, `invariant` (base/errors.ts:77-126), `formatTensor` (tensor/format.ts:45), `supportsDType` (tensor/backend.ts:94), `finfo`/`iinfo` (tensor/dtype.ts:120, :137), `avalOf`, `fullLike`, `tapsFor` (trace/taps.ts), `mergeRuns`, `exportRun`, `resolveTolerance` (base/options.ts:42); missing `@example`: space/tables.ts:64 `optionTable`, trace/logging.ts:181 `logContext`, trace/session.ts:99 `contractOf`; scripts/banned.ts:11-15 checks only `Math.random`, `Date.now`/`performance.now` and `console.*`, so `new Date()`, `crypto.getRandomValues`, `process.hrtime`, `setTimeout` are not banned (the scope is clean today: grep found none, and no `// allow:` opt-outs exist in scope); base/version.ts:6 says "generated from the package manifest" while the constant is typed by hand (it equals `0.1.2`, the manifest version, today).
what: the engine is a library, so unused exports can be deliberate surface, but the unvalidated half of the error vocabulary (`assertShape` etc.) is the part meant to give consistent messages and no engine code uses it (each file throws its own `OmaiError`).
fix: add the three missing `@example`s; extend `banned.ts` with `new Date(`, `crypto.`, `hrtime`, `setTimeout|setInterval|requestAnimationFrame`; add a test that `VERSION` equals `package.json`'s; decide per unused export (keep with a test, or drop): the F34 smoke test keeps the cheap ones alive.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: yes only if exports are dropped

### F36: native arrays in public signatures of the scope, classified

kind: native-array
evidence: `node scripts/review/signatures.mjs` rows for the seven domains (56 functions), read against their signatures:
- kernel (allowed, plain-number helpers below `tensor` in the stack): base/numbers.ts `allClose`, `kahanSum`, `pairwiseSum`, `maxOf`, `minOf`, `argmaxOf`, `argminOf`, `logSumExpOf`, `softmaxOf`, `meanOf`, `varianceOf`, `quantileOf`, `range`, `linspaceNumbers` (14); tensor/pytree.ts:226 `ravel` (the bridge to Float64Array optimisers) and tensor/tensor.ts:213 `toFlat` (a conversion out); random/continuous.ts:294 `gammaVariates` returns a `Float64Array` (a kernel for dists and bandits).
- shape, axis or index (not a vector): tensor `sum`, `prod`, `max`, `min`, `mean`, `variance`, `std`, `norm`, `logsumexp`, `any`, `all`, `transpose`, `squeeze`, `split`, `flip`, `tile`, `repeat`, `slice`, `tensordot`, `slidingWindowView`, `broadcastShapes`, `normaliseAxes`, `stridesOf`, `assertShape` (24); space `box` (shape), `multiDiscrete` (`nvec`).
- tape or JSON data: trace/summaries.ts `summarise`, `subsample`; trace/policy.ts `changed`, `shouldRecord`; trace/serialise.ts `numbersToJson`, `numbersFromJson`; space `encode`, `decode`, `flatten`, `unflatten` (unit-cube coordinates and `SpaceValue` are JSON values by design).
- public vector, migrate: random/continuous.ts:383 `dirichlet(k, alpha: readonly number[], ...)`; random/discrete.ts `categorical(k, weights, ...)`, `multinomial(k, n, probs, ...)`, `aliasTable(weights)`; random/permutation.ts `choice(..., { weights })`, `randomSplit(k, n, fractions)` (6).
  what: only the six random functions are real findings. `random` imports `tensor`, so there is no layering reason for them to take arrays.
  fix: widen those parameters to `TensorLike` and read them with `toFlat` once at entry; keep returns as they are. Widening accepts every existing `number[]` call. Mark the base kernels as kernels in the folder summary so the rule has a stated exception.
  size: S (<1h)
  risk: low (types widen)
  preserves behaviour: yes
  changes public surface: yes (parameter types widen, source compatible)

### F37: small hot spots in tape and tree handling

kind: hot-path
evidence: trace/summaries.ts:23-60 (`summarise` copies all finite values into a boxed `number[]` at :29, then makes three more passes; probe: 39 ms for 1e6 values, run per summarised tensor per scheduled step); tensor/pytree.ts:148 (`treeMap` stringifies both structure definitions on every call to compare them; it has 99 call sites in the engine, 92 of them in optim and nn, i.e. the per-step parameter updates); trace/session.ts `run()` and `stateAt` (`rec.run()` copies the events array, tracer.ts:64-70, then `seek` scans it from the start at checkpoint.ts); trace/query.ts:24-65 (`metricSeries`, `tensorSeries` scan the whole tape per name, so a view with k charts costs k scans per frame); tracer.ts:195-205 (with a finite `maxBytes` every event is `JSON.stringify`-ed to measure it).
what: none is a bug; each is linear work repeated per call, noticeable only with large tensors, many parameters, or long tapes with live views.
fix: single-pass `summarise` over the array-like with no copy; compare tree definitions structurally (a recursive function, no stringify); let `Recorder` expose a live read-only `events` view for `seek` and cache per-name index in `Session`.
size: S (<1h) each
risk: low
preserves behaviour: yes
changes public surface: no
