# The paper: a plan

A plan for describing the library in a peer-reviewed software paper. The target is the open-source track of JMLR
(MLOSS); the Journal of Open Source Education (JOSE) is a companion or a first step. This file records what the venues
ask for, what the paper would say, what we have, what is missing, and the order of work. The library's name is a
placeholder until the rename.

## 1. The venues

**JMLR MLOSS** (https://www.jmlr.org/mloss/mloss-info.html): a description of at most four pages in JMLR format
(references extra), a cover letter (licence, project address, version under review, evidence of an active user
community), and a public repository. Reviewers weigh:
- impact: novelty, breadth and significance; an active user community; openness (public repository, bug tracker, a
  forum); a comparison with previous implementations on run time, memory and features;
- user documentation: installation, tutorials, non-trivial examples, full API documentation viewable on a website;
- openness and sustainability: ease of contribution, a clear design, developer documentation;
- practice: extensive unit and integration tests with reported coverage (ideally near 100%), continuous integration on
  every supported platform and several dependency versions, at least one open-source operating system, no proprietary
  dependencies;
- licensing: an OSI licence, stated in the package and mentioned in each source file.

**JOSE** (https://openjournals.readthedocs.io/en/jose/): open-source educational software or learning modules. The bar
is need rather than novelty ("statement of need"); reviewers build and test the software and judge whether another
instructor could adopt it. Text under CC-BY, code under an OSI licence, a LICENSE file. A short paper; the
documentation lives in the repository. Free, with a DOI.

The two fit different halves of the contribution: MLOSS the software (breadth, correctness, design), JOSE the teaching
(the gallery, the hub, embeddable explanations). JOSE does not require an established community, MLOSS effectively
does, so JOSE can come first.

**Other venues** (checked 2026-10-25):
- **NeurIPS Education track** (new in 2026; https://nips.cc/Conferences/2026/CallforEducationalResources): code
  libraries, notebooks and interactive demos are accepted formats, with a two-page PDF and the materials. But the
  concept must be recent (roughly the past few years; established foundations such as backpropagation are out of
  scope), the resource must be created for the track, and publication is non-archival. It suits one recent topic built
  with the library (conformal prediction, diffusion, an explanation method), not the library itself. The 2026 deadline
  (4 September) has passed; expect the 2027 call around July.
- **NeurIPS has no code or software track**: the Evaluations and Datasets track wants datasets and benchmarks, and the
  reproducibility track (MLRC) requires acceptance at TMLR first. Neither fits.
- **EAAI** (AAAI's Symposium on Educational Advances in AI; https://aaai.org/conference/aaai/aaai-27/eaai-27-call/):
  papers on AI education, and the Model AI Assignments track, a peer-reviewed open archive of assignments, which suits
  hub-style exercises. EAAI-27's deadlines (September 2026) have passed; EAAI-28 should open around September 2027.
  In-person presentation required.
- **VISxAI** (IEEE VIS workshop on visualisation for AI explainability; 9th edition at VIS 2026): interactive
  explainables, which suits individual explorable pages built with the library. The call is posted at visxai.io.
- **JOSS** (Journal of Open Source Software): a short paper about research software, with review of the repository;
  quick and citable, a lighter alternative to MLOSS if a DOI is wanted early.
- **ICLR blogpost track**: Markdown only, and its founding call discourages embedded JavaScript, so not for the
  interactive work.

## 2. The claim

One library, in the browser, spanning the paradigms of AI, where every method returns the data that explains it.
- **Breadth in one design.** Numerics and differentiable programming (JAX-style opt-in autodiff), probability and
  inference (distributions, sampling, probabilistic programs, graphical models), optimisation, classical machine
  learning (linear models, trees and ensembles, kernels and SVMs, Gaussian processes, clustering, manifolds), neural
  networks, search, logic, planning and games, reinforcement learning, signal processing and time series,
  explainability, fairness and trust. One tensor type, one randomness model, one trace model across all of them.
- **Exposition as a contract.** A method does not only return a fitted model: it returns a typed exposition record
  (paths, surfaces through training, posteriors, searches, episodes, attributions, group reports, ...), and every
  record has a view. Iterative methods are `Algorithm`s whose decisions are recorded at steps and played back, so the
  reader watches t-SNE optimise, a tree grow, a policy learn.
- **Correct to the reference.** Every core function is checked against the established Python implementation
  (numpy, scipy, scikit-learn, PyTorch, statsmodels, pgmpy, fairlearn, shap, ...) by golden fixtures at tiered
  tolerances; every primitive's gradients by finite differences; every documented example by execution.
- **No install, no server.** Runs entirely in the browser with a single runtime dependency; every explanation is a
  URL with its state in it, and an embeddable component for a blog post, a lecture or a slide deck.

## 3. The paper (four pages)

1. **Introduction and need** (half a page). Teaching AI spans paradigms that live in separate tools (scikit-learn,
   PyTorch, PyMC, OpenSpiel, shap, fairlearn), each a black box whose internals a learner cannot see; interactive
   explanations (TensorFlow Playground, Distill articles, explorables) are bespoke one-offs that do not compose or
   extend. The library unifies both: the implementations and their explanations.
2. **Design** (one page, one figure). The four layers (core, methods, views, hub) and their rules; the exposition
   contract; `Algorithm`, `runAlgorithm` and trace policies; explicit keys for reproducibility; opt-in
   differentiation; the self-documenting code and the generated documentation. Figure: the layers, with one method
   traced from core to record to view.
3. **Scope** (half a page, one table). The domains grouped by paradigm, with representative methods.
4. **Correctness and quality** (half a page). The oracle (fixtures against reference libraries), property-based tests,
   gradient checks, executed examples, coverage, continuous integration.
5. **Teaching use** (half a page, one figure). The site's three sections; two or three worked cases (an embedding
   learning live; a threshold and its costs; a fairness frontier); embedding in posts and decks. Figure: a strip of
   views exported by the library itself.
6. **Comparison** (half a page, one table). Features against scikit-learn, TensorFlow.js, ml5.js, TensorFlow
   Playground and the explainability and fairness toolkits: browser-native, paradigms covered, process recorded, views
   included, parity checked. Run time and memory for representative methods against the reference (honestly: built for
   exposition, not throughput; within a stated factor of the reference at teaching sizes).
7. **Availability** (a few lines). Licence, repository, documentation site, version.

A draft abstract: _We present [name], an open-source TypeScript library that implements the breadth of artificial
intelligence, from numerical methods and differentiable programming through probabilistic inference, classical
machine learning and neural networks to search, planning and reinforcement learning, so that each method returns the
data needed to explain it. Iterative methods record their decisions as they run, and a companion rendering library
draws and plays every such record in the browser, without a server. Each function is checked against the reference
Python implementation. An accompanying site and embeddable components make the library a resource for teaching and
for interactive writing._

## 4. What we have (2026-10-25)

|               |                                                                                                          |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| core          | 41 domains, about 1,040 source files                                                                     |
| methods       | 50 areas (about 190 files), each method returning exposition records                                     |
| views         | a view for every record kind (over 30), headless and React                                               |
| hub           | 18 collections, about 210 entries, each with live controls                                               |
| site          | gallery (about 110 recipes), hub, generated docs with executable examples                                |
| source        | about 450,000 lines of TypeScript; one runtime dependency (zod)                                          |
| tests         | 1,075 test files (90,000 lines), 117 property-based, 395 golden tests over 547 fixtures                  |
| references    | 115 Python libraries in the oracle (numpy, scipy, scikit-learn, PyTorch, statsmodels, pgmpy, sympy, ...) |
| examples      | about 6,700 documented examples, each executed                                                           |
| documentation | every folder and file summarised in the code; the plan in `docs/structure.md`                            |

## 5. What is missing

For either venue:
- the name (the rename, before anything is public);
- a licence: Apache-2.0 or MIT for code (with an SPDX line in each source file, which MLOSS asks for), CC-BY for the
  documentation and site text; a LICENSE file;
- a public repository with issues and discussions, a CONTRIBUTING guide and a code of conduct;
- releases: versioned packages on npm, a changelog, an archived version (Zenodo DOI);
- the site hosted (static build, e.g. GitHub Pages);
- installation instructions and a tutorial path (first steps, then one tutorial per paradigm).

For MLOSS in particular:
- continuous integration (`make check` and `make fixtures-check` on Linux, macOS and Windows, two Node versions);
- coverage measured and reported, and raised where it is low;
- benchmarks: run time and memory against the references for a representative set (a linear model, a forest, a GP,
  t-SNE, a sampler, an RL agent, autodiff on an MLP), in Node and in the browser;
- evidence of an active user community: the gating item, which only time and use provide (a course or workshop using
  it, posts built from it, outside contributors, issues from users);
- a statement of how the library was built. It was written largely with AI coding agents under human direction;
  check JMLR's and JOSE's current policies and disclose plainly. The oracle (parity with the reference libraries) is
  the answer to how anyone can trust it.

## 6. Order of work

1. **Release readiness** (days): the rename; licence and SPDX headers; README, installation, CONTRIBUTING; CI;
   coverage; npm packages; the hosted site. Freeze a version.
2. **Teaching material** (weeks): a tutorial path through the paradigms, built from hub entries; embeddable
   components documented for posts and decks; two or three blog posts or a lecture built with it.
3. **JOSE submission**: the statement of need, the software and the tutorial path. It establishes the teaching claim
   and gives a citable DOI early.
4. **Benchmarks and comparison** (days): the run-time, memory and feature tables, generated by a script.
5. **Community** (months): use in a course or workshop, posts, issues, contributors; record them for the cover letter.
6. **MLOSS submission**: the four pages (figures exported by the library itself), the cover letter, the archived
   version.
