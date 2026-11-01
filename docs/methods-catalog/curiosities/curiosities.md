# Curiosities (part 9)

Mechanisms worth watching for their own sake: epicycles tracing a drawing, Kepler's equation solved by Newton, orbits and Lagrange points integrated, light bending around a mass, Keplerian shear, the double pendulum, the Cole-Hopf transformation turning Burgers' equation linear, the Kelvin-Helmholtz instability on a lattice, and the rendering mechanisms (ray casting, ray tracing, path tracing, rasterisation) as pictures that build themselves, as sessions over `signal`, `dynamics`, `numerics`, `foundations`, `graphics` and `stochastic`. Each curiosity is a procedure with a knob or two and a striking picture; nothing is shared but the bases and the pictures.

## Structure

Everything here is a procedure watched, so it lives under `simulate/curiosities/` (section 3 of `methods/README.md`), one file per curiosity, grouped by what they have in common: a Fourier series, a central force, a fluid, a renderer, or nothing but a mechanism. A curiosity that is also a lesson elsewhere (the Gibbs overshoot, Newton's basins, maze carving, the Lorenz attractor) lives in its chapter and gets a `see` row here.

```
simulate/curiosities/
  fourier/           epicycles tracing a drawing, the harmonograph
  celestial/         Kepler's equation, orbits and n bodies, Lagrange points, light bending, Keplerian shear, the double pendulum
  fluids/            the Cole-Hopf transformation, the Kelvin-Helmholtz instability
  rendering/         ray casting, Whitted ray tracing, progressive path tracing, rasterisation stage by stage
  mechanisms/        self-avoiding walks, the spirograph, Collatz; the lessons that live elsewhere
  shared/            the drawing slot, the celestial catalogue, the pictures (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a curiosity is whatever makes its picture grow: one more term, one time step, one shot, one ray generation, one triangle. Status is about the method alone; what the library has is in the needs column.

### fourier/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `epicycles` | A drawing's outline (the slot: shapes, glyphs, any SVG path) as its complex Fourier series, a chain of rotating arrows whose tip traces it: the number of terms as the knob or the step (one more term per step, the curve sharpening from a blob to the drawing, the share of energy kept on the tape), the chain at time t with its joints, the order the arrows are chained in as a mode (by frequency, by size, interleaved), the terms kept by count or by energy. | `curve/terms` s, `curve/energyShare` s, `curve/t` s, `curve/joints` t!, `curve/trace` t!, `curve/picture` p (an `epicycles` record), `curve/spectrum` p | `signal/epicycles.ts`: `curveSpectrum`, `keptTerms`, `chainOrder`, `partialSum`, `epicycleJoints`; `graphics/outlines.ts`: `svgOutline`, `normaliseOutline`, `resampleClosed`; `datasets/outlines/shapes.ts`: `outlineShapes`, `shapeOutline`; `exposition/epicycles.ts`: `epicycles`; ready | |
| `harmonograph` | Two or three damped pendulums driving a pen: frequencies, phases and dampings as knobs, the path accumulating one sample per step, near-rational frequency ratios giving the closed figures and slightly off ratios the slowly turning ones; Lissajous as the undamped mode. | `pen/position` t, `pen/trace` t!, `pen/t` s, `pen/picture` p | the closed form in the method (no library piece); `scene/build.ts`: `path` for the trace | |

### celestial/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `keplers-equation` | M = E − e sin E solved for the eccentric anomaly by the mode (Newton, bisection, the secant, fixed-point iteration) at an eccentricity that is the knob: the iterate drawn on the orbit's auxiliary circle, the digits gained per step, fixed-point iteration failing as e → 1 while Newton keeps its quadratic rate; the planet placed at equal time steps along the orbit (the equal areas of Kepler's second law shaded). | `iteration/E` s, `iteration/residual` s, `iteration/error` s, `orbit/position` t, `orbit/sweptArea` s, `orbit/picture` p | `numerics/roots/scalar.ts`: `newton`, `bisect`, `secant`; `numerics/roots/systems.ts`: `fixedPoint`; TODO `dynamics/celestial.ts`: `keplerEquation`, `orbitFromElements` (the elements to a position) | |
| `orbits` | Two bodies, then n bodies under gravity integrated symplectically (Störmer-Verlet against RK4 as the mode: Verlet's bounded energy error, RK4's drift over many orbits) with the initial speed as the knob (circular, elliptic, parabolic, escape), the trails; the figure-eight three-body choreography and the Pythagorean problem from the slot; the inner solar system to scale with time compressed. | `bodies/positions` t!, `bodies/energy` s, `bodies/energyDrift` s, `bodies/angularMomentum` s, `bodies/t` s, `bodies/scene` p (with trails), `bodies/close` e (a near collision) | `dynamics/ode/symplectic.ts`: `symplecticAlgorithm`, `separableHamiltonian`, `energyDrift`; `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm`; TODO `dynamics/celestial.ts`: `nBodyHamiltonian`, `FIGURE_EIGHT`, `PYTHAGOREAN`, the planetary elements | |
| `lagrange-points` | The circular restricted three-body problem in the rotating frame: the effective potential's contours, the five Lagrange points as the field's equilibria (found and classified: the collinear ones saddles, L4 and L5 stable below Routh's mass ratio, the knob that crosses it), a test mass near L4 librating in a tadpole orbit, a horseshoe orbit, the Jacobi constant conserved on the tape and the forbidden region it bounds. | `mass/position` t!, `mass/jacobi` s, `mass/t` s, `frame/potential` p (the contours with the points marked), `frame/equilibria` p, `frame/forbidden` p | `dynamics/fields.ts`: `fixedPoints`, `classifyFixedPoint`; `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm`; TODO `dynamics/celestial.ts`: `restrictedThreeBodyField`, `effectivePotential`, `jacobiConstant` | |
| `light-bending` | Null geodesics in the Schwarzschild metric (the Binet form, integrated): the impact parameter as the knob or the step (one ray per step: a fan of rays bending more as they pass closer, the photon sphere capturing those inside the critical parameter, the Einstein ring), the deflection angle against the weak-field 4GM/c²b; as a lens, a background grid distorted; the geodesic on Flamm's paraboloid as the picture of the curved space. | `ray/impactParameter` s, `ray/path` t!, `ray/deflection` s, `ray/weakField` s, `ray/captured` e, `rays/picture` p, `lens/grid` p, `space/paraboloid` p (a 3-D scene) | `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm`; `dynamics/ode/problem.ts`: `event` (the capture); `foundations/surfaces.ts`: `geodesic`, `surfaceOfRevolution`; `scene/three-d.ts`: `scene3d`; TODO `dynamics/celestial.ts`: `schwarzschildNullGeodesicField`, `deflectionAngle` | |
| `keplerian-shear` | A ring of test particles on circular orbits with ω ∝ r^(−3/2): a radial line of them shearing into a trailing spiral one time step per step (the winding problem of spiral galaxies), a patch of neighbours dispersing (the shearing sheet), a density wave pattern that persists while the particles do not. | `ring/positions` t!, `ring/t` s, `ring/windingAngle` s, `ring/picture` p | the closed form in the method (no library piece); `dynamics/ode/symplectic.ts`: `symplecticAlgorithm` for a self-gravitating mode (TODO) | |
| `double-pendulum` | Two nearby starts diverging (their distance on a log axis: the Lyapunov time), the energy held by Verlet, the second bob's trace, and the flip map: the initial angles coloured by the time to the first flip, a fractal boundary between the orderly and the chaotic (one row of the grid per step). | `pendulum/state` t!, `pendulum/energy` s, `pair/distance` s, `pendulum/trace` t!, `flipMap/row` t!, `flipMap/picture` p | `dynamics/ode/symplectic.ts`: `symplecticAlgorithm`; `dynamics/nonlinear.ts`: `largestLyapunovExponent`; `environments/_mechanics.ts` has the cart version's derivatives (private); TODO `dynamics/named-systems.ts`: `doublePendulumField` with its Hamiltonian form | |

### fluids/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `cole-hopf` | Burgers' equation u_t + u u_x = ν u_xx steepening into a shock by finite volumes beside the heat equation it becomes under φ = exp(−∫u/2ν): the linear solve transformed back and matching exactly (the error on the tape), ν as the knob (the shock sharpening as ν → 0, the transform becoming numerically delicate: the exponent's range as the warning). | `step/t` s, `step/u` t!, `step/phi` t!, `step/transformedBack` t!, `step/error` s, `transform/range` s | `dynamics/pde/finite-volume.ts`: `conservationLaw`, `godunovFlux`, `BURGERS`; `dynamics/pde/finite-difference.ts`: `heat1d`; the transform and its inverse in the method (a few lines); ready | |
| `kelvin-helmholtz` | Two layers shearing past each other on a lattice (lattice Boltzmann, the Reynolds number as the knob, a small perturbation of the interface): the interface rolling up into the cat's-eye vortices one lattice step per step, the vorticity field as the picture, the growth rate of the perturbation against the linear theory's. | `lattice/t` s, `field/velocity` t, `field/vorticity` t!, `field/picture` p, `perturbation/amplitude` s, `perturbation/growthRate` s | `dynamics/pde/operators.ts`: `curlOf`; TODO `lib/dynamics/lattice.ts`: `latticeBoltzmann` (D2Q9, shared with `complex-systems` lattice-gas) | |
| `burgers-shock` | see mathematics (`simulate/numerics/pde/conservation-law`: the fluxes and limiters) | | | |

### rendering/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `ray-casting` | The first-person corridor of the early games: rays marched cell by cell by DDA through a world from `worlds` (the map as the slot), one screen column per step so the frame paints left to right, the wall slice's height from the corrected distance (fish-eye correction on or off as the knob), the top-down view cone with the cells each ray visits; the player moved by knobs, textures and sprites as a mode. | `cast/column` s, `cast/distance` s, `cast/cells` t, `frame/image` t!, `frame/topDown` p (the cone over the map), `frame/picture` p | `graphics/raycasting.ts`: `raycastAlgorithm`, `ddaCast`, `castColumns`, `fisheyeCorrect`, `renderRaycast`, `viewConeScene`; `worlds/continuous.ts`: `castRay`; `problems/maps.ts`: `searchMap`; ready | |
| `ray-tracing` | Whitted ray tracing one ray generation per step (the primary rays, then reflections, refractions and shadow rays spawned at each hit, the image gaining its mirrors and glass generation by generation), Snell's law and Schlick's Fresnel term as the mode's physics, the ray tree of one chosen pixel as a probe, the BVH's intersections counted per ray against brute force. | `generation/depth` s, `generation/rays` s, `generation/intersections` s, `frame/image` t!, `pixel/rayTree` p (a `tree` record), `frame/picture` p | `graphics/raytracing.ts`: `whittedAlgorithm`, `renderWhitted`, `traceRay`, `rayScene`, `refract`, `schlick`; `graphics/bvh.ts`: `buildBvh`, `intersectBvh`; `graphics/scenes.ts`: `cornellBox`; ready | |
| `path-tracing` | Progressive Monte Carlo rendering: the samples per pixel doubling per step and the noise falling away as 1/√n (the pixels' mean standard error on the tape, on a log axis), next-event estimation on or off (the knob: the variance it removes), the tone mapper as a mode (Reinhard, ACES, exposure), a single path's vertices bouncing through the Cornell box as a probe. | `render/samples` s, `render/standardError` s, `render/variance` t, `frame/image` t!, `path/vertices` p, `frame/picture` p | `graphics/pathtracing.ts`: `pathTracerAlgorithm`, `pathTrace`, `tracePath`; `graphics/rendering.ts`: `progressiveRender`; `graphics/tonemapping.ts`: `reinhard`, `acesFilm`, `exposure`; `graphics/lights.ts`: `areaLight`; ready | |
| `rasterisation` | A mesh through the pipeline one triangle per step: vertices to clip space, back faces culled, the triangle clipped against the view volume, pixels covered by edge functions under the top-left rule, attributes interpolated, the depth test keeping the nearest fragment (the depth buffer as an image beside the colour buffer); what happened to each triangle (culled, clipped, drawn) as the tape's messages, the camera's position as a knob. | `triangle/index` s, `triangle/outcome` m, `triangle/fragments` s, `frame/colour` t!, `frame/depth` t!, `frame/picture` p, `camera/frustum` p (a 3-D scene) | `graphics/rasterise.ts`: `rasteriseMesh`, `rasteriseTriangle`, `clipPolygon`, `isBackFacing`, `depthBuffer`, `depthTest`, `mesh`; `graphics/camera.ts`: `pinholeCamera`, `lookAt`, `perspective`, `cameraFrustum`; ready | |

### mechanisms/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `self-avoiding-walk` | A walk that may not cross itself, grown by Rosenbluth sampling one step per step (the attrition: dead ends as events, the weight correcting for them), its mean squared displacement's exponent (ν ≈ 3/4) against the simple walk's 1/2 and the Lévy flight's heavy steps as modes; the exact count of short walks by enumeration beside the estimate. | `walk/position` t, `walk/path` t!, `walk/length` s, `walk/weight` s, `walk/trapped` e, `study/displacement` proj | `stochastic/random-walks.ts`: `selfAvoidingWalkAlgorithm`, `countSelfAvoidingWalks`, `simpleRandomWalk`, `levyFlight`, `meanSquaredDisplacement`; ready | |
| `spirograph` | Hypotrochoids and epitrochoids: the gear ratio and the pen's offset as knobs, the curve drawn one tooth per step, closing after a number of turns set by the ratio's denominator (the petals counted), the curvature along it. | `pen/position` t, `pen/trace` t!, `pen/turns` s, `curve/closed` e, `pen/picture` p | the closed form in the method (no library piece); `foundations/curves.ts`: `curvature`, `arcLength` | |
| `collatz` | Hailstone sequences one step per step for a chosen n (the knob), the stopping time and the peak, the stopping times across n as a sweep (the spiky plot), the tree of predecessors grown outward from 1 as a `tree` record. | `sequence/value` s, `sequence/steps` s, `sequence/peak` s, `study/stoppingTimes` proj, `tree/predecessors` p | an integer recurrence in the method (no library piece); `exposition/trees.ts`: `treeRecord` | |
| `fourier-square-wave` | see mathematics (`simulate/numerics/number-theory/series-and-fourier-series`: the Gibbs overshoot that never shrinks) | | | |
| `newton-basins` | see mathematics (`simulate/numerics/roots/newton-basins`) | | | |
| `maze-carving` | see search-and-planning (`search/state-space/problems/maze-generation`: the carvers watched) | | | |
| `strange-attractors` | see complex-systems (`simulate/dynamics/fractals/attractors/`, with the escape-time sets and the fern beside them) | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `outlines.ts` | The drawing slot: the outline shapes and glyphs of `datasets/outlines/shapes.ts`, any SVG path through `graphics/outlines.ts`, normalised and resampled by arc length, with the number of samples as a knob | `fourier/epicycles` |
| `celestial.ts` | The celestial catalogue: masses, semi-major axes and eccentricities of the planets, the mass ratios of the Earth-Moon and Sun-Jupiter systems, the units and the time compression, and the named n-body configurations (until `dynamics/celestial.ts` holds them) | `celestial/*` |
| `pictures.ts` | From `methods/shared/pictures.ts`: images over steps (the renderers), point traces with trails (the pens, the bodies), 3-D scenes (the orbits, the paraboloid, the frustum), the viewport knob of `complex-systems` for the flip map and the fans of rays | every group |

## Examples to make

- Epicycles redrawing an outline from its Fourier coefficients, the number of terms as the knob (the gallery had this), the arrows chained by size.
- Kepler's equation by Newton for a chosen eccentricity, the iterate on the orbit, fixed-point iteration failing beside it as e grows.
- The restricted three-body problem with a test mass near L4, the Jacobi constant on the tape, the mass ratio pushed past Routh's.
- Light rays bending around a mass as the impact parameter sweeps, the photon sphere catching the close ones.
- The Cornell box rendered progressively, the noise falling as 1/√n, a single path's bounces as a probe.
- The double pendulum's flip map built row by row.

## Rendering

- Scenes and 3-D (`SceneView`, `Scene3DView`), trajectories with trails, images over steps (`imageSteps`), epicycles (`epicycles`), trees (`tree`).
- Missing: nothing specific to the chapter; the zoomable image panel and the accumulating point-cloud panel of `complex-systems` serve the flip map and the fans of rays.

## Problems

- The renderers, the epicycles, the self-avoiding walk and the symplectic integrators are in the library; the celestial pieces (Kepler's elements, the n-body Hamiltonian, the restricted three-body field, the Schwarzschild geodesic, the double pendulum) are one `dynamics/celestial.ts` and one named system away, and the lattice Boltzmann solver is shared with `complex-systems`.
- Where a curiosity is also a lesson (Gibbs, Newton's basins, maze carving, the attractors) it lives in its chapter and is linked here; the test is whether the page has a quantity to compare with the truth (a lesson) or only a picture (a curiosity).
- New chapter mirroring the encyclopedia's part 9; the `foundations/` studies that are curiosities rather than teaching move here. The rendering group is the only home for `graphics` as a subject (`computer-vision` uses its cameras), so it carries those four rows until a graphics chapter is cut.
