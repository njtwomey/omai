import {
  type AtomFigure,
  type AtomQuantity,
  type AtomReference,
  type Belief,
  beliefDensity,
  type MixtureView,
  type RefMarginal,
} from 'omai-engine/probprog/ep'
import type { RefJoint } from 'omai-engine/probprog/validation'
import { add, arange, div, mul, sum, tensor, toFlat } from 'omai-engine/tensor'
import { type Curve, cartesian, contours, type PointSet, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo } from 'react'
import { CurveChart } from '../CurveChart.tsx'

const N = 300

/** The window a marginal is drawn on: where the reference has mass, padded, and including the true value. */
function windowOf(ref: RefMarginal, truth: number | undefined): readonly [number, number] {
  const top = Math.max(...ref.p)
  const keep = ref.x.filter((_, i) => (ref.p[i] ?? 0) >= 1e-3 * top)
  let lo = Math.min(...keep)
  let hi = Math.max(...keep)
  if (truth !== undefined && Number.isFinite(truth)) {
    lo = Math.min(lo, truth)
    hi = Math.max(hi, truth)
  }
  const pad = 0.12 * (hi - lo || 1)
  return [lo - pad, hi + pad]
}

const linspace = (lo: number, hi: number, n: number): number[] => toFlat(add(div(mul(arange(n), hi - lo), n - 1), lo))

/**
 * One quantity's belief against the reference: the reference as a thick curve, EP's belief at the viewed sweep dashed over
 * it (its earlier sweeps faint), and the value the data were made with as a dotted vertical line.
 */
export function BeliefChart(props: {
  readonly quantity: AtomQuantity
  readonly reference: RefMarginal | undefined
  readonly belief: Belief | undefined
  readonly earlier?: readonly (Belief | undefined)[]
}): ReactNode {
  const { quantity: q, reference: ref, belief, earlier } = props
  const view = useMemo(() => {
    if (ref === undefined) return undefined
    const [lo0, hi0] = windowOf(ref, q.truth)
    const lo = q.support === 'real' ? lo0 : Math.max(lo0, 0)
    const hi = q.support === 'unit' ? Math.min(hi0, 1) : hi0
    const xs = linspace(lo, hi, N)
    const draw = (b: Belief | undefined): number[] | undefined =>
      b === undefined || !b.proper ? undefined : xs.map(beliefDensity(q.support, b.mean, b.sd))
    const ep = draw(belief)
    const ghosts = (earlier ?? []).flatMap((b) => {
      const y = draw(b)
      return y === undefined ? [] : [y]
    })
    const inside = ref.x.map((x, i) => [x, ref.p[i] ?? 0] as const).filter(([x]) => x >= lo && x <= hi)
    const top = 1.1 * Math.max(...ref.p, ...(ep ?? []))
    const curves: Curve[] = [
      ...ghosts.map((y) => ({ x: xs, y, colour: 'tab0', width: 1, opacity: 0.18 })),
      {
        name: 'exact posterior',
        colour: 'fg',
        width: 3.4,
        smooth: true,
        x: inside.map((p) => p[0]),
        y: inside.map((p) => p[1]),
      },
      ...(ep === undefined
        ? []
        : [{ name: 'EP belief', colour: 'tab0', width: 2.4, dash: 'dashed' as const, x: xs, y: ep }]),
      ...(q.truth === undefined
        ? []
        : [
            {
              name: 'true value',
              colour: 'tab3',
              width: 1.4,
              dash: 'dotted' as const,
              x: [q.truth, q.truth],
              y: [0, top],
            },
          ]),
    ]
    return { curves, lo, hi, top }
  }, [ref, belief, earlier, q])
  if (view === undefined) return <p className="text-sm text-muted">The reference is being computed...</p>
  return (
    <CurveChart
      curves={view.curves}
      x={{ label: q.label, limits: [view.lo, view.hi] }}
      y={{ label: 'density', limits: [0, view.top] }}
      aspect={2.2}
      narrowAspect={1.6}
    />
  )
}

/** The data of an atom and the answer drawn on them: the exact answer (solid) and EP's (dashed). */
export function DataFigure(props: {
  readonly figure: AtomFigure
  readonly exact: Readonly<Record<string, number>> | undefined
  readonly ep: Readonly<Record<string, number>> | undefined
}): ReactNode {
  const { figure, exact, ep } = props
  const { curves, dots } = useMemo(() => {
    const data: Curve[] = (figure.curves ?? []).map((c) => ({
      name: c.name,
      x: c.x,
      y: c.y,
      colour: c.colour ?? 'muted',
      width: c.colour === undefined ? 1.4 : 2.4,
      opacity: 0.85,
    }))
    const answers = (means: Readonly<Record<string, number>> | undefined): Curve[] =>
      means === undefined || figure.answer === undefined
        ? []
        : figure.answer(means).map((c) => ({ name: c.name, x: c.x, y: c.y }))
    const exactCurves = answers(exact).map((c) => ({
      ...c,
      name: `${c.name} (exact)`,
      colour: 'fg',
      width: 3.4,
    }))
    const epCurves = answers(ep).map((c) => ({
      ...c,
      name: `${c.name} (EP)`,
      colour: 'tab0',
      width: 2,
      dash: 'dashed' as const,
    }))
    const dots: PointSet[] = figure.points.map((p) => ({
      name: p.name,
      x: [...p.x],
      y: [...p.y],
      ...(p.groupFrom !== undefined && ep !== undefined
        ? { groups: [...p.groupFrom(ep)] }
        : p.group === undefined
          ? {}
          : { groups: [...p.group] }),
      ...(p.groupNames === undefined ? {} : { groupNames: p.groupNames as unknown as Record<string, string> }),
      size: 9,
    }))
    return { curves: [...data, ...exactCurves, ...epCurves], dots }
  }, [figure, exact, ep])
  return (
    <CurveChart
      curves={curves}
      dots={dots}
      x={{
        label: figure.xLabel,
        ...(figure.xLimits === undefined ? {} : { limits: [...figure.xLimits] as [number, number] }),
      }}
      y={{
        label: figure.yLabel,
        ...(figure.yLimits === undefined ? {} : { limits: [...figure.yLimits] as [number, number] }),
      }}
      aspect={1.7}
    />
  )
}

const LEVELS = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]

/** The joint posterior of two quantities as contours (solid), and EP's factorised belief (the product of its marginals) dashed. */
export function JointChart(props: {
  readonly joint: RefJoint
  readonly a: AtomQuantity
  readonly b: AtomQuantity
  readonly epA: Belief | undefined
  readonly epB: Belief | undefined
}): ReactNode {
  const { joint, a, b, epA, epB } = props
  const mode = useColourMode()
  const grid = useMemo(() => {
    // Crop to where the posterior has mass (the box of a grid can be many times wider), padded by two cells.
    const top0 = Math.max(...joint.p.flat())
    const live = (v: number): boolean => v > 1e-3 * top0
    const ix = joint.x.map((_, i) => i).filter((i) => (joint.p[i] ?? []).some(live))
    const iy = joint.y.map((_, j) => j).filter((j) => joint.p.some((row) => live(row[j] ?? 0)))
    const i0 = Math.max(0, Math.min(...ix) - 2)
    const i1 = Math.min(joint.x.length - 1, Math.max(...ix) + 2)
    const j0 = Math.max(0, Math.min(...iy) - 2)
    const j1 = Math.min(joint.y.length - 1, Math.max(...iy) + 2)
    const xs = joint.x.slice(i0, i1 + 1)
    const ys = joint.y.slice(j0, j1 + 1)
    const rows = ys.map((_, j) => xs.map((_, i) => joint.p[i0 + i]?.[j0 + j] ?? 0))
    const levels = LEVELS.map((f) => f * top0)
    const ep =
      epA?.proper && epB?.proper
        ? (() => {
            const fa = beliefDensity(a.support, epA.mean, epA.sd)
            const fb = beliefDensity(b.support, epB.mean, epB.sd)
            return ys.map((y) => xs.map((x) => fa(x) * fb(y)))
          })()
        : undefined
    return { rows, ep, levels, xs, ys }
  }, [joint, a, b, epA, epB])
  const build = useCallback(
    (figure: Figure) => {
      const marks: PointSet[] =
        a.truth !== undefined && b.truth !== undefined
          ? [{ name: 'true values', x: [a.truth], y: [b.truth], size: 10, colour: 'tab3' }]
          : []
      return cartesian(
        [
          contours(grid.rows, {
            x: grid.xs,
            y: grid.ys,
            levels: grid.levels,
            colour: 'fg',
            width: 2.4,
            labels: false,
            mode,
          }),
          ...(grid.ep === undefined
            ? []
            : [
                contours(grid.ep, {
                  x: grid.xs,
                  y: grid.ys,
                  levels: grid.levels,
                  colour: 'tab0',
                  width: 1.8,
                  dash: 'dashed',
                  labels: false,
                  mode,
                }),
              ]),
          ...(marks.length === 0 ? [] : [points(marks, { mode, colourOffset: 3 })]),
        ],
        {
          mode,
          figure,
          x: { label: a.label, limits: [grid.xs[0] ?? 0, grid.xs[grid.xs.length - 1] ?? 1] },
          y: { label: b.label, limits: [grid.ys[0] ?? 0, grid.ys[grid.ys.length - 1] ?? 1] },
        },
      )
    },
    [grid, a, b, mode],
  )
  return <ResponsiveChart aspect={1.3} narrowAspect={1.1} build={build} />
}

/** The reference of an atom's quantities by name, for the page. */
export const marginalOf = (reference: AtomReference | undefined, name: string): RefMarginal | undefined =>
  reference?.marginals[name]

/** The belief about each discrete quantity (the probability that a switch is on): the exact one and EP's as two bars. */
export function ProbabilityBars(props: {
  readonly quantities: readonly AtomQuantity[]
  readonly exact: Readonly<Record<string, RefMarginal>> | undefined
  readonly ep: Readonly<Record<string, Belief>> | undefined
}): ReactNode {
  const { quantities, exact, ep } = props
  const bar = (value: number | undefined, colour: string): ReactNode => (
    <div className="flex items-center gap-2">
      <div className="h-3 w-48 rounded-sm border border-line">
        <div
          className="h-full rounded-sm"
          style={{ width: `${Math.max(0, Math.min(1, value ?? 0)) * 100}%`, background: colour }}
        />
      </div>
      <span className="text-xs tabular-nums">
        {value === undefined || !Number.isFinite(value) ? '-' : value.toFixed(4)}
      </span>
    </div>
  )
  return (
    <div className="flex flex-col gap-2">
      {quantities.map((q) => (
        <div key={q.name} className="flex flex-wrap items-center gap-x-6 gap-y-1">
          <span className="w-40 text-xs font-semibold text-muted">{q.label}</span>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-xs text-muted">
              <span className="w-12">exact</span>
              {bar(exact?.[q.name]?.mean, 'var(--wb-fg)')}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted">
              <span className="w-12">EP</span>
              {bar(ep?.[q.name]?.mean, 'var(--wb-tab0, #4c78a8)')}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const BRANCH_COLOURS = ['tab1', 'tab2']

/**
 * A mixture drawn as the gate it is: each exact branch as its Gaussian scaled by its posterior weight (two colours), their sum
 * as the exact posterior, the branches EP's beliefs imply (dashed, same colours) and the single Gaussian EP's belief is.
 */
export function MixtureChart(props: {
  readonly view: MixtureView
  /** EP's belief of the mixed quantity, at the viewed sweep. */
  readonly belief: Belief | undefined
  /** All of EP's beliefs at the viewed sweep, from which the branches it implies are told. */
  readonly beliefs: Readonly<Record<string, Belief>> | undefined
}): ReactNode {
  const { view, belief, beliefs } = props
  const shape = useMemo(() => {
    const all = [...view.exact]
    const lo = Math.min(...all.map((b) => b.mean - 4.5 * b.sd))
    const hi = Math.max(...all.map((b) => b.mean + 4.5 * b.sd))
    const xs = linspace(lo, hi, 500)
    const at = (b: { weight: number; mean: number; sd: number }): number[] =>
      xs.map((x) => b.weight * beliefDensity('real', b.mean, b.sd)(x))
    return { xs, at, lo, hi }
  }, [view])
  const curves = useMemo(() => {
    const { xs, at } = shape
    const exact = view.exact.map((b) => at(b))
    const total = exact.length === 0 ? xs.map(() => 0) : toFlat(sum(tensor(exact), 0))
    const out: Curve[] = [
      ...view.exact.map((b, i) => ({
        name: `${b.label}: ${b.weight.toFixed(3)}`,
        x: xs,
        y: exact[i] ?? [],
        colour: BRANCH_COLOURS[i] ?? 'tab3',
        width: 2,
        area: 0.18,
      })),
      { name: 'exact: the sum', x: xs, y: total, colour: 'fg', width: 3.4 },
    ]
    const ep = view.ep
    return { out, total, ep, xs }
  }, [shape, view])
  const full = useMemo(() => {
    const { xs } = curves
    const out = [...curves.out]
    const branches = beliefs === undefined ? undefined : view.ep(beliefs)
    branches?.forEach((b, i) => {
      out.push({
        name: `EP: ${b.label}`,
        x: xs,
        y: xs.map((x) => b.weight * beliefDensity('real', b.mean, b.sd)(x)),
        colour: BRANCH_COLOURS[i] ?? 'tab3',
        width: 1.4,
        dash: 'dashed',
      })
    })
    if (belief?.proper)
      out.push({
        name: 'EP: one Gaussian',
        x: xs,
        y: xs.map(beliefDensity('real', belief.mean, belief.sd)),
        colour: 'tab0',
        width: 2.6,
        dash: 'dashed',
      })
    return out
  }, [curves, belief, beliefs, view])
  const top =
    Math.max(
      ...curves.total,
      ...(belief?.proper ? curves.xs.map(beliefDensity('real', belief.mean, belief.sd)) : [0]),
    ) * 1.1
  return (
    <CurveChart
      curves={full}
      x={{ label: view.label, limits: [shape.lo, shape.hi] }}
      y={{ label: 'density', limits: [0, top] }}
      aspect={2.2}
      narrowAspect={1.6}
    />
  )
}
