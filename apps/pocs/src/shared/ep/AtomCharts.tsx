import {
  type AtomFigure,
  type AtomQuantity,
  type AtomReference,
  type Belief,
  beliefDensity,
  type RefJoint,
  type RefMarginal,
} from 'omai-engine/bayesian-examples'
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

const linspace = (lo: number, hi: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

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
        width: 6,
        opacity: 0.4,
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
      colour: 'muted',
      width: 1.4,
      opacity: 0.8,
    }))
    const answers = (means: Readonly<Record<string, number>> | undefined): Curve[] =>
      means === undefined || figure.answer === undefined
        ? []
        : figure.answer(means).map((c) => ({ name: c.name, x: c.x, y: c.y }))
    const exactCurves = answers(exact).map((c) => ({
      ...c,
      name: `${c.name} (exact)`,
      colour: 'fg',
      width: 6,
      opacity: 0.4,
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
      ...(p.group === undefined ? {} : { groups: [...p.group] }),
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
    const rows = joint.y.map((_, j) => joint.x.map((_, i) => joint.p[i]?.[j] ?? 0))
    const top = Math.max(...rows.flat())
    const levels = LEVELS.map((f) => f * top)
    const ep =
      epA?.proper && epB?.proper
        ? (() => {
            const fa = beliefDensity(a.support, epA.mean, epA.sd)
            const fb = beliefDensity(b.support, epB.mean, epB.sd)
            return joint.y.map((y) => joint.x.map((x) => fa(x) * fb(y)))
          })()
        : undefined
    return { rows, ep, levels }
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
            x: [...joint.x],
            y: [...joint.y],
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
                  x: [...joint.x],
                  y: [...joint.y],
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
        { mode, figure, x: { label: a.label }, y: { label: b.label } },
      )
    },
    [grid, joint, a, b, mode],
  )
  return <ResponsiveChart aspect={1.3} narrowAspect={1.1} build={build} />
}

/** The reference of an atom's quantities by name, for the page. */
export const marginalOf = (reference: AtomReference | undefined, name: string): RefMarginal | undefined =>
  reference?.marginals[name]
