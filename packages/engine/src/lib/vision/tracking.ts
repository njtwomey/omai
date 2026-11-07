/**
 * Tracking objects and points through frames: SORT (Bewley et al. 2016), a constant-velocity Kalman filter per box
 * (over `dynamics/filtering`) with detections assigned by IoU through the Hungarian algorithm of `optim`, as a traced
 * algorithm over frames; OpenCV's mean-shift and CamShift trackers on a back-projection; and pyramidal Lucas-Kanade
 * point tracking (Bouguet's KLT) over `signal/image` pyramids, frame to frame and as a traced algorithm.
 *
 * @index function SORT tracking, mean shift and pyramidal point tracking over frames.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { filtering } from '../dynamics/index.ts'
import { boxIou } from '../metrics/index.ts'
import { linearSumAssignment } from '../optim/index.ts'
import { image } from '../signal/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { greyOf, roundHalfEven } from './_image.ts'
import { type Box, type BoxesLike, boxesOf } from './boxes.ts'
import type { RotatedRect } from './fitting.ts'
import { imageMoments } from './moments.ts'

// SORT's constant-velocity model of [cx, cy, area, aspect, vcx, vcy, varea] observed as [cx, cy, area, aspect].
const F7 = Array.from({ length: 7 }, (_, i) =>
  Array.from({ length: 7 }, (_, j) => (i === j || (i < 3 && j === i + 4) ? 1 : 0)),
)
const H7 = Array.from({ length: 4 }, (_, i) => Array.from({ length: 7 }, (_, j) => (i === j ? 1 : 0)))
const diag = (d: readonly number[]): number[][] => d.map((v, i) => d.map((_, j) => (i === j ? v : 0)))
const R4 = diag([1, 1, 10, 10])
const Q7 = diag([1, 1, 1, 1, 0.01, 0.01, 0.0001])
const P07 = diag([10, 10, 10, 10, 10000, 10000, 10000])

const toZ = ([x1, y1, x2, y2]: Box): number[] => {
  const w = x2 - x1
  const h = y2 - y1
  return [x1 + w / 2, y1 + h / 2, w * h, w / h]
}

const toBox = (x: readonly number[]): Box => {
  const w = Math.sqrt(Math.max(0, (x[2] ?? 0) * (x[3] ?? 0)))
  const h = w > 0 ? (x[2] ?? 0) / w : 0
  return [(x[0] ?? 0) - w / 2, (x[1] ?? 0) - h / 2, (x[0] ?? 0) + w / 2, (x[1] ?? 0) + h / 2]
}

/** One SORT track: its Kalman state and bookkeeping. */
export interface SortTrack {
  readonly id: number
  readonly x: number[]
  readonly P: number[][]
  readonly hits: number
  readonly hitStreak: number
  readonly age: number
  readonly timeSinceUpdate: number
}

/** The state of {@link sortAlgorithm}: the live tracks, the frame count and the boxes output for the last frame. */
export interface SortState {
  readonly frame: number
  readonly nextId: number
  readonly tracks: readonly SortTrack[]
  /** The confirmed tracks of the last frame: [x1, y1, x2, y2] and the track id (from 1). */
  readonly output: readonly { readonly box: Box; readonly id: number }[]
}

/** Options of {@link sortAlgorithm}. */
export interface SortOptions {
  /** The smallest IoU of a match (default 0.3). */
  readonly iouThreshold?: number
  /** Frames a track survives without a detection (default 1). */
  readonly maxAge?: number
  /** Consecutive hits before a track is reported (default 3). */
  readonly minHits?: number
}

const rowsOfValue = (v: Value, r: number, c: number): number[][] => {
  const f = toFlat(primalOf(v) as Tensor)
  return Array.from({ length: r }, (_, i) => f.slice(i * c, (i + 1) * c))
}

/**
 * SORT, Simple Online and Realtime Tracking (Bewley et al. 2016), as an algorithm whose steps are frames: every track's
 * Kalman filter predicts, detections are assigned to predictions by the Hungarian algorithm on the IoU (pairs below
 * the threshold rejected), matched tracks are updated, unmatched detections start tracks, and tracks unseen for more
 * than maxAge frames end; tracks with minHits consecutive hits (or in the first minHits frames) are reported, as the
 * reference implementation does.
 *
 * @param frames - Per frame, the detections `[n, 4]` as xyxy boxes.
 * @param options - IoU threshold, maximum age and minimum hits.
 * @returns The algorithm (one step per frame).
 * @concept multi-object-tracking
 * @example
 * ```ts
 * import { sortAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const frames = [0, 1, 2, 3].map((t) => [[10 + 2 * t, 10, 30 + 2 * t, 30]])
 * runAlgorithm(sortAlgorithm(frames), { maxSteps: 4 }).state.output.map((o) => o.id) // => [1]
 * ```
 */
export function sortAlgorithm(frames: readonly BoxesLike[], options: SortOptions = {}): Algorithm<SortState> {
  const dets = frames.map((f) => boxesOf(f, 'sortAlgorithm'))
  const thr = options.iouThreshold ?? 0.3
  const maxAge = options.maxAge ?? 1
  const minHits = options.minHits ?? 3
  return {
    name: 'sort',
    config: { iouThreshold: thr, maxAge, minHits, frames: dets.length } as Json,
    schema: channels({
      metrics: {
        tracks: { description: 'live tracks after the frame' },
        reported: { description: 'confirmed tracks output for the frame' },
      },
    }),
    init: () => ({ frame: 0, nextId: 0, tracks: [], output: [] }),
    step: (s) => {
      const frame = s.frame + 1
      const boxes = dets[s.frame] ?? []
      // Predict, keeping the area velocity from driving the area negative.
      const predicted = s.tracks.map((t) => {
        const x0 = [...t.x]
        if ((x0[6] ?? 0) + (x0[2] ?? 0) <= 0) x0[6] = 0
        const p = filtering.kalmanPredict(x0, t.P, F7, Q7)
        return {
          ...t,
          x: toFlat(primalOf(p.x) as Tensor),
          P: rowsOfValue(p.P, 7, 7),
          age: t.age + 1,
          hitStreak: t.timeSinceUpdate > 0 ? 0 : t.hitStreak,
          timeSinceUpdate: t.timeSinceUpdate + 1,
        }
      })
      const live = predicted.filter((t) => t.x.every(Number.isFinite))
      const trkBoxes = live.map((t) => toBox(t.x))
      const matched = new Map<number, number>()
      if (boxes.length > 0 && trkBoxes.length > 0) {
        const iou = boxIou(boxes, trkBoxes)
        const { rowInd, colInd } = linearSumAssignment(iou, { maximize: true })
        rowInd.forEach((d, k) => {
          const t = colInd[k] ?? -1
          if ((iou[d]?.[t] ?? 0) >= thr) matched.set(d, t)
        })
      }
      const updated = live.map((t, ti) => {
        const d = [...matched.entries()].find(([, tt]) => tt === ti)?.[0]
        if (d === undefined) return t
        const u = filtering.kalmanUpdate(t.x, t.P, toZ(boxes[d] as Box), H7, R4)
        return {
          ...t,
          x: toFlat(primalOf(u.x) as Tensor),
          P: rowsOfValue(u.P, 7, 7),
          timeSinceUpdate: 0,
          hits: t.hits + 1,
          hitStreak: t.hitStreak + 1,
        }
      })
      let nextId = s.nextId
      for (const [d, b] of boxes.entries()) {
        if (matched.has(d)) continue
        const z = toZ(b)
        updated.push({
          id: nextId++,
          x: [...z, 0, 0, 0],
          P: P07.map((r) => [...r]),
          hits: 0,
          hitStreak: 0,
          age: 0,
          timeSinceUpdate: 0,
        })
      }
      const output = updated
        .filter((t) => t.timeSinceUpdate < 1 && (t.hitStreak >= minHits || frame <= minHits))
        .map((t) => ({ box: toBox(t.x), id: t.id + 1 }))
      const tracks = updated.filter((t) => t.timeSinceUpdate <= maxAge)
      return { frame, nextId, tracks, output }
    },
    done: (s) => (s.frame >= dets.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { tracks: s.tracks.length, reported: s.output.length } }),
  }
}

/**
 * Track boxes through frames with SORT ({@link sortAlgorithm}, run over every frame).
 *
 * @param frames - Per frame, the detections `[n, 4]` as xyxy.
 * @param options - SORT's options and tracing.
 * @returns Per frame, the reported boxes with their track ids.
 * @concept multi-object-tracking
 * @example
 * ```ts
 * import { sortTrack } from 'mdml-engine/vision'
 * const frames = [0, 1, 2].map((t) => [[10 + t, 10, 30 + t, 30], [60, 60 + t, 80, 80 + t]])
 * sortTrack(frames).map((f) => f.map((o) => o.id)) // => [[1, 2], [1, 2], [1, 2]]
 * ```
 */
export function sortTrack(
  frames: readonly BoxesLike[],
  options: SortOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): { readonly box: Box; readonly id: number }[][] {
  const algorithm = sortAlgorithm(frames, options)
  const out: { readonly box: Box; readonly id: number }[][] = []
  // Step frame by frame to collect each frame's output (the tape records the run when a tracer is given).
  runAlgorithm(
    {
      ...algorithm,
      step: (s, ctx) => {
        const next = algorithm.step(s, ctx)
        out.push([...next.output])
        return next
      },
    },
    {
      maxSteps: frames.length,
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
      ...(options.parent === undefined ? {} : { parent: options.parent }),
    },
  )
  return out
}

/** A window [x, y, width, height]. */
export type Rect = [x: number, y: number, width: number, height: number]

const intersect = (a: Rect, w: number, h: number): Rect => {
  const x1 = Math.max(a[0], 0)
  const y1 = Math.max(a[1], 0)
  const x2 = Math.min(a[0] + a[2], w)
  const y2 = Math.min(a[1] + a[3], h)
  return x2 > x1 && y2 > y1 ? [x1, y1, x2 - x1, y2 - y1] : [0, 0, 0, 0]
}

/** The moments of a window of a plane. */
function windowMoments(p: { h: number; w: number; data: Float64Array }, r: Rect) {
  const rows = Array.from({ length: r[3] }, (_, i) =>
    Array.from({ length: r[2] }, (_, j) => p.data[(r[1] + i) * p.w + r[0] + j] ?? 0),
  )
  return imageMoments(rows)
}

/** The paths the window trackers log, as a type. */
export type TrackPath = 'track/window' | 'track/moment' | 'track/shift' | 'track/box'

/**
 * What the mean-shift trackers log: the window after each update, the mass under it (the zeroth moment), the squared
 * distance it moved, and, for CamShift's last step, the rotated box.
 */
export const trackContract: Contract<unknown, TrackPath> = defineContract({
  'track/window': tensor({ role: 'state', shape: [4], description: 'the window [x, y, width, height]' }),
  'track/moment': scalar({ description: 'the probability mass under the window (the zeroth moment)' }),
  'track/shift': scalar({ description: 'the squared distance the window moved', direction: 'lower' }),
  'track/box': tensor({
    role: 'output',
    shape: [5],
    description: 'the CamShift box: centre x, y, width, height, angle',
  }),
})

/** Options of the mean-shift trackers. */
export interface MeanShiftOptions {
  /** The most window updates (default 100). */
  readonly maxIter?: number
  /** Stop when the window moves less than this distance (default 1). */
  readonly eps?: number
}

/**
 * The state of {@link meanShiftAlgorithm} and {@link camShiftAlgorithm}: the window, the updates made, whether the window
 * stopped moving, and (CamShift) the box once found.
 */
export interface MeanShiftState {
  readonly window: Rect
  readonly iterations: number
  readonly converged: boolean
  /** CamShift only: `shift` while updating the window, `box` before the last step, `done` after. */
  readonly phase: 'shift' | 'box' | 'done'
  readonly box: RotatedRect | null
}

/**
 * Mean shift on a probability image (cv2.meanShift) as a stepped algorithm, one window update per step: the integer
 * window moves to the centroid of the probability mass under it until it moves less than eps or `maxIter` updates pass; the
 * window stays inside the image. It logs the window, the mass under it and the squared shift, and is done (`converged`)
 * when the window stops, `completed` after `maxIter` updates. The state serialises.
 *
 * @param prob - Probability (back-projection) image `[H, W]`.
 * @param window - The starting window [x, y, width, height].
 * @param options - maxIter (default 100) and eps (default 1).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a window with non-positive sizes.
 * @concept mean-shift
 * @example
 * ```ts
 * import { meanShiftAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const prob = Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => (i >= 12 && i < 16 && j >= 10 && j < 14 ? 1 : 0)))
 * runAlgorithm(meanShiftAlgorithm(prob, [6, 8, 6, 6]), { maxSteps: 50 }).state.window // => [8, 10, 6, 6]
 * ```
 */
export function meanShiftAlgorithm(
  prob: TensorLike,
  window: Rect,
  options: MeanShiftOptions = {},
): Algorithm<MeanShiftState, TrackPath> {
  const p = greyOf(prob, 'meanShift')
  if (window[2] <= 0 || window[3] <= 0) throw new MdmlError('value', 'meanShift: the window has non-positive sizes')
  const eps = roundHalfEven((options.eps ?? 1) ** 2)
  const niters = Math.max(options.maxIter ?? 100, 1)
  const ww = window[2]
  const wh = window[3]
  return {
    name: 'vision/mean-shift',
    config: { window, maxIter: niters, eps: options.eps ?? 1 },
    init: () => ({ window: [...window], iterations: 0, converged: false, phase: 'shift', box: null }),
    step: (s, ctx) => {
      let cur = intersect(s.window, p.w, p.h)
      if (cur[2] === 0 && cur[3] === 0) cur = [Math.floor(p.w / 2), Math.floor(p.h / 2), 0, 0]
      cur = [cur[0], cur[1], Math.max(cur[2], 1), Math.max(cur[3], 1)]
      const m = windowMoments(p, cur)
      ctx.log('track/moment', m.m00)
      if (Math.abs(m.m00) < Number.EPSILON) return { ...s, window: cur, converged: true }
      let dx = roundHalfEven(m.m10 / m.m00 - ww * 0.5)
      let dy = roundHalfEven(m.m01 / m.m00 - wh * 0.5)
      const nx = Math.min(Math.max(cur[0] + dx, 0), p.w - cur[2])
      const ny = Math.min(Math.max(cur[1] + dy, 0), p.h - cur[3])
      dx = nx - cur[0]
      dy = ny - cur[1]
      cur = [nx, ny, cur[2], cur[3]]
      ctx.log('track/shift', dx * dx + dy * dy)
      ctx.log('track/window', cur)
      return dx * dx + dy * dy < eps
        ? { ...s, window: cur, converged: true }
        : { ...s, window: cur, iterations: s.iterations + 1 }
    },
    done: (s) => (s.converged ? 'converged' : s.iterations >= niters ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MeanShiftState,
  }
}

/**
 * Mean shift on a probability image (cv2.meanShift): the integer window moves to the centroid of the probability mass
 * under it until it moves less than eps or maxIter iterations pass; the window stays inside the image.
 * {@link meanShiftAlgorithm} run to the end.
 *
 * @param prob - Probability (back-projection) image `[H, W]`.
 * @param window - The starting window [x, y, width, height].
 * @param options - maxIter (default 100) and eps (default 1).
 * @returns The final window and the iterations taken.
 * @concept mean-shift
 * @example
 * ```ts
 * import { meanShift } from 'mdml-engine/vision'
 * const prob = Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => (i >= 12 && i < 16 && j >= 10 && j < 14 ? 1 : 0)))
 * meanShift(prob, [6, 8, 6, 6]).window // => [8, 10, 6, 6]
 * ```
 */
export function meanShift(
  prob: TensorLike,
  window: Rect,
  options: MeanShiftOptions = {},
): { readonly window: Rect; readonly iterations: number } {
  const run = runAlgorithm(meanShiftAlgorithm(prob, window, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return { window: run.state.window, iterations: run.state.iterations }
}

/**
 * Continuously adaptive mean shift (Bradski 1998; cv2.CamShift) as a stepped algorithm: the window updates of mean shift,
 * one per step, then one last step that grows the window by 10 pixels each way, takes the second moments of the
 * probability under it for the object's oriented size, and sets the next window to the box around that ellipse. It logs the
 * window, mass and shift of each update and the box at the end, and is done (`converged`) after the box. The state
 * serialises.
 *
 * @param prob - Probability image `[H, W]`.
 * @param window - The starting window.
 * @param options - Mean shift's maxIter and eps.
 * @returns The algorithm.
 * @concept mean-shift
 * @example
 * ```ts
 * import { camShiftAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const prob = Array.from({ length: 30 }, (_, i) => Array.from({ length: 30 }, (_, j) => (Math.abs(i - 15) < 3 && Math.abs(j - 12) < 7 ? 1 : 0)))
 * runAlgorithm(camShiftAlgorithm(prob, [8, 10, 8, 8]), { maxSteps: 50 }).state.box?.size.map((v) => Math.round(v)) // => [6, 15]
 * ```
 */
export function camShiftAlgorithm(
  prob: TensorLike,
  window: Rect,
  options: MeanShiftOptions = {},
): Algorithm<MeanShiftState, TrackPath> {
  const p = greyOf(prob, 'camShift')
  const inner = meanShiftAlgorithm(prob, window, options)
  const niters = Math.max(options.maxIter ?? 100, 1)
  return {
    ...inner,
    name: 'vision/cam-shift',
    step: (s, ctx) => {
      if (s.phase === 'shift') {
        const next = inner.step(s, ctx)
        return next.converged || next.iterations >= niters ? { ...next, phase: 'box' } : next
      }
      const TOL = 10
      let [x, y, w, h] = s.window
      x = Math.max(0, x - TOL)
      y = Math.max(0, y - TOL)
      w += 2 * TOL
      if (x + w > p.w) w = p.w - x
      h += 2 * TOL
      if (y + h > p.h) h = p.h - y
      const m = windowMoments(p, [x, y, w, h])
      if (Math.abs(m.m00) < Number.EPSILON)
        return { ...s, window: [x, y, w, h], phase: 'done', box: { center: [0, 0], size: [0, 0], angle: 0 } }
      const inv = 1 / m.m00
      const xc = roundHalfEven(m.m10 * inv + x)
      const yc = roundHalfEven(m.m01 * inv + y)
      const a = m.mu20 * inv
      const b = m.mu11 * inv
      const c = m.mu02 * inv
      const square = Math.sqrt(4 * b * b + (a - c) * (a - c))
      let theta = Math.atan2(2 * b, a - c + square)
      let cs = Math.cos(theta)
      let sn = Math.sin(theta)
      const ra = Math.max(0, cs * cs * m.mu20 + 2 * cs * sn * m.mu11 + sn * sn * m.mu02)
      const rc = Math.max(0, sn * sn * m.mu20 - 2 * cs * sn * m.mu11 + cs * cs * m.mu02)
      let length = Math.sqrt(ra * inv) * 4
      let width = Math.sqrt(rc * inv) * 4
      if (length < width) {
        ;[length, width] = [width, length]
        ;[cs, sn] = [sn, cs]
        theta = Math.PI / 2 - theta
      }
      let t0 = Math.max(roundHalfEven(Math.abs(length * cs)), roundHalfEven(Math.abs(width * sn))) + 2
      w = Math.min(t0, (p.w - xc) * 2)
      t0 = Math.max(roundHalfEven(Math.abs(length * sn)), roundHalfEven(Math.abs(width * cs))) + 2
      h = Math.min(t0, (p.h - yc) * 2)
      x = Math.max(0, xc - Math.trunc(w / 2))
      y = Math.max(0, yc - Math.trunc(h / 2))
      w = Math.min(p.w - x, w)
      h = Math.min(p.h - y, h)
      let angle = Math.fround(((Math.PI * 0.5 + theta) * 180) / Math.PI)
      while (angle < 0) angle += 360
      while (angle >= 360) angle -= 360
      if (angle >= 180) angle -= 180
      const box: RotatedRect = {
        center: [Math.fround(x + w * 0.5), Math.fround(y + h * 0.5)],
        size: [Math.fround(width), Math.fround(length)],
        angle,
      }
      ctx.log('track/box', [box.center[0], box.center[1], box.size[0], box.size[1], box.angle])
      ctx.log('track/window', [x, y, w, h])
      return { ...s, window: [x, y, w, h], phase: 'done', box }
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
  }
}

/**
 * Continuously adaptive mean shift (Bradski 1998; cv2.CamShift): mean shift, then the window grown by 10 pixels each
 * way, and the second moments of the probability under it give the object's oriented size; the next window is the
 * box around that ellipse. {@link camShiftAlgorithm} run to the end.
 *
 * @param prob - Probability image `[H, W]`.
 * @param window - The starting window.
 * @param options - Mean shift's maxIter and eps.
 * @returns The rotated box (centre, size [width, height], angle in degrees in [0, 180)) and the next window.
 * @concept mean-shift
 * @example
 * ```ts
 * import { camShift } from 'mdml-engine/vision'
 * const prob = Array.from({ length: 30 }, (_, i) => Array.from({ length: 30 }, (_, j) => (Math.abs(i - 15) < 3 && Math.abs(j - 12) < 7 ? 1 : 0)))
 * camShift(prob, [8, 10, 8, 8]).box.size.map((v) => Math.round(v)) // => [6, 15]
 * ```
 */
export function camShift(
  prob: TensorLike,
  window: Rect,
  options: MeanShiftOptions = {},
): { readonly box: RotatedRect; readonly window: Rect } {
  const run = runAlgorithm(camShiftAlgorithm(prob, window, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return { box: run.state.box as RotatedRect, window: run.state.window }
}

/** Options of {@link calcOpticalFlowPyrLK}. */
export interface PyrLKOptions {
  /** The window side (default 15). */
  readonly winSize?: number
  /** Pyramid levels above the image (default 2). */
  readonly maxLevel?: number
  /** Iterations per level (default 30) and the step size below which they stop (default 0.01 pixels). */
  readonly maxIter?: number
  readonly eps?: number
}

interface Level {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

const bilinear = (p: Level, y: number, x: number): number => {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const fx = x - x0
  const fy = y - y0
  const v = (yy: number, xx: number): number =>
    p.data[Math.min(p.h - 1, Math.max(0, yy)) * p.w + Math.min(p.w - 1, Math.max(0, xx))] ?? 0
  return (
    (1 - fy) * ((1 - fx) * v(y0, x0) + fx * v(y0, x0 + 1)) + fy * ((1 - fx) * v(y0 + 1, x0) + fx * v(y0 + 1, x0 + 1))
  )
}

function pyramid(x: TensorLike, levels: number): Level[] {
  const r = greyOf(x, 'calcOpticalFlowPyrLK')
  return image
    .pyramidGaussian(
      Array.from({ length: r.h }, (_, i) => Array.from(r.data.subarray(i * r.w, (i + 1) * r.w))),
      levels,
    )
    .map((l) => ({ h: l.shape[0] ?? 0, w: l.shape[1] ?? 0, data: Float64Array.from(toFlat(l)) }))
}

/**
 * Sparse optical flow by pyramidal Lucas-Kanade (Bouguet 2001; cv2.calcOpticalFlowPyrLK): each point is tracked from
 * the coarsest pyramid level down, at each level by Gauss-Newton steps that align its window in the next frame (bilinear
 * samples) with the window in the previous one; status 0 marks points lost (leaving the image, or a window without
 * texture).
 *
 * @param prev - Previous frame `[H, W]`.
 * @param next - Next frame `[H, W]`.
 * @param points - Points (x, y) in the previous frame.
 * @param options - Window, levels, iterations and tolerance.
 * @returns The tracked points and per point a status (1 tracked, 0 lost).
 * @concept optical-flow
 * @example
 * ```ts
 * import { calcOpticalFlowPyrLK } from 'mdml-engine/vision'
 * const f = (dx: number) => Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => Math.sin((j - dx) / 3) * Math.cos(i / 4)))
 * calcOpticalFlowPyrLK(f(0), f(1.5), [[16, 16]]).points[0].map((v) => Math.round(v * 10) / 10) // => [17.5, 16]
 * ```
 */
export function calcOpticalFlowPyrLK(
  prev: TensorLike,
  next: TensorLike,
  points: readonly (readonly [number, number])[],
  options: PyrLKOptions = {},
): { readonly points: [number, number][]; readonly status: number[] } {
  const levels = options.maxLevel ?? 2
  const A = pyramid(prev, levels)
  const B = pyramid(next, levels)
  const half = Math.floor((options.winSize ?? 15) / 2)
  const maxIter = options.maxIter ?? 30
  const eps = options.eps ?? 0.01
  const out: [number, number][] = []
  const status: number[] = []
  for (const [px, py] of points) {
    let gx = 0
    let gy = 0
    let ok = true
    for (let l = Math.min(A.length, B.length) - 1; l >= 0 && ok; l--) {
      const a = A[l] as Level
      const b = B[l] as Level
      const sx = (a.w - 1) / Math.max(1, (A[0] as Level).w - 1)
      const sy = (a.h - 1) / Math.max(1, (A[0] as Level).h - 1)
      const ux = px * sx
      const uy = py * sy
      // Spatial gradient matrix of the previous window.
      let gxx = 0
      let gxy = 0
      let gyy = 0
      const ix: number[] = []
      const iy: number[] = []
      const iv: number[] = []
      for (let dy = -half; dy <= half; dy++)
        for (let dx = -half; dx <= half; dx++) {
          const x = ux + dx
          const y = uy + dy
          const fx = (bilinear(a, y, x + 1) - bilinear(a, y, x - 1)) / 2
          const fy = (bilinear(a, y + 1, x) - bilinear(a, y - 1, x)) / 2
          ix.push(fx)
          iy.push(fy)
          iv.push(bilinear(a, y, x))
          gxx += fx * fx
          gxy += fx * fy
          gyy += fy * fy
        }
      const det = gxx * gyy - gxy * gxy
      if (!(det > 1e-12 * Math.max(1, gxx + gyy) ** 2)) {
        ok = false
        break
      }
      let vx = 0
      let vy = 0
      for (let it = 0; it < maxIter; it++) {
        let bx = 0
        let by = 0
        let k = 0
        for (let dy = -half; dy <= half; dy++)
          for (let dx = -half; dx <= half; dx++, k++) {
            const diff = (iv[k] ?? 0) - bilinear(b, uy + gy + vy + dy, ux + gx + vx + dx)
            bx += diff * (ix[k] ?? 0)
            by += diff * (iy[k] ?? 0)
          }
        const ex = (gyy * bx - gxy * by) / det
        const ey = (gxx * by - gxy * bx) / det
        vx += ex
        vy += ey
        if (Math.hypot(ex, ey) < eps) break
      }
      if (l > 0) {
        const n = A[l - 1] as Level
        const fx = (n.w - 1) / Math.max(1, a.w - 1)
        const fy = (n.h - 1) / Math.max(1, a.h - 1)
        gx = (gx + vx) * fx
        gy = (gy + vy) * fy
      } else {
        gx += vx
        gy += vy
      }
    }
    const x = px + gx
    const y = py + gy
    const inside = x >= 0 && y >= 0 && x <= (A[0] as Level).w - 1 && y <= (A[0] as Level).h - 1
    out.push([x, y])
    status.push(ok && inside ? 1 : 0)
  }
  return { points: out, status }
}

/** The state of {@link kltAlgorithm}: the frame reached, the tracked points and which are still alive. */
export interface KltState {
  readonly frame: number
  readonly points: readonly (readonly [number, number])[]
  readonly status: readonly number[]
}

/**
 * KLT point tracking through a frame sequence as an algorithm: each step tracks the live points from one frame to the
 * next by {@link calcOpticalFlowPyrLK}; lost points stay lost.
 *
 * @param frames - Frames `[H, W]`.
 * @param points - Starting points (x, y) in the first frame.
 * @param options - Pyramidal Lucas-Kanade options.
 * @returns The algorithm (one step per frame pair).
 * @concept optical-flow
 * @example
 * ```ts
 * import { kltAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const f = (dx: number) => Array.from({ length: 24 }, (_, i) => Array.from({ length: 24 }, (_, j) => Math.sin((j - dx) / 3) * Math.cos(i / 4)))
 * const s = runAlgorithm(kltAlgorithm([f(0), f(0.5), f(1)], [[12, 12]]), { maxSteps: 5 }).state
 * Math.round(s.points[0][0] * 10) / 10 // => 13
 * ```
 */
export function kltAlgorithm(
  frames: readonly TensorLike[],
  points: readonly (readonly [number, number])[],
  options: PyrLKOptions = {},
): Algorithm<KltState> {
  return {
    name: 'klt',
    config: { frames: frames.length, points: points.length } as Json,
    schema: channels({
      metrics: { alive: { description: 'points still tracked' } },
      tensors: { points: { role: 'state', description: 'tracked points (x, y)', shape: ['n', 2] } },
    }),
    init: () => ({ frame: 0, points: points.map((p) => [p[0], p[1]] as const), status: points.map(() => 1) }),
    step: (s) => {
      const r = calcOpticalFlowPyrLK(
        frames[s.frame] as TensorLike,
        frames[s.frame + 1] as TensorLike,
        s.points,
        options,
      )
      return {
        frame: s.frame + 1,
        points: r.points.map((p, i) => ((s.status[i] ?? 0) === 1 ? p : (s.points[i] as [number, number]))),
        status: r.status.map((v, i) => (v === 1 && (s.status[i] ?? 0) === 1 ? 1 : 0)),
      }
    },
    done: (s) => (s.frame >= frames.length - 1 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { alive: s.status.reduce((a, b) => a + b, 0) },
      tensors: { points: { role: 'state', value: s.points.flat(), shape: [s.points.length, 2] } },
    }),
  }
}
