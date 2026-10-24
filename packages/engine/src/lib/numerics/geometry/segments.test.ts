import { describe, expect, it } from 'vitest'
import { closestPointOnSegment, raySegmentHit, segmentsTouch, sweptCircleHit } from './segments.ts'

describe('closestPointOnSegment', () => {
  it('projects inside the segment and clamps to the ends', () => {
    expect(closestPointOnSegment([1, 2], [0, 0], [4, 0])).toEqual({ point: [1, 0], distance: 2 })
    expect(closestPointOnSegment([-3, 4], [0, 0], [4, 0])).toEqual({ point: [0, 0], distance: 5 })
    expect(closestPointOnSegment([7, 4], [0, 0], [4, 0]).point).toEqual([4, 0])
  })
  it('treats a degenerate segment as its point', () => {
    expect(closestPointOnSegment([3, 4], [0, 0], [0, 0]).distance).toBe(5)
  })
})

describe('raySegmentHit', () => {
  it('returns the ray parameter and a normal facing the origin', () => {
    const hit = raySegmentHit([0, 0], [2, 0], [3, -1], [3, 1])
    expect(hit.t).toBe(1.5)
    expect(hit.normal[0]).toBe(-1)
    expect(Math.abs(hit.normal[1])).toBe(0)
  })
  it('misses parallel rays, rays pointing away and rays past an end', () => {
    expect(raySegmentHit([0, 0], [0, 1], [3, -1], [3, 1]).t).toBe(Infinity)
    expect(raySegmentHit([0, 0], [-1, 0], [3, -1], [3, 1]).t).toBe(Infinity)
    expect(raySegmentHit([0, 2], [1, 0], [3, -1], [3, 1]).t).toBe(Infinity)
  })
})

describe('sweptCircleHit', () => {
  it('stops the circle one radius from the side it hits', () => {
    const r = 0.25
    const hit = sweptCircleHit([0, 0.5], [3, 0], r, [2, -1], [2, 1])
    expect(hit.t).toBeCloseTo((2 - r) / 3, 15)
    expect(hit.normal).toEqual([-1, 0])
    const centre: [number, number] = [3 * hit.t, 0.5]
    expect(closestPointOnSegment(centre, [2, -1], [2, 1]).distance).toBeCloseTo(r, 12)
  })
  it('contacts an endpoint when passing beside the segment', () => {
    const hit = sweptCircleHit([0, 1.3], [4, 0], 0.5, [2, -1], [2, 1])
    expect(hit.t).toBeGreaterThan(0)
    expect(hit.t).toBeLessThan(1)
    const centre: [number, number] = [4 * hit.t, 1.3]
    expect(Math.hypot(centre[0] - 2, centre[1] - 1)).toBeCloseTo(0.5, 12)
    expect(Math.hypot(hit.normal[0], hit.normal[1])).toBeCloseTo(1, 12)
  })
  it('contacts at zero when already touching and moving closer, not when moving away', () => {
    expect(sweptCircleHit([1.8, 0], [1, 0], 0.5, [2, -1], [2, 1]).t).toBe(0)
    expect(sweptCircleHit([1.8, 0], [-1, 0], 0.5, [2, -1], [2, 1]).t).toBe(Infinity)
  })
})

describe('segmentsTouch', () => {
  it('detects crossings, touching ends and collinear overlap', () => {
    expect(segmentsTouch([0, 0], [2, 2], [0, 2], [2, 0])).toBe(true)
    expect(segmentsTouch([0, 0], [1, 0], [1, 0], [1, 1])).toBe(true)
    expect(segmentsTouch([0, 0], [2, 0], [1, 0], [3, 0])).toBe(true)
    expect(segmentsTouch([0, 0], [1, 0], [2, 0], [3, 0])).toBe(false)
    expect(segmentsTouch([0, 0], [1, 1], [0, 1], [-1, 2])).toBe(false)
  })
})
