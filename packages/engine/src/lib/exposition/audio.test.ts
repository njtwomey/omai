import { describe, expect, it } from 'vitest'
import { audioClip, audioDuration, playheadAt, stepAt } from './audio.ts'

const frames = [
  { step: 0, time: 0 },
  { step: 10, time: 1 },
  { step: 30, time: 2 },
]
const clip = audioClip({ sampleRate: 4, samples: new Array(12).fill(0), frames })

describe('the audio clip record and its playhead', () => {
  it('measures the clip', () => {
    expect(audioDuration(clip)).toBe(3)
  })

  it('moves the playhead linearly between frames, clamped at the ends', () => {
    expect(playheadAt(clip, 5)).toBe(0.5)
    expect(playheadAt(clip, 20)).toBe(1.5)
    expect(playheadAt(clip, -4)).toBe(0)
    expect(playheadAt(clip, 100)).toBe(2)
  })

  it('without frames the step is the time in seconds, kept inside the clip', () => {
    const bare = audioClip({ sampleRate: 4, samples: new Array(12).fill(0) })
    expect(playheadAt(bare, 1.5)).toBe(1.5)
    expect(playheadAt(bare, 9)).toBe(3)
    expect(stepAt(bare, 2.4)).toBe(2)
  })

  it('finds the step of a time, the inverse of the playhead on the frames', () => {
    for (const step of [0, 3, 10, 17, 30]) expect(stepAt(clip, playheadAt(clip, step))).toBe(step)
    expect(stepAt(clip, 0.5)).toBe(5)
    expect(stepAt(clip, 99)).toBe(30)
    expect(stepAt(clip, -1)).toBe(0)
  })

  it('checks the rate, the spectrogram and the order of the frames', () => {
    expect(() => audioClip({ sampleRate: 0, samples: [] })).toThrow(/sample rate/)
    expect(() =>
      audioClip({ sampleRate: 1, samples: [], spectrogram: { times: [0], frequencies: [0, 1], magnitude: [[1]] } }),
    ).toThrow(/1 rows/)
    expect(() =>
      audioClip({ sampleRate: 1, samples: [], spectrogram: { times: [0, 1], frequencies: [0], magnitude: [[1]] } }),
    ).toThrow(/row 0/)
    expect(() =>
      audioClip({
        sampleRate: 1,
        samples: [],
        frames: [
          { step: 2, time: 0 },
          { step: 1, time: 1 },
        ],
      }),
    ).toThrow(/out of order/)
  })
})
