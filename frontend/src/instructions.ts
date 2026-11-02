// DRAFT from the Oct 30 research, for Sid to rewrite in his own words (his call: the landlord films "in a specific way").
// Why each step is here: the floor tells SightSeen which way is up; a whole door is the ruler phase 3 needs (UniMetric
// uses an object's full height); and blurry frames are the ones the research says get thrown out.
export const filmingSteps = [
  'Walk slowly around the whole room, so every wall is in the video.',
  'Keep some of the floor in the shot the whole time.',
  'Show each door from the floor all the way to its top.',
]

// The in-page recorder's coach (defaults from the Nov 1 research, his OK "Use the defaults"): one pass of about two
// minutes, phone sideways (frames overlap more when turning), every wall from the doorway and from the middle, the
// ceiling and the floor, slow turns. `at` is seconds into the recording when each line shows.
export const beforeFilming = [
  'Turn the lights on. Turn off fans and TVs.',
  'Make sure nobody else is in the room.',
  'Hold the phone sideways, and move slowly the whole time.',
]

export const coachSteps = [
  { at: 0, say: 'Stand in the doorway. Slowly film the wall in front of you.' },
  { at: 15, say: 'Now turn slowly to the wall on your right.' },
  { at: 30, say: 'Now turn slowly to the wall on your left.' },
  { at: 45, say: 'Walk slowly to the middle of the room.' },
  { at: 55, say: 'Turn slowly all the way round, one wall at a time.' },
  { at: 90, say: 'Tilt up: one slow sweep across the ceiling.' },
  { at: 100, say: 'Tilt down: one slow sweep across the floor.' },
  { at: 110, say: "That's the whole room. Tap Stop." },
]
