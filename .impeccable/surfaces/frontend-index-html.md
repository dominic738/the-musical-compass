---
version: 1
slug: "frontend-index-html"
primary_target: "frontend/index.html"
related_targets: ["frontend/style.css","frontend/script.js"]
---

# Surface: The Musical Compass app (frontend/index.html)

Mode: operate. Single-screen tool: build a compass from four pole words, add songs, read where they land, share the result.

Audience/job: music nerds playing for fun; recruiters skimming a portfolio piece. The payoff is a surprising placement and a compass worth screenshotting.

Constraints: plain HTML/CSS/JS, no build step; light only; existing behaviour (axes, LRCLIB lookup, paste fallback, song list, remove, clear) preserved.

Memorable moment: a song lands and the readout prints its result like the political compass test: two signed scores on the −10…+10 scale and a quadrant flair.

## Direction contract

THESIS: The song map is the political compass chart, played straight and built precisely: pastel quadrants, a 20×20 grid, black axes, a dot with a score. It refuses the card-dashboard arrangement (rounded panels, neutral greys, one accent).

OWN-WORLD: White ground, grey utility sidebar. Quadrants #FF7575 / #42AAFF / #9AED97 / #C09AEC, 2px black frame and axes, faint grid every unit. One Arial-lineage sans everywhere, bold for labels, tabular figures. Square corners, 1px grey inputs, 2px black buttons. Flair chips: small squares in quadrant colours.

STORY: The visitor names four feelings, builds the compass, adds a song and reads its result ("Sad ↔ Happy +6.4, Calm ↔ Angry −2.1, flair CalmHappy"), adds more, then saves the image to share.

FIRST VIEWPORT: Left 360px sidebar: logo and name, 1. Axes (four labelled inputs, Generate), 2. Add a song, Songs list with flair and scores. Right: the square compass centred at about 760px, pole words in bold outside each edge, quadrant flair names in the corners, the result readout and Save image beneath it.

FORM: The Quadrant Meme, rank 1 on the ordered list (taken as Impeccable's pick over the roll). seed: none (the direction roll printed no SEED REFERENCE line, so none was recorded). Signature move: the result readout, a political-compass test result for each song, synced with the selected dot.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
