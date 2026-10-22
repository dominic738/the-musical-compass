# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary:** music nerds playing for fun: people who care about lyrics and want to see where songs they know actually land. Arrive casually (a link, a friend), plot a handful of favorites, poke at the results.
- **Secondary:** recruiters and employers viewing it as Dominic Bankovitch's personal project. They spend a minute or two and should leave convinced the builder has taste and real engineering depth.

## Product Purpose

The Musical Compass places songs on a two-axis map by what their **lyrics** mean, on axes the user chooses (e.g. Sad ↔ Happy, Calm ↔ Angry). It is modeled on the political compass: four poles, four quadrants, a dot that says "you are here".

Success looks like:
1. **Surprise placement:** "wait, *that* song is sad?" An upbeat-sounding track landing in a dark quadrant because of its words.
2. **A screenshot worth sharing:** a finished compass someone wants to post or send to a friend.

## Positioning

Most music maps sort by genre, audio features or popularity. This one reads only the lyrics, and the user defines the axes, so any four feelings become a compass. The pipeline is the differentiator: an LLM writes phrases for each pole word, those are embedded, a linear classifier finds the direction between opposite poles, and songs are projected onto that direction from their own lyric embeddings.

## Operating Context

- Single-page web app. Static frontend (`frontend/`: plain HTML/CSS/JS) hosted on Netlify at mymusicalcompass.com; FastAPI backend (`backend/`) on Google Cloud Run.
- Flow: choose four pole words → build the compass (can take 10–60 s for new words; cached words are fast) → add songs by title + artist → each lands on the map.
- Lyrics come from LRCLIB in the browser when an exact match exists; otherwise the user pastes them. The backend never fetches lyrics and stores only embeddings.

## Capabilities and Constraints

- Axis words: any word or short phrase, max 50 chars; opposite poles must differ.
- Songs: title + artist; any script (non-Latin titles and lyrics supported).
- Placement is relative to the midpoint between each axis's two pole words, squashed with a sigmoid into the [-1, 1] square.
- Plain HTML/CSS/JS, no build step, no framework, no external scripts; the Arimo font is self-hosted.
- Spotify playlist import exists in the backend but is disabled in the UI.
- Undecided: saving/sharing compasses by URL; per-song detail beyond position.

## Brand Commitments

- Name: **The Musical Compass** (keep).
- Mark: a 2×2 grid of four squares, echoing the political compass (keep the idea; colors are free to change).
- The political compass is the explicit reference for the core chart.

## Evidence on Hand

- `frontend/assets/logo.png` (old raster wordmark + 4-square mark), `frontend/assets/favicon-16x16.png`.
- No testimonials, user counts, press or example compasses exist. Do not invent any.

## Product Principles

1. **The map is the product.** Everything else is a way to get songs onto it.
2. **Surprise is the payoff.** Make where a song lands, and why that's interesting, impossible to miss.
3. **Every compass should be screenshot-worthy** on its own, without the surrounding UI.
4. **Show the machinery honestly.** It's a nerd toy and a portfolio piece: the method is part of the charm, never hidden or overstated.
5. **Waiting is part of the experience.** Building a compass takes time; that time should feel intentional.

## Accessibility & Inclusion

- Works at phone width with no horizontal scroll.
- Keyboard operable (Enter submits, Escape closes dialogs); respects `prefers-reduced-motion`.
- Quadrant identity must not rely on color alone.
