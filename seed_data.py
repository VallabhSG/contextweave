"""Seed ContextWeave with realistic sample data.

The demo persona is fictional: a founding engineer at a small studio
building "Wayfarer", an offline-first hiking app. Entries interconnect
(recurring people, projects, decisions) so the knowledge graph, digest,
and cross-document queries all have something real to show.
"""

import httpx
import time

BASE = "https://vallllllllll-contextweave.hf.space"

ENTRIES = [
    # ── MEETINGS ────────────────────────────────────────────────
    {
        "content": """Kickoff meeting for Wayfarer v2 offline mode — March 9, 2026.
Whole team in the room: Sam Okafor (backend), Lena Fischer (design), Ravi Menon (CTO).
The problem: hikers lose signal exactly when they need the map most.
Decisions:
- Offline-first is the headline feature for v2, not an option buried in settings
- Map tiles stored as MBTiles in SQLite on-device; sync over WiFi only by default
- Ship Sierra and Cascades regions first, expand after launch
Action items: Sam to prototype tile pre-fetch by March 20. Lena to sketch the download-region flow.
I'll write the storage budget doc — how many GB is acceptable per region — by Friday.""",
        "metadata": {"source": "meeting", "people": ["Sam Okafor", "Lena Fischer", "Ravi Menon"], "date": "2026-03-09"},
    },
    {
        "content": """Weekly sync with Sam Okafor — March 23, 2026.
Sam's tile pre-fetch prototype works but the numbers are rough: the full Sierra region is 4.1 GB at zoom 15.
Options we discussed:
1. Cap offline detail at zoom 13 (roughly 900 MB) and stream deeper zooms when online
2. Let users draw a corridor around their planned route and only fetch that
3. Both — corridor fetch with a zoom cap
Decision: option 3. The corridor idea came from Elena Marsh's beta feedback — she only ever hikes planned routes.
Sam raised a concern about Mapbox's offline tile licensing. Follow up with their sales team before we commit.""",
        "metadata": {"source": "meeting", "people": ["Sam Okafor", "Elena Marsh"], "date": "2026-03-23"},
    },
    {
        "content": """Design review with Lena Fischer — April 6, 2026.
Lena presented three directions for the region-download flow:
A) Checklist of named regions (simple, boring)
B) Pinch-and-drag a rectangle on the map (flexible, fiddly on small screens)
C) Tap your planned route, and Wayfarer computes the corridor automatically
We picked C — it matches how hikers actually think. Nobody plans "a rectangle".
Lena's insight: "The download screen is the moment users decide whether to trust the app with their safety. It should feel calm."
She'll have an interactive prototype by end of week. I promised to get her real corridor-size numbers from Sam's branch.""",
        "metadata": {"source": "meeting", "people": ["Lena Fischer", "Sam Okafor"], "date": "2026-04-06"},
    },
    {
        "content": """1:1 with Ravi Menon — April 14, 2026.
Ravi pushed back on the Mapbox dependency. Their offline licensing quote came back at $18k/year minimum — brutal for a three-person studio.
Counter-argument: Mapbox's rendering quality and terrain shading are genuinely better.
Ravi's position: OpenStreetMap data with our own MBTiles pipeline gets us 90% of the quality at zero licensing cost, and we own the whole stack.
We agreed to timebox a two-week OSM spike. If the terrain rendering looks acceptable on the Sierra test region, we drop Mapbox.
Also discussed the App Store launch window: aim for the June hiking season, hard deadline June 12.""",
        "metadata": {"source": "meeting", "people": ["Ravi Menon"], "date": "2026-04-14"},
    },

    # ── JOURNAL ENTRIES ─────────────────────────────────────────
    {
        "content": """Journal — February 21, 2026.
Went back through our support inbox from the winter. The same story keeps appearing with different names:
someone downloads a trail map at the trailhead parking lot, loses signal two miles in, and the map goes blank at the fork.
One user got genuinely lost for three hours above Emerald Basin.
We keep polishing features for people sitting on their couch planning trips. The product moment that matters happens where there is no network at all.
Wrote it on a sticky note above my desk: build for the person at the fork in the trail.""",
        "metadata": {"source": "journal", "date": "2026-02-21"},
    },
    {
        "content": """Journal — March 30, 2026.
The corridor-download idea is turning out to be the best decision of the quarter, and it wasn't ours — it came from Elena Marsh's feedback thread.
She wrote: "I don't need the whole mountain range. I need my route plus the escape routes."
That sentence reframed the entire storage problem. A 40-mile route corridor at high zoom is about 300 MB. The full region was 4 GB.
Lesson I keep relearning: beta testers don't give you solutions, they give you constraints you didn't know existed. The solution falls out of the constraint.""",
        "metadata": {"source": "journal", "date": "2026-03-30"},
    },
    {
        "content": """Journal — April 20, 2026.
Week one of the OSM spike done. Honest assessment: the raw OpenStreetMap render looked flat and lifeless next to Mapbox — until Sam added hillshading from the USGS elevation tiles.
With terrain shading it's suddenly 90% of the way there, exactly like Ravi predicted.
The remaining gap is typography on trail labels, which Lena thinks she can fix with a custom style layer.
Feeling good about dropping the $18k dependency. Owning the pipeline also means offline works exactly how we design it, not how a vendor's SDK allows it.""",
        "metadata": {"source": "journal", "date": "2026-04-20"},
    },
    {
        "content": """Journal — May 11, 2026.
Battery testing week. Took Friday off and hiked the Basecamp Ridge loop with four phones in my pack, each running a different GPS sampling strategy.
Results: continuous 1Hz GPS drained 31% over the four-hour loop. Adaptive sampling — 1Hz when moving fast or near forks, one fix per 30s on straight sections — drained 11%.
The fork-detection heuristic Sam wrote (distance to nearest trail junction from the OSM graph) is what makes adaptive sampling safe.
A hiker's phone dying at 2pm is a safety problem, not a UX problem. 11% feels shippable.""",
        "metadata": {"source": "journal", "date": "2026-05-11"},
    },

    # ── LEARNING NOTES ──────────────────────────────────────────
    {
        "content": """Learning notes: MBTiles and offline map storage — March 12, 2026.
MBTiles is just SQLite with a convention: a tiles table keyed by zoom/column/row, blobs of PNG or vector data.
Key facts:
- Vector tiles are 5-10x smaller than raster for the same area, and restyle for free
- Zoom 15 is where file size explodes: each zoom level roughly quadruples tile count
- Deduplication matters: ocean and empty forest tiles are identical blobs, store once with a tile_ref indirection
Rule of thumb from testing: vector corridor at zoom 15, 2km wide, is about 7 MB per trail mile.
That number decides our whole download UX.""",
        "metadata": {"source": "note", "date": "2026-03-12"},
    },
    {
        "content": """Learning notes: GPX parsing pitfalls — April 2, 2026.
GPX is XML, and every device exports it slightly differently.
Gotchas found while importing beta users' recorded tracks:
- Garmin devices write extensions namespaces that break naive parsers
- Some apps export trkpt without elevation; our elevation-gain math must fall back to the terrain model
- Timestamps can be local time without offset — treat all bare timestamps as suspect
- A single file can contain multiple trk segments after GPS dropouts; joining them naively creates teleport lines across valleys
Decision: parse defensively, validate against the trail graph, and show users a preview before import.""",
        "metadata": {"source": "note", "date": "2026-04-02"},
    },
    {
        "content": """Learning notes: OpenStreetMap trail data quality — April 25, 2026.
Spent three days auditing OSM coverage for our launch regions.
Findings:
- Sierra coverage is excellent: 96% of official trails present, mostly accurate
- Cascades are patchier: several decommissioned trails still mapped as active — a real safety issue
- The sac_scale and trail_visibility tags are gold when present: they encode difficulty and how easy the path is to follow
Plan: cross-check OSM against the ranger district's official GIS layers before marking any trail as verified in Wayfarer.
Dana Whitfield at the Cascades ranger district offered to review our trail list — take her up on that.""",
        "metadata": {"source": "note", "date": "2026-04-25"},
    },
    {
        "content": """Learning notes: on-device search for offline mode — May 4, 2026.
Requirement: search trails, peaks, and waypoints with zero network.
Evaluated: bundled SQLite FTS5 index vs a trie baked into the app binary.
FTS5 won easily — 40k named features for the Sierra region indexes to 9 MB, queries return in under 5ms on a mid-range phone, and prefix matching (feature* ) gives type-ahead for free.
Gotcha: FTS5 chokes on special characters in queries; sanitize before matching.
Bonus discovery: bm25 ranking puts exact trail-name matches above partial peak-name matches with zero tuning. Shipping it as-is.""",
        "metadata": {"source": "note", "date": "2026-05-04"},
    },

    # ── CONVERSATIONS ────────────────────────────────────────────
    {
        "content": """Conversation with Dana Whitfield (Cascades district ranger) — May 8, 2026.
Met Dana at the ranger station to talk trail data.
Dana: "Apps are why we do more rescues now, not fewer. People trust a blue dot more than a paper map they'd actually study."
Me: "What would make an app rescue-negative instead?"
Dana: "Show decommissioned trails as gone. Show water sources as seasonal. And stop routing people over Windy Gap in June — it holds snow until July."
We agreed: Dana's team reviews our Cascades trail list before launch, and Wayfarer will show her district's seasonal advisories inline on the map.
Action item: send Dana the trail list export by May 15. This partnership could become the launch story.""",
        "metadata": {"source": "conversation", "people": ["Dana Whitfield"], "date": "2026-05-08"},
    },
    {
        "content": """Beta call with Elena Marsh — March 18, 2026.
Elena has logged 41 hikes on the beta build — our most active tester.
Her top complaints:
1. Downloading a whole region for a single day-hike feels wasteful ("I don't need the whole mountain range. I need my route plus the escape routes.")
2. The elevation profile hides when recording starts — she uses it to pace herself
3. Battery anxiety: she carries a paper map because she doesn't trust the phone to last
Every one of these turned into a roadmap item. The corridor download, the pinned profile, and the adaptive GPS work all trace back to this call.
Need to follow up with Elena once the corridor build is in TestFlight — promised her first access.""",
        "metadata": {"source": "conversation", "people": ["Elena Marsh"], "date": "2026-03-18"},
    },
    {
        "content": """Debate with Sam Okafor over lunch — May 19, 2026.
Sam wants to add live location-sharing between hiking partners before launch. I pushed back.
Sam's case: it's the most-requested feature after offline maps, and the safety angle is real.
My case: it needs a realtime backend, presence handling, and privacy design — a month of work three weeks before the June 12 deadline, for a feature that fails exactly where our users are (no signal).
Middle ground we landed on: post-hike track sharing ships at launch (pure upload, no realtime), live sharing goes to v2.1 with a mesh/satellite investigation.
Wrote it down because we will absolutely relitigate this in July.""",
        "metadata": {"source": "conversation", "people": ["Sam Okafor"], "date": "2026-05-19"},
    },

    # ── DECISIONS & PRIORITIES ───────────────────────────────────
    {
        "content": """Wayfarer v2 launch checklist — locked May 25, 2026.
Scope for June 12 App Store submission:
- Corridor offline downloads (zoom-capped, route + escape routes)
- OSM + USGS hillshade map pipeline (Mapbox fully removed as of May 20)
- Adaptive GPS sampling (11% battery per 4-hour hike, verified on Basecamp Ridge)
- Offline FTS5 search across 40k features
- Seasonal advisories from the Cascades ranger district (Dana Whitfield's team)
- Post-hike track sharing (upload only)
Explicitly cut: live location sharing (v2.1), Android tablet layout, watch app.
Remaining before submission: Dana's trail review back by June 1, Lena's onboarding polish, App Store screenshots.
I'll own the submission itself — reminder set for June 10 to freeze the build.""",
        "metadata": {"source": "note", "date": "2026-05-25"},
    },
]


def ingest(entry: dict, idx: int) -> bool:
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{BASE}/api/ingest/text",
                json={"content": entry["content"], "metadata": entry.get("metadata", {})},
                timeout=60,
            )
            if r.status_code == 429:  # per-IP ingest limit — wait out the window
                print(f"  [{idx+1:02d}] rate limited, waiting 65s (attempt {attempt+1}/3)")
                time.sleep(65)
                continue
            if r.status_code == 200:
                data = r.json()
                print(f"  [{idx+1:02d}] ok    {data['chunks_created']} chunks, {data['entities_extracted']} entities")
                return True
            print(f"  [{idx+1:02d}] FAIL  HTTP {r.status_code}: {r.text[:80]}")
            return False
        except Exception as e:
            print(f"  [{idx+1:02d}] FAIL  {e}")
            return False
    return False


def main():
    print(f"Seeding {len(ENTRIES)} entries into {BASE}\n")

    # Health check first
    try:
        h = httpx.get(f"{BASE}/api/health", timeout=15).json()
        print(f"Current state: {h['events']} events, {h['memories']} memories, {h['entities']} entities\n")
    except Exception as e:
        print(f"Could not reach API: {e}\nMake sure the server is running.\n")
        return

    ok = 0
    for i, entry in enumerate(ENTRIES):
        success = ingest(entry, i)
        if success:
            ok += 1
        time.sleep(6.5)  # the API rate-limits ingest to 10/minute per IP

    print(f"\nDone — {ok}/{len(ENTRIES)} entries ingested successfully.")

    try:
        h = httpx.get(f"{BASE}/api/health", timeout=15).json()
        print(f"New state: {h['events']} events, {h['memories']} memories, {h['entities']} entities, {h['vectors']} vectors")
    except Exception:
        pass


if __name__ == "__main__":
    main()
