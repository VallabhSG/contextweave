# ContextWeave Mobile

Ambient capture companion (Expo SDK 54). Tap **Listen** and put the phone
down — audio records in 45-second segments, each segment is transcribed
server-side by Groq Whisper (`/api/ingest/audio`), and only the words enter
your memory. Audio is never stored.

## Run

```bash
cd mobile
npm install            # if node_modules is missing
cp .env.example .env.local   # point EXPO_PUBLIC_API_URL at your backend
npx expo start         # scan the QR with Expo Go (iOS/Android)
```

Requires the backend to have `CW_GROQ_API_KEY` set — transcription returns
503 otherwise.

## What it does

- **Listen orb** — segment-loop recording via `expo-audio`; keeps capturing
  until you tap Stop. iOS `UIBackgroundModes: audio` is declared so capture
  can continue with the screen off in a dev/standalone build (Expo Go
  itself may suspend in background).
- **Private space** — create or paste a `cw_` API key; stored in
  AsyncStorage and sent as `X-API-Key` on every request.
- **Today's nudge** — shows the proactive digest from `/api/digest`.
- **Capture feed** — live transcript status for each uploaded segment.
