/**
 * ContextWeave — ambient capture companion.
 *
 * Tap Listen and put the phone down: audio is recorded in ~45s segments,
 * transcribed server-side (Groq Whisper), and woven into your private
 * memory. No audio is stored — only the transcript.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  AudioModule,
  RecordingPresets,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

const API_URL = (process.env.EXPO_PUBLIC_API_URL || 'https://vallllllllll-contextweave.hf.space').replace(/\/$/, '');
const KEY_STORAGE = 'cw_api_key';
const SEGMENT_SECONDS = 45;

const serif = Platform.select({ ios: 'Georgia', android: 'serif' });

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function App() {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);

  const [apiKey, setApiKey] = useState(null);
  const [keyInput, setKeyInput] = useState('');
  const [bootstrapped, setBootstrapped] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [freshKey, setFreshKey] = useState(null);

  const [listening, setListening] = useState(false);
  const [segmentElapsed, setSegmentElapsed] = useState(0);
  const [captures, setCaptures] = useState([]);
  const [digest, setDigest] = useState(null);
  const [online, setOnline] = useState(null);

  const listeningRef = useRef(false);
  const capturesRef = useRef(0);

  // ── boot: stored key + health ─────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const stored = await AsyncStorage.getItem(KEY_STORAGE);
        if (stored) setApiKey(stored);
      } finally {
        setBootstrapped(true);
      }
    })();
  }, []);

  const authHeaders = useCallback(
    () => (apiKey ? { 'X-API-Key': apiKey } : {}),
    [apiKey]
  );

  const refreshDigest = useCallback(async () => {
    try {
      const r = await fetch(`${API_URL}/api/digest`, { headers: authHeaders() });
      if (!r.ok) return;
      const d = await r.json();
      setDigest(d.memory_count > 0 ? d : null);
    } catch {
      /* offline — nudge card just stays hidden */
    }
  }, [authHeaders]);

  useEffect(() => {
    if (!bootstrapped) return;
    (async () => {
      try {
        const r = await fetch(`${API_URL}/api/me`, { headers: authHeaders() });
        setOnline(r.ok);
      } catch {
        setOnline(false);
      }
      refreshDigest();
    })();
  }, [bootstrapped, apiKey, authHeaders, refreshDigest]);

  // ── private space ─────────────────────────────────────────
  async function createSpace() {
    setRegistering(true);
    try {
      const r = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'mobile' }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      await AsyncStorage.setItem(KEY_STORAGE, body.api_key);
      setApiKey(body.api_key);
      setFreshKey(body.api_key);
    } catch (e) {
      Alert.alert('Could not create your space', String(e.message || e));
    } finally {
      setRegistering(false);
    }
  }

  async function usePastedKey() {
    const key = keyInput.trim();
    if (!key) return;
    try {
      const r = await fetch(`${API_URL}/api/me`, { headers: { 'X-API-Key': key } });
      if (!r.ok) {
        Alert.alert('Invalid key', 'That key was not accepted by the server.');
        return;
      }
      await AsyncStorage.setItem(KEY_STORAGE, key);
      setApiKey(key);
      setKeyInput('');
    } catch (e) {
      Alert.alert('Could not verify key', String(e.message || e));
    }
  }

  async function leaveSpace() {
    await AsyncStorage.removeItem(KEY_STORAGE);
    setApiKey(null);
    setFreshKey(null);
    setDigest(null);
  }

  // ── ambient capture loop ──────────────────────────────────
  async function uploadSegment(uri) {
    const id = ++capturesRef.current;
    const entry = { id, time: new Date(), status: 'transcribing…', transcript: '' };
    setCaptures((prev) => [entry, ...prev].slice(0, 30));

    const patch = (fields) =>
      setCaptures((prev) => prev.map((c) => (c.id === id ? { ...c, ...fields } : c)));

    try {
      const form = new FormData();
      form.append('file', {
        uri,
        name: `segment-${Date.now()}.m4a`,
        type: 'audio/m4a',
      });
      const r = await fetch(`${API_URL}/api/ingest/audio`, {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        patch({ status: body.detail || `failed (HTTP ${r.status})` });
        return;
      }
      if (!body.transcript) {
        patch({ status: 'no speech detected' });
        return;
      }
      patch({ status: 'remembered', transcript: body.transcript });
      refreshDigest();
    } catch (e) {
      patch({ status: `upload failed: ${e.message || e}` });
    }
  }

  async function segmentLoop() {
    while (listeningRef.current) {
      try {
        await recorder.prepareToRecordAsync();
        recorder.record();
      } catch (e) {
        listeningRef.current = false;
        setListening(false);
        Alert.alert('Recording failed', String(e.message || e));
        return;
      }

      // Tick once a second so Stop reacts immediately
      for (let s = 0; s < SEGMENT_SECONDS && listeningRef.current; s++) {
        setSegmentElapsed(s + 1);
        await sleep(1000);
      }

      try {
        await recorder.stop();
      } catch {
        /* segment lost; keep listening */
      }
      setSegmentElapsed(0);

      const uri = recorder.uri;
      if (uri) uploadSegment(uri);
    }
  }

  async function toggleListening() {
    if (listeningRef.current) {
      listeningRef.current = false;
      setListening(false);
      return;
    }

    const permission = await AudioModule.requestRecordingPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        'Microphone needed',
        'ContextWeave captures ambiently — enable the microphone in Settings.'
      );
      return;
    }
    await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });

    listeningRef.current = true;
    setListening(true);
    segmentLoop();
  }

  // ── UI ────────────────────────────────────────────────────
  if (!bootstrapped) {
    return (
      <View style={[styles.screen, styles.center]}>
        <ActivityIndicator color="#d4a373" />
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.wordmark}>ContextWeave</Text>
        <Text style={styles.tagline}>
          A companion that remembers, so you can be present.
        </Text>

        <View style={styles.statusRow}>
          <View
            style={[
              styles.dot,
              { backgroundColor: online == null ? '#8d8478' : online ? '#7fb069' : '#c0392b' },
            ]}
          />
          <Text style={styles.statusText}>
            {online == null ? 'connecting…' : online ? 'connected' : 'unreachable'}
            {'  ·  '}
            {apiKey ? 'private space' : 'public demo space'}
          </Text>
        </View>

        {/* Listen orb */}
        <View style={styles.orbWrap}>
          <Pressable
            onPress={toggleListening}
            style={[styles.orb, listening && styles.orbLive]}
          >
            <Text style={styles.orbGlyph}>{listening ? '■' : '●'}</Text>
            <Text style={styles.orbLabel}>{listening ? 'Stop' : 'Listen'}</Text>
          </Pressable>
          <Text style={styles.orbHint}>
            {listening
              ? `listening — segment ${segmentElapsed}s / ${SEGMENT_SECONDS}s`
              : 'tap, then just live your life'}
          </Text>
        </View>

        {/* Today's nudge */}
        {digest ? (
          <View style={styles.card}>
            <Text style={styles.cardLabel}>✦ TODAY'S NUDGE</Text>
            <Text style={styles.nudgeHeadline}>{digest.headline}</Text>
            {(digest.commitments || []).slice(0, 3).map((c, i) => (
              <Text key={i} style={styles.nudgeItem}>
                · {c}
              </Text>
            ))}
          </View>
        ) : null}

        {/* Space setup */}
        {!apiKey ? (
          <View style={styles.card}>
            <Text style={styles.cardLabel}>YOUR PRIVATE SPACE</Text>
            <Text style={styles.body}>
              Right now captures land in the shared demo. Create a private space
              so your memory is yours alone.
            </Text>
            <Pressable style={styles.btn} onPress={createSpace} disabled={registering}>
              <Text style={styles.btnText}>
                {registering ? 'Creating…' : 'Create private space'}
              </Text>
            </Pressable>
            <TextInput
              style={styles.input}
              placeholder="…or paste an existing cw_ key"
              placeholderTextColor="#8d8478"
              autoCapitalize="none"
              value={keyInput}
              onChangeText={setKeyInput}
              onSubmitEditing={usePastedKey}
            />
          </View>
        ) : (
          <View style={styles.card}>
            <Text style={styles.cardLabel}>YOUR PRIVATE SPACE</Text>
            {freshKey ? (
              <>
                <Text style={styles.body}>
                  Save this key — it is shown only once and is the only way back
                  into this space:
                </Text>
                <Text selectable style={styles.keyText}>
                  {freshKey}
                </Text>
              </>
            ) : (
              <Text style={styles.body}>Active. Everything you capture here is private.</Text>
            )}
            <Pressable style={styles.btnGhost} onPress={leaveSpace}>
              <Text style={styles.btnGhostText}>Leave space on this phone</Text>
            </Pressable>
          </View>
        )}

        {/* Capture feed */}
        {captures.length > 0 ? (
          <View style={styles.card}>
            <Text style={styles.cardLabel}>CAPTURED</Text>
            {captures.map((c) => (
              <View key={c.id} style={styles.capture}>
                <Text style={styles.captureMeta}>
                  {c.time.toLocaleTimeString()} — {c.status}
                </Text>
                {c.transcript ? (
                  <Text style={styles.captureText}>{c.transcript}</Text>
                ) : null}
              </View>
            ))}
          </View>
        ) : null}

        <Text style={styles.footer}>
          Audio is transcribed and discarded — only words become memory.
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#14110d' },
  center: { alignItems: 'center', justifyContent: 'center' },
  scroll: { padding: 24, paddingTop: 72, paddingBottom: 48 },

  wordmark: { fontFamily: serif, fontSize: 34, color: '#ece5da', letterSpacing: 0.5 },
  tagline: { fontFamily: serif, fontSize: 15, color: '#8d8478', marginTop: 6, fontStyle: 'italic' },

  statusRow: { flexDirection: 'row', alignItems: 'center', marginTop: 18 },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  statusText: { color: '#8d8478', fontSize: 12, letterSpacing: 0.4 },

  orbWrap: { alignItems: 'center', marginVertical: 36 },
  orb: {
    width: 168,
    height: 168,
    borderRadius: 84,
    borderWidth: 1,
    borderColor: '#3a332b',
    backgroundColor: '#1d1915',
    alignItems: 'center',
    justifyContent: 'center',
  },
  orbLive: { borderColor: '#c0392b', backgroundColor: '#231512' },
  orbGlyph: { fontSize: 34, color: '#d4a373', marginBottom: 6 },
  orbLabel: { color: '#ece5da', fontSize: 18, fontFamily: serif },
  orbHint: { color: '#8d8478', fontSize: 12, marginTop: 14 },

  card: {
    backgroundColor: '#1d1915',
    borderColor: '#2c2620',
    borderWidth: 1,
    borderRadius: 14,
    padding: 18,
    marginBottom: 18,
  },
  cardLabel: { color: '#d4a373', fontSize: 11, letterSpacing: 2, marginBottom: 10 },
  body: { color: '#bdb3a5', fontSize: 14, lineHeight: 20, marginBottom: 12 },

  nudgeHeadline: { fontFamily: serif, fontSize: 19, lineHeight: 26, color: '#ece5da', marginBottom: 10 },
  nudgeItem: { color: '#bdb3a5', fontSize: 13, lineHeight: 20 },

  btn: {
    backgroundColor: '#d4a373',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  btnText: { color: '#14110d', fontWeight: '600', fontSize: 14 },
  btnGhost: { paddingVertical: 8, alignItems: 'center' },
  btnGhostText: { color: '#8d8478', fontSize: 13, textDecorationLine: 'underline' },
  input: {
    borderColor: '#2c2620',
    borderWidth: 1,
    borderRadius: 10,
    color: '#ece5da',
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
  },
  keyText: {
    color: '#d4a373',
    fontSize: 12,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
    backgroundColor: '#14110d',
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
  },

  capture: { borderTopColor: '#2c2620', borderTopWidth: 1, paddingVertical: 10 },
  captureMeta: { color: '#8d8478', fontSize: 11, marginBottom: 4 },
  captureText: { color: '#ece5da', fontSize: 14, lineHeight: 20 },

  footer: { color: '#5f574c', fontSize: 11, textAlign: 'center', marginTop: 8 },
});
