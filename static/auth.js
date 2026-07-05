/* CWAuth — Supabase sign-in for the web UI.
 *
 * Discovers the project via GET /api/auth/config; when the backend is not
 * configured for Supabase, `enabled` stays false and the key/demo flows in
 * app.js work exactly as before. The session lives in localStorage (managed
 * by supabase-js, auto-refreshed) and the current access token is cached
 * here so app.js can read it synchronously.
 */
(function () {
  'use strict';

  let client = null;
  let session = null;
  const listeners = [];

  function notify() {
    listeners.forEach(cb => {
      try { cb(); } catch { /* a listener error must not break the auth flow */ }
    });
  }

  window.CWAuth = {
    enabled: false,

    async init() {
      try {
        const r = await fetch('/api/auth/config');
        if (!r.ok) return;
        const cfg = (await r.json()).supabase || {};
        if (!cfg.enabled || !window.supabase) return;
        client = window.supabase.createClient(cfg.url, cfg.anon_key);
        const { data } = await client.auth.getSession();
        session = data.session || null;
        client.auth.onAuthStateChange((_event, s) => { session = s; notify(); });
        this.enabled = true;
      } catch {
        // config unreachable → auth stays disabled, everything else still works
      }
    },

    token() { return session ? session.access_token : null; },
    email() { return session && session.user ? session.user.email : null; },
    onChange(cb) { listeners.push(cb); },

    async signIn(email, password) {
      const { error } = await client.auth.signInWithPassword({ email, password });
      if (error) throw new Error(error.message);
    },

    // Returns true when the account is immediately active; false when the
    // project requires email confirmation first.
    async signUp(email, password) {
      const { data, error } = await client.auth.signUp({ email, password });
      if (error) throw new Error(error.message);
      return !!(data && data.session);
    },

    async signOut() {
      const { error } = await client.auth.signOut();
      if (error) throw new Error(error.message);
    },
  };
})();
