(function () {
  'use strict';

  // Ambient voice capture via the Web Speech API. Transcribes speech live
  // into the ingest textarea; hidden entirely when the browser lacks support.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  function init(btn, textarea, sourceSelect) {
    if (!btn || !textarea) return;
    if (!SR) { btn.classList.add('hidden'); return; }

    let rec = null;
    let base = '';
    let finals = '';

    btn.addEventListener('click', () => {
      if (rec) { try { rec.stop(); } catch { /* already stopping */ } return; }

      rec = new SR();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = navigator.language || 'en-US';
      base = textarea.value ? textarea.value.trimEnd() + '\n' : '';
      finals = '';

      rec.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) finals += t + ' ';
          else interim += t;
        }
        textarea.value = base + finals + interim;
      };

      rec.onend = () => {
        rec = null;
        btn.classList.remove('recording');
        btn.textContent = '🎙 Listen';
        if (finals.trim() && sourceSelect) sourceSelect.value = 'conversation';
      };

      rec.onerror = () => { /* onend fires next and resets state */ };

      btn.classList.add('recording');
      btn.textContent = '■ Stop';
      rec.start();
    });
  }

  window.CWCapture = { init };
})();
