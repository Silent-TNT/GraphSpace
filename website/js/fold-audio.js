/**
 * 折纸音效 — Web Audio 合成（无需外部音频文件）
 */
window.FoldAudio = (function () {
  var ctx = null;
  var enabled = true;
  var unlocked = false;

  function getCtx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return ctx;
  }

  function unlock() {
    if (unlocked) return;
    var c = getCtx();
    if (c.state === "suspended") c.resume();
    unlocked = true;
  }

  function noiseBurst(duration, opts) {
    var c = getCtx();
    var sampleRate = c.sampleRate;
    var len = Math.floor(sampleRate * duration);
    var buffer = c.createBuffer(1, len, sampleRate);
    var data = buffer.getChannelData(0);
    for (var i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / len);
    }
    var src = c.createBufferSource();
    src.buffer = buffer;

    var filter = c.createBiquadFilter();
    filter.type = opts.filterType || "bandpass";
    filter.frequency.value = opts.freq || 1200;
    filter.Q.value = opts.q || 0.8;

    var gain = c.createGain();
    gain.gain.setValueAtTime(0.0001, c.currentTime);
    gain.gain.exponentialRampToValueAtTime(opts.peak || 0.12, c.currentTime + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + duration);

    src.connect(filter);
    filter.connect(gain);
    gain.connect(c.destination);
    src.start();
    src.stop(c.currentTime + duration + 0.02);
  }

  function toneClick(freq, duration) {
    var c = getCtx();
    var osc = c.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = freq;
    var gain = c.createGain();
    gain.gain.setValueAtTime(0.0001, c.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.04, c.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + duration);
    osc.connect(gain);
    gain.connect(c.destination);
    osc.start();
    osc.stop(c.currentTime + duration);
  }

  return {
    setEnabled: function (on) {
      enabled = on;
    },
    isEnabled: function () {
      return enabled;
    },
    unlock: unlock,
    /** 悬停 — 轻折纸摩擦 */
    playHover: function () {
      if (!enabled) return;
      unlock();
      noiseBurst(0.06, { freq: 1800, q: 1.2, peak: 0.05 });
      toneClick(320 + Math.random() * 80, 0.04);
    },
    /** 按下 — 短折痕 */
    playPress: function () {
      if (!enabled) return;
      unlock();
      noiseBurst(0.08, { freq: 900, q: 0.6, peak: 0.1 });
    },
    /** 整页折入 — 双层纸折叠 */
    playFoldAway: function () {
      if (!enabled) return;
      unlock();
      noiseBurst(0.14, { freq: 600, q: 0.5, peak: 0.18 });
      setTimeout(function () {
        noiseBurst(0.1, { freq: 400, q: 0.4, peak: 0.14 });
        toneClick(180, 0.12);
      }, 120);
      setTimeout(function () {
        noiseBurst(0.2, { freq: 250, q: 0.3, peak: 0.22, filterType: "lowpass" });
      }, 380);
    },
  };
})();
