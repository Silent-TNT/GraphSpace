(function () {
  var cardsRoot = document.getElementById("zone-cards");
  if (!cardsRoot) return;

  var cards = Array.from(cardsRoot.querySelectorAll(".zone-card"));
  var transition = document.getElementById("fold-transition");
  var soundToggle = document.getElementById("fold-sound-toggle");
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var isCoarse = window.matchMedia("(hover: none) and (pointer: coarse)").matches;
  var isLeaving = false;
  var lastHoverCard = null;
  var hoverSoundThrottle = 0;

  var zoneColors = {
    portal: "#3b82f6",
    dashboard: "#38bdf8",
    ml: "#a78bfa",
    ai: "#f472b6",
    tools: "#10b981",
  };

  function unlockAudio() {
    if (window.FoldAudio) FoldAudio.unlock();
  }

  document.addEventListener("click", unlockAudio, { once: true });
  document.addEventListener("keydown", unlockAudio, { once: true });

  if (soundToggle && window.FoldAudio) {
    soundToggle.addEventListener("click", function () {
      var on = soundToggle.getAttribute("aria-pressed") !== "true";
      soundToggle.setAttribute("aria-pressed", on ? "true" : "false");
      soundToggle.textContent = on ? "🔊 音效" : "🔇 静音";
      FoldAudio.setEnabled(on);
    });
  }

  cards.forEach(function (card) {
    if (!isCoarse) {
      card.addEventListener("mouseenter", function () {
        if (card === lastHoverCard) return;
        lastHoverCard = card;
        var now = Date.now();
        if (now - hoverSoundThrottle > 180 && window.FoldAudio) {
          hoverSoundThrottle = now;
          FoldAudio.playHover();
        }
      });
    }

    card.addEventListener("mousedown", function () {
      card.classList.add("is-pressed");
      if (window.FoldAudio) FoldAudio.playPress();
    });

    card.addEventListener("mouseup", function () {
      card.classList.remove("is-pressed");
    });

    card.addEventListener("mouseleave", function () {
      card.classList.remove("is-pressed");
      if (lastHoverCard === card) lastHoverCard = null;
    });

    card.addEventListener("click", function (e) {
      if (isLeaving) {
        e.preventDefault();
        return;
      }
      var href = card.getAttribute("href");
      if (!href) return;

      e.preventDefault();
      navigateWithFold(card, href);
    });
  });

  function navigateWithFold(card, href) {
    if (isLeaving) return;
    isLeaving = true;

    var zone = card.getAttribute("data-zone") || "portal";
    var dir = card.getAttribute("data-fold-dir") || "center";
    var color = zoneColors[zone] || "#3b82f6";

    document.documentElement.style.setProperty("--fold-zone-color", color);
    document.body.classList.add("is-transitioning");

    if (window.FoldAudio) FoldAudio.playFoldAway();

    if (reducedMotion || !transition) {
      window.location.href = href;
      return;
    }

    transition.removeAttribute("hidden");
    transition.setAttribute("data-fold-dir", dir);
    transition.setAttribute("aria-hidden", "false");
    requestAnimationFrame(function () {
      transition.classList.add("is-active");
    });

    var duration = isCoarse ? 550 : 900;
    setTimeout(function () {
      window.location.href = href;
    }, duration);
  }

  document.addEventListener("keydown", function (e) {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === "Enter") {
      var focused = document.activeElement;
      if (focused && focused.classList.contains("zone-card")) {
        focused.click();
      }
      return;
    }
    var idx = parseInt(e.key, 10);
    if (idx < 1 || idx > 5) return;
    var card = cards[idx - 1];
    if (card) card.focus();
  });
})();
