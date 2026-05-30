(function () {
  var cfg = window.SITE_CONFIG;
  if (!cfg) return;

  document.querySelectorAll("[data-contact-email]").forEach(function (el) {
    var subject = el.getAttribute("data-subject") || "";
    var href = "mailto:" + cfg.contactEmail;
    if (subject) href += "?subject=" + encodeURIComponent(subject);
    el.setAttribute("href", href);
    if (el.hasAttribute("data-show-email")) {
      el.textContent = cfg.contactEmail;
    }
  });

  document.querySelectorAll("[data-github-url]").forEach(function (el) {
    if (cfg.githubUrl) {
      el.setAttribute("href", cfg.githubUrl);
      el.removeAttribute("aria-disabled");
      el.classList.remove("link-disabled");
    } else {
      el.setAttribute("href", "#");
      el.setAttribute("aria-disabled", "true");
      el.classList.add("link-disabled");
      el.addEventListener("click", function (e) {
        e.preventDefault();
      });
    }
  });

  document.querySelectorAll("[data-author-name]").forEach(function (el) {
    if (cfg.authorName && cfg.authorName !== "请填写你的姓名") {
      el.textContent = cfg.authorName;
    }
  });

  document.querySelectorAll("[data-author-role]").forEach(function (el) {
    if (cfg.authorRole) {
      el.textContent = cfg.authorRole;
    }
  });
})();
