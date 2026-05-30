(function () {
  function injectHeader() {
    var mount = document.querySelector("[data-site-shell]");
    if (!mount) return;

    var header = document.createElement("header");
    header.className = "site-header";
    header.innerHTML =
      '<div class="container header-inner">' +
      '  <a class="logo" href="/">Space<span>Modal</span><span class="logo-sub">GraphSpace · 空间折叠</span></a>' +
      '  <button type="button" class="nav-toggle" aria-expanded="false" aria-controls="site-nav" aria-label="打开导航菜单">' +
      '    <span class="nav-toggle-bar" aria-hidden="true"></span>' +
      '    <span class="nav-toggle-bar" aria-hidden="true"></span>' +
      '    <span class="nav-toggle-bar" aria-hidden="true"></span>' +
      "  </button>" +
      '  <div class="nav-backdrop" id="nav-backdrop" hidden></div>' +
      '  <nav class="nav" id="site-nav" aria-label="站点主导航">' +
      '    <a href="/" data-zone="home">折叠空间</a>' +
      '    <a href="/portal/" data-zone="portal">门户</a>' +
      '    <a href="/viewer/" data-zone="dashboard">看板</a>' +
      '    <a href="/ml/" data-zone="ml">ML</a>' +
      '    <a href="/ai/" data-zone="ai">AI</a>' +
      '    <a href="/tools/" data-zone="tools">工具箱</a>' +
      "  </nav>" +
      "</div>";

    mount.replaceWith(header);

    var toggle = header.querySelector(".nav-toggle");
    var nav = header.querySelector(".nav");
    var backdrop = header.querySelector(".nav-backdrop");

    function setNavOpen(open) {
      header.classList.toggle("nav-open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.setAttribute("aria-label", open ? "关闭导航菜单" : "打开导航菜单");
      backdrop.hidden = !open;
      document.body.classList.toggle("nav-scroll-lock", open);
    }

    toggle.addEventListener("click", function () {
      setNavOpen(!header.classList.contains("nav-open"));
    });

    backdrop.addEventListener("click", function () {
      setNavOpen(false);
    });

    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        setNavOpen(false);
      });
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setNavOpen(false);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectHeader);
  } else {
    injectHeader();
  }
})();
