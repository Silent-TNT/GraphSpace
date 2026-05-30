(function () {
  var mount = document.querySelector("[data-site-shell]");
  if (!mount) return;

  var header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML =
    '<div class="container header-inner">' +
    '  <a class="logo" href="/">Space<span>Modal</span><span class="logo-sub">GraphSpace · 空间折叠</span></a>' +
    '  <nav class="nav" aria-label="站点主导航">' +
    '    <a href="/" data-zone="home">折叠空间</a>' +
    '    <a href="/portal/" data-zone="portal">门户</a>' +
    '    <a href="/viewer/" data-zone="dashboard">看板</a>' +
    '    <a href="/ml/" data-zone="ml">ML</a>' +
    '    <a href="/ai/" data-zone="ai">AI</a>' +
    '    <a href="/tools/" data-zone="tools">工具箱</a>' +
    "  </nav>" +
    "</div>";

  mount.replaceWith(header);
})();
