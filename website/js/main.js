(function () {
  var path = window.location.pathname.replace(/\/$/, "") || "/";

  var zoneRoutes = {
    home: ["/"],
    portal: ["/portal", "/portfolio"],
    dashboard: ["/viewer"],
    ml: ["/ml", "/paper", "/demo", "/product"],
    ai: ["/ai"],
    tools: ["/tools"],
  };

  function pathInZone(p, prefixes) {
    return prefixes.some(function (prefix) {
      if (prefix === "/") return p === "/";
      return p === prefix || p.startsWith(prefix + "/");
    });
  }

  var activeZone = "home";
  var keys = Object.keys(zoneRoutes);
  for (var i = 0; i < keys.length; i++) {
    var zone = keys[i];
    if (pathInZone(path, zoneRoutes[zone])) {
      activeZone = zone;
      break;
    }
  }

  document.querySelectorAll(".nav a[data-zone]").forEach(function (link) {
    if (link.getAttribute("data-zone") === activeZone) {
      link.classList.add("active");
    }
  });
})();
