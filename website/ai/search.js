(function () {
  var input = document.getElementById("search-input");
  var resultsEl = document.getElementById("search-results");
  var statusEl = document.getElementById("search-status");
  if (!input || !resultsEl) return;

  var items = [];

  fetch("/ai/search-index.json")
    .then(function (res) {
      return res.json();
    })
    .then(function (data) {
      items = data.items || [];
      if (statusEl) statusEl.textContent = "已加载 " + items.length + " 条索引";
      var q = new URLSearchParams(window.location.search).get("q");
      if (q) {
        input.value = q;
        render(q);
      }
    })
    .catch(function () {
      if (statusEl) statusEl.textContent = "索引加载失败";
    });

  function score(item, q) {
    var hay = (item.title + " " + item.description + " " + (item.tags || []).join(" ")).toLowerCase();
    if (hay.includes(q)) return 10;
    var parts = q.split(/\s+/).filter(Boolean);
    var s = 0;
    for (var i = 0; i < parts.length; i++) {
      if (hay.includes(parts[i])) s += 3;
    }
    return s;
  }

  function render(query) {
    var q = (query || "").trim().toLowerCase();
    resultsEl.innerHTML = "";
    if (!q) {
      resultsEl.innerHTML = "<li class=\"search-result-item\"><span class=\"search-result-meta\">输入关键词检索页面、方法或 Demo 户型 ID（如 house_2107）</span></li>";
      return;
    }

    var hits = items
      .map(function (item) {
        return { item: item, s: score(item, q) };
      })
      .filter(function (x) {
        return x.s > 0;
      })
      .sort(function (a, b) {
        return b.s - a.s;
      })
      .slice(0, 20);

    if (hits.length === 0) {
      resultsEl.innerHTML = "<li class=\"search-result-item\">未找到与「" + query + "」相关的内容</li>";
      return;
    }

    for (var j = 0; j < hits.length; j++) {
      var it = hits[j].item;
      var li = document.createElement("li");
      li.className = "search-result-item";
      var tags = (it.tags || [])
        .slice(0, 3)
        .map(function (t) {
          return "<span class=\"search-tag\">" + t + "</span>";
        })
        .join("");
      li.innerHTML =
        "<a href=\"" + it.path + "\">" + it.title + "</a>" +
        "<div class=\"search-result-meta\">" + tags + it.description + "</div>";
      resultsEl.appendChild(li);
    }
  }

  input.addEventListener("input", function () {
    render(input.value);
  });
})();
