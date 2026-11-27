(() => {
  const state = {
    overview: null,
    history: [],
    memories: [],
    voiceprints: [],
    reminders: [],
    activeTab: "overview",
    trendMetric: "tokens",
    authRequired: false,
    openGroups: new Set(["memories", "voiceprints", "reminders"]),
  };

  const $ = (id) => document.getElementById(id);

  function formatCompactNum(val) {
    const n = Number(val) || 0;
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 10_000) return `${(n / 1_000).toFixed(1)}k`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(2)}k`;
    return String(Math.round(n));
  }

  function formatUsd(val, digits = 2) {
    const n = Number(val) || 0;
    return `$${n.toFixed(digits)}`;
  }

  function escapeHtml(raw) {
    const s = String(raw ?? "");
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  let toastTimer = null;
  function showToast(msg, isError = false) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.toggle("is-error", isError);
    el.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.classList.remove("is-visible");
    }, 2800);
  }

  function formatTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  async function apiFetch(url, options = {}) {
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await resp.json().catch(() => ({}));
    if (resp.status === 401 && url !== "/api/login" && url !== "/api/auth/status") {
      showLoginGate("登录状态已过期，请重新验证");
      const err = new Error(data.message || "需要登录");
      err.status = 401;
      throw err;
    }
    if (!resp.ok) {
      throw new Error(data.message || data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  function showLoginGate(errMsg = "") {
    $("mainApp").hidden = true;
    $("mobileTabsBar").hidden = true;
    $("logoutBtn").hidden = true;
    $("loginGate").hidden = false;
    $("serverStatus").className = "badge badge--warn";
    $("serverStatusText").textContent = "待验证";
    const errBox = $("loginError");
    if (errMsg) {
      errBox.textContent = errMsg;
      errBox.hidden = false;
    } else {
      errBox.hidden = true;
    }
  }

  function showDashboard(authRequired) {
    state.authRequired = Boolean(authRequired);
    $("loginGate").hidden = true;
    $("mobileTabsBar").hidden = false;
    $("mainApp").hidden = false;
    $("logoutBtn").hidden = !state.authRequired;
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll("[data-tab]").forEach((btn) => {
      btn.setAttribute("aria-selected", btn.dataset.tab === tabName ? "true" : "false");
    });
    document.querySelectorAll("main > section").forEach((sec) => {
      sec.hidden = sec.id !== `panel-${tabName}`;
    });
  }

  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  function syncGroupsUI() {
    document.querySelectorAll("[data-group-toggle]").forEach((btn) => {
      const gid = btn.getAttribute("data-group-toggle");
      const isOpen = state.openGroups.has(gid);
      btn.setAttribute("aria-expanded", isOpen ? "true" : "false");
      btn.textContent = isOpen ? "收起" : "展开";
      const body = $(`group-body-${gid}`);
      if (body) body.hidden = !isOpen;
    });
    const allOpen = state.openGroups.size === 3;
    $("toggleAllGroupsBtn").textContent = allOpen ? "全部收起" : "全部展开";
  }

  document.querySelectorAll("[data-group-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const gid = btn.getAttribute("data-group-toggle");
      if (state.openGroups.has(gid)) {
        state.openGroups.delete(gid);
      } else {
        state.openGroups.add(gid);
      }
      syncGroupsUI();
    });
  });

  $("toggleAllGroupsBtn").addEventListener("click", () => {
    if (state.openGroups.size === 3) {
      state.openGroups.clear();
    } else {
      state.openGroups = new Set(["memories", "voiceprints", "reminders"]);
    }
    syncGroupsUI();
  });

  function formatMs(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    const n = Number(val);
    if (n >= 1000) return `${(n / 1000).toFixed(2)}s`;
    return `${Math.round(n)}ms`;
  }

  function renderTrendChart(data) {
    const metric = state.trendMetric;
    document.querySelectorAll("[data-trend-metric]").forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-trend-metric") === metric ? "true" : "false");
    });
    const usageDaily = (data.usage && Array.isArray(data.usage.daily)) ? data.usage.daily : [];
    const turnDaily = Array.isArray(data.daily_turns) ? data.daily_turns : [];
    const rows = (metric === "turns" || metric === "latency") ? turnDaily : usageDaily;
    if (!rows.length) {
      $("trendChartContainer").innerHTML = `<div class="empty-state">暂无近 14 日趋势数据。</div>`;
      return;
    }
    const values = rows.map((r) => {
      if (metric === "turns") return Number(r.turns) || 0;
      if (metric === "latency") return Number(r.avg_latency_ms) || 0;
      if (metric === "cost") return Number(r.cost_usd) || 0;
      return Number(r.total_tokens) || 0;
    });
    const maxVal = Math.max(...values, 0);
    const totalVal = values.reduce((a, b) => a + b, 0);
    const activeDays = values.filter((v) => v > 0).length;

    const fmtVal = (v) => {
      if (v <= 0) return "";
      if (metric === "latency") return v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`;
      if (metric === "cost") return v >= 0.01 ? `$${v.toFixed(2)}` : `$${v.toFixed(3)}`;
      if (metric === "tokens") return formatCompactNum(v);
      return String(v);
    };

    let summaryLabel = "";
    if (metric === "cost") {
      summaryLabel = !data.usage?.available
        ? "暂未检测到本机 OpenCode 用量数据库，对话后自动记录"
        : `近 14 日累计估算花费 ${formatUsd(totalVal, 4)} · 活跃 ${activeDays} 天`;
    } else if (metric === "tokens") {
      summaryLabel = !data.usage?.available
        ? "暂未检测到本机 OpenCode 用量数据库，对话后自动记录"
        : `近 14 日累计消耗 ${formatCompactNum(totalVal)} Token · 活跃 ${activeDays} 天`;
    } else if (metric === "latency") {
      const weightedTurns = rows.reduce((acc, r) => acc + (Number(r.turns) || 0), 0);
      const weightedSum = rows.reduce((acc, r) => acc + (Number(r.turns) || 0) * (Number(r.avg_latency_ms) || 0), 0);
      const avg14 = weightedTurns > 0 ? weightedSum / weightedTurns : 0;
      const peakP95 = Math.max(...rows.map((r) => Number(r.p95_latency_ms) || 0), 0);
      summaryLabel = weightedTurns > 0
        ? `近 14 日加权平均延迟 ${formatMs(avg14)} · 日峰值 P95 ${formatMs(peakP95)} · 活跃 ${activeDays} 天`
        : "暂无近 14 日延迟记录";
    } else {
      summaryLabel = `近 14 日累计对话 ${totalVal} 轮 · 活跃 ${activeDays} 天`;
    }

    $("trendChartContainer").innerHTML = `
      <div class="trend-bars">
        ${rows
          .map((r, idx) => {
            const v = values[idx];
            const hPct = maxVal > 0 && v > 0 ? Math.max(8, Math.round((v / maxVal) * 100)) : 4;
            const shortDate = String(r.date || "").slice(5);
            const tip =
              metric === "cost"
                ? `${r.date}: ${formatUsd(v, 4)} (${r.calls || 0} 次调用)`
                : metric === "tokens"
                  ? `${r.date}: ${v.toLocaleString()} tokens (${r.calls || 0} 次调用)`
                  : metric === "latency"
                    ? `${r.date}: 平均 ${Math.round(v)} ms · P95 ${Math.round(Number(r.p95_latency_ms) || 0)} ms (${r.turns || 0} 轮)`
                    : `${r.date}: ${v} 轮对话`;
            return `
              <div class="trend-col" title="${escapeHtml(tip)}">
                <span class="trend-val">${escapeHtml(fmtVal(v))}</span>
                <div class="trend-bar-area">
                  <div class="trend-bar ${v > 0 ? "has-val" : ""}" style="height: ${hPct}%"></div>
                </div>
                <span class="trend-date">${escapeHtml(shortDate)}</span>
              </div>
            `;
          })
          .join("")}
      </div>
      <div class="trend-summary-row">
        <span class="num">${escapeHtml(summaryLabel)}</span>
        <span>数据来源：本机 history.json 与隔离 OpenCode SQLite</span>
      </div>
    `;
  }

  function renderRecentTraces(traces) {
    const list = Array.isArray(traces) ? traces.slice(0, 10) : [];
    $("traceCountBadge").innerHTML = `<span class="num">${list.length}</span> / 10`;
    const container = $("recentTracesContainer");
    if (!list.length) {
      container.innerHTML = `<div class="empty-state">暂无对话 Trace。对音箱说话后，这里会自动展示最近 10 次完整延迟拆解。</div>`;
      return;
    }
    container.innerHTML = `<div class="trace-list">${list
      .map((t, idx) => {
        const bd = t.breakdown_ms || {};
        const asrMs = bd.asr_ms != null ? Number(bd.asr_ms) : 0;
        const spkMs = bd.speaker_id_ms != null ? Number(bd.speaker_id_ms) : 0;
        const queueMs = bd.queue_ms != null ? Number(bd.queue_ms) : 0;
        const preLlmTotal = asrMs + spkMs + queueMs;
        const hasLlmTtft = bd.llm_ttft_ms != null;
        const llmTtft = hasLlmTtft ? Number(bd.llm_ttft_ms) : 0;
        const llmTotal = bd.llm_total_ms != null ? Number(bd.llm_total_ms) : 0;
        const llmBarMs = hasLlmTtft ? llmTtft : llmTotal;
        const startPlay = bd.start_play_ms != null ? Number(bd.start_play_ms) : 0;
        const ttsFirst = bd.tts_first_chunk_ms != null ? Number(bd.tts_first_chunk_ms) : 0;
        const ttsHead = Math.max(startPlay, ttsFirst);
        const ttfaVal = bd.ttfa_ms != null ? Number(bd.ttfa_ms) : null;
        const totalRaw = bd.total_ms != null ? Number(bd.total_ms) : 0;
        const total = Math.max(totalRaw, ttfaVal ?? 0, preLlmTotal + llmBarMs + ttsHead);
        const tailMs =
          totalRaw > 0
            ? (ttfaVal != null
                ? Math.max(0, totalRaw - Math.max(ttfaVal, preLlmTotal + llmBarMs + ttsHead))
                : Math.max(0, totalRaw - (preLlmTotal + llmBarMs + ttsHead)))
            : 0;

        const denom = preLlmTotal + llmBarMs + ttsHead + tailMs;
        const pAsr = denom > 0 && preLlmTotal > 0 ? (preLlmTotal / denom) * 100 : 0;
        const pLlm = denom > 0 && llmBarMs > 0 ? (llmBarMs / denom) * 100 : 0;
        const pTts = denom > 0 && ttsHead > 0 ? (ttsHead / denom) * 100 : 0;
        const pRest = denom > 0 && tailMs > 0 ? (tailMs / denom) * 100 : 0;

        const statusCls =
          t.status === "error"
            ? "badge--error"
            : t.status === "interrupted"
              ? "badge--warn"
              : "badge--success";
        const statusText =
          t.status === "error" ? "异常" : t.status === "interrupted" ? "打断" : "正常";
        const transportLabel =
          t.transport === "wan_relay"
            ? `WAN ${t.sample_rate ? `${Math.round(t.sample_rate / 1000)}kHz` : ""}`.trim()
            : t.transport === "lan"
              ? `LAN ${t.sample_rate ? `${Math.round(t.sample_rate / 1000)}kHz` : ""}`.trim()
              : String(t.transport || "voice").toUpperCase();

        const pills = [];
        if (bd.asr_ms != null) pills.push(`<span class="trace-pill"><span class="trace-dot trace-dot--asr"></span>ASR <span class="num">${escapeHtml(formatMs(bd.asr_ms))}</span></span>`);
        if (bd.speaker_id_ms != null) pills.push(`<span class="trace-pill"><span class="trace-dot trace-dot--asr"></span>声纹 <span class="num">${escapeHtml(formatMs(bd.speaker_id_ms))}</span></span>`);
        if (bd.queue_ms != null) pills.push(`<span class="trace-pill">排队/打断清理 <span class="num">${escapeHtml(formatMs(bd.queue_ms))}</span></span>`);
        if (bd.llm_ttft_ms != null) pills.push(`<span class="trace-pill"><span class="trace-dot trace-dot--llm"></span>LLM首句 <span class="num">${escapeHtml(formatMs(bd.llm_ttft_ms))}</span></span>`);
        if (bd.start_play_ms != null) pills.push(`<span class="trace-pill"><span class="trace-dot trace-dot--tts"></span>start_play <span class="num">${escapeHtml(formatMs(bd.start_play_ms))}</span></span>`);
        if (bd.tts_first_chunk_ms != null) pills.push(`<span class="trace-pill"><span class="trace-dot trace-dot--tts"></span>TTS首块 <span class="num">${escapeHtml(formatMs(bd.tts_first_chunk_ms))}</span></span>`);
        if (bd.ack_ms != null) pills.push(`<span class="trace-pill">先导回应 <span class="num">${escapeHtml(formatMs(bd.ack_ms))}</span></span>`);
        if (bd.tts_total_ms != null) pills.push(`<span class="trace-pill">播报总计 <span class="num">${escapeHtml(formatMs(bd.tts_total_ms))}</span></span>`);

        return `
          <details class="trace-item" ${idx === 0 ? "open" : ""}>
            <summary class="trace-summary">
              <div class="trace-head">
                <div class="trace-head-left">
                  <span class="badge ${statusCls}">${escapeHtml(statusText)}</span>
                  <span class="badge badge--accent">${escapeHtml(t.speaker_name || "未知")}</span>
                  <span class="trace-query" title="${escapeHtml(t.query || "")}">${escapeHtml(t.query || "（无文本）")}</span>
                </div>
                <div class="trace-head-right">
                  <span class="badge">${escapeHtml(transportLabel)}</span>
                  ${t.barge_in ? `<span class="badge badge--warn">插话打断</span>` : ""}
                  <span class="trace-kpi">首音 <strong class="num">${escapeHtml(formatMs(ttfaVal))}</strong></span>
                  <span class="trace-kpi">总计 <strong class="num">${escapeHtml(formatMs(bd.total_ms))}</strong></span>
                  <span class="card-meta num">${escapeHtml(formatTime(t.timestamp))}</span>
                </div>
              </div>
              <div class="trace-bar-track" aria-label="阶段延迟占比">
                ${pAsr > 0 ? `<div class="trace-seg trace-seg--asr" style="width: ${pAsr.toFixed(2)}%" title="ASR / 声纹 / 排队: ${escapeHtml(formatMs(preLlmTotal))}"></div>` : ""}
                ${pLlm > 0 ? `<div class="trace-seg trace-seg--llm" style="width: ${pLlm.toFixed(2)}%" title="${hasLlmTtft ? "LLM 首句" : "LLM 总计"}: ${escapeHtml(formatMs(llmBarMs))}"></div>` : ""}
                ${pTts > 0 ? `<div class="trace-seg trace-seg--tts" style="width: ${pTts.toFixed(2)}%" title="TTS 首块 / start_play: ${escapeHtml(formatMs(ttsHead))}"></div>` : ""}
                ${pRest > 0 ? `<div class="trace-seg trace-seg--rest" style="width: ${pRest.toFixed(2)}%" title="后续流式播报: ${escapeHtml(formatMs(tailMs))}"></div>` : ""}
              </div>
              <div class="trace-pills">${pills.join("") || `<span class="trace-pill"><span class="trace-dot trace-dot--llm"></span>LLM总计 <span class="num">${escapeHtml(formatMs(bd.llm_total_ms))}</span></span>`}</div>
            </summary>
            <div class="trace-detail">
              <div class="trace-detail-grid">
                <div class="trace-detail-kv"><span class="trace-detail-k">首音延迟 (TTFA)</span><span class="trace-detail-v num">${escapeHtml(formatMs(bd.ttfa_ms))}</span></div>
                <div class="trace-detail-kv"><span class="trace-detail-k">端到端总耗时</span><span class="trace-detail-v num">${escapeHtml(formatMs(bd.total_ms))}</span></div>
                <div class="trace-detail-kv"><span class="trace-detail-k">LLM 首句 / 总计</span><span class="trace-detail-v num">${escapeHtml(formatMs(bd.llm_ttft_ms))} / ${escapeHtml(formatMs(bd.llm_total_ms))}</span></div>
                <div class="trace-detail-kv"><span class="trace-detail-k">start_play / TTS首块</span><span class="trace-detail-v num">${escapeHtml(formatMs(bd.start_play_ms))} / ${escapeHtml(formatMs(bd.tts_first_chunk_ms))}</span></div>
                <div class="trace-detail-kv"><span class="trace-detail-k">模型</span><span class="trace-detail-v num">${escapeHtml(t.model || "—")}</span></div>
                <div class="trace-detail-kv"><span class="trace-detail-k">音色 · 分句数</span><span class="trace-detail-v num">${escapeHtml(t.voice || "—")} · ${Number(t.sentence_count || 0)} 句</span></div>
              </div>
              ${t.reply ? `<div class="trace-reply-box"><strong>回复：</strong>${escapeHtml(t.reply)}</div>` : ""}
            </div>
          </details>
        `;
      })
      .join("")}</div>`;
  }

  document.querySelectorAll("[data-trend-metric]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.trendMetric = btn.getAttribute("data-trend-metric") || "tokens";
      if (state.overview) renderTrendChart(state.overview);
    });
  });

  $("clearTracesBtn").addEventListener("click", async () => {
    try {
      await apiFetch("/api/traces", { method: "DELETE" });
      if (state.overview) state.overview.recent_traces = [];
      renderRecentTraces([]);
      showToast("已清空最近 Trace 记录");
    } catch (err) {
      showToast(`清空 Trace 失败：${err.message}`, true);
    }
  });

  function renderOverview(data) {
    state.overview = data;
    $("assistantTitle").textContent = `${data.assistant_name || "Nomi"} 控制台`;
    const stats = data.stats || {};
    const usage = data.usage || {};
    $("statTotalTurns").textContent = String(stats.total_turns ?? 0);
    $("statTurnsToday").textContent = String(stats.turns_today ?? 0);
    $("statTurnsTodayInline").textContent = String(stats.turns_today ?? 0);
    $("statAvgLatency").textContent = String(stats.avg_latency_ms ?? 0);
    $("statP95Latency").textContent = String(stats.p95_latency_ms ?? 0);
    $("statTotalTokens").textContent = formatCompactNum(usage.total_tokens || 0);
    $("statTokensSub").textContent = `今日 ${formatCompactNum(usage.today_tokens || 0)} · 输入 ${formatCompactNum(usage.input_tokens || 0)} / 输出 ${formatCompactNum(usage.output_tokens || 0)}`;
    $("statTotalCost").textContent = formatUsd(usage.total_cost_usd || 0, (usage.total_cost_usd || 0) >= 1 ? 2 : 4);
    $("statCostSub").textContent = `今日 ${formatUsd(usage.today_cost_usd || 0, 4)} · 共 ${usage.total_calls || 0} 次调用`;
    $("statCacheTokens").textContent = formatCompactNum(usage.cache_read_tokens || 0);
    $("statReasoningSub").textContent = `思考 Token ${formatCompactNum(usage.reasoning_tokens || 0)}`;

    $("configPathLabel").textContent = data.config_path || "config.yaml";
    $("ttsProviderBadge").textContent = `${data.tts_provider || "edge"} · ${data.speaker_model || "OH2P"}`;
    const spkBadge = $("speakerConnBadge");
    const spkText = $("speakerConnText");
    const spkSt = data.speaker_status || {};
    if (spkSt.speaker_connected === true) {
      spkBadge.hidden = false;
      spkBadge.className = "badge badge--success";
      const via = spkSt.relay_connections > 0 ? "中继在线" : "局域网在线";
      spkText.textContent = `音箱${via}${spkSt.active_connections > 1 ? ` (${spkSt.active_connections})` : ""}`;
    } else if (spkSt.relay_connected === true) {
      spkBadge.hidden = false;
      spkBadge.className = "badge badge--accent";
      spkText.textContent = "中继已连 · 等待音箱";
    } else if (spkSt.speaker_connected === false) {
      spkBadge.hidden = false;
      spkBadge.className = "badge badge--warn";
      spkText.textContent = "音箱离线";
    } else {
      spkBadge.hidden = true;
    }
    $("sidebarModelLabel").textContent = data.model || "—";
    $("sidebarVoiceLabel").textContent = `${data.voice_name || data.voice || "—"} (${data.rate || "+8%"})`;

    renderTrendChart(data);
    renderRecentTraces(data.recent_traces || []);

    const modelSel = $("modelSelect");
    const models = data.available_models || [];
    modelSel.innerHTML = models
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label || m.id)}</option>`)
      .join("");
    if (data.model) modelSel.value = data.model;

    const voiceSel = $("voiceSelect");
    const voices = data.available_voices || [];
    voiceSel.innerHTML = voices
      .map((v) => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.name)} · ${escapeHtml(v.style)}</option>`)
      .join("");
    if (data.voice) voiceSel.value = data.voice;

    const rateSel = $("rateSelect");
    rateSel.querySelectorAll("option[data-custom-rate]").forEach((el) => el.remove());
    const currentRate = data.rate || "+8%";
    if (![...rateSel.options].some((o) => o.value === currentRate)) {
      const opt = document.createElement("option");
      opt.value = currentRate;
      opt.setAttribute("data-custom-rate", "1");
      opt.textContent = `自定义 (${currentRate})`;
      rateSel.appendChild(opt);
    }
    rateSel.value = currentRate;

    const localKws = data.wakewords?.local || [];
    const enKws = data.wakewords?.english || [];
    $("localKeywordsInput").value = localKws.join(", ");
    $("enKeywordsInput").value = enKws.join(", ");
    $("enKwsEnabledCheck").checked = Boolean(data.wakewords?.en_kws_enabled ?? true);

    const wakeLocal = localKws.join(" / ") || "—";
    const wakeEn = data.wakewords?.en_kws_enabled
      ? (enKws.join(" / ") || "已启用")
      : "已关闭";
    const rows = [
      ["当前模型", `<span class="num">${escapeHtml(data.model)}</span>`],
      ["对话后端", escapeHtml(data.default_provider)],
      ["TTS 音色", `${escapeHtml(data.voice_name || data.voice)} <span class="num">(${escapeHtml(data.rate)})</span>`],
      ["ASR 引擎", `${escapeHtml(data.asr_provider)} / ${escapeHtml(data.asr_model)} (${escapeHtml(data.asr_language)})`],
      [
        "硬件 AEC",
        data.aec_enabled
          ? `<span class="badge badge--success">已开启 ${escapeHtml(data.capture_channels)}ch</span>`
          : `<span class="badge">未开启 ${escapeHtml(data.capture_channels)}ch</span>`,
      ],
      ["中文唤醒词", escapeHtml(wakeLocal)],
      ["英文唤醒词", escapeHtml(wakeEn)],
    ];
    $("runtimeSpecList").innerHTML = rows
      .map(
        ([k, v]) => `
          <div class="info-row">
            <span class="info-row-label">${escapeHtml(k)}</span>
            <span class="info-row-value">${v}</span>
          </div>
        `
      )
      .join("");

    const spkEntries = Object.entries(stats.speaker_breakdown || {});
    const modEntries = Object.entries(stats.model_breakdown || {});
    if (!spkEntries.length && !modEntries.length) {
      $("statsBreakdown").innerHTML = `<div class="empty-state">暂无对话统计，唤醒 Nomi 对话后将自动记录。</div>`;
    } else {
      const total = Math.max(1, Number(stats.total_turns) || 1);
      const allRows = [
        ...spkEntries.map(([name, cnt]) => [name, Number(cnt) || 0, "说话人"]),
        ...modEntries.map(([mod, cnt]) => [mod, Number(cnt) || 0, "模型"]),
      ];
      $("statsBreakdown").innerHTML = `
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>类别</th>
                <th>名称</th>
                <th class="is-num">轮次</th>
              </tr>
            </thead>
            <tbody>
              ${allRows
                .map(([label, cnt, kind]) => {
                  const pct = Math.min(100, Math.round((cnt / total) * 100));
                  return `
                    <tr>
                      <td><span class="badge">${escapeHtml(kind)}</span></td>
                      <td>
                        <div>${escapeHtml(label)}</div>
                        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                      </td>
                      <td class="is-num num">${cnt} 轮</td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    renderCapabilities(data.capabilities || []);
    const toggles = data.toggles || {};
    const rc = data.reminders_config || {};
    $("toggleReminders").setAttribute("aria-checked", String(Boolean(toggles.reminders ?? rc.enabled ?? true)));
    $("toggleMemory").setAttribute("aria-checked", String(Boolean(toggles.memory ?? true)));
    $("toggleXiaohongshu").setAttribute("aria-checked", String(Boolean(toggles.xiaohongshu ?? false)));
    $("toggleBrowser").setAttribute("aria-checked", String(Boolean(toggles.browser ?? false)));
    $("toggleOutlook").setAttribute("aria-checked", String(Boolean(toggles.outlook ?? true)));
    $("toggleHostVoiceprint").setAttribute("aria-checked", String(Boolean(toggles.require_host_voiceprint ?? false)));
    if (rc.quiet_start_hhmm) $("quietStartInput").value = rc.quiet_start_hhmm;
    if (rc.quiet_end_hhmm) $("quietEndInput").value = rc.quiet_end_hhmm;
  }

  function renderCapabilities(caps) {
    const enabledCount = caps.filter((c) => c.enabled).length;
    $("navCapsCount").textContent = `${enabledCount}/${caps.length}`;
    if (!caps.length) {
      $("capabilitiesGrid").innerHTML = `<div class="empty-state">未获取到能力配置。</div>`;
      return;
    }
    $("capabilitiesGrid").innerHTML = caps
      .map(
        (c) => `
          <article class="cap-card">
            <div class="cap-head">
              <h2 class="cap-title">${escapeHtml(c.name)}</h2>
              <span class="badge ${c.enabled ? "badge--success" : ""}">${escapeHtml(c.status_label || (c.enabled ? "已启用" : "未启用"))}</span>
            </div>
            <p class="cap-desc">${escapeHtml(c.description || "")}</p>
            <div class="cap-detail num">${escapeHtml(c.detail || "无")}</div>
          </article>
        `
      )
      .join("");
  }

  function renderHistory(data) {
    const turns = data.turns || [];
    state.history = turns;
    $("historyCountLabel").textContent = String(turns.length);
    $("navHistoryCount").textContent = String(turns.length);
    if (!turns.length) {
      $("historyList").innerHTML = `<div class="empty-state">暂无匹配的对话记录。</div>`;
      return;
    }
    $("historyList").innerHTML = `
      <div class="history-list">
        ${turns
          .map((t) => {
            const roleLabel = t.speaker_role ? ` (${t.speaker_role})` : "";
            return `
              <div class="history-item">
                <div class="history-meta">
                  <div class="tags-row">
                    <span class="badge badge--accent">${escapeHtml((t.speaker_name || "未知") + roleLabel)}</span>
                    <span class="badge">${escapeHtml(t.source || "voice")}</span>
                    ${t.voice ? `<span class="badge">${escapeHtml(t.voice)}</span>` : ""}
                    ${t.model ? `<span class="badge num">${escapeHtml(t.model)}</span>` : ""}
                  </div>
                  <span class="num">${formatTime(t.timestamp)} · ${escapeHtml(t.latency_ms)} ms</span>
                </div>
                <div class="history-q">问：${escapeHtml(t.query)}</div>
                <div class="history-a">答：${escapeHtml(t.reply)}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function updateNavMemoryBadge() {
    const total = state.memories.length + state.voiceprints.length + state.reminders.length;
    $("navMemoryCount").textContent = String(total);
  }

  function renderMemories(data) {
    const items = data.memories || [];
    state.memories = items;
    $("memoryCountLabel").textContent = String(items.length);
    $("groupMemoriesSummary").textContent = String(items.length);
    updateNavMemoryBadge();

    const chip = $("memoryStatusChip");
    chip.textContent = data.enabled ? "已启用" : "已禁用";
    chip.className = data.enabled ? "badge badge--success" : "badge";

    if (!items.length) {
      $("memoryList").innerHTML = `<div class="empty-state">暂无长期记忆，可通过语音说“记住……”或在上方添加。</div>`;
      return;
    }
    $("memoryList").innerHTML = `
      <div class="info-list">
        ${items
          .map(
            (m) => `
              <div class="info-row">
                <div class="info-row-main">
                  <div class="tags-row">
                    <span class="info-row-label">${escapeHtml(m.key)}</span>
                    <span class="badge">${escapeHtml(m.category || "fact")}</span>
                  </div>
                  <span class="info-row-sub">${escapeHtml(m.value)}</span>
                </div>
                <button type="button" class="btn btn--sm btn--danger" data-del-memory="${escapeHtml(m.id)}">删除</button>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function renderVoiceprints(data) {
    const items = data.voiceprints || [];
    state.voiceprints = items;
    $("voiceprintCountLabel").textContent = String(items.length);
    $("groupVoiceprintsSummary").textContent = String(items.length);
    updateNavMemoryBadge();

    if (!items.length) {
      $("voiceprintList").innerHTML = `<div class="empty-state">暂无已注册声纹。成员说完话后，你对音箱说“记住刚才和你说话的是…”即可录入。</div>`;
      return;
    }
    $("voiceprintList").innerHTML = `
      <div class="info-list">
        ${items
          .map((vp) => {
            const caps = Array.isArray(vp.capabilities)
              ? vp.capabilities.map((c) => `<span class="badge">${escapeHtml(c)}</span>`).join("")
              : "";
            const meta = [
              `ID: ${vp.id}`,
              vp.sample_count ? `样本 ${vp.sample_count}` : "",
              vp.preferred_voice ? `音色 ${vp.preferred_voice}` : "",
            ]
              .filter(Boolean)
              .join(" / ");
            return `
              <div class="info-row">
                <div class="info-row-main">
                  <div class="tags-row">
                    <span class="info-row-label">${escapeHtml(vp.name || vp.id)}</span>
                    ${vp.role ? `<span class="badge badge--success">${escapeHtml(vp.role)}</span>` : ""}
                    ${caps}
                  </div>
                  <span class="info-row-sub num">${escapeHtml(meta)}</span>
                  ${vp.notes ? `<span class="info-row-sub">${escapeHtml(vp.notes)}</span>` : ""}
                </div>
                <button type="button" class="btn btn--sm btn--danger" data-del-vp="${escapeHtml(vp.id)}">移除</button>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function renderReminders(data) {
    const items = data.reminders || [];
    state.reminders = items;
    $("reminderCountLabel").textContent = String(items.length);
    $("groupRemindersSummary").textContent = String(items.length);
    updateNavMemoryBadge();

    if (!items.length) {
      $("reminderList").innerHTML = `<div class="empty-state">暂无定时提醒。可对音箱说“20 分钟后提醒我…”或“每天早上 8 点半播报早报”。</div>`;
      return;
    }
    $("reminderList").innerHTML = `
      <div class="info-list">
        ${items
          .map(
            (r) => `
              <div class="info-row">
                <div class="info-row-main">
                  <div class="tags-row">
                    <span class="info-row-label">${escapeHtml(r.message || r.title || r.text || "提醒")}</span>
                    <span class="badge ${r.kind === "morning_briefing" ? "badge--accent" : ""}">${escapeHtml(r.kind === "morning_briefing" ? "智能早报" : "定时提醒")}</span>
                  </div>
                  <span class="info-row-sub num">${escapeHtml(r.daily_hhmm ? `每日 ${r.daily_hhmm}` : typeof r.trigger_at === "number" ? formatTime(r.trigger_at) : r.trigger_at || r.due_at || r.time || r.cron || "未设置")}</span>
                </div>
                <div class="toolbar-row">
                  <span class="badge ${r.enabled === false ? "" : "badge--success"}">${escapeHtml(r.status || (r.enabled === false ? "已停用" : "生效中"))}</span>
                  ${r.id ? `<button type="button" class="btn btn--sm btn--danger" data-del-reminder="${escapeHtml(r.id)}">取消</button>` : ""}
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  async function loadAll() {
    try {
      const speaker = $("speakerFilter").value.trim();
      const histUrl = speaker
        ? `/api/history?limit=100&speaker=${encodeURIComponent(speaker)}`
        : "/api/history?limit=100";
      const [overview, history, memories, voiceprints, reminders] = await Promise.all([
        apiFetch("/api/overview"),
        apiFetch(histUrl),
        apiFetch("/api/memories"),
        apiFetch("/api/voiceprints"),
        apiFetch("/api/reminders"),
      ]);
      renderOverview(overview);
      renderHistory(history);
      renderMemories(memories);
      renderVoiceprints(voiceprints);
      renderReminders(reminders);
      $("serverStatus").className = "badge badge--success";
      $("serverStatusText").textContent = "已连接";
      return true;
    } catch (err) {
      if (err && err.status === 401) {
        return false;
      }
      $("serverStatus").className = "badge badge--error";
      $("serverStatusText").textContent = "连接异常";
      showToast(`加载失败：${err.message}`, true);
      return false;
    }
  }

  $("refreshBtn").addEventListener("click", async () => {
    if (await loadAll()) {
      showToast("已刷新实时状态");
    }
  });

  $("settingsForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("saveSettingsBtn");
    btn.disabled = true;
    const payload = {
      model: $("modelSelect").value,
      edge_voice: $("voiceSelect").value,
      edge_rate: $("rateSelect").value,
      local_keywords: $("localKeywordsInput").value.trim(),
      en_keywords: $("enKeywordsInput").value.trim(),
      en_kws_enabled: $("enKwsEnabledCheck").checked,
    };
    try {
      const res = await apiFetch("/api/settings", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      showToast(`已保存并热更新：${res.model} / ${res.edge_voice} (${res.edge_rate})`);
      await loadAll();
    } catch (err) {
      showToast(`保存失败：${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  });

  $("applyFilterBtn").addEventListener("click", async () => {
    const speaker = $("speakerFilter").value.trim();
    const url = speaker
      ? `/api/history?limit=100&speaker=${encodeURIComponent(speaker)}`
      : "/api/history?limit=100";
    try {
      renderHistory(await apiFetch(url));
    } catch (err) {
      showToast(`查询失败：${err.message}`, true);
    }
  });

  $("speakerFilter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("applyFilterBtn").click();
    }
  });

  $("clearHistoryBtn").addEventListener("click", async () => {
    if (!window.confirm("确定清空所有本地对话历史记录吗？")) return;
    try {
      await apiFetch("/api/history", { method: "DELETE" });
      showToast("对话历史已清空");
      await loadAll();
    } catch (err) {
      showToast(`清空失败：${err.message}`, true);
    }
  });

  $("memoryForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const key = $("memoryKey").value.trim();
    const val = $("memoryValue").value.trim();
    if (!val && !key) {
      showToast("请填写记忆内容", true);
      return;
    }
    const body = key && val ? { key, value: val } : { fact: val || key };
    try {
      await apiFetch("/api/memories", {
        method: "POST",
        body: JSON.stringify(body),
      });
      $("memoryKey").value = "";
      $("memoryValue").value = "";
      showToast("长期记忆已保存");
      renderMemories(await apiFetch("/api/memories"));
    } catch (err) {
      showToast(`保存记忆失败：${err.message}`, true);
    }
  });

  $("memoryList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-del-memory]");
    if (!btn) return;
    const id = btn.getAttribute("data-del-memory");
    try {
      await apiFetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
      showToast("已删除该记忆");
      renderMemories(await apiFetch("/api/memories"));
    } catch (err) {
      showToast(`删除失败：${err.message}`, true);
    }
  });

  $("voiceprintList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-del-vp]");
    if (!btn) return;
    const id = btn.getAttribute("data-del-vp");
    try {
      await apiFetch(`/api/voiceprints/${encodeURIComponent(id)}`, { method: "DELETE" });
      showToast("已移除声纹成员");
      renderVoiceprints(await apiFetch("/api/voiceprints"));
    } catch (err) {
      showToast(`移除失败：${err.message}`, true);
    }
  });

  function syncReminderModeFields() {
    const mode = $("reminderMode").value;
    $("reminderDelayField").hidden = mode !== "delay";
    $("reminderDailyField").hidden = mode !== "daily";
    $("reminderOnceField").hidden = mode !== "once";
  }
  $("reminderMode").addEventListener("change", syncReminderModeFields);
  syncReminderModeFields();

  $("reminderForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("addReminderBtn");
    const message = $("reminderMessage").value.trim();
    const mode = $("reminderMode").value;
    const kind = $("reminderKind").value;
    if (!message) {
      showToast("请填写播报内容", true);
      return;
    }
    const payload = { message, kind };
    if (mode === "delay") {
      const mins = Number($("reminderMinutes").value);
      if (!Number.isFinite(mins) || mins < 1 || mins > 43200) {
        showToast("请填写 1 至 43200 分钟之间的倒计时", true);
        return;
      }
      payload.delay_seconds = Math.round(mins * 60);
    } else if (mode === "daily") {
      const hhmm = $("reminderDailyTime").value.trim().slice(0, 5);
      if (!hhmm) {
        showToast("请选择每日触发时间", true);
        return;
      }
      payload.daily_hhmm = hhmm;
    } else {
      const iso = $("reminderOnceTime").value.trim();
      if (!iso) {
        showToast("请选择触发日期时间", true);
        return;
      }
      payload.trigger_at_iso = iso.length === 16 ? `${iso}:00` : iso;
    }
    btn.disabled = true;
    try {
      await apiFetch("/api/reminders", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      $("reminderMessage").value = "";
      showToast("定时提醒已添加");
      renderReminders(await apiFetch("/api/reminders"));
    } catch (err) {
      showToast(`添加提醒失败：${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  });

  $("reminderList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-del-reminder]");
    if (!btn) return;
    const id = btn.getAttribute("data-del-reminder");
    try {
      await apiFetch(`/api/reminders/${encodeURIComponent(id)}`, { method: "DELETE" });
      showToast("已取消该提醒");
      renderReminders(await apiFetch("/api/reminders"));
    } catch (err) {
      showToast(`取消提醒失败：${err.message}`, true);
    }
  });

  document.querySelectorAll("[data-cap-toggle]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const key = btn.getAttribute("data-cap-toggle");
      const nextVal = btn.getAttribute("aria-checked") !== "true";
      btn.disabled = true;
      try {
        await apiFetch("/api/capabilities", {
          method: "POST",
          body: JSON.stringify({ [key]: nextVal }),
        });
        btn.setAttribute("aria-checked", String(nextVal));
        showToast("能力配置已保存并生效");
      } catch (err) {
        showToast(`更新能力配置失败：${err.message}`, true);
        btn.disabled = false;
        return;
      }
      try {
        await loadAll();
      } finally {
        btn.disabled = false;
      }
    });
  });

  $("quietHoursForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("saveQuietBtn");
    btn.disabled = true;
    try {
      await apiFetch("/api/capabilities", {
        method: "POST",
        body: JSON.stringify({
          quiet_start_hhmm: $("quietStartInput").value.trim().slice(0, 5),
          quiet_end_hhmm: $("quietEndInput").value.trim().slice(0, 5),
        }),
      });
      showToast("免打扰时段已保存并生效");
    } catch (err) {
      showToast(`保存免打扰失败：${err.message}`, true);
      btn.disabled = false;
      return;
    }
    try {
      await loadAll();
    } finally {
      btn.disabled = false;
    }
  });

  let lastDebugReply = "";
  let debugAudioObjectUrl = null;

  async function playDebugTts(text) {
    const clean = String(text || "").trim();
    if (!clean) {
      showToast("请先输入试听文本或测试问题", true);
      return;
    }
    const ttsBtn = $("debugTtsBtn");
    const playReplyBtn = $("debugPlayReplyBtn");
    ttsBtn.disabled = true;
    if (playReplyBtn) playReplyBtn.disabled = true;
    try {
      const voice = $("voiceSelect")?.value || state.overview?.voice || "";
      const rate = $("rateSelect")?.value || state.overview?.rate || "";
      const res = await fetch("/api/debug/tts", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean.slice(0, 120), voice, rate }),
      });
      if (res.status === 401) {
        showLoginGate("登录状态已过期，请重新验证");
        return;
      }
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err?.message) msg = err.message;
        } catch {}
        throw new Error(msg);
      }
      const blob = await res.blob();
      if (debugAudioObjectUrl) URL.revokeObjectURL(debugAudioObjectUrl);
      debugAudioObjectUrl = URL.createObjectURL(blob);
      const wrap = $("debugAudioWrap");
      const player = $("debugAudioPlayer");
      wrap.hidden = false;
      player.src = debugAudioObjectUrl;
      await player.play().catch(() => {});
      showToast("音色试听已生成");
    } catch (err) {
      showToast(`音色试听失败：${err.message}`, true);
    } finally {
      ttsBtn.disabled = false;
      if (playReplyBtn) playReplyBtn.disabled = false;
    }
  }

  $("debugAskForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("debugQueryInput");
    const query = input.value.trim();
    if (!query) {
      showToast("请输入测试问题", true);
      return;
    }
    const btn = $("debugAskBtn");
    btn.disabled = true;
    btn.textContent = "思考中…";
    try {
      const res = await apiFetch("/api/debug/ask", {
        method: "POST",
        body: JSON.stringify({ query }),
      });
      lastDebugReply = res.reply || "";
      $("debugResultModel").textContent = `${res.provider || "llm"} · ${res.model || ""}`;
      $("debugResultLatency").textContent = `${Math.round(Number(res.latency_ms) || 0)} ms`;
      $("debugResultText").textContent = lastDebugReply || "（空回复）";
      $("debugPlayReplyBtn").hidden = !lastDebugReply;
      $("debugResultBox").hidden = false;
      showToast(`调试完成 (${Math.round(Number(res.latency_ms) || 0)} ms)`);
      await loadAll();
    } catch (err) {
      showToast(`调试请求失败：${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "发送测试";
    }
  });

  $("debugTtsBtn").addEventListener("click", async () => {
    const sample = $("debugQueryInput").value.trim() || "你好，我是你的语音助手，现在音色和语速都很清晰。";
    await playDebugTts(sample);
  });

  $("debugPlayReplyBtn").addEventListener("click", async () => {
    await playDebugTts(lastDebugReply);
  });

  $("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("loginSubmitBtn");
    btn.disabled = true;
    const pw = $("loginPassword").value;
    try {
      await apiFetch("/api/login", {
        method: "POST",
        body: JSON.stringify({ password: pw }),
      });
      $("loginPassword").value = "";
      showDashboard(true);
      await loadAll();
      showToast("登录成功");
    } catch (err) {
      showLoginGate(err.message || "口令错误");
    } finally {
      btn.disabled = false;
    }
  });

  $("logoutBtn").addEventListener("click", async () => {
    await fetch("/api/logout", { method: "POST" }).catch(() => {});
    showLoginGate("");
    showToast("已退出登录");
  });

  async function bootstrap() {
    try {
      const auth = await apiFetch("/api/auth/status");
      if (auth.authenticated) {
        showDashboard(auth.auth_required);
        await loadAll();
      } else {
        showLoginGate("");
      }
    } catch (err) {
      showLoginGate(err.message || "");
    }
  }

  bootstrap();
})();
