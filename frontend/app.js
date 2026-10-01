const API_BASE = document.documentElement.dataset.apiBase || '';
const CATEGORY_COLORS = {Diet: '#53d9a5', Digital: '#1ea7e1', Electricity: '#32cbbd', Goods: '#ffbd59', Transport: '#ff6b6b', Waste: '#9671ff'};

class ApiClient {
  constructor(baseUrl, timeoutMs = 12000) {
    this.baseUrl = baseUrl;
    this.timeoutMs = timeoutMs;
  }

  async predict(payload) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const requestId = globalThis.crypto?.randomUUID?.() || `web-${Date.now()}`;
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-Request-ID': requestId},
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new ApiError(response.status, body, response.headers.get('X-Request-ID'));
      if (!body?.prediction || !body?.category_breakdown || !Array.isArray(body?.recommendations)) {
        throw new ApiError(502, null, requestId, 'The service returned an unexpected response.');
      }
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new ApiError(408, null, requestId, 'The calculation timed out. Please retry.');
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, null, requestId, navigator.onLine ? 'The service is unavailable. Please retry.' : 'You appear to be offline.');
    } finally {
      clearTimeout(timer);
    }
  }
}

class ApiError extends Error {
  constructor(status, body, requestId, fallback) {
    super(fallback || body?.error?.message || 'Unable to calculate your footprint.');
    this.status = status;
    this.fields = body?.error?.fields || {};
    this.requestId = body?.request_id || requestId;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const api = new ApiClient(API_BASE);
  const form = document.getElementById('carbonForm');
  const predictBtn = document.getElementById('predictBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('loadingSpinner');
  const resultsSection = document.getElementById('resultsSection');
  const statusRegion = document.getElementById('statusRegion');
  const errorSummary = document.getElementById('errorSummary');
  const themeBtn = document.getElementById('themeToggleBtn');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  let categoryChart = null;
  let comparisonChart = null;
  let lastResult = null;

  const preferredTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.dataset.theme = preferredTheme;
  updateThemeButton();

  themeBtn.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('theme', theme);
    updateThemeButton();
    if (lastResult) renderCharts(lastResult.category_breakdown, lastResult.comparison);
  });

  clearHistoryBtn.addEventListener('click', () => {
    localStorage.removeItem('carbonHistory');
    renderLocalHistory();
    statusRegion.textContent = 'Calculation history cleared from this browser.';
  });

  function updateThemeButton() {
    const dark = document.documentElement.dataset.theme === 'dark';
    themeBtn.setAttribute('aria-pressed', String(dark));
    themeBtn.querySelector('span').textContent = dark ? 'Dark theme' : 'Light theme';
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    clearErrors();
    if (!form.reportValidity()) return;
    setLoading(true);
    try {
      const data = await api.predict(readPayload());
      renderResults(data);
      saveLocalHistory(data);
      renderLocalHistory();
      resultsSection.className = 'results-visible';
      statusRegion.textContent = `Calculation complete. Estimated footprint ${data.prediction.total_tco2e_per_year.toFixed(2)} tonnes CO₂ equivalent per year.`;
      resultsSection.focus({preventScroll: true});
      resultsSection.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
    } catch (error) {
      showError(error instanceof ApiError ? error : new ApiError(0, null, null, 'We could not display your result. Please refresh and try again.'));
    } finally {
      setLoading(false);
    }
  });

  function readPayload() {
    const number = id => Number(document.getElementById(id).value);
    return {
      household_size: number('household_size'),
      electricity_usage_kwh: number('electricity_usage_kwh'),
      heating_source: document.getElementById('heating_source').value,
      vehicle_type: document.getElementById('vehicle_type').value,
      vehicle_km: number('vehicle_km'),
      flights_short_haul: number('flights_short_haul'),
      flights_long_haul: number('flights_long_haul'),
      diet_type: document.getElementById('diet_type').value,
      grocery_spend_monthly: number('grocery_spend_monthly'),
      waste_kg_weekly: number('waste_kg_weekly'),
      internet_usage_hours: number('internet_usage_hours'),
    };
  }

  function setLoading(loading) {
    predictBtn.disabled = loading;
    btnText.classList.toggle('hidden', loading);
    spinner.classList.toggle('hidden', !loading);
    predictBtn.setAttribute('aria-busy', String(loading));
    if (loading) statusRegion.textContent = 'Calculating your estimate…';
  }

  function clearErrors() {
    errorSummary.hidden = true;
    errorSummary.textContent = '';
    form.querySelectorAll('[aria-invalid="true"]').forEach(field => {
      field.removeAttribute('aria-invalid');
      field.removeAttribute('aria-describedby');
    });
    form.querySelectorAll('.field-error').forEach(error => error.remove());
  }

  function showError(error) {
    const requestNote = error.requestId ? ` Reference: ${error.requestId}.` : '';
    errorSummary.textContent = `${error.message}${requestNote}`;
    errorSummary.hidden = false;
    for (const [field, message] of Object.entries(error.fields || {})) {
      const input = document.getElementById(field);
      if (!input) continue;
      const id = `${field}-error`;
      const note = document.createElement('span');
      note.id = id;
      note.className = 'field-error';
      note.textContent = message;
      input.setAttribute('aria-invalid', 'true');
      input.setAttribute('aria-describedby', id);
      input.insertAdjacentElement('afterend', note);
    }
    errorSummary.focus();
    statusRegion.textContent = 'Calculation failed.';
  }

  function renderResults(data) {
    lastResult = data;
    const footprint = data.prediction.total_tco2e_per_year;
    document.getElementById('totalScore').textContent = footprint.toFixed(2);
    const grade = data.grade.letter;
    const score = document.getElementById('totalScore');
    score.className = `main-number ${grade === 'A' || grade === 'B' ? 'color-green' : grade === 'C' ? 'color-amber' : 'color-red'}`;
    const badge = document.getElementById('gradeBadge');
    badge.textContent = grade;
    badge.className = `badge ${grade}`;
    document.getElementById('percentileLabel').textContent = data.benchmark.percentile_proxy;
    document.getElementById('modelVersion').textContent = data.model.version;
    renderCharts(data.category_breakdown, data.comparison);
    renderChartAlternatives(data.category_breakdown, data.comparison);
    renderSuggestions(data.recommendations);
    renderInsight(data.category_breakdown, data.recommendations);
  }

  function renderInsight(breakdown, recommendations) {
    const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, value]) => sum + value, 0);
    const [largestName, largestValue] = entries[0];
    const share = total ? Math.round((largestValue / total) * 100) : 0;
    const action = recommendations[0]?.action || 'Review the action plan below';
    document.getElementById('insightText').textContent = `${largestName} represents ${share}% of mapped emissions. ${action}.`;
    document.getElementById('insightCount').textContent = recommendations.length;
  }

  function renderSuggestions(items) {
    const list = document.getElementById('suggestionsList');
    list.replaceChildren(...items.map(item => {
      const card = document.createElement('article');
      card.className = 'suggestion-item';
      const rank = document.createElement('div');
      rank.className = 'suggestion-rank';
      rank.textContent = `#${item.rank}`;
      const content = document.createElement('div');
      content.className = 'suggestion-content';
      const action = document.createElement('p');
      action.textContent = item.action;
      const reason = document.createElement('small');
      reason.textContent = `${item.reason} Difficulty: ${item.difficulty}.`;
      content.append(action, reason);
      const saving = document.createElement('div');
      saving.className = 'suggestion-badge';
      saving.textContent = `≈ ${item.estimated_saving_tco2e.toFixed(2)} tCO₂e`;
      card.append(rank, content, saving);
      return card;
    }));
  }

  function renderCharts(breakdown, comparison) {
    categoryChart?.destroy();
    comparisonChart?.destroy();
    renderCategoryLedger(breakdown);
    if (!globalThis.Chart) {
      document.getElementById('categoryChart').hidden = true;
      document.getElementById('comparisonChart').hidden = true;
      return;
    }
    document.getElementById('categoryChart').hidden = false;
    document.getElementById('comparisonChart').hidden = false;
    const motion = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    const styles = getComputedStyle(document.body);
    const text = styles.getPropertyValue('--paper').trim();
    const muted = styles.getPropertyValue('--paper-dim').trim();
    const line = styles.getPropertyValue('--line').trim();
    const surface = styles.getPropertyValue('--surface').trim();
    const labels = Object.keys(breakdown);
    categoryChart = new Chart(document.getElementById('categoryChart'), {
      type: 'doughnut',
      data: {labels, datasets: [{data: Object.values(breakdown), backgroundColor: labels.map(label => CATEGORY_COLORS[label] || '#8a9a91'), borderColor: surface, borderWidth: 3, hoverOffset: 5}]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: motion ? {duration: 700, easing: 'easeOutQuart'} : false,
        cutout: '72%',
        plugins: {
          legend: {display: false},
          tooltip: {displayColors: true, callbacks: {label: context => ` ${context.label}: ${context.parsed.toFixed(2)} tCO₂e`}},
        },
      },
    });
    comparisonChart = new Chart(document.getElementById('comparisonChart'), {
      type: 'bar',
      data: {
        labels: ['Your estimate', 'India', 'Global'],
        datasets: [{
          label: 'tCO₂e / year',
          data: [comparison.your_value, comparison.india_avg, comparison.world_avg],
          backgroundColor: ['#b7ff4a', '#356052', '#53d9a5'],
          borderRadius: 6,
          borderSkipped: false,
          maxBarThickness: 72,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: motion ? {duration: 700, easing: 'easeOutQuart'} : false,
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: context => ` ${context.parsed.y.toFixed(2)} tCO₂e / year`}},
        },
        scales: {
          y: {beginAtZero: true, border: {display: false}, grid: {color: line}, ticks: {color: muted, font: {family: 'JetBrains Mono', size: 10}}},
          x: {border: {display: false}, grid: {display: false}, ticks: {color: text, font: {family: 'JetBrains Mono', size: 10}}},
        },
      },
    });
  }

  function renderChartAlternatives(breakdown, comparison) {
    const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, value]) => sum + value, 0);
    const [largestName, largestValue] = entries[0];
    const share = total ? Math.round((largestValue / total) * 100) : 0;
    document.getElementById('breakdownTotal').textContent = total.toFixed(2);
    document.getElementById('categorySummary').textContent = `${largestName} is your largest source, accounting for ${share}% of this estimate.`;
    const delta = comparison.your_value - comparison.world_avg;
    const direction = delta >= 0 ? 'above' : 'below';
    document.getElementById('comparisonSummary').textContent = `Your estimate is ${Math.abs(delta).toFixed(2)} tCO₂e ${direction} the configured global benchmark.`;
  }

  function renderCategoryLedger(breakdown) {
    const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
    const ledger = document.getElementById('categoryLegend');
    ledger.replaceChildren(...entries.map(([name, value]) => {
      const item = document.createElement('div');
      item.className = 'ledger-item';
      const label = document.createElement('span');
      label.className = 'ledger-label';
      const swatch = document.createElement('i');
      swatch.style.backgroundColor = CATEGORY_COLORS[name] || '#8a9a91';
      const title = document.createElement('span');
      title.textContent = name;
      label.append(swatch, title);
      const amount = document.createElement('strong');
      amount.textContent = value.toFixed(2);
      item.append(label, amount);
      return item;
    }));
  }

  function saveLocalHistory(data) {
    const history = readLocalHistory();
    history.unshift({createdAt: new Date().toISOString(), total: data.prediction.total_tco2e_per_year, grade: data.grade.letter, breakdown: data.category_breakdown});
    try {
      localStorage.setItem('carbonHistory', JSON.stringify(history.slice(0, 10)));
    } catch {
      // A blocked or full local store must not turn a successful calculation into an error.
    }
  }

  function renderLocalHistory() {
    const history = readLocalHistory();
    const rows = history.map(record => {
      const row = document.createElement('tr');
      const top = Object.entries(record.breakdown).sort((a, b) => b[1] - a[1])[0]?.[0] || '—';
      [new Date(record.createdAt).toLocaleDateString(), record.total.toFixed(2), record.grade, top].forEach(value => {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.appendChild(cell);
      });
      return row;
    });
    document.getElementById('historyBody').replaceChildren(...rows);
    document.getElementById('historyEmpty').hidden = rows.length > 0;
    clearHistoryBtn.hidden = rows.length === 0;
  }

  function readLocalHistory() {
    let stored;
    try {
      stored = JSON.parse(localStorage.getItem('carbonHistory') || '[]');
    } catch {
      return [];
    }
    if (!Array.isArray(stored)) return [];
    const history = stored.map(normalizeHistoryRecord).filter(Boolean).slice(0, 10);
    try {
      localStorage.setItem('carbonHistory', JSON.stringify(history));
    } catch {
      // Reading history still works when browser storage becomes unavailable.
    }
    return history;
  }

  function normalizeHistoryRecord(record) {
    if (!record || typeof record !== 'object') return null;
    const legacy = record.prediction && typeof record.prediction === 'object' ? record.prediction : null;
    const breakdown = record.breakdown || legacy?.category_breakdown;
    const total = Number(record.total ?? legacy?.total_footprint_tco2e);
    const grade = record.grade || legacy?.comparison?.grade;
    const createdAt = record.createdAt || record.timestamp || new Date().toISOString();
    if (!breakdown || typeof breakdown !== 'object' || !Number.isFinite(total) || typeof grade !== 'string') return null;
    const cleanBreakdown = Object.fromEntries(Object.entries(breakdown).filter(([, value]) => Number.isFinite(Number(value))).map(([key, value]) => [key, Number(value)]));
    if (!Object.keys(cleanBreakdown).length) return null;
    return {createdAt, total, grade, breakdown: cleanBreakdown};
  }

  renderLocalHistory();
});
