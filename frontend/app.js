document.addEventListener('DOMContentLoaded', () => {
    
    // -- Theme Handling --
    const themeBtn = document.getElementById('themeToggleBtn');
    const htmlEl = document.documentElement;
    // defaults to dark in HTML, let's allow toggling
    themeBtn.addEventListener('click', () => {
        if(htmlEl.getAttribute('data-theme') === 'dark') {
            htmlEl.removeAttribute('data-theme');
        } else {
            htmlEl.setAttribute('data-theme', 'dark');
        }
    });

    // -- Form Handling --
    const form = document.getElementById('carbonForm');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('loadingSpinner');
    const predictBtn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');

    // Chart instances to destroy and recreate
    let categoryChartInst = null;
    let comparisonChartInst = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Collect data
        const payload = {
            household_size: parseInt(document.getElementById('household_size').value),
            electricity_usage_kwh: parseFloat(document.getElementById('electricity_usage_kwh').value),
            heating_source: document.getElementById('heating_source').value,
            vehicle_type: document.getElementById('vehicle_type').value,
            vehicle_km: parseFloat(document.getElementById('vehicle_km').value),
            flights_short_haul: parseInt(document.getElementById('flights_short_haul').value),
            flights_long_haul: parseInt(document.getElementById('flights_long_haul').value),
            diet_type: document.getElementById('diet_type').value,
            grocery_spend_monthly: parseFloat(document.getElementById('grocery_spend_monthly').value),
            waste_kg_weekly: parseFloat(document.getElementById('waste_kg_weekly').value),
            internet_usage_hours: parseFloat(document.getElementById('internet_usage_hours').value)
        };

        // UI State: Loading
        predictBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if(!response.ok) throw new Error("API Error");

            const data = await response.json();
            
            // Render Results
            renderResults(data);
            
            // Render History
            fetchAndRenderHistory();
            
            // Ensure section is visible smoothly
            resultsSection.classList.remove('results-hidden');
            resultsSection.classList.add('results-visible');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (err) {
            alert('Failed to connect to the prediction API. Ensure backend is running.');
            console.error(err);
        } finally {
            // Revert Button
            predictBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });

    function renderResults(data) {
        // Main score
        const scoreEl = document.getElementById('totalScore');
        const footprint = data.total_footprint_tco2e;
        
        // Animate counter
        const duration = 1500;
        const startTime = performance.now();
        function animateCounter(currentTime) {
            const progress = Math.min((currentTime - startTime) / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            scoreEl.innerText = (footprint * easeOut).toFixed(2);
            if (progress < 1) requestAnimationFrame(animateCounter);
            else scoreEl.innerText = footprint.toFixed(2);
        }
        requestAnimationFrame(animateCounter);
        
        // Color coding
        scoreEl.className = 'main-number'; // Reset
        if(data.comparison.grade === 'A' || data.comparison.grade === 'B') scoreEl.classList.add('color-green');
        else if (data.comparison.grade === 'C') scoreEl.classList.add('color-amber');
        else scoreEl.classList.add('color-red');

        // Badges
        const gradeBdg = document.getElementById('gradeBadge');
        gradeBdg.innerText = data.comparison.grade;
        gradeBdg.className = 'badge ' + data.comparison.grade;
        document.getElementById('percentileLabel').innerText = data.comparison.percentile;

        // Render Charts
        renderCharts(data.category_breakdown, data.comparison);

        // Render Suggestions
        const listDiv = document.getElementById('suggestionsList');
        listDiv.innerHTML = ''; // clear

        data.suggestions.forEach(sug => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.innerHTML = `
                <div class="suggestion-rank">#${sug.rank}</div>
                <div class="suggestion-content">
                    <p>${sug.action}</p>
                    <div class="suggestion-meta">
                        <span>Category: ${sug.category}</span>
                        <span>Complexity: <strong>${sug.difficulty}</strong></span>
                    </div>
                </div>
                <div class="suggestion-badge">- ${sug.co2_saved_tyr} tCO₂e</div>
            `;
            listDiv.appendChild(item);
        });
    }

    function renderCharts(breakdown, comp) {
        // Destroy old ones if exist
        if(categoryChartInst) categoryChartInst.destroy();
        if(comparisonChartInst) comparisonChartInst.destroy();

        // Doughnut Chart Setup
        const catCtx = document.getElementById('categoryChart').getContext('2d');
        categoryChartInst = new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(breakdown),
                datasets: [{
                    data: Object.values(breakdown),
                    backgroundColor: [
                        '#10b981', '#0ea5e9', '#2dd4bf', '#f59e0b', '#ef4444', '#8b5cf6'
                    ],
                    hoverOffset: 20,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                cutout: '70%',
                plugins: {
                    legend: { 
                        position: 'bottom', 
                        labels: { 
                            color: getComputedStyle(document.body).getPropertyValue('--text-main'),
                            padding: 20,
                            font: { family: 'Outfit', size: 12, weight: '600' }
                        } 
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true
                }
            }
        });

        // Bar Chart Setup
        const compCtx = document.getElementById('comparisonChart').getContext('2d');
        comparisonChartInst = new Chart(compCtx, {
            type: 'bar',
            data: {
                labels: ['You', 'India Avg', 'World Avg'],
                datasets: [{
                    label: 'Emissions tCO₂e',
                    data: [comp.your_value, comp.india_avg, comp.world_avg],
                    backgroundColor: [
                        comp.your_value > comp.world_avg ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)',
                        'rgba(14, 165, 233, 0.8)',
                        'rgba(45, 212, 191, 0.8)'
                    ],
                    borderRadius: 12,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { family: 'Inter', weight: '600' } }
                    }
                }
            }
        });
    }

    async function fetchAndRenderHistory() {
        const emptyRow = document.getElementById('historyEmptyRow');
        try {
            // First look in localStorage to cache offline history (Bonus)
            let localHistStr = localStorage.getItem('carbonHistory');
            let localHist = localHistStr ? JSON.parse(localHistStr) : [];
            
            // Fetch from server
            const res = await fetch('/history');
            if(res.ok) {
                const apiHist = await res.json();
                localStorage.setItem('carbonHistory', JSON.stringify(apiHist));
                localHist = apiHist;
            }

            const tbody = document.getElementById('historyBody');
            // Clear all rows except the empty-state row
            Array.from(tbody.querySelectorAll('tr:not(#historyEmptyRow)')).forEach(r => r.remove());
            
            if(!localHist || localHist.length === 0) {
                if(emptyRow) emptyRow.style.display = '';
                return;
            }

            // Has records — hide empty state
            if(emptyRow) emptyRow.style.display = 'none';
            
            localHist.forEach((record, idx) => {
                // Determine top category
                const bd = record.prediction.category_breakdown;
                const topCat = Object.keys(bd).reduce((a, b) => bd[a] > bd[b] ? a : b);
                
                // Use stored timestamp if available, else today
                const dateStr = record.timestamp
                    ? new Date(record.timestamp).toLocaleDateString()
                    : new Date().toLocaleDateString();

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${dateStr}</td>
                    <td><strong>${record.prediction.total_footprint_tco2e.toFixed(2)}</strong></td>
                    <td><span class="badge ${record.prediction.comparison.grade}">${record.prediction.comparison.grade}</span></td>
                    <td>${topCat}</td>
                `;
                tbody.insertBefore(tr, tbody.querySelector('#historyEmptyRow'));
            });

        } catch(err) {
            console.log('Using local storage history due to fetch failure');
            if(emptyRow) emptyRow.style.display = '';
        }
    }

    // Initial fetch so page reload shows history immediately
    fetchAndRenderHistory();
});
