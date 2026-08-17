// Global chart variables are defined in common.js

// Format helper
function formatGoldPrice(price) {
    if (!price) return '--';
    return parseFloat(price).toLocaleString('vi-VN', { maximumFractionDigits: 0 });
}

async function fetchGoldData() {
    if (isFetchingGold) return;
    isFetchingGold = true;
    try {
        const ts = Date.now();
        const [resGold, resHist] = await Promise.all([
            fetch(`/api/gold?_ts=${ts}`),
            fetch(`/api/gold/prediction-history?_ts=${ts}`)
        ]);
        
        if (resGold.ok) {
            goldData = await resGold.json();
        } else {
            console.error('Error fetching gold prices:', resGold.statusText);
            showToast("Lỗi tải dữ liệu giá vàng từ máy chủ", "error");
        }
        
        if (resHist.ok) {
            goldPredictionHistory = await resHist.json();
        } else {
            console.error('Error fetching gold prediction history:', resHist.statusText);
        }
    } catch (err) {
        console.error('Connection error fetching gold prices:', err);
        showToast("Mất kết nối tải dữ liệu giá vàng", "error");
    } finally {
        isFetchingGold = false;
    }
}

async function refreshGoldAnalysis() {
    const btn = document.getElementById('refresh-btn');
    if (!btn || btn.disabled) return;
    const icon = btn.querySelector('i');
    
    btn.disabled = true;
    btn.style.opacity = '0.6';
    btn.style.cursor = 'wait';
    if (icon) icon.classList.add('fa-spin');

    showToast("Đang cập nhật dữ liệu & chạy lại mô hình AI...", "info");

    try {
        const response = await fetch('/api/gold/refresh', { method: 'POST' });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || response.statusText);
        }
        goldData = await response.json();

        // Fetch updated prediction history
        const histRes = await fetch(`/api/gold/prediction-history?_ts=${Date.now()}`);
        if (histRes.ok) {
            goldPredictionHistory = await histRes.json();
        }

        renderGoldSidebar();
        renderGoldPriceDashboard();
        showToast("Cập nhật dữ liệu thành công!", "success");
    } catch (err) {
        console.error('Error refreshing gold prices:', err);
        showToast("Lỗi khi tải lại giá vàng: " + err.message, "error");
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        if (icon) icon.classList.remove('fa-spin');
    }
}

function renderGoldSidebar() {
    const newsContainer = document.getElementById('gold-sidebar-news-list');
    if (!newsContainer) return;
    newsContainer.innerHTML = '';
    
    if (!goldData || !goldData.news || goldData.news.length === 0) {
        newsContainer.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--text-muted);font-size:0.8rem;">Không tìm thấy tin tức vĩ mô liên quan.</div>`;
        return;
    }
    
    goldData.news.forEach((news, idx) => {
        const card = document.createElement('a');
        card.href = news.link;
        card.target = '_blank';
        card.classList.add('news-item-card');
        card.style.animationDelay = `${idx * 0.05}s`;
        card.classList.add('row-fade-in');
        
        const kwBadges = news.keywords.slice(0, 3).map(kw => `
            <span style="font-size: 0.6rem; padding: 0.1rem 0.3rem; background: rgba(251,191,36,0.08); color: #fbbf24; border: 1px solid rgba(251,191,36,0.15); border-radius: 3px; font-weight: 600; text-transform: uppercase;">${kw}</span>
        `).join('');
        
        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 0.7rem; color: var(--text-muted); font-weight: 500;">${news.time}</span>
                <div style="display: flex; gap: 0.25rem;">
                    ${kwBadges}
                </div>
            </div>
            <h4 style="font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin: 0; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">${news.title}</h4>
            <p style="font-size: 0.72rem; color: var(--text-secondary); margin: 0.35rem 0 0 0; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">${news.description}</p>
        `;
        newsContainer.appendChild(card);
    });
}

function renderGoldGLDComparisonChart(history) {
    const canvas = document.getElementById('gold-gld-comparison-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (!history || history.length === 0) return;
    
    const recentHistory = history.slice(-30);
    
    const labels = recentHistory.map(h => {
        const parts = h.date.split('-');
        return `${parts[2]}/${parts[1]}`;
    });
    
    const goldDataPoints = recentHistory.map(h => h.world_price || 0.0);
    const gldDataPoints = recentHistory.map(h => (h.gld || 0.0) * 50); // Convert GLDM (1/50 oz) to 1 oz
    const gldTrustDataPoints = recentHistory.map(h => (h.gld_trust || 0.0) * 10); // Convert GLD (1/10 oz) to 1 oz
    
    if (goldGldChart) {
        goldGldChart.data.labels = labels;
        goldGldChart.data.datasets[0].data = goldDataPoints;
        goldGldChart.data.datasets[1].data = gldDataPoints;
        goldGldChart.data.datasets[2].data = gldTrustDataPoints;
        goldGldChart.update('none');
        return;
    }
    
    goldGldChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Giá Vàng (XAU/USD)',
                    data: goldDataPoints,
                    borderColor: '#fbbf24',
                    backgroundColor: 'rgba(251, 191, 38, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.15,
                    yAxisID: 'y'
                },
                {
                    label: 'SPDR Gold MiniShares (GLDM) x50',
                    data: gldDataPoints,
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.15,
                    yAxisID: 'y'
                },
                {
                    label: 'SPDR Gold Trust (GLD) x10',
                    data: gldTrustDataPoints,
                    borderColor: '#34d399',
                    backgroundColor: 'rgba(52, 211, 153, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.15,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 10, family: 'Inter' }
                    }
                },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            const val = context.parsed.y;
                            if (val !== null) {
                                if (context.datasetIndex === 0) {
                                    label += '$' + val.toLocaleString('en-US', {maximumFractionDigits: 2}) + ' /oz';
                                } else if (context.datasetIndex === 1) {
                                    const orig = val / 50;
                                    label += '$' + orig.toLocaleString('en-US', {maximumFractionDigits: 2}) + ' (Quy đổi: $' + val.toLocaleString('en-US', {maximumFractionDigits: 2}) + ' /oz)';
                                } else if (context.datasetIndex === 2) {
                                    const orig = val / 10;
                                    label += '$' + orig.toLocaleString('en-US', {maximumFractionDigits: 2}) + ' (Quy đổi: $' + val.toLocaleString('en-US', {maximumFractionDigits: 2}) + ' /oz)';
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: { size: 9, family: 'Inter' }
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Giá Quy Đổi sang USD/ounce ($/oz)',
                        color: '#fbbf24',
                        font: { size: 9, family: 'Inter', weight: '600' }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: { size: 9, family: 'Inter' }
                    }
                }
            }
        }
    });
}

function renderDXYHistoryChart(history) {
    const canvas = document.getElementById('dxy-history-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (!history || history.length === 0) return;
    const recentHistory = history.slice(-30);
    
    const labels = recentHistory.map(h => {
        const parts = h.date.split('-');
        return `${parts[2]}/${parts[1]}`;
    });
    
    const dxyData = recentHistory.map(h => h.dxy || 100.0);
    
    if (dxyChart) {
        dxyChart.data.labels = labels;
        dxyChart.data.datasets[0].data = dxyData;
        dxyChart.update('none');
        return;
    }
    
    dxyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Dollar Index (DXY)',
                data: dxyData,
                borderColor: '#60a5fa',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: false,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } }
            }
        }
    });
}

function renderOilHistoryChart(history) {
    const canvas = document.getElementById('oil-history-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (!history || history.length === 0) return;
    const recentHistory = history.slice(-30);
    
    const labels = recentHistory.map(h => {
        const parts = h.date.split('-');
        return `${parts[2]}/${parts[1]}`;
    });
    
    const oilData = recentHistory.map(h => h.brent || 75.0);
    
    if (oilChart) {
        oilChart.data.labels = labels;
        oilChart.data.datasets[0].data = oilData;
        oilChart.update('none');
        return;
    }
    
    oilChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Brent (USD)',
                data: oilData,
                borderColor: '#fb923c',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: false,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } }
            }
        }
    });
}

function renderDJIHistoryChart(history) {
    const canvas = document.getElementById('dji-history-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (!history || history.length === 0) return;
    const recentHistory = history.slice(-30);
    
    const labels = recentHistory.map(h => {
        const parts = h.date.split('-');
        return `${parts[2]}/${parts[1]}`;
    });
    
    const djiData = recentHistory.map(h => h.dji || 35000.0);
    
    if (djiChart) {
        djiChart.data.labels = labels;
        djiChart.data.datasets[0].data = djiData;
        djiChart.update('none');
        return;
    }
    
    djiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Dow Jones (DJI)',
                data: djiData,
                borderColor: '#34d399',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: false,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 } } }
            }
        }
    });
}

function renderGoldPriceDashboard() {
    if (!goldData) return;

    // --- Render US Macro Indicators ---
    if (goldData.macro_indicators) {
        const mi = goldData.macro_indicators;
        const INDICATORS = [
            { key: 'fed_rate',       prefix: 'fed',   isPercent: true },
            { key: 'cpi',            prefix: 'cpi',   isPercent: true },
            { key: 'nonfarm',        prefix: 'nfp',   isPercent: false, suffix: ' nghìn' },
            { key: 'unemployment',   prefix: 'unemp', isPercent: true },
            { key: 'new_home_sales', prefix: 'home',  isPercent: false, suffix: ' nghìn căn/tháng' }
        ];
        
        INDICATORS.forEach(cfg => {
            const ind = mi[cfg.key];
            if (!ind) return;
            
            const valEl = document.getElementById(`macro-${cfg.prefix}-value`);
            const trendEl = document.getElementById(`macro-${cfg.prefix}-trend`);
            const dateEl = document.getElementById(`macro-${cfg.prefix}-date`);
            
            if (valEl) valEl.textContent = ind.value !== null ? `${ind.value}${cfg.isPercent ? '%' : ''}${cfg.suffix || ''}` : '--';
            if (dateEl) dateEl.textContent = ind.date || '--';
            if (trendEl) {
                const trend = ind.trend || 'flat';
                const trendLabels = { up: 'Tăng', down: 'Giảm', flat: 'Không đổi' };
                const trendColors = { up: '#34d399', down: '#f87171', flat: '#94a3b8' };
                const trendBgs = { up: 'rgba(16, 185, 129, 0.1)', down: 'rgba(239, 68, 68, 0.1)', flat: 'rgba(148, 163, 184, 0.1)' };
                
                trendEl.textContent = trendLabels[trend];
                trendEl.style.color = trendColors[trend];
                trendEl.style.background = trendBgs[trend];
            }
        });
    }

    // --- Render Crude Oil Prices ---
    if (goldData.crude_oil) {
        const oil = goldData.crude_oil;
        const wtiEl = document.getElementById('macro-oil-wti');
        const brentEl = document.getElementById('macro-oil-brent');
        if (wtiEl) wtiEl.textContent = oil.wti && oil.wti.price ? `${oil.wti.price.toFixed(2)} USD/thùng` : '-- USD/thùng';
        if (brentEl) brentEl.textContent = oil.brent && oil.brent.price ? `${oil.brent.price.toFixed(2)} USD/thùng` : '-- USD/thùng';
    }

    // --- Render Geopolitical Conflicts ---
    const conflictsList = document.getElementById('macro-conflicts-list');
    if (conflictsList) {
        conflictsList.innerHTML = '';
        const conflicts = goldData.conflict_events || [];
        
        if (conflicts.length === 0) {
            conflictsList.innerHTML = `<div style="font-size:0.78rem;color:var(--text-muted);text-align:center;padding:1rem;">Hiện chưa ghi nhận xung đột mới nổi bật.</div>`;
        } else {
            conflicts.forEach(c => {
                let badgeColor = 'var(--text-secondary)';
                if (c.status === 'Giao tranh quân sự') badgeColor = '#f87171';
                else if (c.status === 'Leo thang căng thẳng') badgeColor = '#fbbf24';
                else if (c.status === 'Đàm phán hòa bình') badgeColor = '#34d399';
                else if (c.status === 'Trừng phạt ngoại giao') badgeColor = '#a78bfa';
                
                const item = document.createElement('div');
                item.style.padding = '0.45rem 0.55rem';
                item.style.background = 'rgba(255,255,255,0.02)';
                item.style.border = '1px solid rgba(255,255,255,0.05)';
                item.style.borderRadius = '5px';
                item.style.fontSize = '0.78rem';
                item.style.marginBottom = '0.35rem';
                
                item.innerHTML = `
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
                        <span style="font-size:0.62rem;font-weight:700;color:#60a5fa;text-transform:uppercase;">${c.parties}</span>
                        <span style="font-size:0.6rem;font-weight:600;color:${badgeColor};">${c.status}</span>
                    </div>
                    <div style="font-weight:600;line-height:1.3;color:var(--text-primary);margin-bottom:0.15rem;">
                        <a href="${c.link}" target="_blank" style="color:inherit;text-decoration:none;">${c.title}</a>
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);display:flex;justify-content:space-between;">
                        <span>${c.time}</span>
                        <a href="${c.link}" target="_blank" style="color:#60a5fa;text-decoration:none;">Chi tiết ➔</a>
                    </div>
                `;
                conflictsList.appendChild(item);
            });
        }
    }
    
    // --- Render Prices ---
    const worldPriceEl = document.getElementById('gold-world-price-val');
    if (worldPriceEl) worldPriceEl.textContent = `${goldData.world.price} USD`;
    
    const chgVal = goldData.world.change || '0.00';
    const chgPct = goldData.world.change_pct || '0.00%';
    const isUp = !chgVal.startsWith('-');
    const chgClass = isUp ? 'text-up' : 'text-down';
    const chgSign = (isUp && !chgVal.startsWith('+')) ? '+' : '';
    
    const worldChangeValEl = document.getElementById('gold-world-change-val');
    if (worldChangeValEl) worldChangeValEl.textContent = `${chgSign}${chgVal} USD`;
    
    const worldChangePctEl = document.getElementById('gold-world-change-pct');
    if (worldChangePctEl) worldChangePctEl.textContent = `(${chgPct})`;
    
    const changeEl = document.getElementById('gold-world-change');
    if (changeEl) changeEl.className = `detail-change ${chgClass}`;
    
    const convertedCayEl = document.getElementById('gold-world-converted-cay');
    if (convertedCayEl) convertedCayEl.textContent = goldData.world.converted_cay ? `${goldData.world.converted_cay}đ` : '-';
    
    const barPricesEl = document.getElementById('gold-sjc-bar-prices');
    if (barPricesEl) {
        const barBuy = goldData.domestic.sjc_bar.buy || '-';
        const barSell = goldData.domestic.sjc_bar.sell || '-';
        barPricesEl.innerHTML = `<span style="color:var(--text-primary)">${barBuy}đ</span> <span style="color:var(--text-muted);font-size:0.9rem;font-weight:400;">/</span> <span style="color:var(--text-primary)">${barSell}đ</span>`;
    }
    
    const barTimeEl = document.getElementById('gold-sjc-bar-time');
    if (barTimeEl) barTimeEl.textContent = `Cập nhật lúc: ${goldData.domestic.time || 'N/A'}`;
    
    const ringPricesEl = document.getElementById('gold-sjc-ring-prices');
    if (ringPricesEl) {
        const ringBuy = goldData.domestic.sjc_ring.buy || '-';
        const ringSell = goldData.domestic.sjc_ring.sell || '-';
        ringPricesEl.innerHTML = `<span style="color:var(--text-primary)">${ringBuy}đ</span> <span style="color:var(--text-muted);font-size:0.9rem;font-weight:400;">/</span> <span style="color:var(--text-primary)">${ringSell}đ</span>`;
    }
    
    const ringTimeEl = document.getElementById('gold-sjc-ring-time');
    if (ringTimeEl) ringTimeEl.textContent = `Cập nhật lúc: ${goldData.domestic.time || 'N/A'}`;
    
    // Premium Diff
    let premiumText = '-';
    if (goldData.domestic.sjc_bar.sell && goldData.world.converted_cay) {
        const sjcSellFloat = parseFloat(goldData.domestic.sjc_bar.sell.replace(/\./g, '')) * 1000;
        const worldCayFloat = parseFloat(goldData.world.converted_cay.replace(/\./g, ''));
        if (!isNaN(sjcSellFloat) && !isNaN(worldCayFloat)) {
            const diff = sjcSellFloat - worldCayFloat;
            const diffM = (diff / 1000000).toFixed(2);
            const sign = diff >= 0 ? '+' : '';
            premiumText = `${sign}${diffM} triệu/lượng`;
            
            const premiumCard = document.getElementById('gold-premium-card');
            const premiumLabel = document.getElementById('gold-premium-label');
            if (premiumCard && premiumLabel) {
                if (diff >= 0) {
                    premiumCard.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(220, 38, 38, 0.05) 100%)';
                    premiumCard.style.borderColor = 'rgba(239, 68, 68, 0.15)';
                    premiumLabel.style.color = '#f87171';
                } else {
                    premiumCard.style.background = 'linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(5, 150, 105, 0.05) 100%)';
                    premiumCard.style.borderColor = 'rgba(16, 185, 129, 0.15)';
                    premiumLabel.style.color = '#34d399';
                }
            }
        }
    }
    const premiumValEl = document.getElementById('gold-premium-val');
    if (premiumValEl) premiumValEl.textContent = premiumText;
    
    // Brands
    const tbodyBrands = document.getElementById('gold-brands-tbody');
    if (tbodyBrands) {
        tbodyBrands.innerHTML = '';
        const brands = goldData.domestic.brands || [];
        if (brands.length === 0) {
            tbodyBrands.innerHTML = `<tr><td colspan="4" style="text-align:center;padding:1.5rem;color:var(--text-muted)">Không có dữ liệu so sánh thương hiệu.</td></tr>`;
        } else {
            brands.forEach(b => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="color:var(--text-muted);font-weight:500;">${b.region}</td>
                    <td style="font-weight:600;">${b.brand}</td>
                    <td class="price-col" style="text-align:right;color:var(--color-buy)">${b.buy}đ</td>
                    <td class="price-col" style="text-align:right;color:var(--color-sell)">${b.sell}đ</td>
                `;
                tbodyBrands.appendChild(tr);
            });
        }
    }
    
    // US Economic Calendar
    const tbodyCal = document.getElementById('gold-calendar-tbody');
    if (tbodyCal && goldData.calendar) {
        tbodyCal.innerHTML = '';
        const todayStr = '2026-06-22';
        
        const upcomingEvents = goldData.calendar.filter(e => e.date >= todayStr).sort((a, b) => a.date.localeCompare(b.date));
        const pastEvents = goldData.calendar.filter(e => e.date < todayStr).sort((a, b) => b.date.localeCompare(a.date));
        const sortedEvents = [...upcomingEvents, ...pastEvents];
        
        sortedEvents.forEach(e => {
            const isUpcoming = e.date >= todayStr;
            const statusBadge = isUpcoming 
                ? `<span class="badge" style="font-size:0.68rem;padding:0.15rem 0.4rem;background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2);">SẮP DIỄN RA</span>`
                : `<span class="badge" style="font-size:0.68rem;padding:0.15rem 0.4rem;background:rgba(255,255,255,0.05);color:var(--text-muted);border:1px solid rgba(255,255,255,0.08);">ĐÃ QUA</span>`;
            
            const catColors = { 'FED': '#60a5fa', 'CPI': '#34d399', 'Việc làm': '#a78bfa' };
            const catColor = catColors[e.category] || 'var(--text-primary)';
            
            let ruleText = e.impact || '';
            if (e.category === 'FED') ruleText = 'Lãi suất tăng ➔ Vàng giảm; Lãi suất giảm ➔ Vàng tăng';
            else if (e.category === 'CPI') ruleText = 'CPI tăng ➔ Vàng tăng; CPI giảm ➔ Vàng giảm';
            else if (e.category === 'Việc làm') ruleText = 'NFP tăng ➔ Vàng giảm; NFP giảm ➔ Vàng tăng';
            
            const tr = document.createElement('tr');
            if (isUpcoming) tr.style.background = 'rgba(16, 185, 129, 0.015)';
            tr.innerHTML = `
                <td style="font-family:'JetBrains Mono', monospace;color:var(--text-muted);font-size:0.78rem;">${e.date}</td>
                <td style="font-weight:600;color:var(--text-primary);">${e.event}</td>
                <td><span style="color:${catColor};font-weight:600;font-size:0.75rem;">${e.category}</span></td>
                <td>${statusBadge}</td>
                <td style="font-size:0.75rem;color:var(--text-secondary);font-weight:500;">${ruleText}</td>
            `;
            tbodyCal.appendChild(tr);
        });
    }

    // Render Macro history charts
    if (goldData.predictions && goldData.predictions.gold_history) {
        renderGoldGLDComparisonChart(goldData.predictions.gold_history);
        renderDXYHistoryChart(goldData.predictions.gold_history);
        renderOilHistoryChart(goldData.predictions.gold_history);
        renderDJIHistoryChart(goldData.predictions.gold_history);
    }
}

function loadTradingViewGoldWidget() {
    if (tradingViewWidgetLoaded) return;
    const container = document.getElementById('tradingview-gold-widget');
    if (!container) return;
    
    tradingViewWidgetLoaded = true;
    container.innerHTML = '';
    
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.innerHTML = JSON.stringify({
        "width": "100%",
        "height": "100%",
        "symbol": "FX_IDC:XAUUSD",
        "interval": "15",
        "timezone": "Asia/Ho_Chi_Minh",
        "theme": "dark",
        "style": "1",
        "locale": "vi_VN",
        "backgroundColor": "#0c1120",
        "gridColor": "rgba(255, 255, 255, 0.04)",
        "withdateranges": true,
        "hide_side_toolbar": true,
        "allow_symbol_change": false,
        "details": false,
        "hotlist": false,
        "calendar": false,
        "support_host": "https://www.tradingview.com"
    });
    container.appendChild(script);
}

// DOM ready initialization
document.addEventListener("DOMContentLoaded", async () => {
    // Show gold price dashboard on load
    const goldDash = document.getElementById('gold-dashboard-content');
    if (goldDash) goldDash.style.display = 'block';
    
    // Fetch and render
    await fetchGoldData();
    renderGoldSidebar();
    renderGoldPriceDashboard();
    loadTradingViewGoldWidget();
    
    // Start continuous update polling loop (every 15 seconds)
    setInterval(async () => {
        try {
            await fetchGoldData();
            renderGoldPriceDashboard();
        } catch (err) {
            console.error("Error polling gold price:", err);
        }
    }, 15000);
});
