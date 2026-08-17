let currentGoldChartTimeframe = 30; // default 30 days

// Format helpers
function formatGoldPrice(price) {
    if (!price) return '--';
    return parseFloat(price).toLocaleString('vi-VN', { maximumFractionDigits: 0 });
}
function formatUSD(val) {
    if (val === undefined || val === null || isNaN(val)) return '';
    return '$' + parseFloat(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
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

function getBestGoldModel() {
    // Returns the key of the single model with the highest R² (excluding ensemble)
    if (!goldData || !goldData.predictions || !goldData.predictions.models) return 'random_forest';
    const models = goldData.predictions.models;
    const singleModels = ['random_forest', 'mlp', 'xgboost', 'linear_regression'];
    let bestKey = 'random_forest';
    let bestR2 = -Infinity;
    singleModels.forEach(k => {
        const r2 = models[k]?.ml_prediction?.r_squared ?? -Infinity;
        if (r2 > bestR2) { bestR2 = r2; bestKey = k; }
    });
    return bestKey;
}

function switchGoldModel(modelName) {
    activeGoldModel = modelName;
    
    // Immediately update menu active state for instant visual feedback
    updateModelMenuActiveState();
    
    updateGoldAIUI();
    
    // Render verification history for selected model
    renderGoldVerificationHistory();
    
    // Sync any legacy select dropdowns that may remain
    const selectEl = document.getElementById('gold-model-select');
    if (selectEl) selectEl.value = modelName;

    // Re-run swing calc with updated model
    runSwingCalc();
}

function swingFillFromCurrentSell() {
    if (!goldData || !goldData.world) return;
    const priceStr = goldData.world.price;
    if (!priceStr) return;
    // priceStr is e.g. "4,143.26" (USD/ounce)
    const numVal = parseFloat(String(priceStr).replace(/[^0-9.]/g, ''));
    if (!isNaN(numVal)) {
        document.getElementById('swing-buy-price').value = numVal;
        if (!document.getElementById('swing-qty').value) {
            document.getElementById('swing-qty').value = 1; // Default to 1 ounce
        }
        if (!document.getElementById('swing-fee').value) {
            document.getElementById('swing-fee').value = 0;
        }
        runSwingCalc();
    }
}

function runSwingCalc() {
    const buyPriceRaw = parseFloat(document.getElementById('swing-buy-price').value);
    const qty         = parseFloat(document.getElementById('swing-qty').value);
    const feePct      = parseFloat(document.getElementById('swing-fee').value) || 0;

    const resultPanel  = document.getElementById('swing-result-panel');
    const breakdownDiv = document.getElementById('swing-breakdown');

    if (!buyPriceRaw || !qty || isNaN(buyPriceRaw) || isNaN(qty)) {
        resultPanel.innerHTML = `<div style="font-size:0.75rem;color:var(--text-muted);">← Nhập thông số để xem kết quả</div>`;
        breakdownDiv.style.display = 'none';
        return;
    }

    // Get T+1 AI prediction (active model price)
    let predPriceT1 = null;
    let modelUsed   = 'ensemble';
    if (goldData && goldData.predictions && goldData.predictions.models) {
        const activeKey = activeGoldModel || 'ensemble';
        const m = goldData.predictions.models[activeKey];
        if (m && m.ml_prediction && m.ml_prediction.predicted_prices && m.ml_prediction.predicted_prices.length > 0) {
            predPriceT1 = m.ml_prediction.predicted_prices[0];
            modelUsed = activeKey === 'ensemble' ? 'Ensemble Hybrid' : activeKey;
        }
    }

    if (predPriceT1 === null) {
        resultPanel.innerHTML = `<div style="font-size:0.75rem;color:var(--text-muted);">Chưa có dữ liệu dự báo AI. Vui lòng tải trang lại.</div>`;
        breakdownDiv.style.display = 'none';
        return;
    }

    // Calculations in USD
    const capital      = buyPriceRaw * qty;                     // tổng vốn bỏ ra (USD)
    const feeCost      = capital * (feePct / 100);               // phí mua + bán (USD)
    const breakeven    = buyPriceRaw * (1 + feePct / 100);       // giá hoà vốn (USD/ounce)
    const revenue      = predPriceT1 * qty;                      // thu về (dự báo, USD)
    const netProfit    = revenue - capital - feeCost;            // lợi nhuận ròng (USD)
    const netProfitPct = (netProfit / (capital + feeCost)) * 100;

    const stopLossPrice  = buyPriceRaw * 0.985;   // -1.5%
    const takeProfitPrice = buyPriceRaw * 1.03;   // +3%

    // Format helpers for USD
    const fmtUSDVal = v => '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const fmtPct = v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

    const isProfit = netProfit > 0;
    const isBreakeven = Math.abs(netProfit) < 0.001;

    // Update result panel
    let verdictText, verdictColor, verdictBg, verdictIcon, aiLabel;
    if (isBreakeven) {
        verdictText  = 'HOÀ VỐN';
        verdictColor = '#94a3b8';
        verdictBg    = 'rgba(148,163,184,0.08)';
        verdictIcon  = 'fa-equals';
        aiLabel      = 'Không lời, không lỗ';
    } else if (isProfit) {
        verdictText  = 'NGÀY MAI LỜI';
        verdictColor = '#34d399';
        verdictBg    = 'rgba(52,211,153,0.08)';
        verdictIcon  = 'fa-arrow-trend-up';
        aiLabel      = `Theo AI ${modelUsed}`;
    } else {
        verdictText  = 'NGÀY MAI LỖ';
        verdictColor = '#f87171';
        verdictBg    = 'rgba(248,113,113,0.08)';
        verdictIcon  = 'fa-arrow-trend-down';
        aiLabel      = `Theo AI ${modelUsed}`;
    }

    resultPanel.innerHTML = `
        <div style="width:100%; display:flex; flex-direction:column; align-items:center; gap:0.4rem; padding:0.5rem 0;">
            <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); font-weight:600;">${aiLabel}</div>
            <div style="padding:0.5rem 1.5rem; background:${verdictBg}; border-radius:8px; border:1px solid ${verdictColor}22; margin:0.3rem 0;">
                <div class="swing-verdict" style="color:${verdictColor}; font-weight: 700;">
                    <i class="fa-solid ${verdictIcon}" style="margin-right:0.4rem; font-size:1.5rem;"></i>${verdictText}
                </div>
            </div>
            <div class="swing-profit-amount" style="color:${verdictColor}; font-weight: 700; font-size: 1.2rem;">
                ${netProfit >= 0 ? '+' : ''}${fmtUSDVal(netProfit)}
            </div>
            <div class="swing-profit-pct" style="color:${verdictColor}; font-size: 0.85rem;">
                ${fmtPct(netProfitPct)} / ${qty} ounce
            </div>
            <div style="margin-top:0.5rem; display:flex; gap:1.5rem; font-size:0.75rem; color:var(--text-muted);">
                <span>Mua: <strong style="color:var(--text-primary); font-family:'JetBrains Mono',monospace;">${fmtUSDVal(buyPriceRaw)}</strong></span>
                <span>→ AI T+1: <strong style="color:#60a5fa; font-family:'JetBrains Mono',monospace;">${fmtUSDVal(predPriceT1)}</strong></span>
            </div>
        </div>
    `;

    // Update breakdown cells
    breakdownDiv.style.display = 'block';
    document.getElementById('swing-d-capital').textContent    = fmtUSDVal(capital);
    document.getElementById('swing-d-fee').textContent         = '−' + fmtUSDVal(feeCost);
    document.getElementById('swing-d-pred').textContent        = fmtUSDVal(predPriceT1);
    document.getElementById('swing-d-pred').style.color        = isProfit ? '#34d399' : '#f87171';
    document.getElementById('swing-d-breakeven').textContent   = fmtUSDVal(breakeven);
    document.getElementById('swing-d-stoploss').textContent    = fmtUSDVal(stopLossPrice);
    document.getElementById('swing-d-takeprofit').textContent  = fmtUSDVal(takeProfitPrice);
}

function renderGoldModelMenu() {
    const menuEl = document.getElementById('gold-model-menu');
    if (!menuEl || !goldData || !goldData.predictions || !goldData.predictions.models) return;

    const models = goldData.predictions.models;

    // Define all 6 models with metadata
    const modelDefs = [
        { key: 'ensemble',          icon: '✨', name: 'Ensemble\nHybrid',    isEnsemble: true  },
        { key: 'random_forest',      icon: '🌲', name: 'Random\nForest',      isEnsemble: false },
        { key: 'mlp',               icon: '🧠', name: 'Mạng\nNơ-ron MLP',   isEnsemble: false },
        { key: 'xgboost',           icon: '🚀', name: 'XGBoost\nRegressor',  isEnsemble: false },
        { key: 'lstm',              icon: '🧠', name: 'Mạng\nLSTM',          isEnsemble: false },
        { key: 'linear_regression', icon: '📈', name: 'Hồi Quy\nTuyến Tính', isEnsemble: false },
    ];

    // Attach R² to each
    modelDefs.forEach(m => {
        const md = models[m.key];
        m.r2 = md ? (md.ml_prediction?.r_squared ?? 0) : 0;
        m.rec = md ? md.recommendation : '—';
        m.actionClass = md ? md.action_class : 'hold';
    });

    // Sort: Ensemble always first, rest sorted by R² desc
    const ensemble = modelDefs.filter(m => m.isEnsemble);
    const others   = modelDefs.filter(m => !m.isEnsemble).sort((a, b) => b.r2 - a.r2);
    const sorted   = [...ensemble, ...others];

    menuEl.innerHTML = '';
    sorted.forEach((m, idx) => {
        const isActive = m.key === activeGoldModel;
        const r2Pct    = (m.r2 * 100).toFixed(1);

        // Badge
        let badge = '';
        if (m.isEnsemble) {
            badge = `<span class="model-menu-badge badge-ensemble">Gom chung 5 mô hình</span>`;
        } else if (idx === 1) {
            badge = `<span class="model-menu-badge badge-top">Cao nhất</span>`;
        }

        // Rank label (only for non-ensemble)
        const rankLabel = m.isEnsemble ? '' : `<span class="model-menu-rank">#${idx}</span>`;

        const card = document.createElement('div');
        card.className = 'model-menu-card' + (isActive ? (m.isEnsemble ? ' active-ensemble' : ' active') : '');
        card.setAttribute('data-model-key', m.key);
        card.innerHTML = `
            ${rankLabel}
            <div class="model-menu-icon">${m.icon}</div>
            <div class="model-menu-name" style="white-space: pre-line;">${m.name}</div>
            ${badge}
            <div class="model-menu-r2">R² ${r2Pct}%</div>
        `;
        card.addEventListener('click', () => switchGoldModel(m.key));
        menuEl.appendChild(card);
    });
}

function updateModelMenuActiveState() {
    const cards = document.querySelectorAll('#gold-model-menu .model-menu-card');
    cards.forEach(card => {
        const key = card.getAttribute('data-model-key');
        const isEnsemble = key === 'ensemble';
        card.className = 'model-menu-card';
        if (key === activeGoldModel) {
            card.classList.add(isEnsemble ? 'active-ensemble' : 'active');
        }
    });
}

function updateGoldAIUI() {
    if (!goldData || !goldData.predictions) return;
    const pred = goldData.predictions;
    if (pred.status === 'error') return;
    
    const activeKey = activeGoldModel || 'ensemble';
    const modelData = pred.models ? pred.models[activeKey] : null;
    if (!modelData) return;
    
    // Save references for redrawing chart when timeframe is toggled
    lastGoldPredData = pred;
    lastGoldModelData = modelData;
    
    // Render Score Gauge (MLP / SVG Circle)
    const score = modelData.score;
    const scoreValEl = document.getElementById('gold-score-val');
    if (scoreValEl) scoreValEl.textContent = score.toFixed(1);
    
    const percentage = Math.max(0, Math.min(100, (score + 5.0) * 10));
    const strokeVal = 345 - (345 * percentage) / 100;
    const fillEl = document.getElementById('gold-score-meter-fill');
    if (fillEl) fillEl.style.strokeDashoffset = strokeVal;
    
    // Recommendation Badge
    const badge = document.getElementById('gold-rec-badge');
    const label = document.getElementById('gold-rec-badge-label');
    const actCls = modelData.action_class;
    
    if (badge) badge.className = `rec-badge-large ${actCls}`;
    if (label) label.textContent = modelData.recommendation.toUpperCase();
    
    // Reasons list
    const reasonsUl = document.getElementById('gold-reasons-list');
    if (reasonsUl) {
        reasonsUl.innerHTML = '';
        const reasons = modelData.reasons || [];
        
        reasons.forEach(r => {
            const li = document.createElement('li');
            if (r.includes('cắt lên') || r.includes('Giao cắt Vàng') || r.includes('đường Tín hiệu (Tín hiệu MUA)') || r.includes('chạm dải dưới') || (r.includes('RSI đạt') && r.includes('Quá Bán')) || r.includes('xu hướng Tăng')) {
                li.classList.add('positive');
            } else if (r.includes('cắt xuống') || r.includes('Tử thần') || r.includes('Tín hiệu BÁN') || r.includes('chạm dải trên') || r.includes('Quá Mua') || r.includes('xu hướng Giảm')) {
                li.classList.add('negative');
            }
            li.textContent = r;
            reasonsUl.appendChild(li);
        });
    }

    /* -- Actionable Advice & Timing (Gold World Price) -- */
    const advice = modelData.actionable_advice || pred.actionable_advice;
    const adviceSection = document.getElementById('gold-actionable-advice-section');
    if (adviceSection) {
        if (advice) {
            adviceSection.style.display = 'block';
            
            const conclusionCard = document.getElementById('gold-advice-conclusion-card');
            const timingCard     = document.getElementById('gold-advice-timing-card');
            
            if (conclusionCard && timingCard) {
                // Reset card classes
                conclusionCard.className = 'advice-card';
                timingCard.className     = 'advice-card';
                
                // Colorize based on conclusion
                if (advice.buy_now_conclusion === "Nên Mua Ngay" || advice.buy_now_conclusion === "Có Thể Giải Ngân") {
                    conclusionCard.classList.add('buy-advice');
                } else if (advice.buy_now_conclusion === "Tuyệt Đối Không Mua") {
                    conclusionCard.classList.add('sell-advice');
                } else {
                    conclusionCard.classList.add('hold-advice');
                }
                
                // Colorize timing card based on value
                if (advice.best_buy_time && advice.best_buy_time.includes("Chờ")) {
                    timingCard.classList.add('sell-advice');
                } else if (advice.best_buy_time && advice.best_buy_time.includes("Sáng")) {
                    timingCard.classList.add('buy-advice');
                } else {
                    timingCard.classList.add('hold-advice');
                }
            }
            
            const concVal = document.getElementById('gold-advice-conclusion-val');
            const concSub = document.getElementById('gold-advice-conclusion-sub');
            const timeVal = document.getElementById('gold-advice-timing-val');
            const timeSub = document.getElementById('gold-advice-timing-sub');
            
            if (concVal) concVal.textContent = advice.buy_now_conclusion;
            if (concSub) concSub.textContent = advice.buy_now_subtext;
            if (timeVal) timeVal.textContent = advice.best_buy_time;
            if (timeSub) timeSub.textContent = advice.best_buy_time_reason;
            
            // Expected Next Price
            const nextPriceValEl = document.getElementById('gold-advice-next-price-val');
            if (nextPriceValEl) {
                const nextChange = advice.next_price_change_pct;
                const nextChangeSign = nextChange > 0 ? '+' : '';
                const nextChangeCls = nextChange > 0 ? 'text-up' : (nextChange < 0 ? 'text-down' : 'text-flat');
                nextPriceValEl.innerHTML = `$${advice.next_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})} <span class="${nextChangeCls}" style="font-size:0.75rem;font-weight:normal;margin-left:0.2rem">(${nextChangeSign}${nextChange.toFixed(2)}%)</span>`;
            }
            
            // Buy Range
            const buyRangeValEl = document.getElementById('gold-advice-buy-range-val');
            if (buyRangeValEl) buyRangeValEl.textContent = advice.target_buy_range;
            
            // Expected Profit
            const profitValEl = document.getElementById('gold-advice-expected-profit-val');
            if (profitValEl) {
                if (advice.expected_profit_pct > 0) {
                    profitValEl.innerHTML = `+${advice.expected_profit_pct.toFixed(2)}% <span style="font-size:0.72rem;font-weight:normal;color:var(--text-secondary)">($${advice.expected_profit_amount.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})})</span>`;
                    profitValEl.className = 'item-value text-up';
                } else {
                    profitValEl.textContent = '0.00% ($0)';
                    profitValEl.className = 'item-value text-flat';
                }
            }
            
            // Stop Loss
            const slEl = document.getElementById('gold-advice-stop-loss-val');
            if (slEl) slEl.textContent = `$${advice.stop_loss_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        } else {
            adviceSection.style.display = 'none';
        }
    }

    // 5-Day forecast grid
    const grid = document.getElementById('gold-forecast-grid');
    if (grid) {
        grid.innerHTML = '';
        
        let currentPrice = pred.current_price;
        if (goldData && goldData.world && goldData.world.price) {
            const liveWorldUsd = parseFloat(String(goldData.world.price).replace(/[^0-9.]/g, ''));
            if (!isNaN(liveWorldUsd) && liveWorldUsd > 0) {
                currentPrice = liveWorldUsd;
            }
        }
        const predPrices = modelData.ml_prediction.predicted_prices || [];
        
        // Helper to get next N trading dates (excluding Sat/Sun)
        const getForecastDates = (startDateStr, count) => {
            let dates = [];
            let curr = new Date(startDateStr);
            if (isNaN(curr.getTime())) {
                curr = new Date();
            }
            let added = 0;
            while (added < count) {
                curr.setDate(curr.getDate() + 1);
                const day = curr.getDay(); // 0: Sunday, 6: Saturday
                if (day !== 0 && day !== 6) {
                    const y = curr.getFullYear();
                    const m = String(curr.getMonth() + 1).padStart(2, '0');
                    const d = String(curr.getDate()).padStart(2, '0');
                    dates.push(`${y}-${m}-${d}`);
                    added++;
                }
            }
            return dates;
        };

        const baseDateStr = (goldData && goldData.latest_prices && goldData.latest_prices.date) || new Date().toISOString().split('T')[0];
        const forecastDates = getForecastDates(baseDateStr, predPrices.length);

        predPrices.forEach((price, idx) => {
            const dayNum = idx + 1;
            const change = price - currentPrice;
            const pctChange = (change / currentPrice) * 100;
            
            const card = document.createElement('div');
            card.className = `forecast-day ${change > 0 ? 'up' : (change < 0 ? 'down' : '')}`;
            card.style.animationDelay = `${idx * 0.05}s`;
            card.classList.add('row-fade-in');
            
            const sign = change > 0 ? '+' : '';
            const trendClass = change > 0 ? 'text-up' : (change < 0 ? 'text-down' : 'text-flat');
            const caret = change > 0 ? '<i class="fa-solid fa-caret-up"></i>' : (change < 0 ? '<i class="fa-solid fa-caret-down"></i>' : '');
            
            // Format vietnamese date label (excluding weekend)
            const dateStr = forecastDates[idx];
            const dateObj = new Date(dateStr);
            const viDay = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'][dateObj.getDay()];
            const formattedDate = `${viDay} ${dateStr.split('-').slice(1).reverse().join('/')}`;

            card.innerHTML = `
                <span class="day-label" style="font-size:0.68rem; font-weight:600;">Ngày T+${dayNum} (${formattedDate})</span>
                <span class="price-val" style="font-size:0.95rem;font-weight:600;color:var(--text-primary);">$${price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
                <span class="change-val ${trendClass}">
                    ${caret} ${sign}${pctChange.toFixed(2)}%
                </span>
            `;
            grid.appendChild(card);
        });
    }
    
    // Render ChartJS history & forecast
    renderGoldHistoryChart(pred, modelData);
    if (pred.gold_history) {
        renderGoldGLDComparisonChart(pred.gold_history);
    }
    
    // Render / refresh the model picker menu with live R² values
    renderGoldModelMenu();

    // Keep MLP architecture diagram always visible as requested
    const mlpArchSection = document.getElementById('gold-mlp-architecture-section');
    if (mlpArchSection) {
        mlpArchSection.style.display = 'block';
    }
    
    // Update active model name in the diagram title
    const activeModelNameEl = document.getElementById('gold-diagram-active-model-name');
    if (activeModelNameEl) {
        const modelLabels = {
            'random_forest': 'Random Forest',
            'linear_regression': 'Hồi quy tuyến tính',
            'mlp': 'Mạng Nơ-ron MLP',
            'xgboost': 'XGBoost',
            'lstm': 'Mạng LSTM (Chuỗi Thời Gian)',
            'ensemble': 'Ensemble Hybrid'
        };
        activeModelNameEl.textContent = modelLabels[activeKey] || activeKey;
    }
    
    // Update model reliability badge
    const reliabilityEl = document.getElementById('gold-model-reliability');
    if (reliabilityEl && modelData.ml_prediction && modelData.ml_prediction.r_squared !== undefined) {
        const r2 = modelData.ml_prediction.r_squared;
        let pct = (r2 * 100).toFixed(1);
        if (pct < 0) pct = 0;
        
        let text = '';
        let bg = '';
        let color = '';
        let border = '';
        
        if (r2 >= 0.85) {
            text = `Độ Tin Cậy: Rất Cao (R² = ${pct}%)`;
            bg = 'rgba(16, 185, 129, 0.15)';
            color = '#10b981';
            border = 'rgba(16, 185, 129, 0.25)';
        } else if (r2 >= 0.70) {
            text = `Độ Tin Cậy: Cao (R² = ${pct}%)`;
            bg = 'rgba(59, 130, 246, 0.15)';
            color = '#3b82f6';
            border = 'rgba(59, 130, 246, 0.25)';
        } else if (r2 >= 0.50) {
            text = `Độ Tin Cậy: Trung Bình (R² = ${pct}%)`;
            bg = 'rgba(245, 158, 11, 0.15)';
            color = '#f59e0b';
            border = 'rgba(245, 158, 11, 0.25)';
        } else {
            text = `Độ Tin Cậy: Thấp (R² = ${pct}%)`;
            bg = 'rgba(239, 68, 68, 0.15)';
            color = '#ef4444';
            border = 'rgba(239, 68, 68, 0.25)';
        }
        
        reliabilityEl.textContent = text;
        reliabilityEl.style.background = bg;
        reliabilityEl.style.color = color;
        reliabilityEl.style.borderColor = border;
    }
    
    // Dynamic macro values rendering in SVG
    if (pred.latest_macro) {
        const lm = pred.latest_macro;
        
        const worldHistoryNode = document.getElementById('gold-macro-world-history');
        if (worldHistoryNode) {
            if (goldData && goldData.world && goldData.world.price) {
                worldHistoryNode.textContent = `TG: ${goldData.world.price} USD`;
            } else {
                worldHistoryNode.textContent = "Lịch sử giá vàng TG";
            }
        }
        
        const dxyEl = document.getElementById('gold-macro-dxy');
        if (dxyEl && lm.dxy !== undefined) {
            dxyEl.textContent = `DXY: ${lm.dxy.toFixed(2)}`;
        }
        
        const us10yEl = document.getElementById('gold-macro-us10y');
        if (us10yEl && lm.us10y !== undefined) {
            us10yEl.textContent = `US10Y: ${lm.us10y.toFixed(3)}%`;
        }
        
        const vixEl = document.getElementById('gold-macro-vix');
        if (vixEl && lm.vix !== undefined) {
            vixEl.textContent = `VIX: ${lm.vix.toFixed(2)}`;
        }
        
        const brentEl = document.getElementById('gold-macro-brent');
        if (brentEl && lm.brent !== undefined) {
            brentEl.textContent = `Brent: $${lm.brent.toFixed(2)}`;
        }
        
        const eventsEl = document.getElementById('gold-macro-events');
        if (eventsEl && lm.days_to_fed !== undefined) {
            eventsEl.textContent = `FED:${lm.days_to_fed}d | CPI:${lm.days_to_cpi}d | NFP:${lm.days_to_nfp}d`;
            eventsEl.setAttribute('font-size', '8.5');
        }
    }
    
    // Dynamic gold price prediction rendering in SVG output nodes
    const predPrices = modelData.ml_prediction?.predicted_prices || [];
    if (predPrices && predPrices.length >= 5) {
        for (let i = 0; i < 5; i++) {
            const valEl = document.getElementById(`gold-output-t${i+1}`);
            const chgEl = document.getElementById(`gold-output-t${i+1}-change`);
            const rectEl = document.getElementById(`gold-rect-t${i+1}`);
            
            if (valEl) {
                const usdVal = predPrices[i];
                valEl.textContent = `T+${i+1}: $${usdVal.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
            
            if (chgEl) {
                const diffUsd = predPrices[i] - currentPrice;
                const pctChange = (diffUsd / currentPrice) * 100;
                
                const sign = diffUsd >= 0 ? '+' : '';
                const textTrend = diffUsd >= 0 ? 'Tăng' : 'Giảm';
                chgEl.textContent = `${textTrend}: ${sign}${pctChange.toFixed(2)}% (${sign}$${diffUsd.toFixed(2)})`;
                
                // Update colors and rect classes
                if (rectEl) {
                    if (diffUsd > 0) {
                        rectEl.setAttribute('fill', '#064e3b'); // dark green
                        rectEl.setAttribute('stroke', '#10b981'); // bright green
                        chgEl.setAttribute('fill', '#a7f3d0'); // light green text
                    } else if (diffUsd < 0) {
                        rectEl.setAttribute('fill', '#7f1d1d'); // dark red
                        rectEl.setAttribute('stroke', '#ef4444'); // bright red
                        chgEl.setAttribute('fill', '#fca5a5'); // light red text
                    } else {
                        rectEl.setAttribute('fill', '#1e293b'); // dark slate
                        rectEl.setAttribute('stroke', '#94a3b8'); // gray
                        chgEl.setAttribute('fill', '#cbd5e1'); // light gray text
                    }
                }
            }
        }
    }
}

function switchGoldChartTimeframe(tf) {
    activeGoldChartTimeframe = tf;
    
    const buttons = document.querySelectorAll('.gold-chart-timeframes .timeframe-btn');
    buttons.forEach(btn => {
        const onclickVal = btn.getAttribute('onclick');
        if (onclickVal && onclickVal.includes(tf)) {
            btn.classList.add('active');
            btn.style.background = 'rgba(251, 191, 36, 0.15)';
            btn.style.borderColor = '#fbbf24';
            btn.style.color = '#fbbf24';
        } else {
            btn.classList.remove('active');
            btn.style.background = 'rgba(255, 255, 255, 0.05)';
            btn.style.borderColor = 'rgba(255, 255, 255, 0.1)';
            btn.style.color = '#94a3b8';
        }
    });
    
    if (lastGoldPredData && lastGoldModelData) {
        renderGoldHistoryChart(lastGoldPredData, lastGoldModelData);
        if (lastGoldPredData.gold_history) {
            renderGoldGLDComparisonChart(lastGoldPredData.gold_history);
        }
    }
}

function renderDXYHistoryChart(history) {
    // No-op here since charts are on /gold, not /gold/prediction
}
function renderOilHistoryChart(history) {
    // No-op here since charts are on /gold, not /gold/prediction
}
function renderDJIHistoryChart(history) {
    // No-op here since charts are on /gold, not /gold/prediction
}
function renderGoldGLDComparisonChart(history) {
    // No-op here since charts are on /gold, not /gold/prediction
}

function renderGoldHistoryChart(pred, modelData) {
    const canvas = document.getElementById('gold-history-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Prepare labels and datasets based on timeframe
    let labels = [];
    let datasets = [];
    
    if (activeGoldChartTimeframe === '24h') {
        const ticks = (goldData && goldData.ticks) ? goldData.ticks : [];
        if (ticks.length === 0) return;
        
        const rawLabels = ticks.map(t => {
            const parts = t.timestamp.split(' ');
            if (parts.length > 1) {
                const timePart = parts[1].substring(0, 5); // "HH:MM"
                const dateParts = parts[0].split('-');
                return `${timePart} (${dateParts[2]}/${dateParts[1]})`;
            }
            return t.timestamp;
        });
        
        // world_price is in USD/ounce
        const worldPrices = ticks.map(t => t.world_price || (t.world_price_vnd / 31900));
        const preds = modelData.ml_prediction.predicted_prices || [];
        
        const histWorldData = [...worldPrices];
        for (let i = 0; i < preds.length; i++) histWorldData.push(null);
        
        const predLine = [...Array(ticks.length - 1).fill(null), worldPrices[worldPrices.length - 1]];
        preds.forEach(p => predLine.push(p));
        
        const chartLabels = [...rawLabels];
        for (let i = 1; i <= preds.length; i++) {
            chartLabels.push(`T+${i}`);
        }
        
        labels = chartLabels;
        
        datasets = [
            {
                label: 'Giá thế giới thực tế (USD/ounce)',
                data: histWorldData,
                borderColor: '#60a5fa', // elegant blue
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: false,
                tension: 0.15
            },
            {
                label: 'Giá dự báo (AI) (USD/ounce)',
                data: predLine,
                borderColor: '#fbbf24', // yellow/gold
                borderWidth: 2.5,
                borderDash: [5, 4],
                pointRadius: 0,
                pointHoverRadius: 5,
                fill: false,
                tension: 0.1
            }
        ];
    } else {
        // '30d' timeframe
        let history = (pred.gold_history || []).slice(-30);
        if (history.length === 0) return;
        
        // Sync with latest World gold price from the price board to ensure exact match
        let domDateStr = null;
        if (goldData && goldData.world && goldData.world.time) {
            const parts = goldData.world.time.split(' ');
            if (parts.length > 1) {
                const dateParts = parts[1].split('/');
                if (dateParts.length === 3) {
                    domDateStr = `${dateParts[2]}-${dateParts[1]}-${dateParts[0]}`;
                }
            }
        }
        if (!domDateStr) {
            domDateStr = new Date().toISOString().split('T')[0];
        }
        
        if (goldData && goldData.world && goldData.world.price) {
            const lastEntry = history[history.length - 1];
            const liveWorldUsd = parseFloat(String(goldData.world.price).replace(/[^0-9.]/g, ''));
            
            if (!isNaN(liveWorldUsd) && liveWorldUsd > 0) {
                if (lastEntry.date !== domDateStr) {
                    history.push({
                        date: domDateStr,
                        world_price: liveWorldUsd
                    });
                } else {
                    lastEntry.world_price = liveWorldUsd;
                }
            }
        }
        
        const rawLabels = history.map(h => {
            const parts = h.date.split('-');
            return `${parts[2]}/${parts[1]}`;
        });
        
        const worldPrices = history.map(h => h.world_price || (h.world_price_vnd / 31900));
        const preds = modelData.ml_prediction.predicted_prices || [];
        const chartLabels = [...rawLabels];
        
        const histWorldData = [...worldPrices];
        for (let i = 0; i < preds.length; i++) histWorldData.push(null);
        
        const predLine = [...Array(history.length - 1).fill(null), worldPrices[worldPrices.length - 1]];
        preds.forEach(p => predLine.push(p));
        
        for (let i = 1; i <= preds.length; i++) {
            chartLabels.push(`T+${i}`);
        }
        labels = chartLabels;
        
        datasets = [
            {
                label: 'Giá thế giới thực tế (USD/ounce)',
                data: histWorldData,
                borderColor: '#3b82f6', // blue
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: false,
                tension: 0.15
            },
            {
                label: 'Giá dự báo (AI) (USD/ounce)',
                data: predLine,
                borderColor: '#fbbf24', // yellow/gold
                borderWidth: 2.5,
                borderDash: [5, 4],
                pointRadius: 0,
                pointHoverRadius: 5,
                fill: false,
                tension: 0.1
            }
        ];
    }
    
    // Seamless update if chart instance exists and timeframe option is identical
    if (goldChart && goldChart.config.options.timeframe === activeGoldChartTimeframe) {
        goldChart.data.labels = labels;
        // Safely update or add datasets
        datasets.forEach((ds, idx) => {
            if (goldChart.data.datasets[idx]) {
                goldChart.data.datasets[idx].data = ds.data;
                goldChart.data.datasets[idx].label = ds.label;
                goldChart.data.datasets[idx].borderColor = ds.borderColor;
                goldChart.data.datasets[idx].borderWidth = ds.borderWidth;
                goldChart.data.datasets[idx].borderDash = ds.borderDash || [];
            } else {
                goldChart.data.datasets.push(ds);
            }
        });
        
        if (goldChart.data.datasets.length > datasets.length) {
            goldChart.data.datasets = goldChart.data.datasets.slice(0, datasets.length);
        }
        
        goldChart.update('none'); // Update smoothly without flashing
        return;
    }
    
    // Otherwise destroy and rebuild
    if (goldChart) {
        goldChart.destroy();
    }
    
    goldChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            timeframe: activeGoldChartTimeframe,
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.02)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.02)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

async function runGoldBackfill() {
    const btn = document.getElementById('gold-backfill-btn');
    const daysSelect = document.getElementById('backfill-days-select');
    const days = daysSelect ? parseInt(daysSelect.value) : 60;
    if (!btn) return;

    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.style.opacity = '0.6';
    btn.style.cursor = 'not-allowed';
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang tính ${days} ngày...`;

    try {
        const res = await fetch(`/api/gold/backfill?days=${days}`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(`Backfill thất bại: ${err.detail || res.statusText}`, 'error');
            return;
        }
        const result = await res.json();
        showToast(`✅ Backfill hoàn tất! Đã lưu ${result.saved} phiên, bỏ qua ${result.skipped} phiên.`, 'success');

        // Reload prediction history & refresh the table
        const histRes = await fetch(`/api/gold/prediction-history?_ts=${Date.now()}`);
        if (histRes.ok) {
            goldPredictionHistory = await histRes.json();
            renderGoldVerificationHistory();
        }
    } catch (err) {
        console.error('Backfill error:', err);
        showToast('Mất kết nối khi thực hiện backfill', 'error');
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        btn.innerHTML = originalHTML;
    }
}

async function runGoldOptimize() {
    const btn = document.getElementById('gold-optimize-btn');
    if (!btn) return;

    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.style.opacity = '0.6';
    btn.style.cursor = 'not-allowed';
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang tối ưu...`;

    try {
        const res = await fetch(`/api/gold/optimize`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(`Tối ưu hóa thất bại: ${err.detail || res.statusText}`, 'error');
            return;
        }
        const result = await res.json();
        showToast(`✅ Tự học & Tối ưu hóa tham số thành công!`, 'success');
        console.log('Optimized parameters:', result.params);
        
        // Refresh prediction history & refresh the page
        const histRes = await fetch(`/api/gold/prediction-history?_ts=${Date.now()}`);
        if (histRes.ok) {
            goldPredictionHistory = await histRes.json();
        if (typeof refreshGoldData === 'function') {
            await refreshGoldData();
        } else {
            location.reload();
        }
    } catch (err) {
        console.error('Optimize error:', err);
        showToast('Mất kết nối khi thực hiện tối ưu hóa', 'error');
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        btn.innerHTML = originalHTML;
    }
}

function formatMacroAnalysisHtml(text) {
    if (!text) return '';
    
    const regex = /Mô hình dự báo (thấp hơn|cao hơn) thực tế ([\d\.]+) USD do chưa phản ánh hết mức độ tác động khi \[(.*)\]\. (Hệ thống đã ghi nhận.*)/i;
    const match = text.match(regex);
    
    if (!match) {
        return `<div style="color: var(--text-secondary); line-height: 1.45; font-style: italic;">${text}</div>`;
    }
    
    const direction = match[1];
    const diffVal = match[2];
    const macroItemsText = match[3];
    const suffix = match[4];
    
    const items = macroItemsText.split(' | ').map(item => item.trim()).filter(Boolean);
    
    let itemsHtml = '';
    items.forEach(item => {
        let cleanItem = item.replace(/➔/g, '<i class="fa-solid fa-right-long" style="margin: 0 0.4rem; color: var(--text-muted); opacity: 0.6;"></i>');
        
        cleanItem = cleanItem.replace(/\(\+\)/g, '<span style="color: #34d399; font-weight: 700; margin-left: 0.15rem;">(+)</span>');
        cleanItem = cleanItem.replace(/\(\-\)/g, '<span style="color: #f87171; font-weight: 700; margin-left: 0.15rem;">(-)</span>');
        
        itemsHtml += `
            <li style="margin-bottom: 0.35rem; display: flex; align-items: flex-start; gap: 0.3rem; font-size: 0.72rem; color: var(--text-secondary);">
                <i class="fa-solid fa-circle-dot" style="font-size: 0.4rem; color: #fbbf24; margin-top: 0.35rem; margin-right: 0.35rem; flex-shrink: 0;"></i>
                <span>${cleanItem}</span>
            </li>
        `;
    });
    
    const prefixIcon = direction === 'thấp hơn' ? 'fa-arrow-trend-down' : 'fa-arrow-trend-up';
    const prefixColor = direction === 'thấp hơn' ? '#f87171' : '#34d399';
    
    return `
        <div style="font-size: 0.74rem; line-height: 1.5; text-align: left; color: var(--text-secondary);">
            <div style="margin-bottom: 0.5rem; font-weight: 600; display: flex; align-items: center; gap: 0.4rem;">
                <i class="fa-solid ${prefixIcon}" style="color: ${prefixColor}; font-size: 0.9rem;"></i>
                <span>Mô hình dự báo <span style="color: ${prefixColor}; font-weight: 700;">${direction}</span> thực tế <strong style="color: var(--text-primary);">$${diffVal} USD</strong> do chưa phản ánh hết tác động vĩ mô:</span>
            </div>
            <ul style="list-style: none; padding-left: 0.6rem; margin: 0.4rem 0 0.6rem 0; border-left: 2px solid rgba(255,255,255,0.06); display: flex; flex-direction: column; gap: 0.2rem;">
                ${itemsHtml}
            </ul>
            <div style="display: flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.6rem; background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.12); border-radius: 4px; color: #34d399; font-weight: 600; font-size: 0.68rem; margin-top: 0.5rem;">
                <i class="fa-solid fa-circle-check"></i>
                <span>${suffix}</span>
            </div>
        </div>
    `;
}

function renderGoldVerificationHistory() {
    const tbodyVerify = document.getElementById('gold-verification-tbody');
    if (!tbodyVerify) return;
    tbodyVerify.innerHTML = '';
    
    if (!goldPredictionHistory || goldPredictionHistory.length === 0) {
        tbodyVerify.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:1.5rem;color:var(--text-muted)">Chưa tải được lịch sử kiểm chứng. Vui lòng làm mới.</td></tr>`;
        return;
    }
    
    const modelNames = {
        random_forest: 'Random Forest (AI)',
        linear_regression: 'Hồi quy tuyến tính',
        mlp: 'Mạng Nơ-ron MLP (Học Sâu)',
        xgboost: 'XGBoost Regressor',
        lstm: 'Mạng LSTM (Chuỗi Thời Gian)',
        ensemble: 'Ensemble Hybrid'
    };

    const activeModelKey = activeGoldModel || 'random_forest';
    const mName = modelNames[activeModelKey];
    
    const activeModelLabel = document.getElementById('verification-active-model-name');
    if (activeModelLabel) activeModelLabel.textContent = mName;

    // Retrieve active model's dynamic adjustment from API payload
    let adj = null;
    if (goldData && goldData.latest_adjustments && goldData.latest_adjustments[activeModelKey]) {
        adj = goldData.latest_adjustments[activeModelKey];
    } else if (goldData && goldData.latest_adjustment && goldData.latest_adjustment.model === activeModelKey) {
        adj = goldData.latest_adjustment;
    }

    // Render learning status notification above the table
    const learningStatusEl = document.getElementById('gold-learning-status');
    if (learningStatusEl) {
        if (adj) {
            const adjModelName = modelNames[adj.model] || adj.model;
            const correction = adj.error * 0.8;
            const sign = correction >= 0 ? '+' : '';
            learningStatusEl.style.display = 'block';
            learningStatusEl.style.background = 'rgba(255, 255, 255, 0.015)';
            learningStatusEl.style.border = '1px solid rgba(255, 255, 255, 0.06)';
            learningStatusEl.style.borderLeft = '4px solid #10b981';
            learningStatusEl.style.color = 'var(--text-primary)';
            learningStatusEl.style.padding = '0.75rem 1rem';
            learningStatusEl.style.borderRadius = '4px';
            learningStatusEl.innerHTML = `
                <div style="display:flex; gap:0.75rem; width: 100%;">
                    <i class="fa-solid fa-brain" style="margin-top:0.2rem; font-size: 1.1rem; color: #10b981; flex-shrink: 0;"></i> 
                    <div style="display:flex; flex-direction:column; gap:0.5rem; flex: 1;">
                        <div style="font-size: 0.78rem; line-height: 1.45; font-weight: 500; text-align: left;">
                            Hệ thống đang <strong>áp dụng bù sai số tự học</strong> từ phiên ngày <strong>${adj.date}</strong> (${adjModelName}): 
                            <strong style="color: #fbbf24; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; margin: 0 0.2rem;">${sign}$${correction.toFixed(2)} USD</strong> cho các dự báo tiếp theo.
                        </div>
                        ${adj.macro_analysis ? `<div style="margin-top:0.3rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.5rem;">${formatMacroAnalysisHtml(adj.macro_analysis)}</div>` : ''}
                    </div>
                </div>`;
        } else {
            learningStatusEl.style.display = 'none';
        }
    }
    
    // Sort in reverse order (newest first) and show only the last 7 sessions (1 week)
    const sortedHistory = [...goldPredictionHistory].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 7);
    
    sortedHistory.forEach(item => {
        const date = item.date;
        const predPrice = item.models ? item.models[activeModelKey] : null;
        const actPrice = item.actual_price;
        
        if (predPrice === null || predPrice === undefined) return;
        
        // Find chronological index of this item in the sorted ascending history to get previous day's price
        const i = goldPredictionHistory.findIndex(x => x.date === date);
        
        let isCorrectDirection = true;
        let trendHTML = '-';
        let isCorrectAfterCorrection = false;
        let correctedPrice = 0;
        
        // Check if this item is the source of the current active adjustment
        const isSourceOfLearning = adj && adj.date === date;
        
        if (isSourceOfLearning) {
            const adj_error = adj.error;
            const correction = adj_error * 0.8;
            correctedPrice = predPrice + correction;
        }

        if (i > 0) {
            let basePrice = null;
            for (let k = i - 1; k >= 0; k--) {
                if (goldPredictionHistory[k] && goldPredictionHistory[k].actual_price && goldPredictionHistory[k].actual_price > 0) {
                    basePrice = goldPredictionHistory[k].actual_price;
                    break;
                }
            }
            if (basePrice && basePrice > 0) {
                const predTrend = predPrice > basePrice ? 'up' : (predPrice < basePrice ? 'down' : 'flat');
                const predTrendIcon = predTrend === 'up' 
                    ? '<span style="color:#34d399; font-weight:600;"><i class="fa-solid fa-arrow-trend-up"></i> Tăng</span>' 
                    : (predTrend === 'down' ? '<span style="color:#f87171; font-weight:600;"><i class="fa-solid fa-arrow-trend-down"></i> Giảm</span>' : '<span style="color:var(--text-muted);">Không đổi</span>');
                
                if (actPrice === null || actPrice === undefined || actPrice === 0.0) {
                    trendHTML = `<div style="display:flex; flex-direction:column; gap:0.15rem; font-size:0.72rem;">
                        <div>Dự báo: ${predTrendIcon}</div>
                        <div style="color:var(--text-muted);">Thực tế: <span style="font-style:italic;">Chờ...</span></div>
                    </div>`;
                } else {
                    const actTrend = actPrice > basePrice ? 'up' : (actPrice < basePrice ? 'down' : 'flat');
                    const actTrendIcon = actTrend === 'up' 
                        ? '<span style="color:#34d399; font-weight:600;"><i class="fa-solid fa-arrow-trend-up"></i> Tăng</span>' 
                        : (actTrend === 'down' ? '<span style="color:#f87171; font-weight:600;"><i class="fa-solid fa-arrow-trend-down"></i> Giảm</span>' : '<span style="color:var(--text-muted);">Không đổi</span>');
                    
                    isCorrectDirection = predTrend === actTrend;
                    trendHTML = `<div style="display:flex; flex-direction:column; gap:0.15rem; font-size:0.72rem;">
                        <div>Dự báo: ${predTrendIcon}</div>
                        <div>Thực tế: ${actTrendIcon}</div>
                    </div>`;
                    
                    if (isSourceOfLearning) {
                        const correctedTrend = correctedPrice > basePrice ? 'up' : (correctedPrice < basePrice ? 'down' : 'flat');
                        isCorrectAfterCorrection = correctedTrend === actTrend;
                    }
                }
            }
        }
        
        let actPriceText = '';
        let diffText = '';
        let errorText = '';
        let ratingText = '';
        let ratingClass = '';
        let borderLeftColor = 'rgba(255, 255, 255, 0.08)';
        let learningBadgeHTML = '';
        
        if (actPrice === null || actPrice === undefined || actPrice === 0.0) {
            actPriceText = '<span style="color:var(--text-muted);font-style:italic;">Chưa diễn ra</span>';
            diffText = '-';
            errorText = '-';
            ratingText = 'Đang chờ...';
            ratingClass = 'text-flat';
        } else {
            actPriceText = `$${actPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            const diffUsd = actPrice - predPrice;
            const errorPct = (diffUsd / (predPrice + 1e-10)) * 100;
            
            diffText = `<span class="${diffUsd >= 0 ? 'text-up' : 'text-down'}">${diffUsd >= 0 ? '+' : ''}$${diffUsd.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>`;
            errorText = `<span class="${errorPct >= 0 ? 'text-up' : 'text-down'}">${errorPct >= 0 ? '+' : ''}${errorPct.toFixed(2)}%</span>`;
            
            const absErr = Math.abs(errorPct);
            if (!isCorrectDirection) {
                ratingText = '❌ Sai hướng (Lệch)';
                ratingClass = 'text-down';
                borderLeftColor = '#ef4444';
                if (isSourceOfLearning) {
                    const correctionText = (adj.error * 0.8) >= 0 ? `+${(adj.error * 0.8).toFixed(2)}` : `${(adj.error * 0.8).toFixed(2)}`;
                    const isCorrectText = isCorrectAfterCorrection ? 'Đúng' : 'Vẫn lệch';
                    const isCorrectIcon = isCorrectAfterCorrection ? 'fa-check text-up' : 'fa-xmark text-down';
                    learningBadgeHTML = `<span style="background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.45rem; border-radius:3px; font-weight:600; font-size:0.62rem; display:inline-flex; align-items:center; gap:0.25rem;">
                        <i class="fa-solid fa-graduation-cap"></i> Tự học lỗi ngày ${adj.date} (Sau bù: $${correctedPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})} - ${isCorrectText} <i class="fa-solid ${isCorrectIcon}"></i>)
                    </span>`;
                } else {
                    learningBadgeHTML = `<span style="background:rgba(59,130,246,0.1); color:#60a5fa; border:1px solid rgba(59,130,246,0.2); padding:0.15rem 0.45rem; border-radius:3px; font-weight:600; font-size:0.62rem; display:inline-flex; align-items:center; gap:0.25rem;" title="Đã được tự động học thông qua quá trình huấn luyện lại mô hình hàng ngày bằng dữ liệu lịch sử."><i class="fa-solid fa-graduation-cap"></i> Đã học (Re-training)</span>`;
                }
            } else {
                if (absErr < 1.0) {
                    ratingText = '⚡ Rất chính xác (<1%)';
                    ratingClass = 'text-up';
                    borderLeftColor = '#10b981';
                } else if (absErr < 2.5) {
                    ratingText = '✅ Tốt (<2.5%)';
                    ratingClass = 'text-up';
                    borderLeftColor = '#10b981';
                } else {
                    ratingText = '⚠️ Sai số lớn';
                    ratingClass = 'text-down';
                    borderLeftColor = '#ef4444';
                }
                
                if (isSourceOfLearning) {
                    const correctionText = (adj.error * 0.8) >= 0 ? `+${(adj.error * 0.8).toFixed(2)}` : `${(adj.error * 0.8).toFixed(2)}`;
                    learningBadgeHTML = `<span style="background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.45rem; border-radius:3px; font-weight:600; font-size:0.62rem; display:inline-flex; align-items:center; gap:0.25rem;">
                        <i class="fa-solid fa-graduation-cap"></i> Tự học (Bù: ${correctionText})
                    </span>`;
                } else {
                    learningBadgeHTML = `<span style="background:rgba(59,130,246,0.1); color:#60a5fa; border:1px solid rgba(59,130,246,0.2); padding:0.15rem 0.45rem; border-radius:3px; font-weight:600; font-size:0.62rem; display:inline-flex; align-items:center; gap:0.25rem;" title="Đã được tự động học thông qua quá trình huấn luyện lại mô hình hàng ngày bằng dữ liệu lịch sử."><i class="fa-solid fa-graduation-cap"></i> Đã học (Re-training)</span>`;
                }
            }
        }
        
        const explanationText = (item.explanations && item.explanations[activeModelKey]) ? item.explanations[activeModelKey] : 'Tín hiệu kỹ thuật & vĩ mô đi ngang.';
        
        const trMain = document.createElement('tr');
        trMain.innerHTML = `
            <td style="font-family:'JetBrains Mono', monospace; font-size:0.78rem; border-bottom: none;">
                <div style="font-weight:600; color:var(--text-primary);">${date}</div>
                <div style="font-size:0.7rem; color:var(--text-muted); font-weight:normal; margin-top:0.15rem;">${mName}</div>
            </td>
            <td style="border-bottom: none;">${trendHTML}</td>
            <td style="color:var(--text-primary); font-weight:500; border-bottom: none;">$${predPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
            <td style="font-weight:600; color:var(--text-primary); border-bottom: none;">${actPriceText}</td>
            <td style="border-bottom: none;">
                <div style="display:flex; flex-direction:column; gap:0.15rem;">
                    <div>${diffText}</div>
                    <div style="font-size:0.7rem;">${errorText}</div>
                </div>
            </td>
        `;
        tbodyVerify.appendChild(trMain);
        
        const trSub = document.createElement('tr');
        trSub.innerHTML = `
            <td colspan="5" style="padding-top: 0px; padding-bottom: 0.65rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem; padding: 0.45rem 0.75rem; background: rgba(255, 255, 255, 0.02); border-radius: 4px; border-left: 3px solid ${borderLeftColor}; font-size: 0.72rem; line-height: 1.4;">
                    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                        <span class="${ratingClass}" style="font-weight: 600;">${ratingText}</span>
                        <span style="color: var(--text-muted); opacity: 0.5;">|</span>
                        <span style="color: var(--text-secondary); text-align: left;">
                            <i class="fa-solid fa-brain" style="color: #fbbf24; margin-right: 0.25rem;"></i>
                            <strong>Lý giải:</strong> ${explanationText}
                        </span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        ${learningBadgeHTML}
                    </div>
                </div>
            </td>
        `;
        tbodyVerify.appendChild(trSub);
    });
}

window.learnFromGoldMistake = async function(date, model) {
    showToast(`Đang phân tích lỗi ngày ${date} & hiệu chỉnh mô hình...`, "info");
    try {
        const response = await fetch('/api/gold/learn-from-mistake', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ date, model })
        });
        
        if (response.ok) {
            const res = await response.json();
            showToast(`Hiệu chỉnh hoàn tất! Đã học từ lỗi và bù sai số $${res.adjustment.toFixed(2)} USD cho các dự báo tiếp theo.`, "success");
            await refreshGoldAnalysis();
        } else {
            const err = await response.json().catch(() => ({}));
            showToast(`Không thể học từ lỗi: ${err.detail || response.statusText}`, "error");
        }
    } catch (err) {
        console.error("Error learning from gold mistake:", err);
        showToast("Lỗi kết nối khi học từ lỗi sai", "error");
    }
};

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

        activeGoldModel = getBestGoldModel();
        renderGoldModelMenu();
        updateGoldAIUI();
        renderGoldVerificationHistory();
        showToast("Cập nhật dữ liệu & phân tích thành công!", "success");
    } catch (err) {
        console.error('Error refreshing gold prices:', err);
        showToast("Lỗi khi phân tích lại: " + err.message, "error");
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
        if (icon) icon.classList.remove('fa-spin');
    }
}

// DOM ready initialization
document.addEventListener("DOMContentLoaded", async () => {
    // Show AI prediction page contents
    const goldDash = document.getElementById('gold-dashboard-content');
    if (goldDash) goldDash.style.display = 'block';

    const loader = document.getElementById('gold-dashboard-loader');
    const body = document.getElementById('gold-dashboard-body');
    if (loader) loader.style.display = 'flex';
    if (body) body.style.display = 'none';

    await fetchGoldData();

    if (loader) loader.style.display = 'none';
    if (body) body.style.display = 'block';

    activeGoldModel = getBestGoldModel();
    renderGoldModelMenu();
    updateGoldAIUI();
    renderGoldVerificationHistory();
    
    // Start continuous update polling loop (every 15 seconds)
    setInterval(async () => {
        try {
            const oldWorldPrice = goldData?.world?.price;
            const oldSJCSell = goldData?.domestic?.sjc_bar?.sell;
            
            await fetchGoldData();
            
            if (goldData && (goldData.world?.price !== oldWorldPrice || goldData.domestic?.sjc_bar?.sell !== oldSJCSell)) {
                updateGoldAIUI();
                renderGoldVerificationHistory();
            }
        } catch (err) {
            console.error("Error polling gold predictions:", err);
        }
    }, 15000);
});
