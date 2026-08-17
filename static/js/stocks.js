        async function selectStocksDashboard() {
            await switchSidebarTab(lastStockTab);
        }

        async function switchSidebarTab(tabName) {
            activeSidebarTab = tabName;
            localStorage.setItem('activeTab', tabName);
            ensureSidebarExpanded();

            const isStock = ['watchlist', 'vn100', 'vnall'].includes(tabName);
            if (isStock) {
                lastStockTab = tabName;
            }

            // Update sub-tabs active state
            ['watchlist', 'vn100', 'vnall'].forEach(t => {
                document.getElementById(`tab-${t}-list`)?.classList.toggle('active', t === tabName);
            });

            // Show/hide stock sub-tabs container
            const stockSubTabs = document.getElementById('stock-sub-tabs');
            if (stockSubTabs) {
                stockSubTabs.style.display = 'flex';
            }

            const isMarket = tabName === 'vnall' || tabName === 'vn100';

            // Lazy load stock data
            if (!marketDataLoaded) {
                const marketTab = document.getElementById('market-tab-content');
                if (marketTab) marketTab.style.display = isMarket ? 'block' : 'none';
                const watchlistTab = document.getElementById('watchlist-tab-content');
                if (watchlistTab) watchlistTab.style.display = tabName === 'watchlist' ? 'block' : 'none';
                
                await loadMarketData();
            }

            const marketTab = document.getElementById('market-tab-content');
            if (marketTab) marketTab.style.display = isMarket ? 'block' : 'none';
            const watchlistTab = document.getElementById('watchlist-tab-content');
            if (watchlistTab) watchlistTab.style.display = tabName === 'watchlist' ? 'block' : 'none';

            // Toggle detail panel visibility
            const detailContent = document.getElementById('detail-content');
            const emptyDetail = document.getElementById('empty-detail-message');
            if (detailContent && emptyDetail) {
                if (activeTicker) {
                    detailContent.style.display = 'flex';
                    emptyDetail.style.display = 'none';
                } else {
                    detailContent.style.display = 'none';
                    emptyDetail.style.display = 'flex';
                }
            }

            const searchEl = document.getElementById('sidebar-search');
            if (searchEl) searchEl.value = '';

            if (isMarket) renderMarketTable();
            else if (tabName === 'watchlist') renderWatchlistTable();

            // Toggle stock navigation menu visibility
            const stockNav = document.getElementById('stock-nav-menu-container');
            if (stockNav) {
                const detailPredicted = document.getElementById('ai-predicted-view') ? document.getElementById('ai-predicted-view').style.display === 'flex' : false;
                stockNav.style.display = (detailPredicted && activeTicker) ? 'block' : 'none';
                if (activeTicker) {
                    const label = document.getElementById('stock-nav-ticker-label');
                    if (label) label.textContent = activeTicker;
                }
            }
        }



        /* ====================================================
           MARKET TABLE RENDER
        ==================================================== */
        function renderMarketTable() {
            const tbody = document.getElementById('market-table-body');
            tbody.innerHTML = '';
            const searchVal = (document.getElementById('sidebar-search')?.value || '').trim().toUpperCase();
            const source = activeSidebarTab === 'vnall' ? vnallData : vn100Data;

            let filtered = source.filter(s => !searchVal || s.ticker.includes(searchVal));

            if (currentSortMode === 'alphabetical') filtered.sort((a,b) => a.ticker.localeCompare(b.ticker));
            else if (currentSortMode === 'gainers')  filtered.sort((a,b) => b.pct_change - a.pct_change);
            else if (currentSortMode === 'losers')   filtered.sort((a,b) => a.pct_change - b.pct_change);

            if (filtered.length === 0) {
                tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;padding:2rem;color:var(--text-muted)">Không tìm thấy mã phù hợp.</td></tr>`;
                return;
            }

            filtered.forEach((stock, idx) => {
                const cls   = stock.change > 0 ? 'text-up' : (stock.change < 0 ? 'text-down' : 'text-flat');
                const sign  = stock.change > 0 ? '+' : '';
                const isWL  = watchlist.includes(stock.ticker);
                const isAct = stock.ticker === activeTicker;

                const btn = isWL
                    ? `<button class="btn-icon btn-icon-remove" onclick="removeFromWatchlist('${stock.ticker}',event)" title="Xoá khỏi Watchlist"><i class="fa-solid fa-minus"></i></button>`
                    : `<button class="btn-icon btn-icon-add" onclick="addToWatchlist('${stock.ticker}',event)" title="Thêm vào Watchlist"><i class="fa-solid fa-plus"></i></button>`;

                const tr = document.createElement('tr');
                if (isAct) tr.classList.add('row-active');
                tr.style.cursor = 'pointer';
                tr.style.animationDelay = `${idx * 0.02}s`;
                tr.classList.add('row-fade-in');
                tr.innerHTML = `
                    <td class="ticker-col">${stock.ticker}</td>
                    <td class="price-col">${formatPrice(stock.current_price)}</td>
                    <td class="price-col ${cls}">${sign}${stock.pct_change.toFixed(2)}%</td>
                    <td style="text-align:right;padding-right:1rem" onclick="event.stopPropagation()">${btn}</td>
                `;
                tr.addEventListener('click', () => selectTicker(stock.ticker));
                tbody.appendChild(tr);
            });
        }

        function filterMarketData() { renderMarketTable(); }

        function setMarketSort(mode) {
            currentSortMode = mode;
            ['alphabetical','gainers','losers'].forEach(m => {
                const id = m === 'alphabetical' ? 'sort-alphabetical' : (m === 'gainers' ? 'sort-top-gainers' : 'sort-top-losers');
                document.getElementById(id)?.classList.toggle('active', m === mode);
            });
            renderMarketTable();
            document.querySelector('#market-tab-content .table-container')?.scrollTo(0, 0);
        }

        /* ====================================================
           WATCHLIST TABLE RENDER
        ==================================================== */
        function setWatchlistSort(mode) {
            watchlistSortMode = mode;
            ['alpha','change','score'].forEach(m => {
                const id = m === 'alpha' ? 'wl-sort-alpha' : (m === 'change' ? 'wl-sort-change' : 'sort-ai-score');
                document.getElementById(id)?.classList.toggle('active', m === mode);
            });
            renderWatchlistTable();
        }

        function renderWatchlistTable() {
            const tbody = document.getElementById('watchlist-table-body');
            tbody.innerHTML = '';

            if (watchlist.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:3rem;color:var(--text-muted)">
                    Watchlist trống. Nhấn <strong style="color:var(--color-buy)">+</strong> ở tab VN100 / VNAllShare để thêm.
                </td></tr>`;
                return;
            }

            // Sort watchlist
            let sorted = [...watchlist];
            if (watchlistSortMode === 'alpha') {
                sorted.sort((a, b) => a.localeCompare(b));
            } else if (watchlistSortMode === 'change') {
                sorted.sort((a, b) => {
                    const sA = getStockData(a), sB = getStockData(b);
                    return (sB?.pct_change || 0) - (sA?.pct_change || 0);
                });
            } else if (watchlistSortMode === 'score') {
                sorted.sort((a, b) => {
                    const pA = getModelScore(a), pB = getModelScore(b);
                    return pB - pA;
                });
            }

            sorted.forEach((ticker, idx) => {
                const stock = getStockData(ticker);
                const pred  = predictions[ticker];

                const cls       = stock && stock.change > 0 ? 'text-up' : (stock && stock.change < 0 ? 'text-down' : 'text-flat');
                const sign      = stock && stock.change > 0 ? '+' : '';
                const priceText = stock ? formatPrice(stock.current_price) : '-';
                const chgText   = stock ? `${sign}${stock.pct_change.toFixed(2)}%` : '-';

                let scoreHtml = '<span style="color:var(--text-muted)">-</span>';
                let recHtml   = `<span class="badge" style="background:rgba(255,255,255,0.03);color:var(--text-muted);border:1px solid var(--panel-border)">Chưa phân tích</span>`;

                if (pred) {
                    if (pred.status === 'loading') {
                        scoreHtml = `<span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;font-size:0.8rem">...</span>`;
                        recHtml   = `<span class="badge" style="background:rgba(59,130,246,0.06);color:#60a5fa;border:1px solid rgba(59,130,246,0.2)"><i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích</span>`;
                    } else if (pred.status === 'success') {
                        const md  = pred.models?.[activeModel] || pred;
                        const sc  = md.score !== undefined ? md.score : pred.score;
                        const ac  = md.action_class || pred.action_class;
                        const rec = md.recommendation || pred.recommendation;

                        const chipClass = sc >= 1.5 ? 'positive' : (sc <= -1.5 ? 'negative' : 'neutral');
                        scoreHtml = `<span class="score-chip ${chipClass}">${sc >= 0 ? '+' : ''}${sc.toFixed(1)}</span>`;

                        let badgeClass = 'badge-hold';
                        if (ac === 'buy')  badgeClass = 'badge-buy';
                        if (ac === 'sell') badgeClass = 'badge-sell';
                        recHtml = `<span class="badge ${badgeClass}">${rec}</span>`;
                    } else if (pred.status === 'error') {
                        scoreHtml = `<span style="color:var(--color-sell);font-size:0.75rem">Lỗi</span>`;
                        recHtml   = `<span class="badge badge-sell" title="${pred.message || ''}">Lỗi dữ liệu</span>`;
                    }
                }

                const isAct = ticker === activeTicker;
                const tr = document.createElement('tr');
                if (isAct) tr.classList.add('row-active');
                tr.style.cursor = 'pointer';
                tr.style.animationDelay = `${idx * 0.03}s`;
                tr.classList.add('row-fade-in');
                tr.innerHTML = `
                    <td class="ticker-col">${ticker}</td>
                    <td class="price-col">${priceText}</td>
                    <td class="price-col ${cls}">${chgText}</td>
                    <td>${scoreHtml}</td>
                    <td>${recHtml}</td>
                    <td style="text-align:right;padding-right:0.75rem" onclick="event.stopPropagation()">
                        <button class="btn-icon btn-icon-remove" onclick="removeFromWatchlist('${ticker}',event)" title="Xoá khỏi Watchlist">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </td>
                `;
                tr.addEventListener('click', () => selectTicker(ticker));
                tbody.appendChild(tr);
            });
        }

        function getStockData(ticker) {
            return vnallData.find(s => s.ticker === ticker) || vn100Data.find(s => s.ticker === ticker) || null;
        }

        function getModelScore(ticker) {
            const pred = predictions[ticker];
            if (!pred || pred.status !== 'success') return -99;
            const md = pred.models?.[activeModel] || pred;
            return md.score !== undefined ? md.score : (pred.score || 0);
        }

        /* ====================================================
           ADD / REMOVE WATCHLIST
        ==================================================== */
        async function addToWatchlist(ticker, event) {
            if (event) event.stopPropagation();
            ticker = ticker.toUpperCase();
            if (watchlist.includes(ticker)) return;

            try {
                const res = await fetch('/api/watchlist', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker })
                });
                if (res.ok) {
                    watchlist.push(ticker);
                    document.getElementById('watchlist-count').textContent = watchlist.length;
                    renderMarketTable();
                    renderWatchlistTable();
                    showToast(`Đã thêm ${ticker} vào Watchlist`, "success");
                    runPredictionForTicker(ticker, true);
                } else {
                    showToast(`Lỗi khi lưu ${ticker}`, "error");
                }
            } catch (e) {
                showToast(`Không thể kết nối máy chủ`, "error");
            }
        }

        async function removeFromWatchlist(ticker, event) {
            if (event) event.stopPropagation();
            ticker = ticker.toUpperCase();

            try {
                const res = await fetch(`/api/watchlist/${ticker}`, { method: 'DELETE' });
                if (res.ok) {
                    watchlist = watchlist.filter(t => t !== ticker);
                    document.getElementById('watchlist-count').textContent = watchlist.length;
                    renderMarketTable();
                    renderWatchlistTable();

                    if (activeTicker === ticker) {
                        if (watchlist.length > 0) selectTicker(watchlist[0]);
                        else { activeTicker = ''; showEmptyDetail(); }
                    }
                    showToast(`Đã xoá ${ticker} khỏi Watchlist`, "info");
                } else {
                    showToast(`Lỗi khi xóa ${ticker}`, "error");
                }
            } catch (e) {
                showToast(`Không thể kết nối máy chủ`, "error");
            }
        }

        function showEmptyDetail() {
            document.getElementById('detail-content').style.display = 'none';
            document.getElementById('detail-empty').style.display = 'flex';
            
            // Hide stock navigation menu
            const stockNav = document.getElementById('stock-nav-menu-container');
            if (stockNav) stockNav.style.display = 'none';
        }

        /* ====================================================
           SELECT TICKER
           ==================================================== */
        function selectTicker(ticker) {
            activeTicker = ticker;
            ensureSidebarExpanded();

            // Switch tab if we are currently looking at gold or calendar dashboard
            if (activeSidebarTab === 'gold' || activeSidebarTab === 'calendar') {
                if (watchlist.includes(ticker)) switchSidebarTab('watchlist');
                else if (vn100Data.some(s => s.ticker === ticker)) switchSidebarTab('vn100');
                else switchSidebarTab('vnall');
            }

            // Hide gold & calendar dashboards
            document.getElementById('gold-dashboard-content').style.display = 'none';
            const calPanel = document.getElementById('calendar-dashboard-content');
            if (calPanel) calPanel.style.display = 'none';

            // Update row active states
            document.querySelectorAll('#market-table-body tr').forEach(row => {
                const col = row.querySelector('.ticker-col');
                if (col) row.classList.toggle('row-active', col.textContent === ticker);
            });
            document.querySelectorAll('#watchlist-table-body tr').forEach(row => {
                const col = row.querySelector('.ticker-col');
                if (col) row.classList.toggle('row-active', col.textContent === ticker);
            });

            const stock = getStockData(ticker);
            if (!stock) { showEmptyDetail(); return; }

            document.getElementById('detail-empty').style.display = 'none';
            document.getElementById('detail-content').style.display = 'flex';

            // Price header
            document.getElementById('detail-ticker-name').textContent = stock.ticker;
            document.getElementById('detail-price-val').textContent = formatPrice(stock.current_price);

            const sign = stock.change > 0 ? '+' : '';
            const cvalEl = document.getElementById('detail-change-val');
            const cpctEl = document.getElementById('detail-change-pct');
            cvalEl.textContent = `${sign}${stock.change.toLocaleString('vi-VN')}đ`;
            cpctEl.textContent = `(${sign}${stock.pct_change.toFixed(2)}%)`;
            const colorCls = stock.change > 0 ? 'text-up' : (stock.change < 0 ? 'text-down' : 'text-flat');
            cvalEl.className = colorCls;
            cpctEl.className = colorCls;

            // Unpredicted label
            const ul = document.getElementById('unpredicted-ticker-label');
            if (ul) ul.textContent = ticker;

            const pred = predictions[ticker];
            if (pred && pred.status === 'loading') {
                setDetailState('loading');
            } else if (pred && pred.status === 'success') {
                setDetailState('predicted');
                renderDetailsPanel(pred);
            } else {
                setDetailState('unpredicted');
            }
        }

        function setDetailState(state) {
            document.getElementById('detail-loader').style.display       = state === 'loading'   ? 'flex' : 'none';
            document.getElementById('ai-unpredicted-state').style.display = state === 'unpredicted' ? 'flex' : 'none';
            document.getElementById('ai-predicted-view').style.display   = state === 'predicted' ? 'flex' : 'none';

            // Toggle stock navigation menu in the left sidebar
            const stockNav = document.getElementById('stock-nav-menu-container');
            if (stockNav) {
                const isStockActive = ['watchlist', 'vn100', 'vnall'].includes(activeSidebarTab);
                stockNav.style.display = (state === 'predicted' && isStockActive) ? 'block' : 'none';
                
                // Update ticker label inside the navigation header
                const label = document.getElementById('stock-nav-ticker-label');
                if (label) label.textContent = activeTicker;
            }
        }

        /* ====================================================
           PREDICTION RUNNER
        ==================================================== */
        async function runPredictionForTicker(ticker, isSilent = false) {
            if (!ticker) return;

            predictions[ticker] = { status: 'loading' };
            renderWatchlistTable();

            if (activeTicker === ticker) setDetailState('loading');
            if (!isSilent) showToast(`Đang phân tích mã ${ticker}...`, "info");

            try {
                const res = await fetch('/api/predict/single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.message || err.detail || "Lỗi máy chủ.");
                }

                const result = await res.json();
                predictions[ticker] = result;

                if (activeTicker === ticker) {
                    activePredictionData = result;
                    setDetailState('predicted');
                    renderDetailsPanel(result);
                }

                renderWatchlistTable();
                if (!isSilent) showToast(`Phân tích ${ticker} thành công!`, "success");

            } catch (err) {
                predictions[ticker] = { status: 'error', message: err.message };
                renderWatchlistTable();
                if (!isSilent) showToast(err.message, "error");
                if (activeTicker === ticker) {
                    setDetailState('unpredicted');
                    const ul = document.getElementById('unpredicted-ticker-label');
                    if (ul) ul.textContent = `${ticker} (Lỗi: ${err.message})`;
                }
            } finally {
                if (activeTicker === ticker) {
                    document.getElementById('detail-loader').style.display = 'none';
                }
            }
        }

        function runPredictionForActiveTicker() {
            runPredictionForTicker(activeTicker, false);
        }

        async function runBatchPredictionForWatchlist() {
            if (watchlist.length === 0) return;

            watchlist.forEach(ticker => {
                if (!predictions[ticker] || predictions[ticker].status === 'error') {
                    predictions[ticker] = { status: 'loading' };
                }
            });
            renderWatchlistTable();

            if (watchlist.includes(activeTicker) && predictions[activeTicker]?.status === 'loading') {
                setDetailState('loading');
            }

            showToast("Đang chạy AI phân tích Watchlist...", "info");

            try {
                const res = await fetch('/api/predict/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tickers: watchlist })
                });

                if (!res.ok) throw new Error("Lỗi phân tích hàng loạt.");

                const results = await res.json();
                Object.assign(predictions, results);

                renderWatchlistTable();

                if (watchlist.includes(activeTicker)) {
                    const ap = predictions[activeTicker];
                    if (ap?.status === 'success') {
                        activePredictionData = ap;
                        setDetailState('predicted');
                        renderDetailsPanel(ap);
                    } else if (ap?.status === 'error') {
                        setDetailState('unpredicted');
                    }
                }

                showToast("Đã phân tích xong toàn bộ Watchlist!", "success");

            } catch (err) {
                showToast(err.message, "error");
                watchlist.forEach(t => {
                    if (predictions[t]?.status === 'loading') delete predictions[t];
                });
                renderWatchlistTable();
                if (watchlist.includes(activeTicker)) {
                    setDetailState('unpredicted');
                }
            }
        }

        /* ====================================================
           MODEL SWITCHING
        ==================================================== */
        function changePredictionModel(modelName) {
            activeModel = modelName;
            if (activePredictionData) renderDetailsPanel(activePredictionData);
            renderWatchlistTable();
        }

        /* ====================================================
           RENDER DETAILS PANEL
        ==================================================== */
        function renderDetailsPanel(data) {
            activePredictionData = data;

            document.getElementById('detail-data-source').textContent = `AI Model — ${data.crawler_source}`;

            const md = data.models?.[activeModel] || data;
            const mlPred  = md.ml_prediction || data.ml_prediction;
            const reasons = md.reasons     || data.reasons;

            // Model select sync
            const msel = document.getElementById('model-select');
            if (msel) msel.value = activeModel;

            // Forecast label
            const modelLabels = {
                random_forest:    '🧠 Random Forest — 5 Phiên Kế Tiếp (1 Tuần)',
                linear_regression:'📈 Linear Regression — 5 Phiên Kế Tiếp (1 Tuần)',
                mlp:              '🧠 Mạng Nơ-ron MLP — 5 Phiên Kế Tiếp (1 Tuần)',
                xgboost:          '🚀 XGBoost Regressor — 5 Phiên Kế Tiếp (1 Tuần)',
                lstm:             '🧠 Mạng LSTM — 5 Phiên Kế Tiếp (1 Tuần)',
                ensemble:         '✨ Ensemble Hybrid — 5 Phiên Kế Tiếp (1 Tuần)'
            };

            const flEl = document.getElementById('forecast-model-label');
            if (flEl) flEl.textContent = modelLabels[activeModel] || 'Dự Báo 5 Phiên Kế Tiếp (1 Tuần)';

            /* -- Score Gauge -- */
            const scoreVal = parseFloat(md.score !== undefined ? md.score : data.score);
            const scoreNumEl  = document.getElementById('detail-score-val');
            const fillCircle  = document.getElementById('score-meter-fill');
            const recBadge    = document.getElementById('rec-badge-large');
            const recBadgeLbl = document.getElementById('rec-badge-label');

            const scorePercent    = (scoreVal + 5) / 10;
            const circumference   = 2 * Math.PI * 55; // r=55
            const offset          = circumference - (scorePercent * circumference);

            scoreNumEl.textContent = (scoreVal >= 0 ? '+' : '') + scoreVal.toFixed(1);
            fillCircle.style.strokeDasharray  = `${circumference}`;
            fillCircle.style.strokeDashoffset = offset;

            let recClass = 'hold', recIcon = 'fa-circle-info', recText = 'THEO DÕI / GIỮ';
            if (scoreVal >= 1.5) {
                fillCircle.style.stroke = 'var(--color-buy)';
                scoreNumEl.style.color  = 'var(--color-buy)';
                recClass = 'buy'; recIcon = 'fa-arrow-trend-up'; recText = 'NÊN MUA';
            } else if (scoreVal <= -1.5) {
                fillCircle.style.stroke = 'var(--color-sell)';
                scoreNumEl.style.color  = 'var(--color-sell)';
                recClass = 'sell'; recIcon = 'fa-arrow-trend-down'; recText = 'NÊN BÁN';
            } else {
                fillCircle.style.stroke = 'var(--color-hold)';
                scoreNumEl.style.color  = 'var(--color-hold)';
            }

            recBadge.className = `rec-badge-large ${recClass}`;
            recBadge.innerHTML = `<i class="fa-solid ${recIcon}"></i><span>${recText}</span>`;

            /* -- Reasons -- */
            const reasonsUl = document.getElementById('detail-reasons-list');
            reasonsUl.innerHTML = '';
            reasons.forEach((r, i) => {
                const li = document.createElement('li');
                li.textContent = r;
                li.style.animationDelay = `${i * 0.06}s`;
                if (r.includes('tăng') || r.includes('MUA') || r.includes('Vàng') || r.includes('hồi phục') || r.includes('bật tăng') || r.includes('Oversold')) {
                    li.classList.add('positive');
                } else if (r.includes('giảm') || r.includes('BÁN') || r.includes('Tử thần') || r.includes('áp lực') || r.includes('chốt lời') || r.includes('Overbought')) {
                    li.classList.add('negative');
                }
                reasonsUl.appendChild(li);
            });

            /* -- Actionable Advice & Timing -- */
            const advice = md.actionable_advice || data.actionable_advice;
            if (advice) {
                document.getElementById('actionable-advice-section').style.display = 'block';
                
                const conclusionCard = document.getElementById('advice-conclusion-card');
                const timingCard     = document.getElementById('advice-timing-card');
                
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
                if (advice.best_buy_time.includes("Chờ")) {
                    timingCard.classList.add('sell-advice');
                } else if (advice.best_buy_time.includes("Sáng")) {
                    timingCard.classList.add('buy-advice');
                } else {
                    timingCard.classList.add('hold-advice');
                }
                
                document.getElementById('advice-conclusion-val').textContent = advice.buy_now_conclusion;
                document.getElementById('advice-conclusion-sub').textContent = advice.buy_now_subtext;
                
                document.getElementById('advice-timing-val').textContent = advice.best_buy_time;
                document.getElementById('advice-timing-sub').textContent = advice.best_buy_time_reason;
                
                // Expected Next Price
                const nextPriceValEl = document.getElementById('advice-next-price-val');
                const nextChange = advice.next_price_change_pct;
                const nextChangeSign = nextChange > 0 ? '+' : '';
                const nextChangeCls = nextChange > 0 ? 'text-up' : (nextChange < 0 ? 'text-down' : 'text-flat');
                nextPriceValEl.innerHTML = `${formatPrice(Math.round(advice.next_price))} <span class="${nextChangeCls}" style="font-size:0.75rem;font-weight:normal;margin-left:0.2rem">(${nextChangeSign}${nextChange.toFixed(2)}%)</span>`;
                
                // Buy Range
                document.getElementById('advice-buy-range-val').textContent = advice.target_buy_range;
                
                // Expected Profit
                const profitValEl = document.getElementById('advice-expected-profit-val');
                if (advice.expected_profit_pct > 0) {
                    profitValEl.innerHTML = `+${advice.expected_profit_pct.toFixed(2)}% <span style="font-size:0.72rem;font-weight:normal;color:var(--text-secondary)">(${formatPrice(Math.round(advice.expected_profit_amount))})</span>`;
                    profitValEl.className = 'item-value text-up';
                } else {
                    profitValEl.textContent = '0.00% (0đ)';
                    profitValEl.className = 'item-value text-flat';
                }
                
                // Stop Loss
                document.getElementById('advice-stop-loss-val').textContent = formatPrice(Math.round(advice.stop_loss_price));
            } else {
                document.getElementById('actionable-advice-section').style.display = 'none';
            }

            /* -- Forecast Grid -- */
            const fGrid = document.getElementById('detail-forecast-grid');
            fGrid.innerHTML = '';
            const r2 = mlPred.r_squared || 0;

            mlPred.predicted_prices.forEach((price, i) => {
                const diff = price - data.current_price;
                const pct  = (diff / data.current_price) * 100;
                const isUp = pct > 0;
                const dayCls   = isUp ? 'text-up' : (pct < 0 ? 'text-down' : 'text-flat');
                const daySgn   = isUp ? '+' : '';
                const dirClass = isUp ? 'up' : (pct < 0 ? 'down' : '');
                const arrow    = isUp ? '▲' : (pct < 0 ? '▼' : '▬');
                const arrowClr = isUp ? 'var(--color-up)' : (pct < 0 ? 'var(--color-down)' : 'var(--text-muted)');

                const confPct  = Math.max(20, Math.min(95, r2 * 80 + 15));
                const confClr  = r2 >= 0.6 ? 'var(--color-buy)' : (r2 >= 0.3 ? 'var(--color-hold)' : 'var(--color-sell)');

                const div = document.createElement('div');
                div.className = `forecast-day ${dirClass}`;
                div.innerHTML = `
                    <span class="day-label">Phiên +${i + 1}</span>
                    <span class="day-arrow" style="color:${arrowClr}">${arrow}</span>
                    <span class="day-price">${formatPrice(Math.round(price))}</span>
                    <span class="day-change ${dayCls}">${daySgn}${pct.toFixed(2)}%</span>
                    <div class="confidence-bar-wrap">
                        <span class="confidence-label">R² ${(r2 * 100).toFixed(0)}%</span>
                        <div class="confidence-bar-track">
                            <div class="confidence-bar-fill" style="width:${confPct}%;background:${confClr}"></div>
                        </div>
                    </div>
                `;
                fGrid.appendChild(div);
            });

            // Render Model Comparison Table
            const compTbody = document.getElementById('model-comparison-tbody');
            if (compTbody && data.models) {
                compTbody.innerHTML = '';
                
                const modelKeys = ['random_forest', 'linear_regression', 'mlp', 'xgboost', 'ensemble'];
                const modelDisplayNames = {
                    random_forest: '🧠 Random Forest',
                    linear_regression: '📈 Linear Regression',
                    mlp: '🧠 MLP Neural Network (Học Sâu)',
                    xgboost: '🚀 XGBoost Regressor',
                    ensemble: '✨ Ensemble Hybrid'
                };
                
                // Detect fallback
                const rfData = data.models.random_forest;
                const isFallback = rfData && rfData.ml_prediction && rfData.ml_prediction.model_name && rfData.ml_prediction.model_name.includes('Fallback');
                
                // Show/hide note container
                const compNote = document.getElementById('model-comparison-note');
                if (compNote) {
                    compNote.style.display = isFallback ? 'block' : 'none';
                }
                
                modelKeys.forEach(key => {
                    const mData = data.models[key];
                    if (!mData) return;
                    
                    const mScore = mData.score;
                    const mRec = mData.recommendation;
                    const mAction = mData.action_class;
                    const mDir = mData.direction;
                    const mDirClass = mData.direction_class;
                    
                    const mMl = mData.ml_prediction || {};
                    const mChange = mMl.predicted_pct_change || 0;
                    const mR2 = mMl.r_squared || 0;
                    const mPrices = mMl.predicted_prices || [];
                    const mTargetPrice = mPrices.length > 0 ? mPrices[mPrices.length - 1] : 0;
                    
                    // Style recommendation and score
                    let recBadgeClass = 'badge-hold';
                    if (mAction === 'buy') recBadgeClass = 'badge-buy';
                    if (mAction === 'sell') recBadgeClass = 'badge-sell';
                    
                    const chipClass = mScore >= 1.5 ? 'positive' : (mScore <= -1.5 ? 'negative' : 'neutral');
                    const scoreHtml = `<span class="score-chip ${chipClass}" style="padding:0.15rem 0.4rem;font-size:0.75rem">${mScore >= 0 ? '+' : ''}${mScore.toFixed(1)}</span>`;
                    
                    // Direction classes and styling
                    const dirCls = mDirClass === 'up' ? 'text-up' : (mDirClass === 'down' ? 'text-down' : 'text-flat');
                    const dirArrow = mDirClass === 'up' ? '▲' : (mDirClass === 'down' ? '▼' : '▬');
                    
                    // Change percent
                    const changeCls = mChange > 0 ? 'text-up' : (mChange < 0 ? 'text-down' : 'text-flat');
                    const changeSign = mChange > 0 ? '+' : '';
                    
                    // R2 color
                    const r2Clr = mR2 >= 0.6 ? 'var(--color-buy)' : (mR2 >= 0.3 ? 'var(--color-hold)' : 'var(--color-sell)');
                    
                    const isActive = key === activeModel;
                    const tr = document.createElement('tr');
                    if (isActive) {
                        tr.classList.add('model-row-active');
                    }
                    tr.style.cursor = 'pointer';
                    
                    let dispName = modelDisplayNames[key];
                    if (key === 'random_forest' && isFallback) {
                        dispName = `🧠 Random Forest <span style="font-size:0.65rem;color:#f59e0b;background:rgba(245,158,11,0.1);padding:1px 4px;border-radius:3px;margin-left:0.2rem;font-weight:normal;" title="Không đủ dữ liệu lịch sử để chạy Random Forest, tự động fallback về Hồi quy tuyến tính">Fallback</span>`;
                        tr.title = `Chuyển sang xem Random Forest (Đang ở chế độ Fallback về Linear Regression)`;
                    } else {
                        tr.title = `Chuyển sang xem ${modelDisplayNames[key]}`;
                    }
                    
                    tr.innerHTML = `
                        <td class="model-name-cell">${dispName} ${isActive ? '<span style="font-size:0.65rem;color:var(--accent-primary);margin-left:0.2rem;font-weight:normal;">(Đang chọn)</span>' : ''}</td>
                        <td><span class="badge ${recBadgeClass}" style="font-size:0.7rem;padding:0.15rem 0.35rem;border-radius:3px">${mRec}</span></td>
                        <td>${scoreHtml}</td>
                        <td class="${dirCls}"><span style="margin-right:0.2rem;font-size:0.7rem">${dirArrow}</span>${mDir}</td>
                        <td class="${changeCls}">${changeSign}${mChange.toFixed(2)}%</td>
                        <td class="price-col">${formatPrice(Math.round(mTargetPrice))}</td>
                        <td style="font-family:'JetBrains Mono',monospace;color:${r2Clr};font-weight:600">${(mR2 * 100).toFixed(0)}%</td>
                    `;
                    
                    tr.addEventListener('click', () => {
                        changePredictionModel(key);
                    });
                    
                    compTbody.appendChild(tr);
                });
            }

            // Render Prediction Verification Table
            const verifyTbody = document.getElementById('prediction-verification-tbody');
            if (verifyTbody) {
                verifyTbody.innerHTML = '';
                
                const modelNames = {
                    random_forest: '🧠 Random Forest',
                    linear_regression: '📈 Linear Regression',
                    mlp: '🧠 MLP Neural Network (Học Sâu)',
                    xgboost: '🚀 XGBoost Regressor',
                    ensemble: '✨ Ensemble Hybrid'
                };
                
                const rows = [];
                const hasOldAnalysis = data.old_analysis && data.old_analysis.models && data.old_analysis.models[activeModel];
                
                // 1. Live Row: Today's real-time comparison
                const stock = getStockData(data.ticker);
                const history = data.history || [];
                
                let showLiveRow = false;
                let predT1 = null;
                let actualPrice = null;
                
                if (stock && history.length > 0) {
                    const latestHistoryRow = history[history.length - 1];
                    const todayStr = new Date().toISOString().slice(0, 10);
                    
                    if (latestHistoryRow.date !== todayStr) {
                        // Case A: History doesn't have today's date. Price is from snapshot, T+1 prediction is from main data.
                        const activeMd = data.models?.[activeModel] || data;
                        const activeMl = activeMd.ml_prediction || data.ml_prediction;
                        predT1 = activeMl.predicted_prices ? activeMl.predicted_prices[0] : null;
                        actualPrice = stock.current_price;
                        showLiveRow = (predT1 !== null && actualPrice !== null);
                    } else if (hasOldAnalysis) {
                        // Case B: History has today's date because it was extended. Price is from snapshot, T+1 prediction is from old_analysis.
                        const oldMd = data.old_analysis.models[activeModel];
                        const oldMl = oldMd.ml_prediction;
                        predT1 = oldMl.predicted_prices ? oldMl.predicted_prices[0] : null;
                        actualPrice = stock.current_price;
                        showLiveRow = (predT1 !== null && actualPrice !== null);
                    }
                }
                
                if (showLiveRow) {
                    const diff = actualPrice - predT1;
                    const errPct = ((actualPrice - predT1) / (predT1 + 1e-10)) * 100;
                    
                    rows.push({
                        type: 'live',
                        label: `Hôm nay (Live - T+1)`,
                        modelName: modelNames[activeModel],
                        predPrice: predT1,
                        actualPrice: actualPrice,
                        diff: diff,
                        errPct: errPct
                    });
                }
                
                // 2. Historical Row: Previous session's comparison from backend
                if (data.backtest && data.backtest.models) {
                    const bModel = data.backtest.models[activeModel];
                    if (bModel) {
                        const actualPrice = data.backtest.actual_price;
                        const predPrice = bModel.predicted_price;
                        const diff = actualPrice - predPrice;
                        const errPct = bModel.error_pct;
                        
                        let dateStr = data.backtest.date;
                        if (dateStr) {
                            const p = dateStr.split('-');
                            if (p.length === 3) dateStr = `${p[2]}/${p[1]}`;
                        }
                        
                        rows.push({
                            type: 'history',
                            label: `Phiên trước (${dateStr} - T+1)`,
                            modelName: modelNames[activeModel],
                            predPrice: predPrice,
                            actualPrice: actualPrice,
                            diff: diff,
                            errPct: errPct
                        });
                    }
                }
                
                // Render rows to table
                if (rows.length === 0) {
                    verifyTbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:1.5rem;color:var(--text-muted)">Không có đủ dữ liệu đối chiếu.</td></tr>`;
                } else {
                    rows.forEach(r => {
                        const diffSign = r.diff > 0 ? '+' : '';
                        const errSign = r.errPct > 0 ? '+' : '';
                        const diffClass = r.diff > 0 ? 'text-up' : (r.diff < 0 ? 'text-down' : 'text-flat');
                        
                        const absErr = Math.abs(r.errPct);
                        let rating = 'Lệch nhiều';
                        let ratingClass = 'badge-sell';
                        if (absErr <= 1.0) {
                            rating = 'Rất chính xác';
                            ratingClass = 'badge-buy';
                        } else if (absErr <= 3.0) {
                            rating = 'Chính xác';
                            ratingClass = 'badge-hold';
                        }
                        
                        const tr = document.createElement('tr');
                        if (r.type === 'live') {
                            tr.style.background = 'rgba(59,130,246,0.04)';
                        }
                        
                        tr.innerHTML = `
                            <td style="font-weight:600;color:${r.type === 'live' ? '#60a5fa' : 'var(--text-primary)'}">
                                <div style="display:flex; flex-direction:column; gap:0.15rem;">
                                    <div>${r.type === 'live' ? '<i class="fa-solid fa-circle-play" style="margin-right:0.3rem;font-size:0.7rem;animation:pulse 2s infinite"></i>' : ''}${r.label}</div>
                                    <div style="font-size:0.7rem; color:var(--text-muted); font-weight:normal;">${r.modelName}</div>
                                </div>
                            </td>
                            <td class="price-col">${formatPrice(Math.round(r.predPrice))}</td>
                            <td class="price-col" style="font-weight:600;color:var(--text-primary)">${formatPrice(Math.round(r.actualPrice))}</td>
                            <td class="price-col">
                                <div style="display:flex; flex-direction:column; gap:0.15rem; font-size:0.78rem;">
                                    <span class="${diffClass}">${diffSign}${Math.round(r.diff).toLocaleString('vi-VN')}đ</span>
                                    <span class="${diffClass}" style="font-size:0.7rem;">${errSign}${r.errPct.toFixed(2)}%</span>
                                </div>
                            </td>
                            <td><span class="badge ${ratingClass}" style="font-size:0.65rem;padding:0.15rem 0.35rem;border-radius:3px">${rating}</span></td>
                        `;
                        verifyTbody.appendChild(tr);
                    });
                }
            }

            /* -- Stats -- */
            document.getElementById('stat-volume').textContent = formatVolume(data.volume);
            const sma20 = data.indicators?.sma_20;
            document.getElementById('stat-sma20').textContent = sma20 ? formatPrice(Math.round(sma20)) : '-';

            const r2el = document.getElementById('stat-r2');
            r2el.textContent = r2.toFixed(2);
            r2el.style.color = r2 >= 0.6 ? 'var(--color-buy)' : (r2 >= 0.3 ? 'var(--color-hold)' : 'var(--color-sell)');

            const mlPct   = mlPred.predicted_pct_change;
            const mlChEl  = document.getElementById('stat-ml-change');
            const mlSign  = mlPct > 0 ? '+' : '';
            mlChEl.textContent = `${mlSign}${mlPct.toFixed(1)}%`;
            mlChEl.className   = `value ${mlPct > 0 ? 'text-up' : (mlPct < 0 ? 'text-down' : 'text-flat')}`;

            /* -- Chart -- */
            renderChart(data, md);

            // Show BB toggle only in price tab
            updateBBToggleVisibility();
        }

        /* ====================================================
           BOLLINGER BANDS TOGGLE
        ==================================================== */
        function toggleBollingerBands(enabled) {
            showBollingerBands = enabled;
            if (activePredictionData) {
                const md = activePredictionData.models?.[activeModel] || activePredictionData;
                renderChart(activePredictionData, md);
            }
        }

        function updateBBToggleVisibility() {
            const wrap = document.getElementById('bb-toggle-wrap');
            if (wrap) wrap.style.display = activeChartTab === 'price' ? 'flex' : 'none';
        }

        /* ====================================================
           CHART RENDERING
        ==================================================== */
        function switchChartTab(tabName) {
            activeChartTab = tabName;
            ['price','rsi','macd'].forEach(t => {
                document.getElementById(`tab-${t}`)?.classList.toggle('active', t === tabName);
            });
            updateBBToggleVisibility();

            const data = predictions[activeTicker];
            if (data) {
                const md = data.models?.[activeModel] || data;
                renderChart(data, md);
            }
        }

        function renderChart(data, modelData) {
            const canvas = document.getElementById('main-stock-chart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');

            if (stockChart) { stockChart.destroy(); stockChart = null; }

            const md      = modelData || data;
            const mlPred  = md.ml_prediction || data.ml_prediction;
            const score   = md.score !== undefined ? md.score : data.score;
            const history = data.history;

            const labels = history.map(h => {
                const p = h.date.split('-');
                return `${p[2]}/${p[1]}`;
            });

            /* ---- PRICE CHART ---- */
            if (activeChartTab === 'price') {
                const stock = getStockData(data.ticker);
                const closePrices = history.map(h => h.close);
                const preds       = mlPred.predicted_prices;
                const chartLabels = [...labels];

                const hasOldAnalysis = data.old_analysis && data.old_analysis.models && data.old_analysis.models[activeModel];

                // 1. Actual price line (solid blue)
                const actualLineData = [...closePrices];
                for (let i = 0; i < preds.length; i++) actualLineData.push(null);

                // 2. New prediction line (starts at last history element, i.e. Monday close, and projects T+1 to T+5)
                const predLine = [...Array(history.length - 1).fill(null), closePrices[closePrices.length - 1]];
                preds.forEach(p => predLine.push(p));

                // Add labels for new forecast T+1 to T+5
                for (let i = 1; i <= preds.length; i++) {
                    chartLabels.push(`T+${i}`);
                }

                // 3. Old prediction line (starts at second-to-last history element, i.e. Friday close, and projects T+1 to T+5)
                let oldPredLine = null;
                if (hasOldAnalysis) {
                    const oldMd = data.old_analysis.models[activeModel];
                    const oldMl = oldMd.ml_prediction;
                    const oldPreds = oldMl.predicted_prices;

                    // Starts at Friday (index history.length - 2)
                    oldPredLine = [...Array(history.length - 2).fill(null), closePrices[history.length - 2]];
                    oldPreds.forEach(p => oldPredLine.push(p));

                    // Pad to match chartLabels length
                    while (oldPredLine.length < chartLabels.length) {
                        oldPredLine.push(null);
                    }
                }

                const modelColorMap = { random_forest: '#a78bfa', linear_regression: '#60a5fa', ensemble: '#34d399' };
                const predColor     = modelColorMap[activeModel] || '#f59e0b';
                const forecastColor = score >= 1.5 ? '#10b981' : (score <= -1.5 ? '#ef4444' : predColor);

                const modelNameMap  = { random_forest: 'Random Forest', linear_regression: 'Linear Regression', ensemble: 'Ensemble' };

                const datasets = [
                    {
                        label: 'Giá đóng cửa thực tế',
                        data: actualLineData,
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        pointRadius: hasOldAnalysis ? [ ...Array(history.length - 1).fill(0), 4, ...Array(preds.length).fill(0) ] : 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#3b82f6',
                        fill: {
                            target: 'origin',
                            above: 'rgba(59,130,246,0.04)'
                        },
                        tension: 0.2
                    },
                    {
                        label: `Dự báo mới (Phiên này)`,
                        data: predLine,
                        borderColor: '#10b981', // vibrant green
                        borderWidth: 2.5,
                        borderDash: [5, 4],
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#fff',
                        fill: false,
                        tension: 0.1
                    }
                ];

                // Append old prediction line if available
                if (hasOldAnalysis) {
                    datasets.push({
                        label: `Dự báo cũ (Phiên trước)`,
                        data: oldPredLine,
                        borderColor: '#f59e0b', // vibrant orange
                        borderWidth: 2,
                        borderDash: [2, 2],
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#fff',
                        fill: false,
                        tension: 0.1
                    });
                }

                // Historical verification point (T+1 prediction for last session)
                const bModel = data.backtest && data.backtest.models ? data.backtest.models[activeModel] : null;
                if (bModel && !hasOldAnalysis) {
                    const prevPredPrice = bModel.predicted_price;
                    const prevPredLine = [...Array(history.length - 1).fill(null), prevPredPrice];
                    for (let i = 0; i < preds.length; i++) prevPredLine.push(null);

                    datasets.push({
                        label: 'Dự báo phiên trước (T+1)',
                        data: prevPredLine,
                        borderColor: '#8b5cf6',
                        backgroundColor: '#8b5cf6',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointStyle: 'triangle',
                        showLine: false,
                        fill: false
                    });
                }

                // Bollinger Bands overlay
                if (showBollingerBands) {
                    const bbUpper = history.map(h => h.bb_upper || null);
                    const bbLower = history.map(h => h.bb_lower || null);
                    const bbMid   = history.map(h => h.bb_middle || null);

                    datasets.push(
                        {
                            label: 'BB Upper',
                            data: [...bbUpper, ...Array(preds.length).fill(null)],
                            borderColor: 'rgba(6,182,212,0.6)',
                            borderWidth: 1,
                            borderDash: [3, 3],
                            pointRadius: 0,
                            fill: false,
                            tension: 0.2
                        },
                        {
                            label: 'BB Middle (SMA20)',
                            data: [...bbMid, ...Array(preds.length).fill(null)],
                            borderColor: 'rgba(245,158,11,0.5)',
                            borderWidth: 1,
                            borderDash: [2, 4],
                            pointRadius: 0,
                            fill: false,
                            tension: 0.2
                        },
                        {
                            label: 'BB Lower',
                            data: [...bbLower, ...Array(preds.length).fill(null)],
                            borderColor: 'rgba(6,182,212,0.6)',
                            borderWidth: 1,
                            borderDash: [3, 3],
                            pointRadius: 0,
                            fill: '-1', // fill between upper and lower
                            backgroundColor: 'rgba(6,182,212,0.04)',
                            tension: 0.2
                        }
                    );
                }

                stockChart = new Chart(ctx, {
                    type: 'line',
                    data: { labels: chartLabels, datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: {
                                    color: '#64748b',
                                    font: { family: 'Inter', size: 11 },
                                    boxWidth: 20,
                                    usePointStyle: true,
                                    pointStyle: 'line'
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(12,17,32,0.95)',
                                borderColor: 'rgba(255,255,255,0.08)',
                                borderWidth: 1,
                                titleColor: '#f1f5f9',
                                bodyColor: '#94a3b8',
                                padding: 10,
                                callbacks: {
                                    label: ctx => {
                                        if (ctx.raw === null) return null;
                                        return ` ${ctx.dataset.label}: ${formatPrice(Math.round(ctx.raw))}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { color: 'rgba(255,255,255,0.03)' },
                                ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 12 }
                            },
                            y: {
                                grid: { color: 'rgba(255,255,255,0.03)' },
                                ticks: {
                                    color: '#64748b', font: { size: 10 },
                                    callback: v => formatPriceShort(v)
                                }
                            }
                        }
                    }
                });

            /* ---- RSI CHART ---- */
            } else if (activeChartTab === 'rsi') {
                const rsiVals = history.map(h => h.rsi);

                stockChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [{
                            label: 'RSI (14)',
                            data: rsiVals,
                            borderColor: '#f59e0b',
                            borderWidth: 2,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            fill: false,
                            tension: 0.25
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(12,17,32,0.95)',
                                borderColor: 'rgba(255,255,255,0.08)',
                                borderWidth: 1,
                                titleColor: '#f1f5f9',
                                bodyColor: '#94a3b8',
                                callbacks: {
                                    label: ctx => ` RSI: ${ctx.raw?.toFixed(1)}`
                                }
                            },
                            annotation: {
                                annotations: {
                                    overbought: {
                                        type: 'line',
                                        yMin: 70, yMax: 70,
                                        borderColor: 'rgba(239,68,68,0.5)',
                                        borderWidth: 1,
                                        borderDash: [4, 3],
                                        label: { display: true, content: 'Quá Mua (70)', position: 'start', color: 'rgba(239,68,68,0.8)', font: { size: 10 } }
                                    },
                                    oversold: {
                                        type: 'line',
                                        yMin: 30, yMax: 30,
                                        borderColor: 'rgba(16,185,129,0.5)',
                                        borderWidth: 1,
                                        borderDash: [4, 3],
                                        label: { display: true, content: 'Quá Bán (30)', position: 'start', color: 'rgba(16,185,129,0.8)', font: { size: 10 } }
                                    },
                                    midline: {
                                        type: 'line',
                                        yMin: 50, yMax: 50,
                                        borderColor: 'rgba(255,255,255,0.06)',
                                        borderWidth: 1
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } },
                            y: {
                                min: 0, max: 100,
                                grid: { color: 'rgba(255,255,255,0.04)' },
                                ticks: { color: '#64748b', stepSize: 10, font: { size: 10 } }
                            }
                        }
                    }
                });

            /* ---- MACD CHART ---- */
            } else if (activeChartTab === 'macd') {
                const macdVal  = history.map(h => h.macd);
                const signalV  = history.map(h => h.macd_signal);
                const histV    = history.map(h => (h.macd - h.macd_signal) || 0);

                stockChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels,
                        datasets: [
                            {
                                type: 'line',
                                label: 'MACD',
                                data: macdVal,
                                borderColor: '#60a5fa',
                                borderWidth: 2,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.15,
                                order: 0
                            },
                            {
                                type: 'line',
                                label: 'Signal',
                                data: signalV,
                                borderColor: '#f43f5e',
                                borderWidth: 2,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.15,
                                order: 0
                            },
                            {
                                type: 'bar',
                                label: 'Histogram',
                                data: histV,
                                backgroundColor: histV.map(v => v >= 0 ? 'rgba(16,185,129,0.35)' : 'rgba(239,68,68,0.35)'),
                                borderWidth: 0,
                                order: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: {
                                labels: { color: '#64748b', font: { size: 11 }, boxWidth: 16, usePointStyle: true }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(12,17,32,0.95)',
                                borderColor: 'rgba(255,255,255,0.08)',
                                borderWidth: 1,
                                titleColor: '#f1f5f9',
                                bodyColor: '#94a3b8'
                            }
                        },
                        scales: {
                            x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } },
                            y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } }
                        }
                    }
                });
            }
        }

        // Onload stocks page initialization
        document.addEventListener("DOMContentLoaded", async () => {
            // Restore last active stock subtab
            const savedTab = localStorage.getItem('activeTab');
            const targetTab = ['watchlist', 'vn100', 'vnall'].includes(savedTab) ? savedTab : 'watchlist';
            switchSidebarTab(targetTab);

            // Continuous polling interval for active stock prediction
            setInterval(async () => {
                try {
                    const detailContent = document.getElementById('detail-content');
                    const predictedView = document.getElementById('ai-predicted-view');
                    if (activeTicker && detailContent && predictedView && detailContent.style.display === 'flex' && predictedView.style.display === 'flex') {
                        await runPredictionForTicker(activeTicker, true);
                    }
                } catch (err) {
                    console.error("Error in stocks continuous polling interval:", err);
                }
            }, 15000);
        });



