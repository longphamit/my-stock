        /* ====================================================
           VIETLOTT LOTTERY FUNCTIONS
        ==================================================== */
        async function selectVietlottDashboard() {
            switchSidebarTab('vietlott');
            
            // Activate the menu button
            selectVietlottGame(activeVietlottGame);
        }

        function selectVietlottGame(gameId) {
            activeVietlottGame = gameId;
            
            // Update active sidebar menu item
            const menuItems = document.querySelectorAll('.vietlott-nav-item');
            menuItems.forEach(item => {
                if (item.id === `vietlott-menu-${gameId}`) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });

            // Show main panel
            document.getElementById('vietlott-dashboard-content').style.display = 'block';
            
            fetchVietlottData(gameId);
        }

        async function fetchVietlottData(gameId) {
            // Guard: prevent duplicate fetches, but reset if stuck for >10 seconds
            const now = Date.now();
            if (isFetchingVietlott && (now - isFetchingVietlottTime) < 10000) {
                console.log('[Vietlott] Fetch already in progress, skipping...');
                return;
            }
            isFetchingVietlott = true;
            isFetchingVietlottTime = now;
            console.log('[Vietlott] Fetching data for:', gameId);
            
            const loader = document.getElementById('vietlott-dashboard-loader');
            const body = document.getElementById('vietlott-dashboard-body');
            
            if (loader) loader.style.display = 'flex';
            if (body) body.style.display = 'none';
            
            try {
                const response = await fetch(`/api/vietlott/${gameId}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                const data = await response.json();
                vietlottData[gameId] = data;
                
                if (loader) loader.style.display = 'none';
                if (body) body.style.display = 'block';
                
                try {
                    renderVietlottDashboard(data, gameId);
                } catch (renderErr) {
                    console.error(`Render error for ${gameId}:`, renderErr);
                    // Show partial error in body instead of leaving blank
                    const errEl = document.getElementById('vietlott-latest-draw-section');
                    if (errEl) errEl.innerHTML = `<div style="padding:1rem; color:#f43f5e; font-size:0.85rem;"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi hiển thị: ${renderErr.message}</div>`;
                }
            } catch (err) {
                console.error(`Error fetching Vietlott ${gameId} data:`, err);
                if (loader) loader.style.display = 'none';
                if (body) body.style.display = 'block';
                const errEl = document.getElementById('vietlott-latest-draw-section');
                if (errEl) errEl.innerHTML = `<div style="padding:1.5rem; text-align:center; color:#f43f5e;"><i class="fa-solid fa-circle-exclamation" style="font-size:2rem; margin-bottom:0.75rem; display:block;"></i>Không thể tải dữ liệu xổ số.<br><small style="color:var(--text-muted); font-family:'JetBrains Mono',monospace;">${err.message}</small></div>`;
                showToast(`Không thể tải dữ liệu ${gameId}: ` + err.message, "error");
            } finally {
                isFetchingVietlott = false;
            }
        }

        async function refreshVietlottData() {
            const btn = document.getElementById('vietlott-manual-refresh-btn');
            const icon = document.getElementById('vietlott-refresh-icon');
            const statusBadge = document.getElementById('vietlott-sync-status');
            
            if (!btn) return;
            btn.disabled = true;
            btn.style.opacity = '0.6';
            btn.style.cursor = 'not-allowed';
            if (icon) icon.classList.add('fa-spin');
            if (statusBadge) statusBadge.style.display = 'inline-block';
            
            showToast("Bắt đầu đồng bộ dữ liệu Vietlott từ GitHub...", "info");
            
            try {
                const response = await fetch('/api/vietlott/refresh', { method: 'POST' });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const resData = await response.json();
                showToast("Đồng bộ thành công dữ liệu xổ số từ GitHub!", "success");
                
                // Refetch active game
                await fetchVietlottData(activeVietlottGame);
            } catch (err) {
                console.error('Error syncing Vietlott:', err);
                showToast("Lỗi khi đồng bộ dữ liệu: " + err.message, "error");
            } finally {
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
                if (icon) icon.classList.remove('fa-spin');
                if (statusBadge) statusBadge.style.display = 'none';
            }
        }

        function renderVietlottDashboard(data, gameId) {
            const gameNames = {
                "power655": "Power 6/55 - Thống Kê &amp; Phân Tích",
                "mega645": "Mega 6/45 - Thống Kê &amp; Phân Tích",
                "keno": "Keno - Tần Suất &amp; Thống Kê Nhanh",
                "bingo18": "Bingo 18 - Kết Quả &amp; Tài Xỉu",
                "max3d": "Max 3D - Phân Tích Vị Trí Chữ Số",
                "max3dpro": "Max 3D Pro - Phân Tích Số Trúng",
                "power535": "Power 5/35 - Tần Suất Trúng Giải"
            };
            
            // Title and Info
            document.getElementById('vietlott-title').innerHTML = gameNames[gameId] || gameId.toUpperCase();
            document.getElementById('vietlott-update-time').innerText = data.updated_at || '--';
            document.getElementById('vietlott-total-draws').innerText = data.total_draws ? Number(data.total_draws).toLocaleString('vi-VN') : '--';
            
            renderVietlottLatestDraw(data, gameId);
            renderVietlottKPIs(data, gameId);
            renderVietlottHeatmap(data, gameId);
            renderVietlottHistoryTable(data, gameId);

            // Handle forecast subtab visibility
            const forecastTabBtn = document.getElementById('tab-vietlott-forecast');
            if (data.forecast) {
                if (forecastTabBtn) forecastTabBtn.style.display = 'block';
                renderVietlottForecast(data, gameId, activeVietlottModel);
            } else {
                if (forecastTabBtn) forecastTabBtn.style.display = 'none';
                if (activeVietlottSubTab === 'forecast') {
                    activeVietlottSubTab = 'stats';
                }
            }
            
            switchVietlottSubTab(activeVietlottSubTab);
        }

        function switchVietlottSubTab(subTabName) {
            activeVietlottSubTab = subTabName;
            
            // Toggle active classes on subtab buttons
            const statsBtn = document.getElementById('tab-vietlott-stats');
            const forecastBtn = document.getElementById('tab-vietlott-forecast');
            
            if (statsBtn) statsBtn.classList.toggle('active', subTabName === 'stats');
            if (forecastBtn) forecastBtn.classList.toggle('active', subTabName === 'forecast');
            
            // Toggle visibility of the two sub-views
            const statsView = document.getElementById('vietlott-subview-stats');
            const forecastView = document.getElementById('vietlott-subview-forecast');
            
            if (statsView) statsView.style.display = subTabName === 'stats' ? 'block' : 'none';
            if (forecastView) forecastView.style.display = subTabName === 'forecast' ? 'block' : 'none';
            
            // If forecast is selected, render forecast
            if (subTabName === 'forecast') {
                const gameId = activeVietlottGame;
                const data = vietlottData[gameId];
                if (data) {
                    renderVietlottForecast(data, gameId, activeVietlottModel);
                }
            }
        }

        function changeVietlottModel(modelName) {
            activeVietlottModel = modelName;
            const gameId = activeVietlottGame;
            const data = vietlottData[gameId];
            if (data) {
                renderVietlottForecast(data, gameId, modelName);
            }
        }

        function renderVietlottForecast(data, gameId, modelName) {
            const forecast = data.forecast;
            if (!forecast) return;
            
            // Check if game supports Monte Carlo
            const supportsMC = ['mega645', 'power655', 'power535'].includes(gameId);
            if (modelName === 'monte_carlo' && !supportsMC) {
                modelName = 'ensemble';
                activeVietlottModel = 'ensemble';
            }
            
            const modelData = forecast.models[modelName];
            if (!modelData) return;
            
            // Synchronize model selector value
            const modelSelect = document.getElementById('vietlott-model-select');
            if (modelSelect) {
                const mcOption = modelSelect.querySelector('option[value="monte_carlo"]');
                if (mcOption) {
                    mcOption.style.display = supportsMC ? 'block' : 'none';
                }
                modelSelect.value = modelName;
            }
            
            // 1. Render Lucky Numbers
            const ballsContainer = document.getElementById('vietlott-forecast-balls');
            if (!ballsContainer) return;
            
            const luckyNumbers = modelData.lucky_numbers || [];
            let ballsHTML = '';
            
            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                let ballClass = 'vietlott-ball forecast-glow';
                if (gameId === 'keno') ballClass += ' keno-ball';
                
                ballsHTML = luckyNumbers.map((num, idx) => {
                    let isSpecial = (gameId === 'power655' && idx === 5) || (gameId === 'power535' && idx === 4);
                    let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                    return `<span class="${cls}">${String(num).padStart(2, '0')}</span>`;
                }).join(' ');
            } else if (gameId === 'bingo18') {
                const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                ballsHTML = luckyNumbers.map(num => {
                    return `<span class="vietlott-ball bingo-ball forecast-glow"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                }).join(' ');
                
                ballsHTML += `
                <div style="margin-left: 1rem; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; background: rgba(251,191,36,0.1); padding: 0.3rem 0.6rem; border-radius: 6px; border: 1px solid rgba(251,191,36,0.2);">
                    <span style="color:var(--text-secondary);">AI Dự đoán tổng:</span>
                    <strong style="color: #fbbf24; font-size: 1.15rem; font-family:'JetBrains Mono', monospace;">${modelData.lucky_sum}</strong>
                    <span style="padding: 0.1rem 0.3rem; background: #fbbf24; border-radius: 4px; color:#1e293b; font-size:0.68rem; font-weight:700;">${modelData.lucky_sum <= 10 ? 'Nhỏ' : 'Lớn'}</span>
                </div>
                `;
            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                ballsHTML = luckyNumbers.map(combo => {
                    return `
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem;">
                        <div style="display: flex; gap: 0.2rem;">
                            <span class="vietlott-ball max3d-ball forecast-glow" style="width: 2.2rem; height: 2.2rem; font-size: 1rem; border-radius: 6px;">${combo[0]}</span>
                            <span class="vietlott-ball max3d-ball forecast-glow" style="width: 2.2rem; height: 2.2rem; font-size: 1rem; border-radius: 6px;">${combo[1]}</span>
                            <span class="vietlott-ball max3d-ball forecast-glow" style="width: 2.2rem; height: 2.2rem; font-size: 1rem; border-radius: 6px;">${combo[2]}</span>
                        </div>
                    </div>
                    `;
                }).join('<div style="color:var(--text-muted); font-size:0.8rem; font-weight:bold; align-self:center;">hoặc</div>');
            }
            
            ballsContainer.innerHTML = ballsHTML;
            
            // Set Target Draw Label
            const targetLabel = document.getElementById('vietlott-forecast-target-label');
            if (targetLabel && data.latest_draw) {
                let latestId = data.latest_draw.id;
                let nextId = "";
                if (latestId) {
                    if (latestId.startsWith('#')) {
                        let numPart = parseInt(latestId.substring(1));
                        nextId = `#${String(numPart + 1).padStart(7, '0')}`;
                    } else if (latestId.includes('-')) {
                        // Keno or Bingo IDs can be like 2024-12-03 or id 0083123
                        let numPart = parseInt(latestId);
                        nextId = isNaN(numPart) ? latestId + " + 1" : String(numPart + 1).padStart(latestId.length, '0');
                    } else {
                        let numPart = parseInt(latestId);
                        nextId = isNaN(numPart) ? latestId + " + 1" : String(numPart + 1).padStart(latestId.length, '0');
                    }
                }
                targetLabel.innerHTML = `Bộ số gợi ý cho kỳ tiếp theo <span style="color:#f43f5e; font-weight:700;">${nextId ? 'Kỳ ' + nextId : ''}</span>`;
            }

            // 2. Render Backtest Card
            const backtestCard = document.getElementById('vietlott-backtest-card');
            const latestBacktest = (forecast.backtest_history && forecast.backtest_history.length > 0) ? forecast.backtest_history[0] : null;
            if (backtestCard && latestBacktest) {
                const backtestModel = latestBacktest.models[modelName];
                const latestDraw = data.latest_draw || {};
                
                if (backtestModel) {
                    let accuracyHTML = '';
                    let detailsHTML = '';
                    
                    if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                        let matchedCount = backtestModel.matched_count || 0;
                        let totalCount = gameId === 'keno' ? 6 : (gameId === 'mega645' || gameId === 'power655' ? 6 : 5);
                        let rate = ((matchedCount / totalCount) * 100).toFixed(0);
                        
                        let predictedNums = backtestModel.predicted_numbers || [];
                        let actualNums = latestBacktest.actual_numbers || latestDraw.result || [];
                        let matchedSet = new Set(backtestModel.matched_numbers || []);
                        
                        let predictedBallsHTML = predictedNums.map((num, idx) => {
                            let isMatched = matchedSet.has(num);
                            let ballClass = 'vietlott-ball';
                            if (gameId === 'keno') ballClass += ' keno-ball';
                            let isSpecial = (gameId === 'power655' && idx === 5) || (gameId === 'power535' && idx === 4);
                            let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                            
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 8px rgba(52,211,153,0.8); border: none;';
                            } else {
                                extraStyle = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); color: var(--text-muted); box-shadow: none; opacity: 0.55;';
                            }
                            return `<span class="${cls}" style="font-size:0.72rem; width:1.5rem; height:1.5rem; ${extraStyle}">${String(num).padStart(2, '0')}</span>`;
                        }).join(' ');
                        
                        let actualBallsHTML = actualNums.map((num, idx) => {
                            let isMatched = matchedSet.has(num);
                            let ballClass = 'vietlott-ball';
                            if (gameId === 'keno') ballClass += ' keno-ball';
                            let isSpecial = (gameId === 'power655' && idx === 6) || (gameId === 'power535' && idx === 5);
                            let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                            
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 8px rgba(52,211,153,0.8); border: none;';
                            } else {
                                extraStyle = '';
                            }
                            return `<span class="${cls}" style="font-size:0.72rem; width:1.5rem; height:1.5rem; ${extraStyle}">${String(num).padStart(2, '0')}</span>`;
                        }).join(' ');
                        
                        accuracyHTML = `
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:0.5rem; margin-bottom:0.5rem;">
                            <span style="font-size:0.8rem; font-weight:600; color:var(--text-primary);">
                                <i class="fa-solid fa-square-poll-vertical"></i> Kiểm Thử Lịch Sử T-1 (Kỳ #${latestDraw.id})
                            </span>
                            <span style="font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; background: rgba(16, 185, 129, 0.15); color: #10b981; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.25); white-space:nowrap;">
                                Khớp: ${matchedCount}/${totalCount} số (${rate}%)
                            </span>
                        </div>
                        `;
                        
                        detailsHTML = `
                        <div style="display:flex; flex-direction:column; gap:0.6rem;">
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">AI dự báo (T-1):</span>
                                <div style="display:flex; gap:0.25rem; flex-wrap:wrap;">${predictedBallsHTML}</div>
                            </div>
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">Kết quả thực tế (T-1):</span>
                                <div style="display:flex; gap:0.25rem; flex-wrap:wrap;">${actualBallsHTML}</div>
                            </div>
                        </div>
                        `;
                    } else if (gameId === 'bingo18') {
                        let matchedCount = backtestModel.matched_count || 0;
                        let isTotalMatched = backtestModel.total_matched;
                        
                        let predictedDice = backtestModel.predicted_numbers || [];
                        let actualDice = latestBacktest.actual_numbers || latestDraw.result || [];
                        let matchedSet = new Set(backtestModel.matched_numbers || []);
                        const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                        
                        let predictedDiceHTML = predictedDice.map(num => {
                            let isMatched = matchedSet.has(num);
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 8px rgba(52,211,153,0.8); border: none;';
                            } else {
                                extraStyle = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); color: var(--text-muted); opacity: 0.55;';
                            }
                            return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.75rem; width:1.5rem; height:1.5rem; display:inline-flex; align-items:center; justify-content:center; ${extraStyle}"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                        }).join(' ');
                        
                        let actualDiceHTML = actualDice.map(num => {
                            let isMatched = matchedSet.has(num);
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 8px rgba(52,211,153,0.8); border: none;';
                            } else {
                                extraStyle = 'background: radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%);';
                            }
                            return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.75rem; width:1.5rem; height:1.5rem; display:inline-flex; align-items:center; justify-content:center; ${extraStyle}"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                        }).join(' ');
                        
                        accuracyHTML = `
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:0.5rem; margin-bottom:0.5rem;">
                            <span style="font-size:0.8rem; font-weight:600; color:var(--text-primary);">
                                <i class="fa-solid fa-square-poll-vertical"></i> Kiểm Thử Lịch Sử T-1 (Kỳ #${latestDraw.id})
                            </span>
                            <div style="display:flex; gap:0.4rem;">
                                <span style="font-size: 0.68rem; padding: 0.15rem 0.45rem; border-radius: 4px; background: rgba(16, 185, 129, 0.15); color: #10b981; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.25);">Dice khớp: ${matchedCount}/3</span>
                                <span style="font-size: 0.68rem; padding: 0.15rem 0.45rem; border-radius: 4px; background: ${isTotalMatched ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)'}; color: ${isTotalMatched ? '#10b981' : '#f43f5e'}; font-weight: 600; border: 1px solid ${isTotalMatched ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)'};">Tổng: ${isTotalMatched ? 'ĐÚNG' : 'SAI'}</span>
                            </div>
                        </div>
                        `;
                        
                        detailsHTML = `
                        <div style="display:flex; flex-direction:column; gap:0.6rem;">
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">AI dự báo (T-1):</span>
                                <div style="display:flex; gap:0.25rem; align-items:center;">
                                    ${predictedDiceHTML}
                                    <span style="font-size:0.7rem; color:var(--text-secondary); background:rgba(251,191,36,0.1); padding:0.1rem 0.35rem; border-radius:4px; font-weight:600; font-family:'JetBrains Mono';">Tổng ${backtestModel.predicted_total}</span>
                                </div>
                            </div>
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">Kết quả thực tế (T-1):</span>
                                <div style="display:flex; gap:0.25rem; align-items:center;">
                                    ${actualDiceHTML}
                                    <span style="font-size:0.7rem; color:var(--text-secondary); background:rgba(251,191,36,0.1); padding:0.1rem 0.35rem; border-radius:4px; font-weight:600; font-family:'JetBrains Mono';">Tổng ${latestDraw.total} (${latestDraw.large_small})</span>
                                </div>
                            </div>
                        </div>
                        `;
                    } else if (['max3d', 'max3dpro'].includes(gameId)) {
                        let matchedCount = backtestModel.matched_count || 0;
                        
                        let predictedCombos = backtestModel.predicted_numbers || [];
                        let actualCombos = latestBacktest.actual_numbers || [];
                        let matchedSet = new Set(backtestModel.matched_numbers || []);
                        
                        let predictedCombosHTML = predictedCombos.map(combo => {
                            let isMatched = matchedSet.has(combo);
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; font-weight: 700;';
                            } else {
                                extraStyle = 'background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); color: var(--text-muted); opacity: 0.55;';
                            }
                            return `<span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; padding:0.1rem 0.35rem; border-radius:4px; ${extraStyle}">${combo}</span>`;
                        }).join(' ');
                        
                        let actualCombosHTML = actualCombos.map(combo => {
                            let isMatched = matchedSet.has(combo);
                            let extraStyle = '';
                            if (isMatched) {
                                extraStyle = 'background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; font-weight: 700;';
                            } else {
                                extraStyle = 'background: rgba(147,51,234,0.1); border: 1px solid rgba(147,51,234,0.2); color: #c084fc;';
                            }
                            return `<span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; padding:0.1rem 0.35rem; border-radius:4px; ${extraStyle}">${combo}</span>`;
                        }).join(' ');
                        
                        accuracyHTML = `
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:0.5rem; margin-bottom:0.5rem;">
                            <span style="font-size:0.8rem; font-weight:600; color:var(--text-primary);">
                                <i class="fa-solid fa-square-poll-vertical"></i> Kiểm Thử Lịch Sử T-1 (Kỳ #${latestDraw.id})
                            </span>
                            <span style="font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; background: ${matchedCount > 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)'}; color: ${matchedCount > 0 ? '#10b981' : 'var(--text-muted)'}; font-weight: 600; border: 1px solid ${matchedCount > 0 ? 'rgba(16, 185, 129, 0.25)' : 'rgba(255, 255, 255, 0.1)'};">
                                Trúng: ${matchedCount} bộ 3 số
                            </span>
                        </div>
                        `;
                        
                        detailsHTML = `
                        <div style="display:flex; flex-direction:column; gap:0.6rem;">
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">AI dự báo (T-1):</span>
                                <div style="display:flex; gap:0.25rem; flex-wrap:wrap;">${predictedCombosHTML}</div>
                            </div>
                            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.74rem; flex-wrap:wrap;">
                                <span style="color:var(--text-muted); width:130px;">Bộ số thực tế ra (T-1):</span>
                                <div style="display:flex; gap:0.25rem; flex-wrap:wrap; max-height: 80px; overflow-y: auto; padding-right: 0.5rem;">${actualCombosHTML}</div>
                            </div>
                        </div>
                        `;
                    }
                    
                    backtestCard.innerHTML = accuracyHTML + detailsHTML;
                }
            }

            // 3. Render Probabilities Grid
            const probTitle = document.getElementById('vietlott-probability-title');
            const probSubtitle = document.getElementById('vietlott-probability-subtitle');
            const probContainer = document.getElementById('vietlott-forecast-probabilities');
            
            if (probContainer) {
                if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                    if (probTitle) probTitle.innerHTML = `<i class="fa-solid fa-chart-column"></i> Xác Suất Xuất Hiện Chi Tiết Của Từng Con Số`;
                    if (probSubtitle) probSubtitle.innerText = `Xác suất dự báo (%) cho kỳ tiếp theo dựa trên các đặc trưng tần số và độ trễ`;
                    
                    const probs = modelData.number_probabilities || [];
                    const sortedProbs = [...probs].sort((a,b) => a.number - b.number);
                    
                    let probHTML = '';
                    for (let p of sortedProbs) {
                        let isLucky = luckyNumbers.includes(p.number);
                        let barColor = isLucky ? '#f43f5e' : 'rgba(255,255,255,0.15)';
                        let textColor = isLucky ? '#f43f5e' : 'var(--text-secondary)';
                        let weight = isLucky ? '700' : '500';
                        let pct = (p.prob * 100).toFixed(1);
                        
                        probHTML += `
                        <div style="padding: 0.5rem 0.6rem; background: rgba(255,255,255,0.01); border: 1px solid ${isLucky ? 'rgba(244,63,94,0.15)' : 'rgba(255,255,255,0.03)'}; border-radius: 8px; display: flex; flex-direction: column; gap: 0.25rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="vietlott-ball" style="font-size:0.72rem; width:1.35rem; height:1.35rem; background:${isLucky ? '' : 'rgba(255,255,255,0.05)'}; border:1px solid ${isLucky ? 'transparent' : 'rgba(255,255,255,0.1)'}; box-shadow:none;">${p.number}</span>
                                <span style="font-size:0.74rem; font-weight:${weight}; color:${textColor}; font-family:'JetBrains Mono', monospace;">${pct}%</span>
                            </div>
                            <div style="height: 4px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; overflow: hidden; margin-top:0.1rem;">
                                <div style="width: ${pct}%; background: ${barColor}; height: 100%;"></div>
                            </div>
                        </div>
                        `;
                    }
                    
                    probContainer.style.gridTemplateColumns = 'repeat(auto-fill, minmax(105px, 1fr))';
                    probContainer.innerHTML = probHTML;
                } else if (gameId === 'bingo18') {
                    if (probTitle) probTitle.innerHTML = `<i class="fa-solid fa-chart-column"></i> Xác Suất Cho Từng Mặt Xúc Xắc và Tổng Điểm`;
                    if (probSubtitle) probSubtitle.innerText = `Biểu đồ xác suất cho các mặt xúc xắc 1-6 và tổng tiềm năng`;
                    
                    // Dice Face Probabilities
                    const diceProbs = modelData.number_probabilities || [];
                    const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                    let diceHTML = '<div style="grid-column: 1 / -1; font-size: 0.76rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.25rem;"><i class="fa-solid fa-dice"></i> Xác suất từng mặt xúc xắc (1 - 6)</div>';
                    
                    for (let p of diceProbs) {
                        let isLucky = luckyNumbers.includes(p.number);
                        let barColor = isLucky ? '#fbbf24' : 'rgba(255,255,255,0.15)';
                        let textColor = isLucky ? '#fbbf24' : 'var(--text-secondary)';
                        let pct = (p.prob * 100).toFixed(1);
                        
                        diceHTML += `
                        <div style="padding: 0.5rem 0.6rem; background: rgba(255,255,255,0.01); border: 1px solid ${isLucky ? 'rgba(251,191,36,0.15)' : 'rgba(255,255,255,0.03)'}; border-radius: 8px; display: flex; flex-direction: column; gap: 0.25rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="vietlott-ball" style="border-radius:4px; font-size:0.85rem; width:1.35rem; height:1.35rem; background:${isLucky ? 'radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%)' : 'rgba(255,255,255,0.05)'}; border:none; box-shadow:none;"><i class="fa-solid fa-dice-${diceIcons[p.number-1]}"></i></span>
                                <span style="font-size:0.74rem; font-weight:600; color:${textColor}; font-family:'JetBrains Mono', monospace;">${pct}%</span>
                            </div>
                            <div style="height: 4px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; overflow: hidden; margin-top:0.1rem;">
                                <div style="width: ${pct}%; background: ${barColor}; height: 100%;"></div>
                            </div>
                        </div>
                        `;
                    }
                    
                    // Sum Probabilities
                    const sumProbs = modelData.sum_probabilities || [];
                    const sortedSums = [...sumProbs].sort((a,b) => a.number - b.number);
                    let sumHTML = '<div style="grid-column: 1 / -1; font-size: 0.76rem; color: var(--text-secondary); font-weight: 600; margin-top: 1rem; margin-bottom: 0.25rem;"><i class="fa-solid fa-calculator"></i> Xác suất tổng điểm tiềm năng (3 - 18)</div>';
                    
                    for (let s of sortedSums) {
                        let isLucky = s.number === modelData.lucky_sum;
                        let barColor = isLucky ? '#f43f5e' : 'rgba(255,255,255,0.12)';
                        let textColor = isLucky ? '#f43f5e' : 'var(--text-secondary)';
                        let pct = (s.prob * 100).toFixed(1);
                        
                        sumHTML += `
                        <div style="padding: 0.5rem 0.6rem; background: rgba(255,255,255,0.01); border: 1px solid ${isLucky ? 'rgba(244,63,94,0.15)' : 'rgba(255,255,255,0.03)'}; border-radius: 8px; display: flex; flex-direction: column; gap: 0.25rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; font-weight:700; color:${textColor};">${s.number}</span>
                                <span style="font-size:0.72rem; font-weight:600; color:${textColor}; font-family:'JetBrains Mono', monospace;">${pct}%</span>
                            </div>
                            <div style="height: 4px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; overflow: hidden; margin-top:0.1rem;">
                                <div style="width: ${pct}%; background: ${barColor}; height: 100%;"></div>
                            </div>
                        </div>
                        `;
                    }
                    
                    probContainer.style.gridTemplateColumns = 'repeat(6, 1fr)';
                    probContainer.innerHTML = diceHTML + sumHTML;
                } else if (['max3d', 'max3dpro'].includes(gameId)) {
                    if (probTitle) probTitle.innerHTML = `<i class="fa-solid fa-chart-column"></i> Xác Suất Chữ Số Tại Từng Vị Trí (0 - 9)`;
                    if (probSubtitle) probSubtitle.innerText = `Tỷ lệ phần trăm dự báo cho chữ số tại hàng Trăm, Chục và Đơn vị`;
                    
                    const posProbs = modelData.position_probabilities || {};
                    let posHTML = '';
                    
                    const positions = [
                        { key: 'hundreds', title: 'Hàng Trăm', color: '#8b5cf6' },
                        { key: 'tens', title: 'Hàng Chục', color: '#06b6d4' },
                        { key: 'units', title: 'Hàng Đơn Vị', color: '#10b981' }
                    ];
                    
                    positions.forEach(pos => {
                        let digits = posProbs[pos.key] || [];
                        let sortedDigits = [...digits].sort((a,b) => a.number - b.number);
                        
                        posHTML += `
                        <div style="grid-column: 1 / -1; font-size: 0.76rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.25rem; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 0.25rem;">
                            <span style="color: ${pos.color};">${pos.title}</span>
                        </div>
                        `;
                        
                        for (let d of sortedDigits) {
                            let pct = (d.prob * 100).toFixed(1);
                            
                            posHTML += `
                            <div style="padding: 0.5rem 0.6rem; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 8px; display: flex; flex-direction: column; gap: 0.25rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span class="vietlott-ball" style="width:1.3rem; height:1.3rem; font-size:0.72rem; border-radius:4px; background:${pos.color}; box-shadow:none;">${d.number}</span>
                                    <span style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); font-family:'JetBrains Mono', monospace;">${pct}%</span>
                                </div>
                                <div style="height: 3px; background: rgba(255, 255, 255, 0.05); border-radius: 1.5px; overflow: hidden; margin-top:0.1rem;">
                                    <div style="width: ${pct}%; background: ${pos.color}; height: 100%;"></div>
                                </div>
                            </div>
                            `;
                        }
                    });
                    
                    probContainer.style.gridTemplateColumns = 'repeat(5, 1fr)';
                    probContainer.innerHTML = posHTML;
                }
            }

            // 4. Render Model Comparison Table
            const comparisonTbody = document.getElementById('vietlott-model-comparison-tbody');
            if (comparisonTbody) {
                let compHTML = '';
                const modelsMeta = {
                    "ensemble": { name: "✨ Ensemble Hybrid", icon: "✨" },
                    "monte_carlo": { name: "🎲 Monte Carlo Simulator", icon: "🎲" },
                    "random_forest": { name: "🌲 Random Forest", icon: "🌲" },
                    "mlp": { name: "🧠 MLP Neural Network", icon: "🧠" },
                    "xgboost": { name: "🚀 XGBoost Classifier", icon: "🚀" },
                    "linear_regression": { name: "📈 Linear Regression", icon: "📈" }
                };
                
                for (let [mKey, mMeta] of Object.entries(modelsMeta)) {
                    const mData = forecast.models[mKey];
                    if (!mData) continue;
                    
                    const luckyNums = mData.lucky_numbers || [];
                    
                    // Render tiny balls for this model's lucky numbers
                    let mBallsHTML = '';
                    if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                        let ballClass = 'vietlott-ball';
                        if (gameId === 'keno') ballClass += ' keno-ball';
                        
                        mBallsHTML = `<div class="vietlott-balls-container">` + luckyNums.map((num, idx) => {
                            let isSpecial = (gameId === 'power655' && idx === 5) || (gameId === 'power535' && idx === 4);
                            let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                            return `<span class="${cls}" style="font-size:0.7rem; width:1.35rem; height:1.35rem; box-shadow:none; margin:0 1px;">${String(num).padStart(2, '0')}</span>`;
                        }).join(' ') + `</div>`;
                    } else if (gameId === 'bingo18') {
                        const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                        mBallsHTML = `<div class="vietlott-balls-container">` + luckyNums.map(num => {
                            return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.75rem; width:1.35rem; height:1.35rem; background: radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%); box-shadow:none; display:inline-flex; align-items:center; justify-content:center;"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                        }).join(' ') + ` <span style="font-size:0.72rem; color:var(--text-secondary); background:rgba(251,191,36,0.1); padding:0.1rem 0.35rem; border-radius:4px; margin-left:0.25rem; font-weight:600; font-family:'JetBrains Mono';">Tổng ${mData.lucky_sum} (${mData.lucky_sum <= 10 ? 'Nhỏ' : 'Lớn'})</span></div>`;
                    } else if (['max3d', 'max3dpro'].includes(gameId)) {
                        mBallsHTML = `<div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">` + luckyNums.map(combo => {
                            return `<span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#c084fc; font-weight:700; background:rgba(147,51,234,0.1); border:1px solid rgba(147,51,234,0.2); padding:0.1rem 0.35rem; border-radius:4px;">${combo}</span>`;
                        }).join('<span style="color:var(--text-muted); font-size:0.68rem; align-self:center;">|</span>') + `</div>`;
                    }
                    
                    // Calculate Average Confidence scaled relative to prior baseline (random guess = 50%)
                    let avgConfidence = 0.0;
                    
                    const getNormalizedConfidence = (prob, prior) => {
                        if (prior <= 0 || prior >= 1) return prob;
                        const ratio = prior / (1 - prior);
                        return prob / (prob + (1 - prob) * ratio);
                    };
                    
                    if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                        const numProbs = mData.number_probabilities || [];
                        const luckySet = new Set(luckyNums);
                        const selectedProbs = numProbs.filter(p => luckySet.has(p.number)).map(p => p.prob);
                        
                        const numDrawMap = { mega645: 6, power655: 6, power535: 5, keno: 6 };
                        const maxNumMap = { mega645: 45, power655: 55, power535: 35, keno: 80 };
                        const prior = (numDrawMap[gameId] || 6) / (maxNumMap[gameId] || 45);
                        
                        const normalizedScores = selectedProbs.map(p => getNormalizedConfidence(p, prior));
                        avgConfidence = normalizedScores.length > 0 ? (normalizedScores.reduce((a,b)=>a+b, 0) / normalizedScores.length) : 0.0;
                    } else if (gameId === 'bingo18') {
                        const faceProbs = mData.number_probabilities || [];
                        const sumProbs = mData.sum_probabilities || [];
                        const luckyFaceSet = new Set(luckyNums);
                        
                        const selectedFaceProbs = faceProbs.filter(p => luckyFaceSet.has(p.number)).map(p => p.prob);
                        const selectedSumProb = sumProbs.find(s => s.number === mData.lucky_sum)?.prob || 0.0;
                        
                        const facePrior = 0.5; // 3 out of 6 dice faces
                        const sumPrior = 0.0625; // 1 out of 16 sums
                        
                        const faceScores = selectedFaceProbs.map(p => getNormalizedConfidence(p, facePrior));
                        const sumScore = getNormalizedConfidence(selectedSumProb, sumPrior);
                        
                        const allScores = [...faceScores, sumScore];
                        avgConfidence = allScores.length > 0 ? (allScores.reduce((a,b)=>a+b, 0) / allScores.length) : 0.0;
                    } else if (['max3d', 'max3dpro'].includes(gameId)) {
                        const posProbs = mData.position_probabilities || {};
                        const hMap = Object.fromEntries((posProbs.hundreds || []).map(x => [x.number, x.prob]));
                        const tMap = Object.fromEntries((posProbs.tens || []).map(x => [x.number, x.prob]));
                        const uMap = Object.fromEntries((posProbs.units || []).map(x => [x.number, x.prob]));
                        
                        let comboProbs = luckyNums.map(combo => {
                            let h = parseInt(combo[0]) || 0;
                            let t = parseInt(combo[1]) || 0;
                            let u = parseInt(combo[2]) || 0;
                            return (hMap[h] || 0.0) * (tMap[t] || 0.0) * (uMap[u] || 0.0);
                        });
                        
                        const prior = gameId === 'max3dpro' ? 0.04 : 0.02; // Max 3D Pro draws 40 combos, Max 3D draws 20 combos
                        const normalizedScores = comboProbs.map(p => getNormalizedConfidence(p, prior));
                        avgConfidence = normalizedScores.length > 0 ? (normalizedScores.reduce((a,b)=>a+b, 0) / normalizedScores.length) : 0.0;
                    }
                    
                    let confPct = (avgConfidence * 100).toFixed(1) + "%";
                    
                    // Backtest accuracy for T-1
                    let backtestAccuracyHTML = '';
                    if (forecast.backtest) {
                        const backtestModel = forecast.backtest.models[mKey];
                        if (backtestModel) {
                            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                                let matched = backtestModel.matched_count || 0;
                                let total = gameId === 'keno' ? 6 : (gameId === 'mega645' || gameId === 'power655' ? 6 : 5);
                                let rate = ((matched / total) * 100).toFixed(0);
                                backtestAccuracyHTML = `<span style="color:#10b981; font-weight:600; font-family:'JetBrains Mono';">${rate}%</span> <span style="font-size:0.68rem; color:var(--text-muted);">(${matched}/${total} số)</span>`;
                            } else if (gameId === 'bingo18') {
                                let matched = backtestModel.matched_count || 0;
                                let isTotalMatched = backtestModel.total_matched;
                                backtestAccuracyHTML = `<span style="color:#10b981; font-weight:600; font-family:'JetBrains Mono';">${matched}/3 mặt</span> <span style="font-size:0.68rem; color:${isTotalMatched ? '#10b981' : '#f43f5e'}; font-weight:600;">(Tổng: ${isTotalMatched ? 'ĐÚNG' : 'SAI'})</span>`;
                            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                                let matched = backtestModel.matched_count || 0;
                                backtestAccuracyHTML = `<span style="color:${matched > 0 ? '#10b981' : 'var(--text-muted)'}; font-weight:600;">Trúng ${matched} bộ</span>`;
                            }
                        }
                    }
                    
                    const isSelected = mKey === modelName;
                    const rowStyle = isSelected ? 'background: rgba(244,63,94,0.08); border-left: 3px solid #f43f5e;' : 'cursor: pointer;';
                    const activeBadge = isSelected ? ` <span style="font-size:0.58rem; padding:0.08rem 0.25rem; background:#f43f5e; color:#ffffff; border-radius:4px; font-weight:700; margin-left:0.25rem;">ĐANG CHỌN</span>` : '';
                    
                    compHTML += `
                    <tr style="${rowStyle}" onclick="changeVietlottModel('${mKey}')" onmouseover="if(!${isSelected}) this.style.background='rgba(255,255,255,0.02)';" onmouseout="if(!${isSelected}) this.style.background='transparent';">
                        <td style="padding: 0.6rem 0.8rem; font-size:0.76rem; font-weight:600; color:${isSelected ? '#f43f5e' : 'var(--text-primary)'}; display:flex; align-items:center; gap:0.4rem; border:none; height:100%;">
                            <span>${mMeta.name}</span>${activeBadge}
                        </td>
                        <td style="padding: 0.6rem 0.8rem; border:none; vertical-align:middle;">${mBallsHTML}</td>
                        <td style="padding: 0.6rem 0.8rem; text-align: center; font-size:0.74rem; border:none; vertical-align:middle;">${backtestAccuracyHTML || '--'}</td>
                        <td style="padding: 0.6rem 0.8rem; text-align: center; font-family:'JetBrains Mono', monospace; font-size:0.76rem; font-weight:700; color:#fbbf24; border:none; vertical-align:middle;">${confPct}</td>
                    </tr>
                    `;
                }
                
                comparisonTbody.innerHTML = compHTML;
            }

            // 4. Render 20-Draw Backtest History
            const historyTbody = document.getElementById('vietlott-backtest-history-tbody');
            const activeModelLabel = document.getElementById('vietlott-backtest-active-model');
            
            if (activeModelLabel) {
                const modelDisplayNames = {
                    ensemble: 'Ensemble Hybrid',
                    monte_carlo: 'Monte Carlo Simulator',
                    random_forest: 'Random Forest',
                    mlp: 'Neural Network MLP',
                    xgboost: 'XGBoost',
                    linear_regression: 'Linear Regression'
                };
                activeModelLabel.innerText = modelDisplayNames[modelName] || modelName;
            }

            if (historyTbody) {
                if (!forecast.backtest_history || forecast.backtest_history.length === 0) {
                    historyTbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:2rem; color:var(--text-muted);">Không có dữ liệu lịch sử backtest. Vui lòng nhấn Làm mới để chạy phân tích AI.</td></tr>`;
                } else {
                    let historyHTML = '';
                    
                    forecast.backtest_history.forEach(item => {
                        const mBack = item.models ? item.models[modelName] : null;
                        if (!mBack) return;
                        
                        let actualNums = item.actual_numbers || [];
                        let predictedNums = mBack.predicted_numbers || [];
                        let matchedSet = new Set(mBack.matched_numbers || []);
                        
                        let actualHTML = '';
                        let predictedHTML = '';
                        let statusHTML = '';
                        
                        if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                            // Render actual balls
                            actualHTML = `<div style="display:flex; gap:0.2rem; flex-wrap:wrap; justify-content:center;">` + 
                                actualNums.map((num, idx) => {
                                    let isMatched = matchedSet.has(num);
                                    let isSpecial = (gameId === 'power655' && idx === 6) || (gameId === 'power535' && idx === 5);
                                    let bg = isMatched 
                                        ? 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 5px rgba(52,211,153,0.5); border:none;' 
                                        : (isSpecial ? 'background: radial-gradient(circle at 30% 30%, #f43f5e 0%, #be123c 50%, #881337 100%);' : 'background: rgba(255,255,255,0.06);');
                                    return `<span class="vietlott-ball" style="width:1.35rem; height:1.35rem; font-size:0.65rem; ${bg}">${String(num).padStart(2, '0')}</span>`;
                                }).join('') + `</div>`;
                                
                            // Render predicted balls
                            predictedHTML = `<div style="display:flex; gap:0.2rem; flex-wrap:wrap; justify-content:center;">` + 
                                predictedNums.map((num, idx) => {
                                    let isMatched = matchedSet.has(num);
                                    let isSpecial = (gameId === 'power655' && idx === 5) || (gameId === 'power535' && idx === 4);
                                    let bg = isMatched 
                                        ? 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%); box-shadow: 0 0 5px rgba(52,211,153,0.5); border:none;' 
                                        : (isSpecial ? 'background: radial-gradient(circle at 30% 30%, #f43f5e 0%, #be123c 50%, #881337 100%);' : 'background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); color:var(--text-muted); opacity:0.5;');
                                    return `<span class="vietlott-ball" style="width:1.35rem; height:1.35rem; font-size:0.65rem; ${bg}">${String(num).padStart(2, '0')}</span>`;
                                }).join('') + `</div>`;
                                
                            // Matched stats
                            let totalCount = gameId === 'keno' ? 6 : (gameId === 'mega645' || gameId === 'power655' ? 6 : 5);
                            let matchedCount = mBack.matched_count || 0;
                            let rate = ((matchedCount / totalCount) * 100).toFixed(0);
                            
                            statusHTML = `
                                <div style="display:flex; flex-direction:column; align-items:center; gap:0.1rem;">
                                    <span style="font-size:0.72rem; font-weight:700; color:${matchedCount > 0 ? '#34d399' : 'var(--text-muted)'};">${matchedCount}/${totalCount} số</span>
                                    <span style="font-size:0.62rem; color:var(--text-muted); font-family:'JetBrains Mono';">${rate}%</span>
                                </div>
                            `;
                        } else if (gameId === 'bingo18') {
                            const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                            
                            // Render actual dice
                            actualHTML = `<div style="display:flex; gap:0.2rem; justify-content:center; align-items:center;">` + 
                                actualNums.map(num => {
                                    let isMatched = matchedSet.has(num);
                                    let bg = isMatched 
                                        ? 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%);' 
                                        : 'background: radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%);';
                                    return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.68rem; width:1.25rem; height:1.25rem; display:inline-flex; align-items:center; justify-content:center; ${bg}"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                                }).join('') + 
                                `<span style="font-size:0.68rem; font-weight:600; color:var(--text-secondary); background:rgba(255,255,255,0.06); padding:0.05rem 0.25rem; border-radius:3px; font-family:'JetBrains Mono'; margin-left:0.25rem;">Tổng ${item.actual_total}</span>` + 
                                `</div>`;
                                
                            // Render predicted dice
                            predictedHTML = `<div style="display:flex; gap:0.2rem; justify-content:center; align-items:center;">` + 
                                predictedNums.map(num => {
                                    let isMatched = matchedSet.has(num);
                                    let bg = isMatched 
                                        ? 'background: radial-gradient(circle at 30% 30%, #34d399 0%, #059669 50%, #064e3b 100%);' 
                                        : 'background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); color:var(--text-muted); opacity:0.5;';
                                    return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.68rem; width:1.25rem; height:1.25rem; display:inline-flex; align-items:center; justify-content:center; ${bg}"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                                }).join('') + 
                                `<span style="font-size:0.68rem; font-weight:600; color:var(--text-secondary); background:rgba(251,191,36,0.1); padding:0.05rem 0.25rem; border-radius:3px; font-family:'JetBrains Mono'; margin-left:0.25rem;">Tổng ${mBack.predicted_total}</span>` + 
                                `</div>`;
                                
                            let matchedCount = mBack.matched_count || 0;
                            let isTotalMatched = mBack.total_matched;
                            statusHTML = `
                                <div style="display:flex; flex-direction:column; align-items:center; gap:0.1rem;">
                                    <span style="font-size:0.7rem; font-weight:700; color:${matchedCount > 0 ? '#34d399' : 'var(--text-muted)'};">Dice: ${matchedCount}/3</span>
                                    <span style="font-size:0.58rem; padding:0.02rem 0.2rem; border-radius:3px; background:${isTotalMatched ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)'}; color:${isTotalMatched ? '#34d399' : '#f43f5e'}; font-weight:700;">Tổng: ${isTotalMatched ? 'ĐÚNG' : 'SAI'}</span>
                                </div>
                            `;
                        } else if (['max3d', 'max3dpro'].includes(gameId)) {
                            // Render actual combos
                            actualHTML = `<div style="display:flex; gap:0.2rem; flex-wrap:wrap; justify-content:center;">` + 
                                actualNums.slice(0, 3).map(combo => {
                                    let isMatched = matchedSet.has(combo);
                                    let bg = isMatched 
                                        ? 'background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; font-weight: 700;' 
                                        : 'background: rgba(147,51,234,0.1); border: 1px solid rgba(147,51,234,0.2); color: #c084fc;';
                                    return `<span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; padding:0.05rem 0.25rem; border-radius:3px; ${bg}">${combo}</span>`;
                                }).join('') + (actualNums.length > 3 ? `<span style="font-size:0.62rem; color:var(--text-muted); align-self:center;">+${actualNums.length - 3}</span>` : '') + `</div>`;
                                
                            // Render predicted combos
                            predictedHTML = `<div style="display:flex; gap:0.2rem; flex-wrap:wrap; justify-content:center;">` + 
                                predictedNums.map(combo => {
                                    let isMatched = matchedSet.has(combo);
                                    let bg = isMatched 
                                        ? 'background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; font-weight: 700;' 
                                        : 'background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); color:var(--text-muted); opacity:0.5;';
                                    return `<span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; padding:0.05rem 0.25rem; border-radius:3px; ${bg}">${combo}</span>`;
                                }).join('') + `</div>`;
                                
                            let matchedCount = mBack.matched_count || 0;
                            statusHTML = `
                                <div style="display:flex; flex-direction:column; align-items:center; gap:0.1rem;">
                                    <span style="font-size:0.7rem; font-weight:700; color:${matchedCount > 0 ? '#34d399' : 'var(--text-muted)'};">Trúng: ${matchedCount} bộ</span>
                                </div>
                            `;
                        }

                        historyHTML += `
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <td style="padding:0.5rem 0.8rem; text-align:center; font-family:'JetBrains Mono',monospace; font-size:0.74rem; font-weight:600; color:var(--text-primary); border:none;">
                                    ${item.draw_id || '--'}
                                </td>
                                <td style="padding:0.5rem 0.8rem; text-align:center; font-size:0.74rem; color:var(--text-secondary); border:none;">
                                    ${item.date || '--'}
                                </td>
                                <td style="padding:0.5rem 0.8rem; border:none; vertical-align:middle;">
                                    ${actualHTML}
                                </td>
                                <td style="padding:0.5rem 0.8rem; border:none; vertical-align:middle;">
                                    ${predictedHTML}
                                </td>
                                <td style="padding:0.5rem 0.8rem; text-align:center; border:none; vertical-align:middle;">
                                    ${statusHTML}
                                </td>
                            </tr>
                        `;
                    });
                    
                    historyTbody.innerHTML = historyHTML;
                }
            }
        }

        function renderVietlottLatestDraw(data, gameId) {
            const container = document.getElementById('vietlott-latest-draw-section');
            if (!container) return;
            
            const latest = data.latest_draw;
            if (!latest) {
                container.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-muted);">Không có dữ liệu kỳ quay gần nhất.</div>`;
                return;
            }
            
            let ballsHTML = '';
            let extraHTML = '';
            
            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                let result = latest.result || [];
                let isPower655 = gameId === 'power655';
                let isPower535 = gameId === 'power535';
                
                let ballClass = 'vietlott-ball';
                if (gameId === 'keno') ballClass += ' keno-ball';
                
                ballsHTML = result.map((num, idx) => {
                    let isSpecial = (isPower655 && idx === 6) || (isPower535 && idx === 5);
                    let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                    return `<span class="${cls}">${String(num).padStart(2, '0')}</span>`;
                }).join(' ');
                
                // Add distribution quick summary
                let odds = result.filter(n => n % 2 === 1).length;
                let evens = result.length - odds;
                extraHTML = `
                <div style="font-size: 0.72rem; color: var(--text-secondary); display: flex; gap: 0.6rem; margin-top: 0.35rem;">
                    <span style="padding: 0.1rem 0.4rem; background: rgba(244,63,94,0.1); border-radius: 4px; color:#f43f5e; font-weight:600;">Lẻ: ${odds}</span>
                    <span style="padding: 0.1rem 0.4rem; background: rgba(59,130,246,0.1); border-radius: 4px; color:#3b82f6; font-weight:600;">Chẵn: ${evens}</span>
                </div>
                `;
            } else if (gameId === 'bingo18') {
                let result = latest.result || [];
                ballsHTML = result.map(num => {
                    const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                    return `<span class="vietlott-ball bingo-ball"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                }).join(' ');
                
                extraHTML = `
                <div style="font-size: 0.85rem; color: var(--text-primary); font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
                    <span>Tổng điểm: <strong style="color: #fbbf24; font-size: 1.15rem; font-family:'JetBrains Mono', monospace;">${latest.total}</strong></span>
                    <span style="padding: 0.15rem 0.45rem; background: rgba(244,63,94,0.1); border-radius: 4px; color:#f43f5e; font-size:0.72rem;">${latest.large_small}</span>
                </div>
                `;
            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                let result = latest.result || {};
                let prizesHTML = '';
                
                for (let [prizeName, values] of Object.entries(result)) {
                    if (!Array.isArray(values)) continue;
                    let balls = values.map(val => `<span class="vietlott-ball max3d-ball">${val}</span>`).join(' ');
                    prizesHTML += `
                    <div style="background: rgba(255, 255, 255, 0.015); border: 1px solid rgba(255, 255, 255, 0.04); padding: 0.55rem 0.75rem; border-radius: 8px; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
                        <span style="font-size: 0.74rem; color: var(--text-muted); font-weight: 600; min-width: 90px;">${prizeName}</span>
                        <div style="display: flex; gap: 0.35rem; flex-wrap: wrap; justify-content: flex-end;">${balls}</div>
                    </div>
                    `;
                }
                
                container.innerHTML = `
                <div style="padding: 1.25rem 1.5rem; background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #f43f5e; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.4rem;">
                        <span>KẾT QUẢ KỲ QUAY MỚI NHẤT (Kỳ #${latest.id})</span>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--text-muted);">${latest.date}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.6rem;">
                        ${prizesHTML}
                    </div>
                </div>
                `;
                return;
            }
            
            container.innerHTML = `
            <div style="padding: 1.25rem 1.5rem; background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Kỳ Quay Gần Nhất</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-top: 0.2rem;">
                        Kỳ #${latest.id} <span style="color: var(--text-muted); font-weight: 500; font-size: 0.82rem;">(${latest.date})</span>
                    </div>
                    ${extraHTML}
                </div>
                <div class="vietlott-balls-container">
                    ${ballsHTML}
                </div>
            </div>
            `;
        }

        function renderVietlottKPIs(data, gameId) {
            const container = document.getElementById('vietlott-kpis-section');
            if (!container) return;
            
            const stats = data.stats;
            if (!stats) {
                container.innerHTML = '';
                return;
            }
            
            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                const dist = stats.distribution;
                
                // Hot HTML
                const hotHTML = stats.hot_numbers.slice(0, 5).map(n => `
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.35rem 0.5rem; background: rgba(244, 63, 94, 0.03); border-radius: 6px; border: 1px solid rgba(244, 63, 94, 0.08);">
                        <span class="vietlott-ball keno-ball" style="font-size:0.75rem; width:1.6rem; height:1.6rem;">${n.number}</span>
                        <span style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">${n.freq_100} lần</span>
                    </div>
                `).join('');
                
                // Cold HTML
                const coldHTML = stats.cold_numbers.slice(0, 5).map(n => `
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.35rem 0.5rem; background: rgba(59, 130, 246, 0.03); border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.08);">
                        <span class="vietlott-ball keno-ball power-special" style="font-size:0.75rem; width:1.6rem; height:1.6rem;">${n.number}</span>
                        <span style="font-size: 0.7rem; color: var(--text-muted);">${n.draws_since_last} kỳ vắng</span>
                    </div>
                `).join('');
                
                container.innerHTML = `
                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #f43f5e; font-weight: 600;">Số Hay Ra (Hot)</span>
                        <span class="stat-icon" style="background: rgba(244, 63, 94, 0.1); color: #f43f5e; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-fire"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
                        ${hotHTML}
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #3b82f6; font-weight: 600;">Số Lâu Chưa Ra (Cold)</span>
                        <span class="stat-icon" style="background: rgba(59, 130, 246, 0.1); color: #3b82f6; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-snowflake"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
                        ${coldHTML}
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #10b981; font-weight: 600;">Phân Bố Kết Quả (100 kỳ)</span>
                        <span class="stat-icon" style="background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-chart-pie"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.65rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
                                <span style="color: var(--text-secondary);">Chẵn: ${dist.even_percent}%</span>
                                <span style="color: var(--text-secondary);">Lẻ: ${dist.odd_percent}%</span>
                            </div>
                            <div style="height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden; display: flex;">
                                <div style="width: ${dist.even_percent}%; background: #f43f5e; height: 100%;"></div>
                                <div style="width: ${dist.odd_percent}%; background: #3b82f6; height: 100%;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
                                <span style="color: var(--text-secondary);">Nhỏ: ${dist.small_percent}%</span>
                                <span style="color: var(--text-secondary);">Lớn: ${dist.large_percent}%</span>
                            </div>
                            <div style="height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden; display: flex;">
                                <div style="width: ${dist.small_percent}%; background: #10b981; height: 100%;"></div>
                                <div style="width: ${dist.large_percent}%; background: #fbbf24; height: 100%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            } else if (gameId === 'bingo18') {
                const dist = stats.distribution;
                
                // Hot dice numbers
                const sortedDice = [...stats.number_stats].sort((a,b) => b.freq_100 - a.freq_100);
                const diceHTML = sortedDice.slice(0, 3).map(n => {
                    const icons = ['one','two','three','four','five','six'];
                    return `
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.35rem 0.5rem; background: rgba(245, 158, 11, 0.03); border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.08);">
                        <span class="vietlott-ball keno-ball" style="border-radius: 4px; width:1.5rem; height:1.5rem; background: radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%);"><i class="fa-solid fa-dice-${icons[n.number-1]}"></i></span>
                        <span style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">${n.freq_100} lần</span>
                    </div>
                    `;
                }).join('');

                // Top Sums
                const sortedSums = [...stats.sum_stats].sort((a,b) => b.freq_100 - a.freq_100);
                const sumsHTML = sortedSums.slice(0, 3).map(s => `
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.35rem 0.5rem; background: rgba(244, 63, 94, 0.03); border-radius: 6px; border: 1px solid rgba(244, 63, 94, 0.08);">
                        <span style="font-family:'JetBrains Mono', monospace; font-weight:700; color:#f43f5e; font-size:0.85rem;">Tổng ${s.sum}</span>
                        <span style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">${s.freq_100} lần</span>
                    </div>
                `).join('');
                
                container.innerHTML = `
                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #f59e0b; font-weight: 600;">Xúc Xắc Ra Nhiều (Hot)</span>
                        <span class="stat-icon" style="background: rgba(245, 158, 11, 0.1); color: #f59e0b; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-dice"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
                        ${diceHTML}
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #f43f5e; font-weight: 600;">Tổng Điểm Ra Nhiều (Hot Sum)</span>
                        <span class="stat-icon" style="background: rgba(244, 63, 94, 0.1); color: #f43f5e; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-calculator"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
                        ${sumsHTML}
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-card-header">
                        <span class="label" style="color: #fbbf24; font-weight: 600;">Tài Xỉu &amp; Trùng Lặp (100 kỳ)</span>
                        <span class="stat-icon" style="background: rgba(251, 191, 36, 0.1); color: #fbbf24; padding: 0.2rem 0.4rem; border-radius: 4px;"><i class="fa-solid fa-chart-bar"></i></span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.65rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
                                <span style="color: var(--text-secondary);">Xỉu (3-10): ${dist.small_percent}%</span>
                                <span style="color: var(--text-secondary);">Tài (11-18): ${dist.large_percent}%</span>
                            </div>
                            <div style="height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden; display: flex;">
                                <div style="width: ${dist.small_percent}%; background: #10b981; height: 100%;"></div>
                                <div style="width: ${dist.large_percent}%; background: #f43f5e; height: 100%;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
                                <span style="color: var(--text-secondary);">Trùng đôi: ${dist.double_percent}%</span>
                                <span style="color: var(--text-secondary);">Trùng ba: ${dist.triple_percent}%</span>
                            </div>
                            <div style="height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden; display: flex;">
                                <div style="width: ${dist.double_percent}%; background: #fbbf24; height: 100%;"></div>
                                <div style="width: ${dist.triple_percent}%; background: #8b5cf6; height: 100%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                // position stats
                const posKeyMap = {
                    'hundreds': { title: 'Hàng Trăm (Hot)', color: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.1)' },
                    'tens': { title: 'Hàng Chục (Hot)', color: '#06b6d4', bg: 'rgba(6, 182, 212, 0.1)' },
                    'units': { title: 'Hàng Đơn Vị (Hot)', color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)' }
                };
                
                let cardsHTML = '';
                for (let [pos, posCfg] of Object.entries(posKeyMap)) {
                    let sortedPos = [...stats.position_stats[pos]].sort((a,b) => b.freq_100 - a.freq_100);
                    let posHTML = sortedPos.slice(0, 3).map(d => `
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.35rem 0.5rem; background: rgba(255,255,255,0.015); border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                            <span class="vietlott-ball max3d-ball" style="font-size:0.75rem; width:1.5rem; height:1.5rem; border-radius:4px; background: radial-gradient(circle at 30% 30%, #c084fc 0%, #9333ea 50%, #581c87 100%);">${d.digit}</span>
                            <span style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">${d.freq_100} lần</span>
                        </div>
                    `).join('');
                    
                    cardsHTML += `
                    <div class="stat-card">
                        <div class="stat-card-header">
                            <span class="label" style="color: ${posCfg.color}; font-weight: 600;">${posCfg.title}</span>
                            <span class="stat-icon" style="background: ${posCfg.bg}; color: ${posCfg.color}; padding: 0.2rem 0.4rem; border-radius: 4px;">0-9</span>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
                            ${posHTML}
                        </div>
                    </div>
                    `;
                }
                container.innerHTML = cardsHTML;
            }
        }

        function renderVietlottHeatmap(data, gameId) {
            const container = document.getElementById('vietlott-heatmap-container');
            const heatmapTitle = document.getElementById('vietlott-heatmap-title');
            const heatmapSubtitle = document.getElementById('vietlott-heatmap-subtitle');
            
            if (!container) return;
            
            const stats = data.stats;
            if (!stats) {
                container.innerHTML = '';
                return;
            }
            
            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                if (heatmapTitle) heatmapTitle.innerHTML = `<i class="fa-solid fa-fire"></i> Ma Trận Tần Suất &amp; Thời Gian Vắng Bóng`;
                if (heatmapSubtitle) heatmapSubtitle.innerText = 'Màu đậm: ra nhiều 100 kỳ | Số nhỏ bên dưới: số kỳ chưa về';
                
                let maxFreq = Math.max(...stats.number_stats.map(n => n.freq_100), 1);
                
                // Sort number stats by number asc
                const sortedStats = [...stats.number_stats].sort((a,b) => a.number - b.number);
                
                let html = '<div class="heatmap-grid">';
                for (let n of sortedStats) {
                    let isHot = stats.hot_numbers.some(hn => hn.number === n.number);
                    let isCold = stats.cold_numbers.some(cn => cn.number === n.number);
                    let cellClass = 'heatmap-cell';
                    if (isHot) cellClass += ' hot-cell';
                    if (isCold) cellClass += ' cold-cell';
                    
                    let opacity = 0.03 + 0.55 * (n.freq_100 / maxFreq);
                    let bgStyle = `background-color: rgba(244, 63, 94, ${opacity});`;
                    if (isCold) {
                        bgStyle = `background-color: rgba(59, 130, 246, ${opacity});`;
                    }
                    
                    html += `
                    <div class="${cellClass}" style="${bgStyle}" title="Xuất hiện 100 kỳ qua: ${n.freq_100} lần (Tổng: ${n.total_occurrences}) | Số kỳ chưa ra: ${n.draws_since_last}">
                        <span class="heatmap-num">${String(n.number).padStart(2, '0')}</span>
                        <span class="heatmap-sub">${n.draws_since_last} kỳ</span>
                    </div>
                    `;
                }
                html += '</div>';
                container.innerHTML = html;
            } else if (gameId === 'bingo18') {
                if (heatmapTitle) heatmapTitle.innerHTML = `<i class="fa-solid fa-dice"></i> Tần Suất Xúc Xắc &amp; Điểm Số`;
                if (heatmapSubtitle) heatmapSubtitle.innerText = 'Màu đậm: số/tổng xuất hiện nhiều hơn';
                
                // Dice 1-6
                let html = '<div style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.3rem;"><i class="fa-solid fa-dice"></i> Tần suất xúc xắc (1 - 6)</div>';
                html += '<div class="heatmap-grid" style="margin-bottom: 1.5rem; grid-template-columns: repeat(6, 1fr);">';
                let maxNumFreq = Math.max(...stats.number_stats.map(n => n.freq_100), 1);
                for (let n of stats.number_stats) {
                    let opacity = 0.03 + 0.55 * (n.freq_100 / maxNumFreq);
                    const diceIcons = ['one','two','three','four','five','six'];
                    html += `
                    <div class="heatmap-cell" style="background-color: rgba(245, 158, 11, ${opacity}); padding: 0.6rem 0;" title="Số lần xuất hiện: ${n.freq_100}">
                        <span class="heatmap-num" style="font-size: 1.3rem; color: #fbbf24;"><i class="fa-solid fa-dice-${diceIcons[n.number-1]}"></i></span>
                        <span class="heatmap-sub" style="color: var(--text-primary); font-weight:600; font-family:'JetBrains Mono';">${n.freq_100} lần</span>
                    </div>
                    `;
                }
                html += '</div>';
                
                // Sums 3-18
                html += '<div style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.3rem;"><i class="fa-solid fa-calculator"></i> Tần suất tổng điểm (3 - 18)</div>';
                html += '<div class="heatmap-grid">';
                let maxSumFreq = Math.max(...stats.sum_stats.map(s => s.freq_100), 1);
                for (let s of stats.sum_stats) {
                    let opacity = 0.03 + 0.55 * (s.freq_100 / maxSumFreq);
                    html += `
                    <div class="heatmap-cell" style="background-color: rgba(244, 63, 94, ${opacity});" title="Số lần xuất hiện: ${s.freq_100}">
                        <span class="heatmap-num">${s.sum}</span>
                        <span class="heatmap-sub">${s.freq_100} lần</span>
                    </div>
                    `;
                }
                html += '</div>';
                
                container.innerHTML = html;
            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                if (heatmapTitle) heatmapTitle.innerHTML = `<i class="fa-solid fa-cubes"></i> Ma Trận Chữ Số Theo Vị Trí (0 - 9)`;
                if (heatmapSubtitle) heatmapSubtitle.innerText = 'Phân bổ tần suất xuất hiện của chữ số tại từng hàng (Trăm - Chục - Đơn vị)';
                
                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem;">';
                const positions = [
                    { key: 'hundreds', title: 'Hàng Trăm', color: 'rgba(139, 92, 246, ' },
                    { key: 'tens', title: 'Hàng Chục', color: 'rgba(6, 182, 212, ' },
                    { key: 'units', title: 'Hàng Đơn Vị', color: 'rgba(16, 185, 129, ' }
                ];
                
                positions.forEach(pos => {
                    let maxPosFreq = Math.max(...stats.position_stats[pos.key].map(d => d.freq_100), 1);
                    html += `
                    <div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 0.3rem;">${pos.title}</div>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.4rem;">
                    `;
                    for (let d of stats.position_stats[pos.key]) {
                        let opacity = 0.05 + 0.55 * (d.freq_100 / maxPosFreq);
                        html += `
                        <div class="heatmap-cell" style="background-color: ${pos.color}${opacity}); padding: 0.5rem 0;" title="Số lần: ${d.freq_100} | Số kỳ chưa ra: ${d.draws_since_last}">
                            <span class="heatmap-num">${d.digit}</span>
                            <span class="heatmap-sub" style="font-size: 0.55rem; color: var(--text-primary); font-weight:600;">${d.freq_100} lần</span>
                        </div>
                        `;
                    }
                    html += `
                        </div>
                    </div>
                    `;
                });
                html += '</div>';
                container.innerHTML = html;
            }
        }

        function renderVietlottHistoryTable(data, gameId) {
            const tbody = document.getElementById('vietlott-history-tbody');
            const extraHeader = document.getElementById('vietlott-history-extra-header');
            
            if (!tbody) return;
            
            const draws = data.recent_draws || [];
            if (draws.length === 0) {
                tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; padding: 2rem; color: var(--text-muted);">Không có dữ liệu lịch sử.</td></tr>`;
                return;
            }
            
            if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                if (extraHeader) extraHeader.innerText = 'Phân Tích Lẻ/Chẵn';
            } else if (gameId === 'bingo18') {
                if (extraHeader) extraHeader.innerText = 'Tổng Điểm';
            } else if (['max3d', 'max3dpro'].includes(gameId)) {
                if (extraHeader) extraHeader.innerText = 'Giải Nhất & Khác';
            }
            
            let html = '';
            for (let draw of draws) {
                let ballsHTML = '';
                let extraValue = '';
                
                if (['mega645', 'power655', 'power535', 'keno'].includes(gameId)) {
                    let result = draw.result || [];
                    let isPower655 = gameId === 'power655';
                    let isPower535 = gameId === 'power535';
                    
                    let ballClass = 'vietlott-ball';
                    if (gameId === 'keno') ballClass += ' keno-ball';
                    
                    ballsHTML = `<div class="vietlott-balls-container">` + result.map((num, idx) => {
                        let isSpecial = (isPower655 && idx === 6) || (isPower535 && idx === 5);
                        let cls = isSpecial ? `${ballClass} power-special` : ballClass;
                        return `<span class="${cls}" style="font-size:0.76rem; width:1.6rem; height:1.6rem;">${String(num).padStart(2, '0')}</span>`;
                    }).join(' ') + `</div>`;
                    
                    let odds = result.filter(n => n % 2 === 1).length;
                    let evens = result.length - odds;
                    extraValue = `<span style="font-family:'JetBrains Mono', monospace; font-size:0.76rem; font-weight:600; color:var(--text-secondary);">${odds} Lẻ - ${evens} Chẵn</span>`;
                } else if (gameId === 'bingo18') {
                    let result = draw.result || [];
                    ballsHTML = `<div class="vietlott-balls-container">` + result.map(num => {
                        const diceIcons = ['one', 'two', 'three', 'four', 'five', 'six'];
                        return `<span class="vietlott-ball keno-ball" style="border-radius:4px; font-size:0.85rem; width:1.5rem; height:1.5rem; background: radial-gradient(circle at 30% 30%, #fbbf24 0%, #d97706 50%, #78350f 100%);"><i class="fa-solid fa-dice-${diceIcons[num-1]}"></i></span>`;
                    }).join(' ') + `</div>`;
                    
                    extraValue = `<span style="font-size:0.78rem; font-weight:600; color:var(--text-primary);">Tổng ${draw.total} (${draw.large_small})</span>`;
                } else if (['max3d', 'max3dpro'].includes(gameId)) {
                    let result = draw.result || {};
                    let dbPrizes = result["Giải Đặc biệt"] || [];
                    ballsHTML = `<div class="vietlott-balls-container">` + dbPrizes.map(val => `<span class="vietlott-ball max3d-ball" style="height: 1.5rem; width:2.2rem; font-size: 0.74rem; border-radius:4px;">${val}</span>`).join(' ') + `</div>`;
                    
                    let firstPrize = result["Giải Nhất"] || [];
                    extraValue = `<span style="font-size:0.7rem; color:var(--text-secondary);" title="${firstPrize.join(', ')}">Nhất: ${firstPrize.slice(0, 3).join(', ')}${firstPrize.length > 3 ? '...' : ''}</span>`;
                }
                
                html += `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); transition: background var(--transition-fast);">
                    <td style="padding: 0.55rem 0.8rem; font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:var(--text-secondary);">${draw.date}</td>
                    <td style="padding: 0.55rem 0.8rem; font-family:'JetBrains Mono', monospace; font-size:0.74rem; font-weight:600; color:var(--text-primary);">#${draw.id}</td>
                    <td style="padding: 0.55rem 0.8rem;">${ballsHTML}</td>
                    <td style="padding: 0.55rem 0.8rem;">${extraValue}</td>
                </tr>
                `;
            }
            tbody.innerHTML = html;
        }

        // Onload Vietlott page initialization
        document.addEventListener("DOMContentLoaded", async () => {
            // Show vietlott dashboard Immediately
            const vlDash = document.getElementById('vietlott-dashboard-content');
            if (vlDash) vlDash.style.display = 'block';

            // Load Mega 6/45 as default
            await selectVietlottGame('mega645');
        });


