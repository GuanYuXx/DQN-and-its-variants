document.addEventListener('DOMContentLoaded', () => {
    const widthInput = document.getElementById('width');
    const heightInput = document.getElementById('height');
    const epochsInput = document.getElementById('epochs');
    const trainBtn = document.getElementById('train-btn');
    const randomizeBtn = document.getElementById('randomize-btn');
    
    const gridContainer = document.getElementById('grid-container');
    const replayGridContainer = document.getElementById('replay-grid-container');
    const routeSvg = document.getElementById('route-svg');
    const routeSelectors = document.getElementById('route-selectors');
    
    const trainStatusEl = document.getElementById('training-status');
    const replayStatusEl = document.getElementById('replay-status');
    const replayBtn = document.getElementById('replay-btn');
    
    const modeSelect = document.getElementById('mode-select');
    const basicModeContent = document.getElementById('basic-mode-content');
    const compareModeContent = document.getElementById('compare-mode-content');
    const replayGridContainerDb = document.getElementById('replay-grid-container-db');
    const replayGridContainerDu = document.getElementById('replay-grid-container-du');
    const trainStatusDb = document.getElementById('training-status-db');
    const trainStatusDu = document.getElementById('training-status-du');
    const finalRouteSvgDb = document.getElementById('final-route-svg-db');
    const finalRouteSvgDu = document.getElementById('final-route-svg-du');
    
    let lossChart = null;
    let lossChartDb = null;
    let lossChartDu = null;
    let finalGame = null;
    let finalGameDb = null;
    let finalGameDu = null;
    let capturedRoutes = [];
    
    const colors = ['#3b82f6', '#f43f5e', '#8b5cf6', '#10b981', '#f59e0b'];

    function makeDraggable(container) {
        const cells = container.querySelectorAll('.cell');
        let draggedCell = null;
        
        cells.forEach(cell => {
            cell.setAttribute('draggable', 'true');
            cell.style.cursor = 'grab';
            
            cell.addEventListener('dragstart', function(e) {
                if (!this.classList.contains('player') && 
                    !this.classList.contains('goal') && 
                    !this.classList.contains('pit') && 
                    !this.classList.contains('wall')) {
                    e.preventDefault();
                    return;
                }
                draggedCell = this;
                e.dataTransfer.setData('text/plain', this.id);
                setTimeout(() => this.style.opacity = '0.5', 0);
            });
            
            cell.addEventListener('dragend', function() {
                setTimeout(() => this.style.opacity = '1', 0);
                draggedCell = null;
            });
            
            cell.addEventListener('dragover', function(e) {
                e.preventDefault();
            });
            
            cell.addEventListener('drop', function(e) {
                e.preventDefault();
                if (this === draggedCell) return;
                
                const tempClasses = [...this.classList];
                const tempText = this.innerText;
                
                this.className = draggedCell.className;
                this.innerText = draggedCell.innerText;
                
                draggedCell.className = tempClasses.join(' ');
                draggedCell.innerText = tempText;
                
                routeSvg.innerHTML = '';
            });
        });
    }

    function initGrid(container, w, h, isDraggable = false) {
        container.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
        container.innerHTML = '';
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `${container.id}-cell-${r}-${c}`;
                
                if (r === 0 && c === 0) {
                    cell.classList.add('goal');
                    cell.innerText = '+';
                } else if (r === 0 && c === 1) {
                    cell.classList.add('pit');
                    cell.innerText = '-';
                } else if (r === 1 && c === 1) {
                    cell.classList.add('wall');
                    cell.innerText = 'W';
                }
                
                container.appendChild(cell);
            }
        }
        
        if (isDraggable) makeDraggable(container);
    }

    function randomizePositions() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const totalCells = w * h;
        
        const indices = [];
        while(indices.length < 4) {
            const r = Math.floor(Math.random() * totalCells);
            if(indices.indexOf(r) === -1) indices.push(r);
        }
        
        const cells = gridContainer.querySelectorAll('.cell');
        cells.forEach(c => {
            c.className = 'cell';
            c.innerText = '';
        });
        
        const types = [
            { class: 'player', text: 'P' },
            { class: 'goal', text: '+' },
            { class: 'pit', text: '-' },
            { class: 'wall', text: 'W' }
        ];
        
        for(let i=0; i<4; i++) {
            const cell = cells[indices[i]];
            cell.classList.add(types[i].class);
            cell.innerText = types[i].text;
        }
        routeSvg.innerHTML = '';
    }

    function updatePlayerPos(container, r, c) {
        container.querySelectorAll('.player').forEach(el => {
            el.classList.remove('player');
            if(el.innerText === 'P') el.innerText = '';
        });
        
        const cell = document.getElementById(`${container.id}-cell-${r}-${c}`);
        if (cell) {
            cell.classList.add('player');
            if(cell.innerText === '') cell.innerText = 'P';
        }
    }

    function initChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        if (lossChart) lossChart.destroy();
        
        lossChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Training Loss (Moving Avg)',
                data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });
        
        const ctxDb = document.getElementById('lossChartDb').getContext('2d');
        if (lossChartDb) lossChartDb.destroy();
        lossChartDb = new Chart(ctxDb, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Double DQN Loss',
                data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });

        const ctxDu = document.getElementById('lossChartDu').getContext('2d');
        if (lossChartDu) lossChartDu.destroy();
        lossChartDu = new Chart(ctxDu, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Dueling DQN Loss',
                data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });
    }

    function updateChart(epoch, loss) {
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
    }
    
    function updateChartCompare(epoch, lossDb, lossDu) {
        lossChartDb.data.labels.push(epoch);
        lossChartDb.data.datasets[0].data.push(lossDb);
        lossChartDb.update();
        
        lossChartDu.data.labels.push(epoch);
        lossChartDu.data.datasets[0].data.push(lossDu);
        lossChartDu.update();
    }
    
    function drawRoutes() {
        routeSvg.innerHTML = '';
        
        const checkedBoxes = Array.from(routeSelectors.querySelectorAll('input[type="checkbox"]:checked'));
        if (checkedBoxes.length === 0) return;
        
        const cellSize = 60;
        const gapSize = 4;
        
        checkedBoxes.forEach(box => {
            const index = parseInt(box.value);
            const route = capturedRoutes[index].route;
            const color = colors[index % colors.length];
            
            if (route.length < 2) return;
            
            let d = '';
            for(let i=0; i<route.length; i++) {
                const r = route[i].pos[0];
                const c = route[i].pos[1];
                const x = c * (cellSize + gapSize) + cellSize/2 + 8;
                const y = r * (cellSize + gapSize) + cellSize/2 + 8;
                
                const ox = x + (Math.random() * 8 - 4);
                const oy = y + (Math.random() * 8 - 4);
                
                if (i === 0) d += `M ${ox} ${oy} `;
                else d += `L ${ox} ${oy} `;
            }
            
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", d);
            path.setAttribute("fill", "none");
            path.setAttribute("stroke", color);
            path.setAttribute("stroke-width", "4");
            path.setAttribute("stroke-linejoin", "round");
            path.setAttribute("stroke-linecap", "round");
            path.setAttribute("opacity", "0.8");
            
            if (route.length > 0) {
                const last = route[route.length-1];
                const lx = last.pos[1] * (cellSize + gapSize) + cellSize/2 + 8;
                const ly = last.pos[0] * (cellSize + gapSize) + cellSize/2 + 8;
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", lx);
                circle.setAttribute("cy", ly);
                circle.setAttribute("r", "5");
                circle.setAttribute("fill", color);
                routeSvg.appendChild(circle);
            }
            
            routeSvg.appendChild(path);
        });
    }

    function buildRouteSelectors() {
        routeSelectors.innerHTML = '';
        if (capturedRoutes.length === 0) {
            routeSelectors.innerHTML = '<p class="placeholder-text">No routes captured.</p>';
            return;
        }
        
        capturedRoutes.forEach((cr, i) => {
            const label = document.createElement('label');
            const color = colors[i % colors.length];
            label.style.color = color;
            label.style.fontWeight = 'bold';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = i;
            checkbox.addEventListener('change', drawRoutes);
            
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` Epoch ${cr.epoch}`));
            routeSelectors.appendChild(label);
        });
    }

    function applyInitPos(initPos, container) {
        if (!initPos) return;
        const cells = container.querySelectorAll('.cell');
        cells.forEach(c => {
            c.className = 'cell';
            c.innerText = '';
        });
        const w = parseInt(widthInput.value);
        
        function setCell(type, pos, symbol) {
            const idx = pos[0] * w + pos[1];
            if (idx >= 0 && idx < cells.length) {
                cells[idx].classList.add(type);
                cells[idx].innerText = symbol;
            }
        }
        setCell('player', initPos.player, 'P');
        setCell('goal', initPos.goal, '+');
        setCell('pit', initPos.pit, '-');
        setCell('wall', initPos.wall, 'W');
    }

    async function playFinalGame() {
        if (!finalGame) return;
        
        replayBtn.disabled = true;
        replayStatusEl.innerText = 'Playing...';
        replayStatusEl.className = 'status running badge';
        
        // Sync starting position with the top grid
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        
        // Find player start in final game
        if (finalGame.length > 0) {
            const start = finalGame[0].pos;
            updatePlayerPos(replayGridContainer, start[0], start[1]);
            await new Promise(r => setTimeout(r, 500));
            
            for (let step of finalGame) {
                updatePlayerPos(replayGridContainer, step.pos[0], step.pos[1]);
                await new Promise(r => setTimeout(r, 200));
            }
        }
        
        replayBtn.disabled = false;
        replayStatusEl.innerText = 'Finished';
        replayStatusEl.className = 'status success badge';
    }

    function drawStaticRoute(route, svgElement, color) {
        svgElement.innerHTML = '';
        if (!route || route.length < 2) return;
        
        const cellSize = 60;
        const gapSize = 4;
        let d = '';
        
        for(let i=0; i<route.length; i++) {
            const r = route[i].pos[0];
            const c = route[i].pos[1];
            const x = c * (cellSize + gapSize) + cellSize/2 + 8;
            const y = r * (cellSize + gapSize) + cellSize/2 + 8;
            
            if (i === 0) d += `M ${x} ${y} `;
            else d += `L ${x} ${y} `;
        }
        
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "4");
        path.setAttribute("stroke-linejoin", "round");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("opacity", "0.9");
        
        const last = route[route.length-1];
        const lx = last.pos[1] * (cellSize + gapSize) + cellSize/2 + 8;
        const ly = last.pos[0] * (cellSize + gapSize) + cellSize/2 + 8;
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", lx);
        circle.setAttribute("cy", ly);
        circle.setAttribute("r", "5");
        circle.setAttribute("fill", color);
        
        svgElement.appendChild(path);
        svgElement.appendChild(circle);
    }

    function showFinalRoutesCompare() {
        syncReplayGrid();
        drawStaticRoute(finalGameDb, finalRouteSvgDb, '#f59e0b');
        drawStaticRoute(finalGameDu, finalRouteSvgDu, '#f59e0b');
    }

    function setupGrids() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        initGrid(gridContainer, w, h, true);
        initGrid(replayGridContainer, w, h, false);
        initGrid(replayGridContainerDb, w, h, false);
        initGrid(replayGridContainerDu, w, h, false);
        updatePlayerPos(gridContainer, 0, w - 1);
        
        // Copy layout from top grid to replay grid
        syncReplayGrid();
        
        routeSvg.innerHTML = '';
        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';
        routeSelectors.innerHTML = '<p class="placeholder-text">Routes will appear here after training.</p>';
    }
    
    function syncReplayGrid() {
        const cells = gridContainer.querySelectorAll('.cell');
        const replayCells = replayGridContainer.querySelectorAll('.cell');
        cells.forEach((cell, i) => {
            replayCells[i].className = cell.className;
            replayCells[i].innerText = cell.innerText;
        });
        // Compare grids are managed independently in compare mode — do NOT sync
        // the main grid's Player into them. We only copy Goal/Pit/Wall here as a
        // safety net for non-compare modes that still expect parity.
        if (modeSelect.value !== 'compare') {
            const replayCellsDb = replayGridContainerDb.querySelectorAll('.cell');
            const replayCellsDu = replayGridContainerDu.querySelectorAll('.cell');
            cells.forEach((cell, i) => {
                if (replayCellsDb[i]) {
                    replayCellsDb[i].className = cell.className;
                    replayCellsDb[i].innerText = cell.innerText;
                }
                if (replayCellsDu[i]) {
                    replayCellsDu[i].className = cell.className;
                    replayCellsDu[i].innerText = cell.innerText;
                }
            });
        }
    }

    // Init
    setupGrids();
    initChart();
    
    modeSelect.addEventListener('change', (e) => {
        const isLightning = e.target.value === 'lightning_random';
        const isCompare   = e.target.value === 'compare';
        const isRainbow   = e.target.value === 'rainbow_random';
        const rainbowModeContent = document.getElementById('rainbow-mode-content');
        if (isCompare) {
            basicModeContent.style.display = 'none';
            compareModeContent.style.display = 'flex';
            if (rainbowModeContent) rainbowModeContent.style.display = 'none';
            resetCompare();
        } else if (isRainbow) {
            basicModeContent.style.display = 'none';
            compareModeContent.style.display = 'none';
            if (rainbowModeContent) rainbowModeContent.style.display = 'flex';
            const rw = parseInt(document.getElementById('rainbow-width').value);
            const rh = parseInt(document.getElementById('rainbow-height').value);
            if (typeof initRainbowValidationGrid === 'function') initRainbowValidationGrid(rw, rh);
        } else {
            basicModeContent.style.display = 'block';
            compareModeContent.style.display = 'none';
            if (rainbowModeContent) rainbowModeContent.style.display = 'none';
            if (lossChart && !isLightning) lossChart.resize();
        }
        // Hide Start Training / Randomize / epochs in compare/rainbow mode (offline pre-trained)
        const hideTrainUI = isCompare || isRainbow;
        if (trainBtn)        trainBtn.style.display      = hideTrainUI ? 'none' : '';
        if (randomizeBtn)    randomizeBtn.style.display  = hideTrainUI ? 'none' : '';
        const epochsGroupCmp = document.getElementById('epochs-group');
        if (epochsGroupCmp)  epochsGroupCmp.style.display = hideTrainUI ? 'none' : '';
        // Hide top width/height inputs in rainbow / compare mode
        // (rainbow uses its own X/Y selects; compare is fixed 4×4 pre-trained)
        const widthGroup  = widthInput  ? widthInput.closest('.control-group')  : null;
        const heightGroup = heightInput ? heightInput.closest('.control-group') : null;
        const hideGridSize = isRainbow || isCompare;
        if (widthGroup)  widthGroup.style.display  = hideGridSize ? 'none' : '';
        if (heightGroup) heightGroup.style.display = hideGridSize ? 'none' : '';
        
        const lightningInfoPanel = document.getElementById('lightning-info-panel');
        if (lightningInfoPanel) {
            lightningInfoPanel.style.display = isLightning ? 'block' : 'none';
        }
        // basic-info-panel: show only in basic mode (HW3-1)
        const basicInfoPanel = document.getElementById('basic-info-panel');
        if (basicInfoPanel) {
            const isBasic = !isLightning && !isCompare && !isRainbow;
            basicInfoPanel.style.display = isBasic ? 'block' : 'none';
        }
        const lightningControls = document.getElementById('lightning-controls');
        if (lightningControls) {
            lightningControls.style.display = 'none'; // no training config in random mode
        }

        // Hide/show training-only panels
        const sampledRoutesPanel = document.getElementById('sampled-routes-panel');
        if (sampledRoutesPanel) sampledRoutesPanel.style.display = isLightning ? 'none' : 'block';
        const replayPanelWrapper = document.getElementById('replay-panel-wrapper');
        if (replayPanelWrapper) replayPanelWrapper.style.display = isLightning ? 'none' : 'block';
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.style.display = isLightning ? 'none' : 'grid';
            mainContent.style.gridTemplateColumns = '1fr 1fr';
        }
        const lightningImagePanel = document.getElementById('lightning-image-panel');
        if (lightningImagePanel) lightningImagePanel.style.display = isLightning ? 'block' : 'none';

        // Hide Start Training button and epochs in random mode OR compare mode
        const trainBtn_ = document.getElementById('train-btn');
        if (trainBtn_) trainBtn_.style.display = (isLightning || isCompare) ? 'none' : '';
        const epochsGroup = document.getElementById('epochs-group');
        if (epochsGroup) epochsGroup.style.display = (isLightning || isCompare) ? 'none' : '';

        // Show validation grid immediately in lightning_random mode
        const interactiveValidationPanel = document.getElementById('interactive-validation-panel');
        if (interactiveValidationPanel) {
            interactiveValidationPanel.style.display = isLightning ? 'block' : 'none';
        }
        if (isLightning) {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
            const verifyBtnEl = document.getElementById('verify-btn');
            if (verifyBtnEl) verifyBtnEl.disabled = false;
        }

        if (!isLightning) initChart();
        setupGrids();
        // Enable drag attribute on compare grids so the Player marker can be dragged.
        if (isCompare && typeof window.enableCompareDragAttr === 'function') {
            window.enableCompareDragAttr();
        }
        trainStatusEl.innerText = 'Awaiting Training...';
        trainStatusEl.className = 'status badge';
        replayStatusEl.innerText = 'Awaiting Training...';
        replayStatusEl.className = 'status badge';
        if (isCompare) {
            trainStatusDb.innerText = 'Pre-trained Loaded';
            trainStatusDb.className = 'status success badge';
            trainStatusDu.innerText = 'Pre-trained Loaded';
            trainStatusDu.className = 'status success badge';
        } else {
            trainStatusDb.innerText = 'Awaiting Training...';
            trainStatusDb.className = 'status badge';
            trainStatusDu.innerText = 'Awaiting Training...';
            trainStatusDu.className = 'status badge';
        }
        if (!isLightning) {
            trainBtn.innerText = 'Start Training';
            trainBtn.disabled = false;
        }
        replayBtn.disabled = true;
        const lrStatus = document.getElementById('lr-status');
        if (lrStatus) lrStatus.style.display = 'none';
    });

    function onGridSizeChange() {
        setupGrids();
        if (modeSelect.value === 'lightning_random') {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
            document.getElementById('validation-route-svg').innerHTML = '';
        }
    }
    widthInput.addEventListener('change', onGridSizeChange);
    heightInput.addEventListener('change', onGridSizeChange);

    randomizeBtn.addEventListener('click', () => {
        randomizePositions();
        syncReplayGrid();
    });
    replayBtn.addEventListener('click', playFinalGame);

    trainBtn.addEventListener('click', () => {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const epochs = parseInt(epochsInput.value);

        let playerPos = '', goalPos = '', pitPos = '', wallPos = '';
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.getElementById(`${gridContainer.id}-cell-${r}-${c}`);
                if (cell.classList.contains('player')) playerPos = `${r},${c}`;
                if (cell.classList.contains('goal')) goalPos = `${r},${c}`;
                if (cell.classList.contains('pit')) pitPos = `${r},${c}`;
                if (cell.classList.contains('wall')) wallPos = `${r},${c}`;
            }
        }
        
        syncReplayGrid();
        initChart();

        trainBtn.disabled = true;
        randomizeBtn.disabled = true;
        trainBtn.innerText = 'Training...';
        trainStatusEl.innerText = 'Training in progress...';
        trainStatusEl.className = 'status running badge';
        replayBtn.disabled = true;
        replayStatusEl.innerText = 'Awaiting Training...';
        replayStatusEl.className = 'status badge';
        
        const mode = modeSelect.value;
        if (mode === 'compare') {
            trainStatusDb.innerText = 'Training...';
            trainStatusDb.className = 'status running badge';
            trainStatusDu.innerText = 'Training...';
            trainStatusDu.className = 'status running badge';
        }
        
        finalGame = null;
        finalGameDb = null;
        finalGameDu = null;
        capturedRoutes = [];

        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';

        const gradClipNode = document.getElementById('grad-clip');
        const gradClip = gradClipNode ? gradClipNode.value : '1.0';
        
        const lrStatus = document.getElementById('lr-status');
        if (lrStatus && mode === 'lightning_random') {
            lrStatus.style.display = 'inline-block';
            lrStatus.innerText = 'LR: --';
        }

        const source = new EventSource(`/api/stream_train?mode=${mode}&width=${w}&height=${h}&epochs=${epochs}&player=${playerPos}&goal=${goalPos}&pit=${pitPos}&wall=${wallPos}&grad_clip=${gradClip}`);
        
        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                if (mode === 'compare') {
                    updateChartCompare(data.epoch, data.loss_double, data.loss_dueling);
                } else {
                    updateChart(data.epoch, data.loss);
                    if (data.lr !== undefined && document.getElementById('lr-status')) {
                        document.getElementById('lr-status').innerText = `LR: ${data.lr.toExponential(2)}`;
                    }
                }
            } else if (data.type === 'complete') {
                source.close();
                
                trainBtn.disabled = false;
                randomizeBtn.disabled = false;
                trainBtn.innerText = 'Start Training';
                trainStatusEl.innerText = 'Training Complete';
                trainStatusEl.className = 'status success badge';
                
                if (mode === 'compare') {
                    finalGameDb = data.final_game_double;
                    finalGameDu = data.final_game_dueling;
                    trainStatusDb.innerText = 'Complete';
                    trainStatusDb.className = 'status success badge';
                    trainStatusDu.innerText = 'Complete';
                    trainStatusDu.className = 'status success badge';
                    
                    showFinalRoutesCompare();
                } else {
                    capturedRoutes = data.routes;
                    finalGame = data.final_game;
                    
                    if (mode === 'lightning_random') {
                        document.getElementById('interactive-validation-panel').style.display = 'block';
                        const verifyBtn = document.getElementById('verify-btn');
                        if (verifyBtn) verifyBtn.disabled = false;
                        if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
                        
                        setTimeout(() => {
                            document.getElementById('interactive-validation-panel').scrollIntoView({behavior: 'smooth'});
                        }, 100);
                    } else {
                        replayBtn.disabled = false;
                        replayStatusEl.innerText = 'Ready';
                        buildRouteSelectors();
                        playFinalGame();
                    }
                }
            }
        };
        
        source.onerror = function() {
            source.close();
            console.log("SSE connection closed.");
        };
    });
    
    // ─────────────────────────────────────────────────────────────────────
    // Compare-mode (offline pre-trained Double DQN + Dueling DQN)
    // ─────────────────────────────────────────────────────────────────────
    let comparePlayerPos = null;     // [r, c] or null
    let compareIsLoading = false;

    function resetCompare() {
        comparePlayerPos = null;
        // Remove any Player markers from the two compare grids; keep Goal/Pit/Wall
        [replayGridContainerDb, replayGridContainerDu].forEach(grid => {
            if (!grid) return;
            grid.querySelectorAll('.cell.player').forEach(el => {
                el.classList.remove('player');
                if (el.innerText === 'P') el.innerText = '';
            });
        });
        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';

        const posDisp = document.getElementById('compare-player-pos');
        if (posDisp) { posDisp.innerText = '未選擇'; posDisp.style.color = '#f59e0b'; }
        const resDb = document.getElementById('compare-result-db');
        const resDu = document.getElementById('compare-result-du');
        if (resDb) resDb.innerText = '';
        if (resDu) resDu.innerText = '';
        const cmpStatus = document.getElementById('compare-status');
        if (cmpStatus) {
            cmpStatus.innerText = '已載入預訓練模型';
            cmpStatus.className = 'status success badge';
        }
    }

    function handleCompareCellClick(r, c) {
        // Only allow Player start on empty cells (not on Goal/Pit/Wall)
        const refId = `${replayGridContainerDb.id}-cell-${r}-${c}`;
        const refCell = document.getElementById(refId);
        if (!refCell) return;
        if (refCell.classList.contains('goal') ||
            refCell.classList.contains('pit')  ||
            refCell.classList.contains('wall')) {
            const cmpStatus = document.getElementById('compare-status');
            if (cmpStatus) {
                cmpStatus.innerText = '此格被 Goal/Pit/Wall 佔用，請點空格';
                cmpStatus.className = 'status badge';
                cmpStatus.style.background = '#f43f5e';
                cmpStatus.style.color = '#fff';
                setTimeout(() => {
                    cmpStatus.style.background = '';
                    cmpStatus.style.color = '';
                    cmpStatus.innerText = '已載入預訓練模型';
                    cmpStatus.className = 'status success badge';
                }, 2000);
            }
            return;
        }
        comparePlayerPos = [r, c];
        [replayGridContainerDb, replayGridContainerDu].forEach(grid => {
            grid.querySelectorAll('.cell.player').forEach(el => {
                el.classList.remove('player');
                if (el.innerText === 'P') el.innerText = '';
            });
            const cell = document.getElementById(`${grid.id}-cell-${r}-${c}`);
            if (cell) {
                cell.classList.add('player');
                cell.innerText = 'P';
            }
        });
        // Old routes/results are stale once the player moves
        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';
        const resDb = document.getElementById('compare-result-db');
        const resDu = document.getElementById('compare-result-du');
        if (resDb) resDb.innerText = '';
        if (resDu) resDu.innerText = '';

        const posDisp = document.getElementById('compare-player-pos');
        if (posDisp) { posDisp.innerText = `(${r}, ${c})`; posDisp.style.color = '#f59e0b'; }

        // Re-mark the new player cells as draggable (and other cells as drop targets).
        if (typeof window.enableCompareDragAttr === 'function') window.enableCompareDragAttr();
    }

    function _renderCompareSummary(elemId, route, label) {
        const el = document.getElementById(elemId);
        if (!el || !route || route.length === 0) return;
        const last = route[route.length - 1];
        const steps = route.length - 1;
        let msg;
        if (last.reward === 10)       msg = `${label}: ✅ 抵達 Goal（${steps} 步）`;
        else if (last.reward === -10) msg = `${label}: ❌ 掉入 Pit（${steps} 步）`;
        else                          msg = `${label}: ⏱ 超出步數上限（${steps} 步）`;
        el.innerText = msg;
    }

    async function playCompare() {
        const cmpStatus = document.getElementById('compare-status');
        const playBtn = document.getElementById('compare-play-btn');
        if (comparePlayerPos === null) {
            if (cmpStatus) {
                cmpStatus.innerText = '請先在任一網格點擊空白格設定 Player 起點';
                cmpStatus.className = 'status badge';
                cmpStatus.style.background = '#f43f5e';
                cmpStatus.style.color = '#fff';
                setTimeout(() => {
                    cmpStatus.style.background = '';
                    cmpStatus.style.color = '';
                    cmpStatus.innerText = '已載入預訓練模型';
                    cmpStatus.className = 'status success badge';
                }, 2500);
            }
            return;
        }
        if (compareIsLoading) return;
        compareIsLoading = true;
        if (playBtn) { playBtn.disabled = true; playBtn.innerText = 'Verifying...'; }
        if (cmpStatus) {
            cmpStatus.innerText = 'Verifying...';
            cmpStatus.className = 'status running badge';
        }
        const payload = {
            width:  parseInt(widthInput.value),
            height: parseInt(heightInput.value),
            player_pos: comparePlayerPos,
        };
        try {
            const [rDb, rDu] = await Promise.all([
                fetch('/api/verify_double',  { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }),
                fetch('/api/verify_dueling', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }),
            ]);
            const dataDb = await rDb.json();
            const dataDu = await rDu.json();
            if (dataDb.status === 'success') {
                drawStaticRoute(dataDb.route, finalRouteSvgDb, '#f59e0b');
                _renderCompareSummary('compare-result-db', dataDb.route, 'Double');
            } else { console.error('verify_double error:', dataDb.message); }
            if (dataDu.status === 'success') {
                drawStaticRoute(dataDu.route, finalRouteSvgDu, '#f59e0b');
                _renderCompareSummary('compare-result-du', dataDu.route, 'Dueling');
            } else { console.error('verify_dueling error:', dataDu.message); }
            if (cmpStatus) {
                cmpStatus.innerText = 'Verification Complete';
                cmpStatus.className = 'status success badge';
            }
        } catch (err) {
            console.error(err);
            if (cmpStatus) {
                cmpStatus.innerText = 'Error';
                cmpStatus.className = 'status badge';
            }
        } finally {
            compareIsLoading = false;
            if (playBtn) { playBtn.disabled = false; playBtn.innerText = '▶ Play'; }
        }
    }

    // Event delegation: cells inside compare grids get re-rendered by initGrid(),
    // so bind on the container once instead of per-cell.
    let compareDragged = null;
    [replayGridContainerDb, replayGridContainerDu].forEach(grid => {
        if (!grid) return;
        grid.addEventListener('click', (e) => {
            if (modeSelect.value !== 'compare') return;
            const cell = e.target.closest('.cell');
            if (!cell || !cell.id) return;
            const m = cell.id.match(/-cell-(\d+)-(\d+)$/);
            if (!m) return;
            handleCompareCellClick(parseInt(m[1]), parseInt(m[2]));
        });

        // Drag support: Player marker can be dragged onto any empty cell.
        grid.addEventListener('dragstart', (e) => {
            if (modeSelect.value !== 'compare') return;
            const cell = e.target.closest('.cell');
            if (!cell || !cell.classList.contains('player')) { e.preventDefault(); return; }
            compareDragged = cell;
            e.dataTransfer.effectAllowed = 'move';
            setTimeout(() => cell.style.opacity = '0.5', 0);
        });
        grid.addEventListener('dragend', (e) => {
            const cell = e.target.closest('.cell');
            if (cell) cell.style.opacity = '1';
            compareDragged = null;
        });
        grid.addEventListener('dragover', (e) => {
            if (modeSelect.value !== 'compare' || !compareDragged) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });
        grid.addEventListener('drop', (e) => {
            if (modeSelect.value !== 'compare' || !compareDragged) return;
            e.preventDefault();
            const cell = e.target.closest('.cell');
            if (!cell || !cell.id) return;
            if (cell.classList.contains('goal') ||
                cell.classList.contains('pit')  ||
                cell.classList.contains('wall')) return;
            const m = cell.id.match(/-cell-(\d+)-(\d+)$/);
            if (!m) return;
            handleCompareCellClick(parseInt(m[1]), parseInt(m[2]));
        });
    });

    // Make every cell on the two compare grids draggable so dragstart fires;
    // the handler above filters to allow only .player cells to actually start.
    function enableCompareDragAttr() {
        [replayGridContainerDb, replayGridContainerDu].forEach(grid => {
            if (!grid) return;
            grid.querySelectorAll('.cell').forEach(cell => {
                cell.setAttribute('draggable', 'true');
                if (!cell.classList.contains('goal') &&
                    !cell.classList.contains('pit')  &&
                    !cell.classList.contains('wall')) {
                    cell.style.cursor = cell.classList.contains('player') ? 'grab' : 'pointer';
                } else {
                    cell.style.cursor = 'default';
                }
            });
        });
    }
    window.enableCompareDragAttr = enableCompareDragAttr;

    function randomComparePlayerPos() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const empties = [];
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.getElementById(`${replayGridContainerDb.id}-cell-${r}-${c}`);
                if (cell &&
                    !cell.classList.contains('goal') &&
                    !cell.classList.contains('pit')  &&
                    !cell.classList.contains('wall')) {
                    empties.push([r, c]);
                }
            }
        }
        if (empties.length === 0) return;
        const [r, c] = empties[Math.floor(Math.random() * empties.length)];
        handleCompareCellClick(r, c);
        enableCompareDragAttr();
    }

    const comparePlayBtn   = document.getElementById('compare-play-btn');
    if (comparePlayBtn)   comparePlayBtn.addEventListener('click', playCompare);
    const compareResetBtn  = document.getElementById('compare-reset-btn');
    if (compareResetBtn)  compareResetBtn.addEventListener('click', () => { resetCompare(); enableCompareDragAttr(); });
    const compareRandomBtn = document.getElementById('compare-random-btn');
    if (compareRandomBtn) compareRandomBtn.addEventListener('click', randomComparePlayerPos);

    // Validation Grid Logic
    let draggedElement = null;
    function initValidationGrid(w, h) {
        const maxObstacles = Math.min(w, h) - 1;

        // Update max hint display
        const maxDisplay = document.getElementById('max-obstacles-display');
        if (maxDisplay) maxDisplay.innerText = maxObstacles;

        // Read user inputs, default to 1/1; both W and Pit must be at least 1
        const numWallsInput = document.getElementById('num-walls');
        const numPitsInput  = document.getElementById('num-pits');
        let numWalls = numWallsInput ? Math.max(1, parseInt(numWallsInput.value) || 1) : 1;
        let numPits  = numPitsInput  ? Math.max(1, parseInt(numPitsInput.value)  || 1) : 1;

        // Clamp: walls in [1, maxObstacles-1] (reserve 1 for pit), then pits in [1, remaining]
        numWalls = Math.min(numWalls, Math.max(1, maxObstacles - 1));
        numPits  = Math.min(numPits,  Math.max(1, maxObstacles - numWalls));

        // Write clamped values back
        if (numWallsInput) numWallsInput.value = numWalls;
        if (numPitsInput)  numPitsInput.value  = numPits;

        const container = document.getElementById('validation-grid-container');
        if (!container) return;

        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${w}, 60px)`;
        document.getElementById('validation-route-svg').innerHTML = '';

        let pieces = ['player', 'goal'];
        for(let i=0; i<numPits; i++) pieces.push('pit');
        for(let i=0; i<numWalls; i++) pieces.push('wall');
        while(pieces.length < w * h) pieces.push('');
        
        pieces.sort(() => Math.random() - 0.5);
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `val-cell-${r}-${c}`;
                
                const piece = pieces.shift();
                if (piece) {
                    cell.classList.add(piece);
                    if (piece === 'player') cell.innerText = 'P';
                    if (piece === 'goal') cell.innerText = '+';
                    if (piece === 'pit') cell.innerText = '-';
                    if (piece === 'wall') cell.innerText = 'W';
                    cell.draggable = true;
                    cell.style.cursor = 'grab';
                } else {
                    cell.draggable = false;
                    cell.style.cursor = 'default';
                }

                cell.addEventListener('dragstart', function(e) {
                    if (!this.draggable) { e.preventDefault(); return; }
                    draggedElement = this;
                    e.dataTransfer.effectAllowed = 'move';
                    this.style.cursor = 'grabbing';
                    setTimeout(() => this.style.opacity = '0.5', 0);
                });

                cell.addEventListener('dragend', function() {
                    setTimeout(() => this.style.opacity = '1', 0);
                    if (this.draggable) this.style.cursor = 'grab';
                    draggedElement = null;
                });

                cell.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                });

                cell.addEventListener('drop', function(e) {
                    e.preventDefault();
                    if (draggedElement && this !== draggedElement) {
                        const tempClass = draggedElement.className;
                        const tempText = draggedElement.innerText;
                        const tempDraggable = draggedElement.draggable;
                        const tempCursor = draggedElement.style.cursor;

                        draggedElement.className = this.className;
                        draggedElement.innerText = this.innerText;
                        draggedElement.draggable = this.draggable;
                        draggedElement.style.cursor = this.style.cursor;

                        this.className = tempClass;
                        this.innerText = tempText;
                        this.draggable = tempDraggable;
                        this.style.cursor = tempCursor;

                        document.getElementById('validation-route-svg').innerHTML = '';
                        document.getElementById('validation-status').innerText = 'Ready to Verify';
                        document.getElementById('validation-status').className = 'status badge';
                    }
                });
                
                container.appendChild(cell);
            }
        }
    }

    // Obstacle count inputs — re-render validation grid on change
    const numWallsInput = document.getElementById('num-walls');
    const numPitsInput  = document.getElementById('num-pits');

    if (numWallsInput) {
        numWallsInput.addEventListener('change', () => {
            if (modeSelect.value !== 'lightning_random') return;
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            const maxObs = Math.min(w, h) - 1;
            let nW = Math.max(1, parseInt(numWallsInput.value) || 1);
            let nP = Math.max(1, parseInt(numPitsInput.value)  || 1);
            nW = Math.min(nW, Math.max(1, maxObs - 1));
            nP = Math.min(nP, Math.max(1, maxObs - nW));
            numWallsInput.value = nW;
            numPitsInput.value  = nP;
            initValidationGrid(w, h);
        });
    }

    if (numPitsInput) {
        numPitsInput.addEventListener('change', () => {
            if (modeSelect.value !== 'lightning_random') return;
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            const maxObs = Math.min(w, h) - 1;
            let nW = Math.max(1, parseInt(numWallsInput.value) || 1);
            let nP = Math.max(1, parseInt(numPitsInput.value)  || 1);
            nP = Math.min(nP, Math.max(1, maxObs - 1));
            nW = Math.min(nW, Math.max(1, maxObs - nP));
            numWallsInput.value = nW;
            numPitsInput.value  = nP;
            initValidationGrid(w, h);
        });
    }

    const verifyBtn = document.getElementById('verify-btn');
    if (verifyBtn) {
        verifyBtn.addEventListener('click', async () => {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            let custom_positions = {};
            
            for (let r = 0; r < h; r++) {
                for (let c = 0; c < w; c++) {
                    const cell = document.getElementById(`val-cell-${r}-${c}`);
                    if (!cell) continue;
                    if (cell.classList.contains('player')) custom_positions['Player'] = [r, c];
                    if (cell.classList.contains('goal'))   custom_positions['Goal']   = [r, c];
                    if (cell.classList.contains('pit'))  {
                        if (!custom_positions['Pit']) custom_positions['Pit'] = [];
                        custom_positions['Pit'].push([r, c]);
                    }
                    if (cell.classList.contains('wall')) {
                        if (!custom_positions['Wall']) custom_positions['Wall'] = [];
                        custom_positions['Wall'].push([r, c]);
                    }
                }
            }
            
            const validationStatus = document.getElementById('validation-status');
            validationStatus.innerText = 'Verifying...';
            validationStatus.className = 'status running badge';
            verifyBtn.disabled = true;
            
            try {
                const response = await fetch('/api/verify_random', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({width: w, height: h, custom_positions: custom_positions})
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    drawStaticRoute(data.route, document.getElementById('validation-route-svg'), '#10b981');
                    validationStatus.innerText = 'Verification Complete';
                    validationStatus.className = 'status success badge';
                } else {
                    validationStatus.innerText = 'Error';
                    validationStatus.className = 'status badge';
                    console.error(data.message);
                }
            } catch (err) {
                console.error(err);
            } finally {
                verifyBtn.disabled = false;
            }
        });
    }

    // ── Rainbow Validation Grid ─────────────────────────────────────────────
    // 4 fixed pieces (Player / Goal / Pit / Wall), all draggable, no overlap.
    // Drop onto an occupied cell -> revert + .shake animation as feedback.
    let rainbowDragged = null;

    function _rainbowShake(cell) {
        if (!cell) return;
        cell.classList.remove('shake');
        // force reflow so re-adding the class restarts the animation
        void cell.offsetWidth;
        cell.classList.add('shake');
        setTimeout(() => cell.classList.remove('shake'), 450);
    }

    function _rainbowSetupCell(cell) {
        cell.addEventListener('dragstart', function(e) {
            if (!this.draggable) { e.preventDefault(); return; }
            rainbowDragged = this;
            e.dataTransfer.effectAllowed = 'move';
            this.style.cursor = 'grabbing';
            setTimeout(() => this.style.opacity = '0.5', 0);
        });
        cell.addEventListener('dragend', function() {
            setTimeout(() => this.style.opacity = '1', 0);
            if (this.draggable) this.style.cursor = 'grab';
            rainbowDragged = null;
        });
        cell.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });
        cell.addEventListener('drop', function(e) {
            e.preventDefault();
            if (!rainbowDragged || this === rainbowDragged) return;

            // Overlap guard: target must be a blank cell (no piece classes).
            const occupied = ['player', 'goal', 'pit', 'wall'].some(c => this.classList.contains(c));
            if (occupied) {
                _rainbowShake(rainbowDragged);
                return;
            }

            // Move: copy piece classes/text from source to target, blank the source.
            const cls = rainbowDragged.className;
            const txt = rainbowDragged.innerText;
            this.className = cls;
            this.innerText = txt;
            this.draggable = true;
            this.style.cursor = 'grab';

            rainbowDragged.className = 'cell';
            rainbowDragged.innerText = '';
            rainbowDragged.draggable = false;
            rainbowDragged.style.cursor = 'default';

            const svg = document.getElementById('rainbow-route-svg');
            if (svg) svg.innerHTML = '';
            const status = document.getElementById('rainbow-validation-status');
            if (status) {
                status.innerText = 'Ready to Verify';
                status.className = 'status badge';
            }
        });
    }

    function initRainbowValidationGrid(w, h) {
        const container = document.getElementById('rainbow-validation-grid');
        if (!container) return;
        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${w}, 60px)`;
        const svg = document.getElementById('rainbow-route-svg');
        if (svg) svg.innerHTML = '';

        // Random non-overlapping positions for the 4 pieces.
        const idx = [];
        for (let i = 0; i < w * h; i++) idx.push(i);
        idx.sort(() => Math.random() - 0.5);
        const [pIdx, gIdx, pitIdx, wIdx] = idx.slice(0, 4);
        const pieceAt = {};
        pieceAt[pIdx]   = { cls: 'player', txt: 'P' };
        pieceAt[gIdx]   = { cls: 'goal',   txt: '+' };
        pieceAt[pitIdx] = { cls: 'pit',    txt: '-' };
        pieceAt[wIdx]   = { cls: 'wall',   txt: 'W' };

        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const flat = r * w + c;
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `rainbow-cell-${r}-${c}`;
                const piece = pieceAt[flat];
                if (piece) {
                    cell.classList.add(piece.cls);
                    cell.innerText = piece.txt;
                    cell.draggable = true;
                    cell.style.cursor = 'grab';
                } else {
                    cell.draggable = false;
                    cell.style.cursor = 'default';
                }
                _rainbowSetupCell(cell);
                container.appendChild(cell);
            }
        }

        const status = document.getElementById('rainbow-validation-status');
        if (status) {
            status.innerText = 'Awaiting Verification...';
            status.className = 'status badge';
        }
    }
    window.initRainbowValidationGrid = initRainbowValidationGrid;

    const rainbowWidthSel  = document.getElementById('rainbow-width');
    const rainbowHeightSel = document.getElementById('rainbow-height');
    function _rainbowSizeChange() {
        if (modeSelect.value !== 'rainbow_random') return;
        const w = parseInt(rainbowWidthSel.value);
        const h = parseInt(rainbowHeightSel.value);
        initRainbowValidationGrid(w, h);
    }
    if (rainbowWidthSel)  rainbowWidthSel.addEventListener('change', _rainbowSizeChange);
    if (rainbowHeightSel) rainbowHeightSel.addEventListener('change', _rainbowSizeChange);

    const rainbowShuffleBtn = document.getElementById('rainbow-shuffle-btn');
    if (rainbowShuffleBtn) {
        rainbowShuffleBtn.addEventListener('click', () => {
            if (modeSelect.value !== 'rainbow_random') return;
            const w = parseInt(rainbowWidthSel.value);
            const h = parseInt(rainbowHeightSel.value);
            initRainbowValidationGrid(w, h);
        });
    }

    const rainbowVerifyBtn = document.getElementById('rainbow-verify-btn');
    if (rainbowVerifyBtn) {
        rainbowVerifyBtn.addEventListener('click', async () => {
            const w = parseInt(rainbowWidthSel.value);
            const h = parseInt(rainbowHeightSel.value);
            const positions = {};
            for (let r = 0; r < h; r++) {
                for (let c = 0; c < w; c++) {
                    const cell = document.getElementById(`rainbow-cell-${r}-${c}`);
                    if (!cell) continue;
                    if (cell.classList.contains('player')) positions['Player'] = [r, c];
                    if (cell.classList.contains('goal'))   positions['Goal']   = [r, c];
                    if (cell.classList.contains('pit'))    positions['Pit']    = [r, c];
                    if (cell.classList.contains('wall'))   positions['Wall']   = [r, c];
                }
            }
            const required = ['Player', 'Goal', 'Pit', 'Wall'];
            const missing = required.filter(k => !positions[k]);
            const status = document.getElementById('rainbow-validation-status');
            if (missing.length) {
                if (status) {
                    status.innerText = `Missing: ${missing.join(', ')}`;
                    status.className = 'status badge';
                }
                return;
            }
            if (status) {
                status.innerText = 'Verifying...';
                status.className = 'status running badge';
            }
            rainbowVerifyBtn.disabled = true;
            try {
                const response = await fetch('/api/verify_rainbow', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ width: w, height: h, positions: positions })
                });
                const data = await response.json();
                const svg = document.getElementById('rainbow-route-svg');
                if (data.status === 'success') {
                    drawStaticRoute(data.route, svg, '#a78bfa');
                    if (status) {
                        status.innerText = 'Verification Complete';
                        status.className = 'status success badge';
                    }
                } else {
                    if (status) {
                        status.innerText = 'Error';
                        status.className = 'status badge';
                    }
                    console.error(data.message);
                    alert(`Rainbow verify failed:\n${data.message}`);
                }
            } catch (err) {
                console.error(err);
                if (status) {
                    status.innerText = 'Network Error';
                    status.className = 'status badge';
                }
            } finally {
                rainbowVerifyBtn.disabled = false;
            }
        });
    }

});
