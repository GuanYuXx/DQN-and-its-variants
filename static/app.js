document.addEventListener('DOMContentLoaded', () => {
    const widthInput = document.getElementById('width');
    const heightInput = document.getElementById('height');
    const epochsInput = document.getElementById('epochs');
    const trainBtn = document.getElementById('train-btn');
    const gridContainer = document.getElementById('grid-container');
    const statusEl = document.getElementById('simulation-status');
    
    let lossChart = null;

    // Initialize Grid
    function initGrid(w, h) {
        gridContainer.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
        gridContainer.innerHTML = '';
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `cell-${r}-${c}`;
                
                // Set static items (assuming minimum 4x4)
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
                
                gridContainer.appendChild(cell);
            }
        }
    }

    // Set player position
    function updatePlayerPos(r, c) {
        // Remove existing player
        document.querySelectorAll('.player').forEach(el => {
            el.classList.remove('player');
            if(el.innerText === 'P') el.innerText = '';
        });
        
        const cell = document.getElementById(`cell-${r}-${c}`);
        if (cell) {
            cell.classList.add('player');
            if(cell.innerText === '') cell.innerText = 'P';
        }
    }

    // Initial draw
    initGrid(parseInt(widthInput.value), parseInt(heightInput.value));
    updatePlayerPos(0, parseInt(widthInput.value) - 1);

    // Event listeners
    widthInput.addEventListener('change', () => {
        initGrid(parseInt(widthInput.value), parseInt(heightInput.value));
        updatePlayerPos(0, parseInt(widthInput.value) - 1);
    });
    
    heightInput.addEventListener('change', () => {
        initGrid(parseInt(widthInput.value), parseInt(heightInput.value));
        updatePlayerPos(0, parseInt(widthInput.value) - 1);
    });

    // Chart init
    function initChart(data) {
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        if (lossChart) {
            lossChart.destroy();
        }

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: data.length}, (_, i) => i + 1),
                datasets: [{
                    label: 'Training Loss (Moving Avg)',
                    data: data,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f8fafc' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    // Playback games
    async function playGames(games) {
        for (let i = 0; i < games.length; i++) {
            const game = games[i];
            statusEl.innerText = `Replaying Game ${i+1}/${games.length}...`;
            statusEl.className = 'status running badge';
            
            // reset to initial static mode position
            updatePlayerPos(0, parseInt(widthInput.value) - 1);
            await new Promise(r => setTimeout(r, 400));
            
            for (let step of game) {
                updatePlayerPos(step.pos[0], step.pos[1]);
                await new Promise(r => setTimeout(r, 200)); // Delay between moves
            }
            await new Promise(r => setTimeout(r, 1000)); // Delay between games
        }
        
        statusEl.innerText = 'Training & Replay Complete!';
        statusEl.className = 'status success badge';
    }

    // Train
    trainBtn.addEventListener('click', async () => {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const epochs = parseInt(epochsInput.value);

        initGrid(w, h);
        updatePlayerPos(0, w - 1);

        trainBtn.disabled = true;
        trainBtn.innerText = 'Training...';
        statusEl.innerText = 'Training in progress...';
        statusEl.className = 'status running badge';

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ width: w, height: h, epochs: epochs })
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                initChart(result.data.losses);
                await playGames(result.data.games);
            } else {
                alert('Training failed: ' + result.message);
                statusEl.innerText = 'Training Failed';
                statusEl.className = 'status badge';
            }
        } catch (e) {
            console.error(e);
            alert('An error occurred during training.');
            statusEl.innerText = 'Error';
            statusEl.className = 'status badge';
        } finally {
            trainBtn.disabled = false;
            trainBtn.innerText = 'Start Training';
        }
    });
});
