// Agentic Poker - Client-side JavaScript

// WebSocket connection
let ws = null;
let autoPlayEnabled = false;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'game_update') {
            updateGameState(message.data);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 1000);
    };
}

function updateGameState(gameData) {
    // Update pot
    const potEl = document.getElementById('pot');
    if (potEl) potEl.textContent = `$${gameData.pot}`;

    // Update current bet
    const betEl = document.getElementById('current-bet');
    if (betEl) betEl.textContent = `$${gameData.current_bet}`;

    // Update phase
    const phaseEl = document.getElementById('phase');
    if (phaseEl) phaseEl.textContent = gameData.phase;

    // Update hand count
    const handCountEl = document.getElementById('hand-count');
    if (handCountEl) handCountEl.textContent = gameData.hand_count;

    // Update community cards
    updateCommunityCards(gameData.community_cards, gameData.phase);

    // Update players
    updatePlayers(gameData.players);
}

function updateCommunityCards(cards, phase) {
    const cardsDisplay = document.querySelector('.cards-display');
    if (!cardsDisplay) return;

    if (cards && cards.length > 0) {
        cardsDisplay.innerHTML = cards.map(card =>
            `<span class="card">${card}</span>`
        ).join('');
    } else {
        cardsDisplay.innerHTML = `<span class="no-cards">⏳ ${phase}</span>`;
    }
}

function updatePlayers(players) {
    players.forEach((player, index) => {
        updatePlayerCard(player, index);
    });
}

function updatePlayerCard(player, index) {
    // Find player card elements
    const playerCards = document.querySelectorAll('.player-card');
    if (index >= playerCards.length) return;

    const playerCard = playerCards[index];

    // Update classes
    playerCard.className = 'player-card';
    if (player.status === 'FOLDED') {
        playerCard.classList.add('folded');
    }
    if (player.is_current) {
        playerCard.classList.add('current');
    }

    // Update status emoji
    const statusEl = playerCard.querySelector('.player-status');
    if (statusEl) {
        if (player.is_current) {
            statusEl.textContent = '👉';
        } else if (player.status === 'ACTIVE') {
            statusEl.textContent = '🟢';
        } else {
            statusEl.textContent = '🔴';
        }
    }

    // Update current marker
    const currentMarker = playerCard.querySelector('.current-marker');
    if (currentMarker) {
        currentMarker.style.display = player.is_current ? 'inline' : 'none';
    }

    // Update cards
    const cardsEl = playerCard.querySelector('.player-cards');
    if (cardsEl && player.cards && player.cards.length > 0) {
        cardsEl.innerHTML = player.cards.map(card =>
            `<span class="card">${card}</span>`
        ).join('');
    }

    // Update chips
    const statsEl = playerCard.querySelector('.player-stats');
    if (statsEl) {
        let statsHTML = `<div class="stat">💰 $${player.chips}</div>`;
        if (player.current_bet > 0) {
            statsHTML += `<div class="stat bet">🎲 Bet: $${player.current_bet}</div>`;
        }
        statsEl.innerHTML = statsHTML;
    }

    // Update recent actions
    const actionsEl = playerCard.querySelector('.player-actions');
    if (actionsEl) {
        let actionsHTML = '<div class="actions-header">Recent Actions:</div>';

        if (player.recent_actions && player.recent_actions.length > 0) {
            player.recent_actions.forEach(action => {
                actionsHTML += `
                    <div class="action-item">
                        <code>{${action.phase}} → ${action.message}</code>
                    </div>
                `;
            });
        } else {
            actionsHTML += '<div class="no-actions">No actions yet</div>';
        }

        actionsEl.innerHTML = actionsHTML;
    }
}

async function newHand() {
    try {
        const response = await fetch('/api/game/new', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();
        updateGameState(data);
        console.log('New hand started');
    } catch (error) {
        console.error('Error starting new hand:', error);
    }
}

async function stepGame() {
    try {
        const response = await fetch('/api/game/step', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();
        updateGameState(data);
        console.log('Game stepped');
    } catch (error) {
        console.error('Error stepping game:', error);
    }
}

async function toggleAutoPlay(enabled) {
    autoPlayEnabled = enabled;

    try {
        const response = await fetch('/api/game/auto-play', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enabled })
        });

        const data = await response.json();
        console.log('Auto-play:', data.auto_play ? 'enabled' : 'disabled');
    } catch (error) {
        console.error('Error toggling auto-play:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();

    // Keep WebSocket alive with ping
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
        }
    }, 30000); // Ping every 30 seconds
});
