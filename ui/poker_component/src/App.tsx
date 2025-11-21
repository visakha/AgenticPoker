import { useEffect, useState, useCallback } from "react"
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"
import type { ComponentProps } from "streamlit-component-lib"
import Table from "./components/Table"
import InfoPanel from "./components/InfoPanel"

// Mock Data for Dev (if not connected to Streamlit)
const MOCK_STATE = {
  players: [
    { name: "The Rock", chips: 1000, cards: [{ rank: "A", suit: "♥" }, { rank: "K", suit: "♥" }], status: "ACTIVE", personality: "tight_passive", vpip: 0.1, pfr: 0.05, aggression: 0.5 },
    { name: "Maniac", chips: 500, cards: [{ rank: "2", suit: "♦" }, { rank: "7", suit: "♣" }], status: "FOLDED", personality: "loose_aggressive", vpip: 0.6, pfr: 0.4, aggression: 3.0 },
    { name: "Prof", chips: 1200, cards: [{ rank: "Q", suit: "♠" }, { rank: "Q", suit: "♣" }], status: "ACTIVE", personality: "analytical", vpip: 0.25, pfr: 0.2, aggression: 1.5 },
    { name: "Fish", chips: 800, cards: [{ rank: "9", suit: "♥" }, { rank: "8", suit: "♥" }], status: "ACTIVE", personality: "loose_passive", vpip: 0.45, pfr: 0.05, aggression: 0.2 },
    { name: "Bot5", chips: 1000, cards: [], status: "ACTIVE", personality: "neutral", vpip: 0.2, pfr: 0.1, aggression: 1.0 },
    { name: "Bot6", chips: 1000, cards: [], status: "ACTIVE", personality: "neutral", vpip: 0.2, pfr: 0.1, aggression: 1.0 },
    { name: "Bot7", chips: 1000, cards: [], status: "ACTIVE", personality: "neutral", vpip: 0.2, pfr: 0.1, aggression: 1.0 },
  ],
  communityCards: [{ rank: "Q", suit: "♠" }, { rank: "J", suit: "♠" }, { rank: "T", suit: "♦" }],
  pot: 150,
  logs: [
    { sender: "Dealer", message: "Dealing Flop", type: "game" },
    { sender: "Maniac", message: "I'm all in!", type: "chat" },
  ]
}

const App = (props: ComponentProps) => {
  // Use props.args.game_state if available, otherwise mock
  const gameState = props.args?.game_state || MOCK_STATE

  const [focusedPlayerIndex, setFocusedPlayerIndex] = useState(0)
  const [isPaused, setIsPaused] = useState(false)

  // Keyboard Navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "ArrowRight") {
      setFocusedPlayerIndex(prev => (prev + 1) % gameState.players.length)
    } else if (e.key === "ArrowLeft") {
      setFocusedPlayerIndex(prev => (prev - 1 + gameState.players.length) % gameState.players.length)
    } else if (e.code === "Space") {
      setIsPaused(prev => !prev)
      Streamlit.setComponentValue({ action: "TOGGLE_PAUSE" })
    }
  }, [gameState.players.length])

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])

  // Adjust height
  useEffect(() => {
    Streamlit.setFrameHeight(800)
  }, [])

  const focusedPlayer = gameState.players[focusedPlayerIndex]

  return (
    <div className="w-full h-screen bg-[#020617] text-white overflow-hidden relative select-none">
      {/* Header / Control Bar */}
      <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-50 pointer-events-none">
        <div className="text-xs text-gray-500 font-mono">
          Use ARROWS to navigate players • SPACE to Pause
        </div>
        {isPaused && (
          <div className="bg-red-500/20 text-red-400 px-4 py-1 rounded-full border border-red-500/50 animate-pulse font-bold">
            PAUSED
          </div>
        )}
      </div>

      {/* Main Table */}
      <div className="flex items-center justify-center h-full pb-32"> {/* Padding for InfoPanel */}
        <Table
          players={gameState.players.map((p: any, i: number) => ({
            ...p,
            isDealer: i === 0, // Mock dealer pos
            isCurrentTurn: i === 2, // Mock turn
            isActive: i === focusedPlayerIndex
          }))}
          communityCards={gameState.communityCards}
          pot={gameState.pot}
          focusedPlayerIndex={focusedPlayerIndex}
        />
      </div>

      {/* Info Panel */}
      <InfoPanel player={focusedPlayer} logs={gameState.logs} />
    </div>
  )
}

export default withStreamlitConnection(App)
