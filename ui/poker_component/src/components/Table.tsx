import React from 'react';
import Player from './Player';
import type { PlayerProps } from './Player';

interface TableProps {
    players: PlayerProps[];
    communityCards: { rank: string; suit: string }[];
    pot: number;
    focusedPlayerIndex: number;
}

const Table: React.FC<TableProps> = ({ players, communityCards, pot, focusedPlayerIndex }) => {
    // Arrange players in a circle/oval
    // For 7 players, we can position them absolutely or using a grid layout
    // A simple approach for now is a top/bottom/sides layout

    return (
        <div className="relative w-full h-[600px] bg-[#0f172a] rounded-[3rem] border-8 border-[#1e293b] shadow-2xl flex items-center justify-center overflow-hidden">
            {/* Felt */}
            <div className="absolute inset-4 bg-gradient-to-b from-green-900 to-green-950 rounded-[2.5rem] shadow-inner border border-green-800/30"></div>

            {/* Logo */}
            </div>
            <div className="absolute right-8 top-1/2 transform -translate-y-1/2 z-20">
                {players[3] && <Player {...players[3]} isActive={3 === focusedPlayerIndex} />}
            </div>

            {/* Bottom Row */ }
    <div className="absolute bottom-8 flex gap-12 z-20">
        {players.slice(4).map((p, i) => (
            <Player key={i + 4} {...p} isActive={(i + 4) === focusedPlayerIndex} />
        ))}
    </div>
        </div >
    );
};

export default Table;
