import React from 'react';

export interface PlayerProps {
  name: string;
  chips: number;
  cards: { rank: string; suit: string }[];
  status: string;
  personality: string;
  vpip?: number;
  pfr?: number;
  aggression?: number;
  isActive?: boolean;
}

const Player: React.FC<PlayerProps> = ({
  name,
  chips,
  cards,
  status,
  personality,
  vpip = 0,
  pfr = 0,
  aggression = 0,
  isActive = false,
}) => {
  const statusColors: Record<string, string> = {
    ACTIVE: 'bg-green-600',
    FOLDED: 'bg-gray-600',
    ALL_IN: 'bg-red-600',
    SITTING_OUT: 'bg-gray-400',
  };

  return (
    <div
      className={`
                relative p-4 rounded-xl backdrop-blur-sm border-2 
                transition-all duration-300 ease-out
                ${isActive ? 'bg-blue-900/80 border-blue-400 shadow-lg shadow-blue-500/50 scale-105' : 'bg-gray-900/60 border-gray-700'}
                hover:scale-110 hover:shadow-xl
                animate-fade-in
            `}
      style={{ minWidth: '180px' }}
    >
      {/* Active indicator pulse */}
      {isActive && (
        <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-400 rounded-full animate-ping"></div>
      )}

      {/* Player Name & Status */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-white font-bold text-sm truncate">{name}</span>
        <span
          className={`${statusColors[status] || 'bg-gray-500'} text-white text-xs px-2 py-1 rounded-full font-semibold`}
        >
          {status}
        </span>
      </div>

      {/* Chips */}
      <div className="text-yellow-400 font-mono text-lg mb-2 flex items-center gap-1">
        <span className="text-yellow-500">💰</span>
        ${chips.toLocaleString()}
      </div>

      {/* Cards */}
      <div className="flex gap-1 mb-2">
        {cards.length > 0 ? (
          cards.map((card, i) => (
            <div
              key={i}
              className="w-10 h-14 bg-white rounded shadow-md flex items-center justify-center text-sm font-bold border border-gray-300 transform transition-transform hover:scale-110 hover:-translate-y-1"
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <span className={['♥', '♦'].includes(card.suit) ? 'text-red-600' : 'text-black'}>
                {card.rank}
                {card.suit}
              </span>
            </div>
          ))
        ) : (
          <div className="w-10 h-14 bg-gray-700 rounded border border-gray-600 border-dashed"></div>
        )}
      </div>

      {/* Personality & Stats (on hover) */}
      <div className="text-xs text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="font-semibold text-purple-400 mb-1">{personality}</div>
        <div className="grid grid-cols-3 gap-1 text-[10px]">
          <div>
            <span className="text-gray-500">VPIP:</span> {(vpip * 100).toFixed(0)}%
          </div>
          <div>
            <span className="text-gray-500">PFR:</span> {(pfr * 100).toFixed(0)}%
          </div>
          <div>
            <span className="text-gray-500">AGG:</span> {aggression.toFixed(1)}
          </div>
        </div>
      </div>

      <style>{`
                @keyframes fade-in {
                    from {
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                .animate-fade-in {
                    animation: fade-in 0.4s ease-out;
                }
            `}</style>
    </div>
  );
};

export default Player;
