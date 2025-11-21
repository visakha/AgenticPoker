import React from 'react';
import { Activity, MessageSquare, TrendingUp } from 'lucide-react';

interface InfoPanelProps {
    player: {
        name: string;
        personality: string;
        vpip: number;
        pfr: number;
        aggression: number;
        handStrength?: number;
    } | null;
    logs: { sender: string; message: string; type: 'game' | 'chat' }[];
}

const InfoPanel: React.FC<InfoPanelProps> = ({ player, logs }) => {
    return (
        <div className="absolute bottom-0 left-0 right-0 h-1/4 bg-black/80 backdrop-blur-md border-t border-cyan-900/50 flex overflow-hidden">
            {/* Left: Player Stats (if hovered/focused) */}
            <div className="w-1/3 p-4 border-r border-gray-800">
                <h3 className="text-cyan-400 font-bold mb-2 flex items-center gap-2">
                    <Activity size={18} />
                    AGENT ANALYSIS
                </h3>
                {player ? (
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-gray-400">Name:</span>
                            <span className="text-white font-bold">{player.name}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-gray-400">Personality:</span>
                            <span className="text-purple-400 text-sm">{player.personality}</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 mt-2">
                            <div className="bg-gray-900 p-2 rounded text-center">
                                <div className="text-xs text-gray-500">VPIP</div>
                                <div className="text-cyan-300 font-mono">{(player.vpip * 100).toFixed(0)}%</div>
                            </div>
                            <div className="bg-gray-900 p-2 rounded text-center">
                                <div className="text-xs text-gray-500">PFR</div>
                                <div className="text-cyan-300 font-mono">{(player.pfr * 100).toFixed(0)}%</div>
                            </div>
                            <div className="bg-gray-900 p-2 rounded text-center">
                                <div className="text-xs text-gray-500">AGG</div>
                                <div className="text-cyan-300 font-mono">{player.aggression.toFixed(1)}</div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="text-gray-500 italic flex h-full items-center justify-center">
                        Hover over a player to analyze
                    </div>
                )}
            </div>

            {/* Middle: Win Probability / Hand Strength (Placeholder Graph) */}
            <div className="w-1/3 p-4 border-r border-gray-800 flex flex-col">
                <h3 className="text-green-400 font-bold mb-2 flex items-center gap-2">
                    <TrendingUp size={18} />
                    WIN PROBABILITY
                </h3>
                <div className="flex-1 bg-gray-900/50 rounded relative overflow-hidden flex items-end gap-1 p-2">
                    {/* Mock Bars */}
                    {[20, 35, 45, 30, 60, 75, 50, 40, 65, 80].map((h, i) => (
                        <div key={i} className="flex-1 bg-green-500/30 hover:bg-green-400 transition-all" style={{ height: `${h}%` }}></div>
                    ))}
                </div>
            </div>

            {/* Right: Logs / Dialogue */}
            <div className="w-1/3 p-4 flex flex-col">
                <h3 className="text-yellow-400 font-bold mb-2 flex items-center gap-2">
                    <MessageSquare size={18} />
                    COMMS LOG
                </h3>
                <div className="flex-1 overflow-y-auto space-y-1 pr-2 font-mono text-xs">
                    {logs.map((log, i) => (
                        <div key={i} className={`${log.type === 'chat' ? 'text-white' : 'text-gray-500'}`}>
                            <span className="opacity-50">[{new Date().toLocaleTimeString()}]</span>{' '}
                            <span className={log.type === 'chat' ? 'text-cyan-300' : 'text-yellow-600'}>{log.sender}:</span>{' '}
                            {log.message}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default InfoPanel;
