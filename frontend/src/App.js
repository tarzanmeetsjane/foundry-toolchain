import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConnectionProvider, WalletProvider } from '@solana/wallet-adapter-react';
import { WalletAdapterNetwork } from '@solana/wallet-adapter-base';
import { PhantomWalletAdapter } from '@solana/wallet-adapter-phantom';
import { WalletModalProvider, WalletMultiButton } from '@solana/wallet-adapter-react-ui';
import { clusterApiUrl } from '@solana/web3.js';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, Brain, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import { ToastContainer, toast } from 'react-toastify';
import axios from 'axios';
import './App.css';
import 'react-toastify/dist/ReactToastify.css';
require('@solana/wallet-adapter-react-ui/styles.css');

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Main Dashboard Component
const Dashboard = () => {
  const [pools, setPools] = useState([]);
  const [signals, setSignals] = useState([]);
  const [arbitrageOps, setArbitrageOps] = useState([]);
  const [stats, setStats] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [poolsRes, statsRes, arbitrageRes] = await Promise.all([
        axios.get(`${API}/pools/data`),
        axios.get(`${API}/dashboard/stats`),
        axios.get(`${API}/mev/opportunities`)
      ]);

      setPools(poolsRes.data);
      setStats(statsRes.data);
      setArbitrageOps(arbitrageRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Dashboard data fetch error:', error);
      toast.error('Failed to fetch dashboard data');
      setLoading(false);
    }
  };

  const getAISignal = async (poolAddress) => {
    try {
      const response = await axios.get(`${API}/ai/analysis/${poolAddress}`);
      const signal = response.data;
      setSignals(prev => [...prev.filter(s => s.pool_address !== poolAddress), signal]);
      
      if (signal.confidence > 0.7) {
        toast.success(`Strong ${signal.signal_type} signal detected! Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
      }
    } catch (error) {
      console.error('AI signal error:', error);
      toast.error('Failed to get AI analysis');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading DeFi Trading Bot...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Solana DeFi Trading Bot
          </h1>
          <p className="text-gray-400 mt-2">Zero-Configuration Automated Trading</p>
        </div>
        <div className="flex items-center space-x-4">
          <WalletMultiButton className="!bg-purple-600 hover:!bg-purple-700" />
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard 
            icon={<Activity className="h-8 w-8" />}
            title="Pools Monitored"
            value={stats.total_pools_monitored}
            color="blue"
          />
          <StatCard 
            icon={<DollarSign className="h-8 w-8" />}
            title="24h Volume"
            value={`$${(stats.total_volume_24h / 1000000).toFixed(1)}M`}
            color="green"
          />
          <StatCard 
            icon={<Brain className="h-8 w-8" />}
            title="AI Signals"
            value={stats.active_ai_signals}
            color="purple"
          />
          <StatCard 
            icon={<Zap className="h-8 w-8" />}
            title="Trades Executed"
            value={stats.total_trades_executed}
            color="yellow"
          />
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Liquidity Pools */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <Activity className="h-6 w-6 mr-2" />
              Liquidity Pools
            </h2>
            <div className="space-y-4">
              {pools.map((pool) => (
                <PoolCard 
                  key={pool.pool_address} 
                  pool={pool} 
                  onGetSignal={() => getAISignal(pool.pool_address)}
                  signal={signals.find(s => s.pool_address === pool.pool_address)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* AI Signals & MEV */}
        <div className="space-y-6">
          
          {/* AI Signals Panel */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <Brain className="h-5 w-5 mr-2" />
              AI Trading Signals
            </h3>
            <div className="space-y-3">
              {signals.length === 0 ? (
                <p className="text-gray-400">Click "Get AI Signal" on pools above</p>
              ) : (
                signals.map((signal) => (
                  <SignalCard key={signal.id} signal={signal} />
                ))
              )}
            </div>
          </div>

          {/* MEV Opportunities */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <Zap className="h-5 w-5 mr-2" />
              MEV Opportunities
            </h3>
            <div className="space-y-3">
              {arbitrageOps.slice(0, 3).map((op) => (
                <div key={op.id} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{op.token_pair}</span>
                    <span className="text-green-400">+{(op.profit_potential).toFixed(2)}%</span>
                  </div>
                  <div className="text-sm text-gray-400 mt-1">
                    {op.dex_a} ↔ {op.dex_b}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Component: Stats Card
const StatCard = ({ icon, title, value, color }) => {
  const colorClasses = {
    blue: 'text-blue-400 bg-blue-900/20',
    green: 'text-green-400 bg-green-900/20',
    purple: 'text-purple-400 bg-purple-900/20',
    yellow: 'text-yellow-400 bg-yellow-900/20'
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className={`inline-flex p-3 rounded-lg ${colorClasses[color]} mb-4`}>
        {icon}
      </div>
      <h3 className="text-2xl font-bold">{value}</h3>
      <p className="text-gray-400">{title}</p>
    </div>
  );
};

// Component: Pool Card
const PoolCard = ({ pool, onGetSignal, signal }) => {
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="font-bold text-lg">{pool.token_a}/{pool.token_b}</h3>
          <p className="text-sm text-gray-400">{pool.pool_address.slice(0, 8)}...</p>
        </div>
        <button 
          onClick={onGetSignal}
          className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
        >
          Get AI Signal
        </button>
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-400">Liquidity:</span>
          <div className="font-medium">${(pool.liquidity / 1000000).toFixed(1)}M</div>
        </div>
        <div>
          <span className="text-gray-400">24h Volume:</span>
          <div className="font-medium">${(pool.volume_24h / 1000000).toFixed(1)}M</div>
        </div>
        <div>
          <span className="text-gray-400">Correlation:</span>
          <div className="font-medium">{(pool.correlation_ratio * 100).toFixed(1)}%</div>
        </div>
        <div>
          <span className="text-gray-400">Participation:</span>
          <div className="font-medium">{(pool.participation_score * 100).toFixed(1)}%</div>
        </div>
      </div>

      {signal && (
        <div className="mt-4 p-3 bg-gray-600 rounded-lg">
          <div className="flex items-center justify-between">
            <span className={`font-bold ${
              signal.signal_type === 'BUY' ? 'text-green-400' : 
              signal.signal_type === 'SELL' ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {signal.signal_type}
            </span>
            <span className="text-sm">
              Confidence: {(signal.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">{signal.reasoning}</p>
        </div>
      )}
    </div>
  );
};

// Component: Signal Card
const SignalCard = ({ signal }) => {
  const getSignalIcon = () => {
    if (signal.signal_type === 'BUY') return <TrendingUp className="h-4 w-4 text-green-400" />;
    if (signal.signal_type === 'SELL') return <TrendingDown className="h-4 w-4 text-red-400" />;
    return <Activity className="h-4 w-4 text-yellow-400" />;
  };

  const getSignalColor = () => {
    if (signal.signal_type === 'BUY') return 'border-green-400 bg-green-900/20';
    if (signal.signal_type === 'SELL') return 'border-red-400 bg-red-900/20';
    return 'border-yellow-400 bg-yellow-900/20';
  };

  return (
    <div className={`border-l-4 p-3 rounded ${getSignalColor()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {getSignalIcon()}
          <span className="font-medium">{signal.signal_type}</span>
        </div>
        <span className="text-sm">{(signal.confidence * 100).toFixed(1)}%</span>
      </div>
      <p className="text-xs text-gray-400 mt-1">{signal.reasoning}</p>
      <div className="text-xs text-gray-500 mt-2">
        Expected: ${signal.expected_profit.toFixed(4)}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const network = WalletAdapterNetwork.Mainnet;
  const endpoint = useMemo(() => clusterApiUrl(network), [network]);
  
  const wallets = useMemo(
    () => [new PhantomWalletAdapter()],
    []
  );

  return (
    <ConnectionProvider endpoint={endpoint}>
      <WalletProvider wallets={wallets} autoConnect>
        <WalletModalProvider>
          <div className="App">
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Dashboard />} />
              </Routes>
            </BrowserRouter>
            <ToastContainer
              position="top-right"
              autoClose={5000}
              hideProgressBar={false}
              newestOnTop={false}
              closeOnClick
              rtl={false}
              pauseOnFocusLoss
              draggable
              pauseOnHover
              theme="dark"
            />
          </div>
        </WalletModalProvider>
      </WalletProvider>
    </ConnectionProvider>
  );
}

export default App;
