import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Server, 
  Shield, 
  Database, 
  Stethoscope, 
  Play, 
  Square, 
  RefreshCw, 
  Terminal, 
  FileUp, 
  CheckCircle, 
  AlertCircle,
  Lock,
  Network
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// --- Types & mock Data ---

type LogMessage = {
  id: number;
  source: 'Server' | 'Hospital 1' | 'Hospital 2' | 'System';
  message: string;
  timestamp: string;
  type: 'info' | 'success' | 'warning' | 'error';
};

type TrainingMetric = {
  round: number;
  accuracy: number;
  loss: number;
};

// --- Main Application Component ---

export default function FedMedApp() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'hospitals' | 'doctor'>('dashboard');
  const [isTraining, setIsTraining] = useState(false);
  const [round, setRound] = useState(0);
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [hospital1Status, setHospital1Status] = useState<'idle' | 'training' | 'uploading'>('idle');
  const [hospital2Status, setHospital2Status] = useState<'idle' | 'training' | 'uploading'>('idle');
  const logEndRef = useRef<HTMLDivElement>(null);

  // --- Simulation Logic ---

  const addLog = (source: LogMessage['source'], message: string, type: LogMessage['type'] = 'info') => {
    const newLog: LogMessage = {
      id: Date.now(),
      source,
      message,
      timestamp: new Date().toLocaleTimeString(),
      type
    };
    setLogs(prev => [...prev, newLog]);
  };

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Simulation Loop
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isTraining && round < 10) {
      interval = setInterval(() => {
        const currentRound = round + 1;
        
        // Sequence of events for a round
        setTimeout(() => {
          addLog('Server', `Starting Round ${currentRound}/10`, 'info');
          setHospital1Status('training');
          setHospital2Status('training');
        }, 500);

        setTimeout(() => {
          addLog('Hospital 1', 'Loading real chest X-rays...', 'info');
          addLog('Hospital 2', 'Loading real chest X-rays...', 'info');
        }, 1500);

        setTimeout(() => {
          addLog('Hospital 1', 'Privacy Engine Active! (Noise Multiplier: 0.3)', 'warning');
          addLog('Hospital 2', 'Privacy Engine Active! (Noise Multiplier: 0.3)', 'warning');
        }, 2500);

        setTimeout(() => {
          addLog('Hospital 1', `Epoch 5/5 Completed. Loss: ${(0.8 - currentRound * 0.05 + Math.random() * 0.1).toFixed(4)}`, 'success');
          addLog('Hospital 2', `Epoch 5/5 Completed. Loss: ${(0.9 - currentRound * 0.06 + Math.random() * 0.1).toFixed(4)}`, 'success');
          setHospital1Status('uploading');
          setHospital2Status('uploading');
        }, 4500);

        setTimeout(() => {
          const newAccuracy = 0.5 + (currentRound * 0.045) + (Math.random() * 0.02);
          const newLoss = 1.0 - (currentRound * 0.08);
          
          setMetrics(prev => [...prev, { round: currentRound, accuracy: newAccuracy, loss: newLoss }]);
          addLog('Server', `Aggregated weights from 2 clients. Global Accuracy: ${(newAccuracy * 100).toFixed(2)}%`, 'success');
          
          setRound(currentRound);
          setHospital1Status('idle');
          setHospital2Status('idle');

          if (currentRound === 10) {
            setIsTraining(false);
            addLog('Server', 'Federated Training Completed. Model saved as pneumonia_model.pth', 'success');
          }
        }, 6000);

      }, 7000); // 7 seconds per round
    }

    return () => clearInterval(interval);
  }, [isTraining, round]);

  const handleStart = () => {
    setIsTraining(true);
    if (round === 0) {
      addLog('System', 'Initializing FedMed Network...', 'info');
      addLog('Server', '🚀 Server is starting... Waiting for 2 Hospitals to connect.', 'warning');
    }
  };

  const handleStop = () => setIsTraining(false);
  
  const handleReset = () => {
    setIsTraining(false);
    setRound(0);
    setLogs([]);
    setMetrics([]);
    setHospital1Status('idle');
    setHospital2Status('idle');
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col shadow-xl z-20">
        <div className="p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="bg-blue-500 p-2 rounded-lg">
              <Network className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">FedMed</h1>
              <p className="text-xs text-slate-400">Secure FL Platform</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          <SidebarButton 
            active={activeTab === 'dashboard'} 
            onClick={() => setActiveTab('dashboard')} 
            icon={<Activity size={20} />} 
            label="Live Dashboard" 
          />
          <SidebarButton 
            active={activeTab === 'hospitals'} 
            onClick={() => setActiveTab('hospitals')} 
            icon={<Database size={20} />} 
            label="Connected Hospitals" 
          />
          <SidebarButton 
            active={activeTab === 'doctor'} 
            onClick={() => setActiveTab('doctor')} 
            icon={<Stethoscope size={20} />} 
            label="Doctor's Portal" 
          />
        </nav>

        <div className="p-4 bg-slate-800 m-4 rounded-xl border border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <div className={`w-3 h-3 rounded-full ${isTraining ? 'bg-green-400 animate-pulse' : 'bg-slate-500'}`}></div>
            <span className="text-sm font-medium text-slate-300">System Status</span>
          </div>
          <div className="text-xs text-slate-400">
            {isTraining ? `Training Round ${round + 1}/10` : 'Ready to start'}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        {/* Top Header */}
        <header className="bg-white h-16 border-b border-slate-200 flex items-center justify-between px-8 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-800">
            {activeTab === 'dashboard' && 'Network Overview'}
            {activeTab === 'hospitals' && 'Hospital Nodes Status'}
            {activeTab === 'doctor' && 'AI Diagnosis Assistant'}
          </h2>
          <div className="flex items-center gap-4">
            <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium border border-blue-200">
              Model: ResNet18 (Privacy Enabled)
            </span>
            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium border border-emerald-200">
              Clients: 2/2 Active
            </span>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-8">
          {activeTab === 'dashboard' && (
            <DashboardView 
              metrics={metrics} 
              logs={logs} 
              logRef={logEndRef} 
              onStart={handleStart} 
              onStop={handleStop} 
              onReset={handleReset}
              isTraining={isTraining}
              round={round}
              h1Status={hospital1Status}
              h2Status={hospital2Status}
            />
          )}
          {activeTab === 'hospitals' && <HospitalsView h1Status={hospital1Status} h2Status={hospital2Status} round={round} />}
          {activeTab === 'doctor' && <DoctorPortal round={round} />}
        </div>
      </main>
    </div>
  );
}

// --- Sub-Components ---

function SidebarButton({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
        active 
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
          : 'text-slate-400 hover:bg-slate-800 hover:text-white'
      }`}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}

function DashboardView({ metrics, logs, logRef, onStart, onStop, onReset, isTraining, round, h1Status, h2Status }: any) {
  return (
    <div className="grid grid-cols-12 gap-6 h-full">
      {/* Main Control Panel */}
      <div className="col-span-8 space-y-6">
        {/* Network Topology Visualization */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-600"></div>
          <div className="flex justify-between items-center mb-6">
             <h3 className="font-semibold text-lg flex items-center gap-2">
               <Server className="text-indigo-600" size={20}/> Federated Network Topology
             </h3>
             <div className="flex gap-2">
               {!isTraining ? (
                 <button onClick={onStart} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                   <Play size={16} /> Start Training
                 </button>
               ) : (
                 <button onClick={onStop} className="flex items-center gap-2 bg-amber-500 hover:bg-amber-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                   <Square size={16} /> Pause
                 </button>
               )}
               <button onClick={onReset} className="flex items-center gap-2 bg-slate-100 hover:bg-slate-200 text-slate-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                 <RefreshCw size={16} /> Reset
               </button>
             </div>
          </div>

          <div className="flex justify-center items-center py-8 relative">
            {/* Connecting Lines */}
            <svg className="absolute w-full h-full pointer-events-none" style={{zIndex: 0}}>
              <line x1="50%" y1="50%" x2="25%" y2="80%" className={`stroke-2 ${h1Status !== 'idle' ? 'stroke-blue-400 animate-pulse' : 'stroke-slate-200'}`} strokeDasharray={h1Status !== 'idle' ? "5,5" : "0"} />
              <line x1="50%" y1="50%" x2="75%" y2="80%" className={`stroke-2 ${h2Status !== 'idle' ? 'stroke-blue-400 animate-pulse' : 'stroke-slate-200'}`} strokeDasharray={h2Status !== 'idle' ? "5,5" : "0"} />
            </svg>

            {/* Central Server */}
            <div className="relative z-10 flex flex-col items-center">
              <div className="w-20 h-20 bg-indigo-50 border-4 border-indigo-100 rounded-full flex items-center justify-center shadow-lg mb-2">
                <Server className="text-indigo-600 w-8 h-8" />
              </div>
              <span className="font-semibold text-slate-700">Central Server</span>
              <span className="text-xs text-slate-500">Aggregator</span>
            </div>

            {/* Hospital Nodes - Absolute positioned relative to center */}
          </div>
          
          <div className="flex justify-around mt-4 relative z-10">
            {/* Hospital 1 */}
            <div className="flex flex-col items-center">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-md border-2 transition-all duration-300 ${h1Status === 'training' ? 'bg-blue-50 border-blue-400 scale-110' : 'bg-white border-slate-100'}`}>
                {h1Status === 'training' ? <Activity className="text-blue-500 animate-bounce" /> : <Database className="text-slate-400" />}
              </div>
              <div className="mt-2 text-center">
                <p className="font-medium text-slate-700">Hospital 1</p>
                <div className="flex items-center gap-1 text-xs text-slate-500">
                  <Shield size={10} className="text-emerald-500" />
                  <span>Secure Enclave</span>
                </div>
              </div>
            </div>

            {/* Hospital 2 */}
            <div className="flex flex-col items-center">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-md border-2 transition-all duration-300 ${h2Status === 'training' ? 'bg-blue-50 border-blue-400 scale-110' : 'bg-white border-slate-100'}`}>
                {h2Status === 'training' ? <Activity className="text-blue-500 animate-bounce" /> : <Database className="text-slate-400" />}
              </div>
              <div className="mt-2 text-center">
                <p className="font-medium text-slate-700">Hospital 2</p>
                <div className="flex items-center gap-1 text-xs text-slate-500">
                   <Shield size={10} className="text-emerald-500" />
                   <span>Secure Enclave</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h4 className="text-sm font-semibold text-slate-500 mb-4 uppercase tracking-wider">Global Model Accuracy</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metrics}>
                  <defs>
                    <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="round" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Area type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorAcc)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h4 className="text-sm font-semibold text-slate-500 mb-4 uppercase tracking-wider">Global Loss</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="round" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Terminal / Logs Sidebar */}
      <div className="col-span-4 bg-slate-900 rounded-2xl overflow-hidden flex flex-col shadow-lg border border-slate-800">
        <div className="bg-slate-800 p-3 border-b border-slate-700 flex items-center gap-2">
          <Terminal size={16} className="text-slate-400" />
          <span className="text-xs font-mono text-slate-300">server_logs.txt</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-3 custom-scrollbar">
          {logs.length === 0 && (
            <div className="text-slate-600 italic text-center mt-10">Waiting for training to start...</div>
          )}
          {logs.map((log: LogMessage) => (
            <div key={log.id} className="flex gap-2">
              <span className="text-slate-500 shrink-0">[{log.timestamp}]</span>
              <div>
                <span className={`font-bold mr-2 ${
                  log.source === 'Server' ? 'text-indigo-400' : 
                  log.source.includes('Hospital') ? 'text-blue-400' : 'text-slate-300'
                }`}>
                  {log.source}:
                </span>
                <span className={`${
                  log.type === 'error' ? 'text-red-400' :
                  log.type === 'success' ? 'text-emerald-400' :
                  log.type === 'warning' ? 'text-amber-400' :
                  'text-slate-300'
                }`}>
                  {log.message}
                </span>
              </div>
            </div>
          ))}
          <div ref={logRef} />
        </div>
      </div>
    </div>
  );
}

function HospitalsView({ h1Status, h2Status, round }: any) {
  return (
    <div className="grid grid-cols-2 gap-8">
      {[1, 2].map((id) => {
        const status = id === 1 ? h1Status : h2Status;
        return (
          <div key={id} className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="h-2 bg-blue-500"></div>
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center">
                    <Database className="text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-800">Hospital {id}</h3>
                    <p className="text-sm text-slate-500">Connected Client</p>
                  </div>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1 ${
                  status === 'training' ? 'bg-amber-100 text-amber-700' : 
                  status === 'uploading' ? 'bg-blue-100 text-blue-700' : 
                  'bg-slate-100 text-slate-600'
                }`}>
                  {status === 'training' && <Activity size={12} className="animate-spin" />}
                  {status === 'idle' ? 'IDLE' : status.toUpperCase()}
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-slate-600">Privacy Engine (Opacus)</span>
                    <Lock size={14} className="text-emerald-500" />
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>Noise Multiplier</span>
                    <span className="font-mono bg-white px-2 py-1 rounded border">0.3</span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 mt-2">
                    <span>Max Grad Norm</span>
                    <span className="font-mono bg-white px-2 py-1 rounded border">2.0</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white p-3 rounded-xl border border-slate-100 shadow-sm">
                    <p className="text-xs text-slate-400 mb-1">Local Samples</p>
                    <p className="text-lg font-semibold text-slate-800">1,240</p>
                  </div>
                  <div className="bg-white p-3 rounded-xl border border-slate-100 shadow-sm">
                    <p className="text-xs text-slate-400 mb-1">Status</p>
                    <p className="text-lg font-semibold text-slate-800">
                      {status === 'training' ? `Epoch ${Math.min(5, Math.ceil(Math.random() * 5))}/5` : 'Ready'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DoctorPortal({ round }: { round: number }) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<'NORMAL' | 'PNEUMONIA' | null>(null);
  const [confidence, setConfidence] = useState(0);

  const handleImageUpload = () => {
    // Simulate upload
    setSelectedImage('https://prod-images-static.radiopaedia.org/images/157210/332ea0911765c71a3407e770c80c2f_jumbo.jpg');
    setResult(null);
  };

  const runDiagnosis = () => {
    setAnalyzing(true);
    setTimeout(() => {
      setAnalyzing(false);
      // Determine result based on "Global Model" quality (simulated by round number)
      const isAccurate = round > 2; 
      setResult(isAccurate ? 'PNEUMONIA' : (Math.random() > 0.5 ? 'NORMAL' : 'PNEUMONIA'));
      setConfidence(0.75 + (round * 0.02));
    }, 2000);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
        <div className="p-8 border-b border-slate-100 text-center">
          <h2 className="text-2xl font-bold text-slate-800">AI Diagnostic Assistant</h2>
          <p className="text-slate-500 mt-2">Upload a patient chest X-ray to use the Federated Global Model.</p>
        </div>

        <div className="p-8 grid grid-cols-2 gap-8">
          {/* Upload Area */}
          <div className="space-y-4">
            <div 
              onClick={handleImageUpload}
              className={`h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-colors ${
                selectedImage ? 'border-blue-300 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
              }`}
            >
              {selectedImage ? (
                <img src={selectedImage} alt="X-ray" className="h-full object-contain p-2" />
              ) : (
                <>
                  <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mb-3">
                    <FileUp className="text-slate-400" />
                  </div>
                  <p className="font-medium text-slate-600">Click to load Patient X-ray</p>
                  <p className="text-xs text-slate-400 mt-1">Simulated Upload</p>
                </>
              )}
            </div>
            
            <button 
              onClick={runDiagnosis}
              disabled={!selectedImage || analyzing}
              className={`w-full py-3 rounded-lg font-semibold transition-all ${
                !selectedImage 
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'
              }`}
            >
              {analyzing ? 'Analyzing with Global Model...' : 'Run Diagnosis'}
            </button>
          </div>

          {/* Results Area */}
          <div className="flex flex-col justify-center">
            {!result && !analyzing && (
              <div className="text-center text-slate-400">
                <Stethoscope size={48} className="mx-auto mb-4 opacity-20" />
                <p>Waiting for analysis...</p>
              </div>
            )}

            {analyzing && (
              <div className="text-center space-y-4">
                <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
                <p className="text-slate-600 animate-pulse">Consulting Federated Nodes...</p>
              </div>
            )}

            {result && (
              <div className="animate-fade-in space-y-6">
                <div className={`p-6 rounded-xl border-l-4 shadow-sm ${
                  result === 'PNEUMONIA' ? 'bg-red-50 border-red-500' : 'bg-emerald-50 border-emerald-500'
                }`}>
                  <h3 className="text-sm uppercase tracking-wide font-semibold text-slate-500 mb-1">Model Prediction</h3>
                  <div className="flex items-center gap-3">
                    {result === 'PNEUMONIA' ? <AlertCircle className="text-red-500" size={32} /> : <CheckCircle className="text-emerald-500" size={32} />}
                    <span className={`text-3xl font-bold ${
                      result === 'PNEUMONIA' ? 'text-red-700' : 'text-emerald-700'
                    }`}>{result}</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-medium text-slate-600">
                    <span>Confidence Score</span>
                    <span>{(confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-600 transition-all duration-1000" 
                      style={{ width: `${confidence * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    * Result based on global model trained across {round > 0 ? '2' : '0'} hospitals.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}