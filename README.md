# 🚀 TradeBot Frontend Dashboard

A modern, responsive, and real-time web dashboard for the **TradeBot** cryptocurrency trading system. Built with React, TypeScript, and Tailwind CSS, it provides a seamless interface to monitor, configure, and control your algorithmic trading strategies.

![React](https://img.shields.io/badge/React-18.3-61dafb?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178c6?style=flat-square&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?style=flat-square&logo=tailwindcss)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)

---

## 📖 About The Project

TradeBot is an advanced algorithmic trading system utilizing Machine Learning (Random Forest, XGBoost, TensorFlow) and Reinforcement Learning to trade crypto assets on testnets.

This **Frontend Dashboard** serves as the control center, allowing you to:
- 🟢 **Start/Stop** the bot in various modes (Live, Simple Backtest, Enhanced ML Backtest, Train).
- 📊 **Monitor** real-time performance metrics, equity curves, and open positions.
- ⚙️ **Configure** trading parameters, ML hyperparameters, and per-symbol risk settings on the fly.
- 📜 **View** live, color-coded streaming logs via WebSocket.
- 🧪 **Trigger** backtests and view detailed performance reports (Sharpe Ratio, Max Drawdown, Win Rate).

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State & Data**: React Hooks, Axios, Recharts (for charts), Lucide React (for icons)

### Backend (TradeBot)
- **Framework**: FastAPI (Python 3.10+)
- **ML/Data**: Pandas, Scikit-Learn, TensorFlow, XGBoost, CCXT
- **Server**: Uvicorn (ASGI)

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed on your system:

1. **Python 3.10+**
2. **Node.js 18+** and **npm** (LTS recommended)
3. **Python Virtual Environment** (Recommended)

Verify installations:
```bash
python3 --version
node --version
npm --version
```

---

## ⚡ Quick Start (TL;DR)

If you already have dependencies installed, open **two separate terminals**:

**Terminal 1 (Backend):**
```bash
cd path/to/TradeBot
source path/to/your/venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd path/to/TradeBot/frontend
npm run dev
```
Open your browser to: **http://localhost:3000**

---

## 📦 Detailed Installation & Setup

### Step 1: Install Backend Dependencies
Navigate to your project directory, activate your virtual environment, and install required Python packages:
```bash
cd path/to/TradeBot
source path/to/your/venv/bin/activate

pip install numpy pandas ccxt scikit-learn matplotlib shap imblearn tensorflow xgboost fastapi uvicorn
```

### Step 2: Install Frontend Dependencies
Navigate to the frontend folder and install Node packages:
```bash
cd path/to/TradeBot/frontend
npm install
```
*(This installs React, Vite, Tailwind, Recharts, Axios, and all necessary dev dependencies).*

---

## 🚀 How to Run the Project

You **must** run the backend and frontend simultaneously in **two separate terminal windows**.

### Terminal 1: Start the FastAPI Backend
```bash
# 1. Navigate to the project root
cd path/to/TradeBot

# 2. Activate the virtual environment
source path/to/your/venv/bin/activate

# 3. Start the server (using python -m ensures the correct venv is used)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
✅ *Success looks like:* `INFO: Application startup complete.`

### Terminal 2: Start the React Frontend
```bash
# 1. Navigate to the frontend directory
cd path/to/TradeBot/frontend

# 2. Start the Vite development server
npm run dev
```
✅ *Success looks like:* `Local: http://localhost:3000/`

### Step 3: Access the Dashboard
Open your web browser and go to: **http://localhost:3000**

---

## 📁 Project Structure

```text
TradeBot/
├── api/                          # FastAPI Backend
│   ├── main.py                   # App entry point & route definitions
│   └── ...
├── core/                         # Core trading logic
├── ml/                           # Machine Learning models & training
├── frontend/                     # 🌐 REACT FRONTEND (This Project)
│   ├── src/
│   │   ├── api/                  # Axios API client & endpoints
│   │   ├── components/           # Reusable UI components
│   │   │   ├── Layout/           # Sidebar, Header, Main Layout
│   │   │   ├── Dashboard/        # Status cards, Metrics, Equity Chart
│   │   │   ├── Bot/              # Start/Stop controls, Status display
│   │   │   ├── Config/           # Settings editor (Trade, ML, Symbols)
│   │   │   ├── Positions/        # Live open positions table
│   │   │   ├── Backtest/         # Backtest triggers & results
│   │   │   ├── Reports/          # Performance analytics
│   │   │   └── Logs/             # Real-time WebSocket log viewer
│   │   ├── hooks/                # Custom hooks (useWebSocket, useBotStatus)
│   │   ├── types/                # TypeScript interfaces
│   │   ├── pages/                # Page-level components (Dashboard, Bot, etc.)
│   │   ├── styles/               # Global CSS & Tailwind imports
│   │   ├── App.tsx               # Main routing setup
│   │   └── main.tsx              # React entry point
│   ├── index.html                # HTML template
│   ├── package.json              # Node dependencies
│   ├── vite.config.ts            # Vite config (includes API proxy)
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── tsconfig.json             # TypeScript config
└── requirements.txt              # Python dependencies
```

---

## 🔌 API & WebSocket Endpoints

The frontend communicates with the backend via Vite's built-in proxy (`/api` → `http://localhost:8000`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/bot/start` | Start bot (payload: `{ mode: 'live' \| 'simple' \| 'enhanced' \| 'train' }`) |
| `POST` | `/bot/stop` | Stop the running bot |
| `GET` | `/bot/status` | Get current bot status, PID, and uptime |
| `GET` | `/config` | Fetch current trading and ML configuration |
| `PUT` | `/config` | Update configuration parameters |
| `GET` | `/positions` | Get list of currently open positions |
| `POST` | `/backtest/simple` | Trigger a simple backtest job |
| `POST` | `/backtest/enhanced` | Trigger an ML-enhanced backtest job |
| `GET` | `/backtest/result` | Fetch results of the last backtest job |
| `GET` | `/reports/latest` | Get latest performance report metrics |
| `GET` | `/chart/equity` | Get historical equity curve data |
| `WS` | `/ws/logs` | **WebSocket**: Stream live bot logs in real-time |

---

## 🐛 Troubleshooting

### 1. `ModuleNotFoundError: No module named 'fastapi'`
**Cause:** The system is using the global Python installation instead of your virtual environment.
**Fix:** Always activate the venv and use `python -m`:
```bash
source path/to/your/venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend shows "Unexpected token" or Babel parsing errors
**Cause:** Template literals (`` ` ``) or variables (`${}`) were accidentally escaped during file generation.
**Fix:** Ensure your `.tsx` files do not contain `\`` or `\${`. If you used an automated script to generate files, run a find-and-replace in your IDE to remove the backslashes before backticks and dollar signs.

### 3. Frontend shows "No data" or Network Errors
**Cause:** The frontend cannot reach the backend.
**Fix:**
1. Verify the backend is running: `curl http://localhost:8000/bot/status`
2. Check the browser console (F12) for CORS or 404 errors.
3. Ensure `vite.config.ts` has the correct proxy settings pointing to `http://localhost:8000`.

### 4. Port 3000 or 8000 is already in use
**Fix:** Kill the process or use a different port.
```bash
# Kill process on port 3000 (Linux/Mac)
lsof -ti:3000 | xargs kill -9

# Or run frontend on a different port
npm run dev -- --port 3001
```

---

## 💻 Development & Building

### Available NPM Scripts
```bash
npm run dev      # Start local development server with Hot Module Replacement (HMR)
npm run build    # Compile TypeScript and bundle for production (outputs to /dist)
npm run preview  # Locally preview the production build
```

### Adding a New Feature
1. **New Page:** Create `src/pages/NewPage.tsx`, add it to `src/App.tsx` routes, and add a link in `src/components/Layout/Sidebar.tsx`.
2. **New API Call:** Add the endpoint function in `src/api/client.ts` and define its return type in `src/types/index.ts`.

---

## 📝 Notes
- The Vite proxy (`vite.config.ts`) automatically handles CORS during development. Do not call `http://localhost:8000` directly from React; use `/api/...` so the proxy forwards it correctly.
- The dashboard polls `/bot/status` every 5 seconds and `/positions` every 10 seconds to keep the UI fresh.
- Logs are streamed via WebSocket for zero-latency updates.

---

**Happy Algorithmic Trading! 📈🤖**
