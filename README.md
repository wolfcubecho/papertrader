# Crypto Paper Traders

Structure-based crypto paper traders for 24/7 VPS deployment.

## Strategies

### Scalp Trader (`npm run scalp`)
- **Timeframe:** 5m primary
- **Entry:** Dual-mode (TREND breakouts / RANGE mean reversion)
- **Stops:** Structure-based (swing high/low, max 2% risk)
- **Target:** Middle Bollinger Band (min 1.5:1 R:R)
- **Symbols:** 20 major coins

### Swing Trader (`npm run swing`)
- **Timeframe:** 4H primary
- **Entry:** SMC/ICT (fresh Order Blocks, OTE zones, displacement)
- **Stops:** 2x ATR with structure
- **Target:** Quality-tiered (1.5:1 to 4:1 R:R)
- **Symbols:** 20 major coins

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/crypto-paper-traders.git
cd crypto-paper-traders

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run scalp trader
npm run scalp

# Or run swing trader
npm run swing

# Or run both simultaneously
npm run start:both
```

## VPS Deployment (Ubuntu/Debian)

### 1. Install Node.js 20+
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### 2. Install Python 3.10+ (REQUIRED for auto-learning)
```bash
sudo apt install -y python3 python3-pip
pip3 install pandas numpy scikit-learn lightgbm
```

**Note:** The scalp trader auto-trains every 100 trades. Without Python + LightGBM installed, training will fail (but trading continues).

### 3. Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/crypto-paper-traders.git
cd crypto-paper-traders
npm install
npm run build
```

### 4. Run with PM2 (recommended for 24/7)
```bash
# Install PM2
npm install -g pm2

# Start scalp trader
pm2 start dist/multi-coin-paper-trader-scalp.js --name "scalp-trader"

# Start swing trader
pm2 start dist/multi-coin-paper-trader-swing.js --name "swing-trader"

# Save PM2 config for auto-restart on reboot
pm2 save
pm2 startup
```

### 5. Monitor
```bash
# View logs
pm2 logs scalp-trader
pm2 logs swing-trader

# Status
pm2 status

# Restart
pm2 restart scalp-trader
```

## Data Storage

Trade data is stored locally in JSON files:
- `data/paper-trades-scalp/*.json` - Scalp trader state per symbol
- `data/paper-trades-swing/*.json` - Swing trader state per symbol
- `data/models/` - ML model files (optional)

## No API Keys Required

These traders use **public Binance API** only (candle data). No API keys needed.
All trades are simulated locally - no real orders placed.

## Configuration

Edit the CONFIG object in each trader file to adjust:
- Risk per trade (default: 1%)
- Cooldown between trades
- Min signals required
- Max hold time
- etc.

## License

MIT
