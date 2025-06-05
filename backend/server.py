from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import httpx
import websockets
import json
import base64

# Solana imports
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.publickey import PublicKey
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
import ta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configuration
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Solana Configuration
SOLANA_RPC = os.environ.get('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
JUPITER_API = os.environ.get('JUPITER_V6_API', 'https://quote-api.jup.ag/v6')
MIN_PROFIT = float(os.environ.get('MIN_PROFIT_THRESHOLD', '0.01'))
MAX_SLIPPAGE = int(os.environ.get('MAX_SLIPPAGE_BPS', '50'))

# Create Solana client
solana_client = AsyncClient(SOLANA_RPC, commitment=Confirmed)

# Create the main app without a prefix
app = FastAPI(title="Solana DeFi Trading Bot", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Data Models
class WalletConnection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wallet_address: str
    network: str = "mainnet-beta"
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    
class WalletConnectionCreate(BaseModel):
    wallet_address: str

class PoolData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pool_address: str
    token_a: str
    token_b: str
    liquidity: float
    volume_24h: float
    fee_rate: float
    correlation_ratio: float
    participation_score: float
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class TradeSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    pool_address: str
    suggested_amount: float
    expected_profit: float
    risk_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ArbitrageOpportunity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dex_a: str
    dex_b: str
    token_pair: str
    price_diff: float
    profit_potential: float
    volume_required: float
    execution_complexity: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

class TransactionHistory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tx_hash: str
    wallet_address: str
    action: str  # 'swap', 'add_liquidity', 'remove_liquidity'
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    slippage: float
    profit_loss: float
    gas_fee: float
    status: str  # 'pending', 'confirmed', 'failed'
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# AI Trading Engine
class SophisticatedTradingAI:
    def __init__(self):
        self.price_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    async def analyze_market_conditions(self, pool_data: Dict) -> TradeSignal:
        """Sophisticated market analysis using multiple indicators"""
        try:
            # Get historical price data
            price_data = await self.get_price_history(pool_data['pool_address'])
            
            if len(price_data) < 50:  # Need enough data
                return TradeSignal(
                    signal_type="HOLD",
                    confidence=0.3,
                    reasoning="Insufficient data for analysis",
                    pool_address=pool_data['pool_address'],
                    suggested_amount=0.0,
                    expected_profit=0.0,
                    risk_score=0.5
                )
            
            # Calculate technical indicators
            df = pd.DataFrame(price_data)
            df['rsi'] = ta.momentum.RSIIndicator(close=df['price']).rsi()
            df['macd'] = ta.trend.MACD(close=df['price']).macd()
            df['bb_upper'], df['bb_lower'] = ta.volatility.BollingerBands(close=df['price']).bollinger_hband(), ta.volatility.BollingerBands(close=df['price']).bollinger_lband()
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(close=df['price'], volume=df['volume']).volume_sma()
            
            # Current market indicators
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_price = df['price'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # Volume analysis
            volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
            
            # Trend analysis
            price_change_24h = (current_price - df['price'].iloc[-24]) / df['price'].iloc[-24]
            volatility = df['price'].pct_change().std()
            
            # AI Decision Logic
            signals = []
            confidence_factors = []
            
            # RSI signals
            if current_rsi < 30:  # Oversold
                signals.append("BUY")
                confidence_factors.append(0.8)
            elif current_rsi > 70:  # Overbought
                signals.append("SELL")
                confidence_factors.append(0.8)
            
            # MACD signals
            if current_macd > 0 and df['macd'].iloc[-2] <= 0:  # Bullish crossover
                signals.append("BUY")
                confidence_factors.append(0.7)
            elif current_macd < 0 and df['macd'].iloc[-2] >= 0:  # Bearish crossover
                signals.append("SELL")
                confidence_factors.append(0.7)
            
            # Bollinger Bands
            if current_price < bb_lower:  # Oversold
                signals.append("BUY")
                confidence_factors.append(0.6)
            elif current_price > bb_upper:  # Overbought
                signals.append("SELL")
                confidence_factors.append(0.6)
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                confidence_factors = [cf * 1.2 for cf in confidence_factors]
            
            # Determine final signal
            if not signals:
                signal_type = "HOLD"
                confidence = 0.4
                reasoning = "No clear signals detected"
            else:
                # Most common signal
                buy_count = signals.count("BUY")
                sell_count = signals.count("SELL")
                
                if buy_count > sell_count:
                    signal_type = "BUY"
                    confidence = min(np.mean(confidence_factors) * (buy_count / len(signals)), 0.95)
                    reasoning = f"Bullish indicators: RSI={current_rsi:.1f}, Volume ratio={volume_ratio:.1f}"
                elif sell_count > buy_count:
                    signal_type = "SELL" 
                    confidence = min(np.mean(confidence_factors) * (sell_count / len(signals)), 0.95)
                    reasoning = f"Bearish indicators: RSI={current_rsi:.1f}, Price near resistance"
                else:
                    signal_type = "HOLD"
                    confidence = 0.5
                    reasoning = "Mixed signals, waiting for clarity"
            
            # Calculate suggested amount and profit
            base_amount = float(os.environ.get('BASE_TRADE_SIZE', '0.1'))
            suggested_amount = base_amount * confidence
            expected_profit = suggested_amount * abs(price_change_24h) * confidence
            risk_score = volatility * (1 - confidence)
            
            return TradeSignal(
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                pool_address=pool_data['pool_address'],
                suggested_amount=suggested_amount,
                expected_profit=expected_profit,
                risk_score=risk_score
            )
            
        except Exception as e:
            logging.error(f"AI analysis error: {e}")
            return TradeSignal(
                signal_type="HOLD",
                confidence=0.2,
                reasoning=f"Analysis error: {str(e)}",
                pool_address=pool_data.get('pool_address', ''),
                suggested_amount=0.0,
                expected_profit=0.0,
                risk_score=0.8
            )
    
    async def get_price_history(self, pool_address: str) -> List[Dict]:
        """Get historical price data for analysis"""
        # Mock data for now - in production, integrate with actual DEX APIs
        now = datetime.utcnow()
        prices = []
        base_price = 100.0
        
        for i in range(100):
            # Simulate price movement with some trend and noise
            trend = 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # Random noise
            price = base_price * (1 + trend + noise)
            volume = np.random.uniform(10000, 50000)
            
            prices.append({
                'timestamp': now - timedelta(hours=i),
                'price': price,
                'volume': volume
            })
        
        return list(reversed(prices))  # Chronological order

# Initialize AI engine
trading_ai = SophisticatedTradingAI()

# Jupiter DEX Integration
class JupiterSwapService:
    def __init__(self):
        self.api_url = JUPITER_API
        
    async def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = MAX_SLIPPAGE):
        """Get swap quote from Jupiter"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": amount,
                    "slippageBps": slippage_bps
                }
            )
            return response.json()
    
    async def get_swap_transaction(self, quote_response: Dict, user_public_key: str):
        """Get swap transaction from Jupiter"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/swap",
                json={
                    "quoteResponse": quote_response,
                    "userPublicKey": user_public_key,
                    "wrapAndUnwrapSol": True
                }
            )
            return response.json()

jupiter_service = JupiterSwapService()

# MEV Detection Service
class MEVDetector:
    async def detect_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Detect MEV arbitrage opportunities across DEXs"""
        opportunities = []
        
        # Common trading pairs to monitor
        pairs = [
            ("SOL", "USDC"),
            ("SOL", "USDT"), 
            ("USDC", "USDT"),
            ("RAY", "SOL"),
            ("SRM", "SOL")
        ]
        
        for token_a, token_b in pairs:
            try:
                # Get prices from different DEXs (simplified)
                jupiter_price = await self.get_price_from_jupiter(token_a, token_b)
                raydium_price = await self.get_price_from_raydium(token_a, token_b)
                
                if jupiter_price and raydium_price:
                    price_diff = abs(jupiter_price - raydium_price) / min(jupiter_price, raydium_price)
                    
                    if price_diff > MIN_PROFIT:
                        opportunities.append(ArbitrageOpportunity(
                            dex_a="Jupiter",
                            dex_b="Raydium", 
                            token_pair=f"{token_a}/{token_b}",
                            price_diff=price_diff,
                            profit_potential=price_diff * 10000,  # Example calculation
                            volume_required=1000.0,
                            execution_complexity="Medium"
                        ))
                        
            except Exception as e:
                logging.error(f"MEV detection error for {token_a}/{token_b}: {e}")
                
        return opportunities
    
    async def get_price_from_jupiter(self, token_a: str, token_b: str) -> Optional[float]:
        """Get price from Jupiter DEX"""
        try:
            # Mock implementation - replace with actual Jupiter API calls
            return np.random.uniform(0.95, 1.05)
        except:
            return None
    
    async def get_price_from_raydium(self, token_a: str, token_b: str) -> Optional[float]:
        """Get price from Raydium DEX"""
        try:
            # Mock implementation - replace with actual Raydium API calls
            return np.random.uniform(0.95, 1.05)
        except:
            return None

mev_detector = MEVDetector()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Solana DeFi Trading Bot API", "version": "1.0.0", "status": "active"}

@api_router.post("/wallet/connect", response_model=WalletConnection)
async def connect_wallet(input: WalletConnectionCreate):
    """Connect a Phantom wallet"""
    try:
        # Validate wallet address
        pubkey = PublicKey(input.wallet_address)
        
        wallet_data = WalletConnection(wallet_address=input.wallet_address)
        await db.wallets.insert_one(wallet_data.dict())
        
        return wallet_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid wallet address: {str(e)}")

@api_router.get("/wallet/{wallet_address}/balance")
async def get_wallet_balance(wallet_address: str):
    """Get wallet SOL balance"""
    try:
        pubkey = PublicKey(wallet_address)
        balance = await solana_client.get_balance(pubkey)
        sol_balance = balance.value / 1e9  # Convert lamports to SOL
        
        return {
            "wallet_address": wallet_address,
            "sol_balance": sol_balance,
            "lamports": balance.value
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching balance: {str(e)}")

@api_router.get("/pools/data", response_model=List[PoolData])
async def get_pool_data():
    """Get liquidity pool data with analytics"""
    # Mock pool data - replace with actual DEX API calls
    pools = [
        PoolData(
            pool_address="8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6",
            token_a="SOL",
            token_b="USDC",
            liquidity=5000000.0,
            volume_24h=1200000.0,
            fee_rate=0.003,
            correlation_ratio=0.85,
            participation_score=0.92
        ),
        PoolData(
            pool_address="HCKZBrwkVa2qa3TGCsJ8bdrxF4KjJPRbVZSgggL6PcKL",
            token_a="RAY",
            token_b="SOL", 
            liquidity=2300000.0,
            volume_24h=800000.0,
            fee_rate=0.0025,
            correlation_ratio=0.78,
            participation_score=0.88
        )
    ]
    
    # Store in database
    for pool in pools:
        await db.pools.replace_one(
            {"pool_address": pool.pool_address},
            pool.dict(),
            upsert=True
        )
    
    return pools

@api_router.get("/ai/analysis/{pool_address}", response_model=TradeSignal)
async def get_ai_analysis(pool_address: str):
    """Get AI trading signal for a specific pool"""
    try:
        # Get pool data
        pool = await db.pools.find_one({"pool_address": pool_address})
        if not pool:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        # Get AI analysis
        signal = await trading_ai.analyze_market_conditions(pool)
        
        # Store signal in database
        await db.trade_signals.insert_one(signal.dict())
        
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@api_router.get("/mev/opportunities", response_model=List[ArbitrageOpportunity])
async def get_arbitrage_opportunities():
    """Detect MEV arbitrage opportunities"""
    try:
        opportunities = await mev_detector.detect_arbitrage_opportunities()
        
        # Store opportunities
        for opp in opportunities:
            await db.arbitrage_opportunities.insert_one(opp.dict())
        
        return opportunities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MEV detection failed: {str(e)}")

@api_router.post("/swap/quote")
async def get_swap_quote(input_mint: str, output_mint: str, amount: float, slippage_bps: int = MAX_SLIPPAGE):
    """Get swap quote from Jupiter"""
    try:
        amount_lamports = int(amount * 1e9)  # Convert to lamports
        quote = await jupiter_service.get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote failed: {str(e)}")

@api_router.post("/trading/execute")
async def execute_trade(
    wallet_address: str,
    signal_id: str,
    background_tasks: BackgroundTasks
):
    """Execute trade based on AI signal"""
    try:
        # Get signal from database
        signal = await db.trade_signals.find_one({"id": signal_id})
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Check confidence threshold
        confidence_threshold = float(os.environ.get('AI_CONFIDENCE_THRESHOLD', '0.7'))
        if signal['confidence'] < confidence_threshold:
            raise HTTPException(status_code=400, detail="Signal confidence too low")
        
        # Add background task for execution
        background_tasks.add_task(execute_trade_background, wallet_address, signal)
        
        return {"message": "Trade execution started", "signal_id": signal_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")

async def execute_trade_background(wallet_address: str, signal: Dict):
    """Background task for trade execution"""
    try:
        # Create transaction record
        tx_record = TransactionHistory(
            tx_hash="pending",
            wallet_address=wallet_address,
            action="swap",
            token_in="SOL",
            token_out="USDC",
            amount_in=signal['suggested_amount'],
            amount_out=0.0,  # Will be updated after execution
            slippage=0.0,
            profit_loss=0.0,
            gas_fee=0.0,
            status="pending"
        )
        
        await db.transactions.insert_one(tx_record.dict())
        
        # Simulate trade execution (replace with actual implementation)
        await asyncio.sleep(2)  # Simulate network delay
        
        # Update transaction status
        tx_record.status = "confirmed"
        tx_record.tx_hash = f"5{uuid.uuid4().hex[:62]}"  # Mock transaction hash
        
        await db.transactions.replace_one(
            {"id": tx_record.id},
            tx_record.dict()
        )
        
        logging.info(f"Trade executed: {tx_record.tx_hash}")
        
    except Exception as e:
        logging.error(f"Background trade execution failed: {e}")

@api_router.get("/trading/history/{wallet_address}", response_model=List[TransactionHistory])
async def get_trading_history(wallet_address: str):
    """Get trading history for a wallet"""
    try:
        transactions = await db.transactions.find(
            {"wallet_address": wallet_address}
        ).sort("timestamp", -1).to_list(100)
        
        return [TransactionHistory(**tx) for tx in transactions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Get various statistics
        total_pools = await db.pools.count_documents({})
        total_trades = await db.transactions.count_documents({})
        active_signals = await db.trade_signals.count_documents({
            "created_at": {"$gte": datetime.utcnow() - timedelta(hours=1)}
        })
        
        # Calculate total volume (mock data)
        total_volume = await db.transactions.aggregate([
            {"$group": {"_id": None, "total": {"$sum": "$amount_in"}}}
        ]).to_list(1)
        
        volume = total_volume[0]["total"] if total_volume else 0.0
        
        return {
            "total_pools_monitored": total_pools,
            "total_trades_executed": total_trades,
            "active_ai_signals": active_signals,
            "total_volume_24h": volume,
            "system_status": "operational",
            "last_updated": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Solana DeFi Trading Bot API starting up...")
    logger.info(f"Connected to Solana RPC: {SOLANA_RPC}")
    logger.info(f"Jupiter API: {JUPITER_API}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    await solana_client.close()
    logger.info("Connections closed")
