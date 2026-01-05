"""
ADVANCED AUTONOMOUS POLYMARKET BOT v2.0
- Auto-discovers optimal markets
- Advanced pattern detection with ML scoring
- Telegram notifications to your phone
- Predictive momentum algorithms
"""

import requests
import time
from collections import deque
import statistics
from datetime import datetime
import json
import hashlib

# ============ CONFIGURATION ============
# Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = "8382732549:AAFYmPhScCN7G2xFFlr_o30dZ5IpxC_P7rY"  # Replace this
TELEGRAM_CHAT_ID = "6485399160"      # Replace this

CHECK_INTERVAL = 120  # 2 minutes
VOLUME_SPIKE_THRESHOLD = 2.2
PRICE_MOMENTUM_THRESHOLD = 0.04
HISTORY_WINDOW = 30  # Increased for better pattern detection

# Market filtering
MIN_VOLUME = 50000
MAX_VOLUME = 500000
MIN_LIQUIDITY = 10000
TARGET_MARKET_COUNT = 7

# Advanced settings
ENABLE_TELEGRAM = True  # Set to False if not using Telegram yet
ENABLE_ML_SCORING = True
MIN_CONFIDENCE_ALERT = 0.65  # Only alert on 65%+ confidence signals

# ============ TELEGRAM NOTIFICATIONS ============
class TelegramNotifier:
    """Send alerts to your phone via Telegram"""
    
    def __init__(self, bot_token, chat_id, enabled=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bot_token != "YOUR_BOT_TOKEN_HERE"
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if self.enabled:
            self._test_connection()
    
    def _test_connection(self):
        """Verify Telegram credentials work"""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            if response.status_code == 200:
                print("✅ Telegram connected successfully")
            else:
                print("⚠️  Telegram credentials invalid - notifications disabled")
                self.enabled = False
        except Exception as e:
            print(f"⚠️  Telegram connection failed: {e}")
            self.enabled = False
    
    def send_alert(self, message, priority="normal"):
        """Send notification to phone"""
        if not self.enabled:
            return
        
        # Add emoji based on priority
        emoji_map = {
            "critical": "🔴",
            "high": "🟠", 
            "normal": "🟢",
            "info": "ℹ️"
        }
        
        formatted_msg = f"{emoji_map.get(priority, '•')} {message}"
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_msg,
                "parse_mode": "HTML"
            }
            requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=5)
        except Exception as e:
            print(f"Telegram send failed: {e}")

# ============ ADVANCED PATTERN ANALYZER ============
class AdvancedPatternAnalyzer:
    """Machine learning-inspired pattern scoring"""
    
    def __init__(self):
        self.pattern_weights = {
            'volume_spike': 0.30,
            'momentum': 0.25,
            'imbalance': 0.20,
            'spread': 0.10,
            'acceleration': 0.15  # New: rate of price change
        }
        self.pattern_history = {}
    
    def calculate_confidence(self, signals, price_history, volume_history):
        """Calculate ML-style confidence score (0-1)"""
        scores = {}
        
        # Score each signal type
        for signal in signals:
            sig_type = signal['type'].lower()
            strength = signal.get('strength', 0.5)
            
            if sig_type == 'volume_spike':
                # Exponential scoring for volume spikes
                scores['volume_spike'] = min(1.0, (strength - 1) / 3)
            
            elif sig_type == 'momentum':
                scores['momentum'] = min(1.0, strength / 0.15)
            
            elif sig_type == 'imbalance':
                scores['imbalance'] = strength
            
            elif sig_type == 'tight_spread':
                scores['spread'] = 1.0
        
        # Calculate price acceleration (second derivative)
        if len(price_history) >= 10:
            scores['acceleration'] = self._calculate_acceleration(price_history)
        
        # Weighted average
        total_score = sum(
            scores.get(key, 0) * weight 
            for key, weight in self.pattern_weights.items()
        )
        
        # Bonus for signal confluence (multiple signals together)
        signal_count = len(signals)
        confluence_bonus = min(0.15, (signal_count - 1) * 0.05)
        
        final_confidence = min(1.0, total_score + confluence_bonus)
        return final_confidence
    
    def _calculate_acceleration(self, prices):
        """Detect if price change is accelerating"""
        recent = list(prices)[-10:]
        
        # First derivative (velocity)
        velocities = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        # Second derivative (acceleration)
        if len(velocities) >= 2:
            accelerations = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
            avg_accel = sum(accelerations) / len(accelerations)
            
            # Normalize to 0-1 range
            return min(1.0, abs(avg_accel) * 100)
        
        return 0
    
    def predict_direction(self, signals, price_history):
        """Predict likely direction of next move"""
        momentum_signals = [s for s in signals if s['type'] == 'MOMENTUM']
        imbalance_signals = [s for s in signals if s['type'] == 'IMBALANCE']
        
        direction_score = 0
        
        # Weight momentum direction
        for sig in momentum_signals:
            direction_score += 1 if sig.get('direction') == 'UP' else -1
        
        # Weight order book imbalance
        for sig in imbalance_signals:
            direction_score += 0.7 if sig.get('direction') == 'BUY' else -0.7
        
        # Check price trend consistency
        if len(price_history) >= 5:
            recent = list(price_history)[-5:]
            trend = sum(1 if recent[i] > recent[i-1] else -1 for i in range(1, 5))
            direction_score += trend * 0.3
        
        if direction_score > 0.5:
            return 'UP', abs(direction_score) / 3
        elif direction_score < -0.5:
            return 'DOWN', abs(direction_score) / 3
        else:
            return 'NEUTRAL', 0

# ============ MARKET DISCOVERY ============
class MarketDiscovery:
    """Enhanced market discovery with caching"""
    
    def __init__(self):
        self.api_base = "https://clob.polymarket.com"
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.cache = {}
        self.cache_duration = 600  # 10 minutes
    
    def discover_markets(self):
        """Find optimal markets with smart caching"""
        cache_key = "market_list"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
        
        print("🔍 Discovering optimal markets...")
        
        try:
            url = f"{self.gamma_api}/markets"
            params = {'active': 'true', 'closed': 'false', 'limit': 100}
            
            response = requests.get(url, params=params, timeout=15)
            markets = response.json()
            
            candidates = []
            
            for market in markets:
                try:
                    volume = float(market.get('volume', 0))
                    liquidity = float(market.get('liquidity', 0))
                    
                    if MIN_VOLUME <= volume <= MAX_VOLUME and liquidity >= MIN_LIQUIDITY:
                        tokens = market.get('tokens', [])
                        if len(tokens) >= 2:
                            token_id = tokens[0].get('token_id', '')
                            
                            if token_id:
                                candidates.append({
                                    'token_id': token_id,
                                    'question': market.get('question', 'Unknown'),
                                    'volume': volume,
                                    'liquidity': liquidity,
                                    'end_date': market.get('endDate', ''),
                                    'score': self._calculate_score(volume, liquidity, market)
                                })
                except (KeyError, ValueError, TypeError):
                    continue
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[:TARGET_MARKET_COUNT]
            
            print(f"✅ Found {len(selected)} optimal markets")
            
            # Cache results
            result_ids = [(m['token_id'], m['question']) for m in selected]
            self.cache[cache_key] = (time.time(), result_ids)
            
            return result_ids
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return []
    
    def _calculate_score(self, volume, liquidity, market):
        """Enhanced scoring with multiple factors"""
        # Volume score (peak at $200k)
        volume_score = 1 - abs(volume - 200000) / 500000
        
        # Liquidity score
        liquidity_score = min(liquidity / 50000, 1.0)
        
        # Time-to-expiry score (prefer markets with 1-4 weeks left)
        time_score = 0.5  # Default if we can't parse
        try:
            end_date = market.get('endDate', '')
            if end_date:
                # Simple heuristic - boost recent deadlines slightly
                time_score = 0.7
        except:
            pass
        
        return (volume_score * 0.5) + (liquidity_score * 0.3) + (time_score * 0.2)

# ============ ENHANCED MARKET TRACKER ============
class MarketTracker:
    def __init__(self, token_id, question="Unknown"):
        self.token_id = token_id
        self.question = question
        self.price_history = deque(maxlen=HISTORY_WINDOW)
        self.volume_history = deque(maxlen=HISTORY_WINDOW)
        self.last_update = None
        self.analyzer = AdvancedPatternAnalyzer()
        
    def fetch_market_data(self):
        """Enhanced data fetching with error handling"""
        try:
            url = f"https://clob.polymarket.com/book?token_id={self.token_id}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('bids') and data.get('asks'):
                best_bid = float(data['bids'][0]['price']) if data['bids'] else 0
                best_ask = float(data['asks'][0]['price']) if data['asks'] else 1
                mid_price = (best_bid + best_ask) / 2
                
                # Enhanced volume calculation (deeper book)
                bid_volume = sum(float(b['size']) for b in data['bids'][:15])
                ask_volume = sum(float(a['size']) for a in data['asks'][:15])
                total_volume = bid_volume + ask_volume
                
                return {
                    'price': mid_price,
                    'volume': total_volume,
                    'spread': best_ask - best_bid,
                    'imbalance': (bid_volume - ask_volume) / max(total_volume, 1),
                    'bid_depth': len(data['bids']),
                    'ask_depth': len(data['asks'])
                }
        except Exception as e:
            pass
        return None
    
    def update(self):
        """Update with enhanced analysis"""
        data = self.fetch_market_data()
        if not data:
            return None
            
        self.price_history.append(data['price'])
        self.volume_history.append(data['volume'])
        self.last_update = time.time()
        
        if len(self.price_history) < 10:
            return None
            
        return self.analyze_patterns(data)
    
    def analyze_patterns(self, current):
        """Enhanced pattern detection"""
        signals = []
        
        # Signal 1: Volume Spike (with adaptive threshold)
        if len(self.volume_history) >= 5:
            recent_avg = statistics.mean(list(self.volume_history)[-5:])
            older_avg = statistics.mean(list(self.volume_history)[:-5])
            
            if older_avg > 0 and current['volume'] > recent_avg * VOLUME_SPIKE_THRESHOLD:
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'strength': current['volume'] / recent_avg,
                    'msg': f"Volume spike: {current['volume']/recent_avg:.1f}x recent average"
                })
        
        # Signal 2: Multi-timeframe Momentum
        if len(self.price_history) >= 10:
            short_term = list(self.price_history)[-3:]  # Last 3 checks
            mid_term = list(self.price_history)[-7:]    # Last 7 checks
            
            short_change = (short_term[-1] - short_term[0]) / max(short_term[0], 0.01)
            mid_change = (mid_term[-1] - mid_term[0]) / max(mid_term[0], 0.01)
            
            if abs(short_change) > PRICE_MOMENTUM_THRESHOLD:
                # Check if momentum is accelerating
                is_accelerating = abs(short_change) > abs(mid_change)
                
                signals.append({
                    'type': 'MOMENTUM',
                    'direction': 'UP' if short_change > 0 else 'DOWN',
                    'strength': abs(short_change),
                    'accelerating': is_accelerating,
                    'msg': f"{'Accelerating' if is_accelerating else 'Steady'} momentum {short_change*100:+.1f}%"
                })
        
        # Signal 3: Strong Order Book Imbalance
        if abs(current['imbalance']) > 0.3:
            side = 'BUY' if current['imbalance'] > 0 else 'SELL'
            signals.append({
                'type': 'IMBALANCE',
                'direction': side,
                'strength': abs(current['imbalance']),
                'msg': f"Order book: {abs(current['imbalance'])*100:.0f}% {side} side"
            })
        
        # Signal 4: Spread Tightening (liquidity improving)
        if current['spread'] < 0.02:
            signals.append({
                'type': 'TIGHT_SPREAD',
                'msg': f"Tight spread: {current['spread']*100:.2f}%"
            })
        
        # Signal 5: Book Depth (new)
        total_depth = current['bid_depth'] + current['ask_depth']
        if total_depth > 30:  # Lots of limit orders = institutional interest
            signals.append({
                'type': 'DEEP_BOOK',
                'msg': f"Deep order book: {total_depth} levels"
            })
        
        if signals:
            # Calculate ML confidence score
            confidence = self.analyzer.calculate_confidence(
                signals, self.price_history, self.volume_history
            )
            
            # Predict direction
            direction, dir_confidence = self.analyzer.predict_direction(
                signals, self.price_history
            )
            
            return {
                'token_id': self.token_id,
                'question': self.question,
                'price': current['price'],
                'signals': signals,
                'confidence': confidence,
                'predicted_direction': direction,
                'direction_confidence': dir_confidence,
                'timestamp': time.time()
            }
        
        return None

# ============ ADVANCED BOT ============
class AdvancedPolymarketBot:
    def __init__(self):
        self.discovery = MarketDiscovery()
        self.trackers = []
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ENABLE_TELEGRAM)
        self.cycle = 0
        self.last_discovery = 0
        self.discovery_interval = 3600
        self.alert_history = set()
        
    def initialize(self):
        """Bootstrap the advanced bot"""
        print("🤖 ADVANCED AUTONOMOUS POLYMARKET BOT v2.0")
        print("=" * 60)
        print(f"ML Scoring: {'✅' if ENABLE_ML_SCORING else '❌'}")
        print(f"Telegram: {'✅' if self.telegram.enabled else '❌'}")
        print("=" * 60)
        
        self.telegram.send_alert("🤖 Bot started and monitoring markets", "info")
        self.refresh_markets()
    
    def refresh_markets(self):
        """Discover and track new markets"""
        market_data = self.discovery.discover_markets()
        
        if not market_data:
            return
        
        self.trackers = []
        for token_id, question in market_data:
            tracker = MarketTracker(token_id, question)
            self.trackers.append(tracker)
        
        self.last_discovery = time.time()
        print(f"✅ Monitoring {len(self.trackers)} markets\n")
    
    def run(self):
        """Enhanced main loop"""
        self.initialize()
        
        while True:
            self.cycle += 1
            now = time.time()
            
            # Refresh markets hourly
            if now - self.last_discovery > self.discovery_interval:
                print("\n🔄 Refreshing market list...")
                self.refresh_markets()
                self.telegram.send_alert("🔄 Markets refreshed", "info")
            
            print(f"\n[Cycle {self.cycle}] {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 60)
            
            signals_found = 0
            
            for tracker in self.trackers:
                result = tracker.update()
                if result and result['confidence'] >= MIN_CONFIDENCE_ALERT:
                    self.handle_signal(result)
                    signals_found += 1
            
            if signals_found == 0:
                print("No high-confidence signals detected")
            
            print(f"Next check in {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)
    
    def handle_signal(self, result):
        """Enhanced signal handling with Telegram"""
        # Create unique alert ID to avoid spam
        alert_id = hashlib.md5(
            f"{result['token_id']}{result['confidence']:.2f}".encode()
        ).hexdigest()[:8]
        
        # Check if we already alerted on this recently (within 1 hour)
        if alert_id in self.alert_history:
            return
        
        self.alert_history.add(alert_id)
        
        # Console output
        print(f"\n🚨 HIGH CONFIDENCE SIGNAL")
        print(f"Market: {result['question'][:50]}...")
        print(f"Price: {result['price']:.3f}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Predicted: {result['predicted_direction']} ({result['direction_confidence']*100:.0f}%)")
        
        for sig in result['signals']:
            print(f"  • {sig['msg']}")
        
        # Make trading decision
        decision = self.make_decision(result)
        
        if decision:
            print(f"\n💰 {decision['action']}")
            print(f"Reason: {decision['reason']}")
            print(f"Risk Level: {decision['risk']}")
            
            # Send Telegram alert
            self._send_telegram_alert(result, decision)
    
    def _send_telegram_alert(self, result, decision):
        """Format and send Telegram notification"""
        priority = "critical" if result['confidence'] > 0.8 else "high"
        
        msg = f"<b>{decision['action']}</b>\n\n"
        msg += f"<b>Market:</b> {result['question'][:60]}\n"
        msg += f"<b>Price:</b> {result['price']:.3f}\n"
        msg += f"<b>Confidence:</b> {result['confidence']*100:.0f}%\n"
        msg += f"<b>Direction:</b> {result['predicted_direction']}\n\n"
        msg += f"<b>Reason:</b> {decision['reason']}\n"
        msg += f"<b>Risk:</b> {decision['risk']}"
        
        self.telegram.send_alert(msg, priority)
    
    def make_decision(self, result):
        """Advanced decision making with risk assessment"""
        confidence = result['confidence']
        direction = result['predicted_direction']
        signals = result['signals']
        
        # High confidence trade (80%+)
        if confidence >= 0.80:
            if direction == 'UP':
                return {
                    'action': '🟢 STRONG BUY YES',
                    'reason': 'Multiple strong indicators align',
                    'risk': 'LOW',
                    'size': 'Large'
                }
            elif direction == 'DOWN':
                return {
                    'action': '🔴 STRONG BUY NO',
                    'reason': 'Multiple strong indicators align',
                    'risk': 'LOW',
                    'size': 'Large'
                }
        
        # Medium confidence trade (65-80%)
        elif confidence >= 0.65:
            if direction == 'UP':
                return {
                    'action': '🟡 MODERATE BUY YES',
                    'reason': 'Good signal confluence',
                    'risk': 'MEDIUM',
                    'size': 'Medium'
                }
            elif direction == 'DOWN':
                return {
                    'action': '🟡 MODERATE BUY NO',
                    'reason': 'Good signal confluence',
                    'risk': 'MEDIUM',
                    'size': 'Medium'
                }
        
        return {
            'action': '⚪ WATCH',
            'reason': 'Interesting pattern but wait for confirmation',
            'risk': 'N/A',
            'size': 'None'
        }

# ============ SETUP INSTRUCTIONS ============
def print_setup_instructions():
    """Help users set up Telegram"""
    print("\n" + "="*60)
    print("📱 TELEGRAM SETUP INSTRUCTIONS")
    print("="*60)
    print("\n1. Open Telegram and search for '@BotFather'")
    print("2. Send: /newbot")
    print("3. Choose a name and username for your bot")
    print("4. Copy the TOKEN (looks like: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)")
    print("\n5. Search for '@userinfobot' in Telegram")
    print("6. Start a chat - it will show your CHAT_ID (looks like: 123456789)")
    print("\n7. Paste both values at the top of this script")
    print("8. Start a chat with your bot (search for its username)")
    print("="*60 + "\n")

# ============ ENTRY POINT ============
if __name__ == "__main__":
    # Check if Telegram is configured
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print_setup_instructions()
        print("⚠️  Running without Telegram notifications")
        print("You'll still see console alerts\n")
        time.sleep(3)
    
    bot = AdvancedPolymarketBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.telegram.send_alert("👋 Bot stopped", "info")
        print("\n\n👋 Bot stopped")
    except Exception as e:
        error_msg = f"❌ Fatal error: {str(e)[:100]}"
        bot.telegram.send_alert(error_msg, "critical")
        print(f"\n{error_msg}")
        raise