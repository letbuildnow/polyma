"""
RAILWAY-OPTIMIZED POLYMARKET BOT
Clean version without keep-alive hacks
Railway handles the infrastructure automatically
"""

import requests
import time
from collections import deque
import statistics
from datetime import datetime
import os
import hashlib

# ============ CONFIGURATION ============
# Railway automatically loads these from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

CHECK_INTERVAL = 120
VOLUME_SPIKE_THRESHOLD = 2.2
PRICE_MOMENTUM_THRESHOLD = 0.04
HISTORY_WINDOW = 30

MIN_VOLUME = 50000
MAX_VOLUME = 500000
MIN_LIQUIDITY = 10000
TARGET_MARKET_COUNT = 7

ENABLE_TELEGRAM = True
MIN_CONFIDENCE_ALERT = 0.65

# ============ TELEGRAM NOTIFICATIONS ============
class TelegramNotifier:
    def __init__(self, bot_token, chat_id, enabled=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if self.enabled:
            self._test_connection()
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            if response.status_code == 200:
                bot_info = response.json()
                print(f"✅ Telegram connected: @{bot_info['result']['username']}")
                return True
            else:
                print("⚠️  Telegram credentials invalid")
                self.enabled = False
                return False
        except Exception as e:
            print(f"⚠️  Telegram connection failed: {e}")
            self.enabled = False
            return False
    
    def send_alert(self, message, priority="normal"):
        if not self.enabled:
            return
        
        emoji_map = {"critical": "🔴", "high": "🟠", "normal": "🟢", "info": "ℹ️"}
        formatted_msg = f"{emoji_map.get(priority, '•')} {message}"
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_msg,
                "parse_mode": "HTML"
            }
            response = requests.post(
                f"{self.base_url}/sendMessage", 
                json=payload, 
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return False

# ============ ADVANCED PATTERN ANALYZER ============
class AdvancedPatternAnalyzer:
    def __init__(self):
        self.pattern_weights = {
            'volume_spike': 0.30,
            'momentum': 0.25,
            'imbalance': 0.20,
            'spread': 0.10,
            'acceleration': 0.15
        }
    
    def calculate_confidence(self, signals, price_history, volume_history):
        scores = {}
        
        for signal in signals:
            sig_type = signal['type'].lower()
            strength = signal.get('strength', 0.5)
            
            if sig_type == 'volume_spike':
                scores['volume_spike'] = min(1.0, (strength - 1) / 3)
            elif sig_type == 'momentum':
                scores['momentum'] = min(1.0, strength / 0.15)
            elif sig_type == 'imbalance':
                scores['imbalance'] = strength
            elif sig_type == 'tight_spread':
                scores['spread'] = 1.0
        
        if len(price_history) >= 10:
            scores['acceleration'] = self._calculate_acceleration(price_history)
        
        total_score = sum(
            scores.get(key, 0) * weight 
            for key, weight in self.pattern_weights.items()
        )
        
        signal_count = len(signals)
        confluence_bonus = min(0.15, (signal_count - 1) * 0.05)
        
        return min(1.0, total_score + confluence_bonus)
    
    def _calculate_acceleration(self, prices):
        recent = list(prices)[-10:]
        velocities = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        if len(velocities) >= 2:
            accelerations = [
                velocities[i] - velocities[i-1] 
                for i in range(1, len(velocities))
            ]
            avg_accel = sum(accelerations) / len(accelerations)
            return min(1.0, abs(avg_accel) * 100)
        return 0
    
    def predict_direction(self, signals, price_history):
        momentum_signals = [s for s in signals if s['type'] == 'MOMENTUM']
        imbalance_signals = [s for s in signals if s['type'] == 'IMBALANCE']
        
        direction_score = 0
        
        for sig in momentum_signals:
            direction_score += 1 if sig.get('direction') == 'UP' else -1
        
        for sig in imbalance_signals:
            direction_score += 0.7 if sig.get('direction') == 'BUY' else -0.7
        
        if len(price_history) >= 5:
            recent = list(price_history)[-5:]
            trend = sum(
                1 if recent[i] > recent[i-1] else -1 
                for i in range(1, 5)
            )
            direction_score += trend * 0.3
        
        if direction_score > 0.5:
            return 'UP', abs(direction_score) / 3
        elif direction_score < -0.5:
            return 'DOWN', abs(direction_score) / 3
        else:
            return 'NEUTRAL', 0

# ============ MARKET DISCOVERY ============
class MarketDiscovery:
    def __init__(self):
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.cache = {}
        self.cache_duration = 600
    
    def discover_markets(self):
        cache_key = "market_list"
        
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                print("📦 Using cached market list")
                return cached_data
        
        print("🔍 Discovering optimal markets...")
        
        try:
            url = f"{self.gamma_api}/markets"
            params = {
                'active': 'true',
                'closed': 'false',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
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
                                    'score': self._calculate_score(volume, liquidity)
                                })
                except (KeyError, ValueError, TypeError):
                    continue
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[:TARGET_MARKET_COUNT]
            
            print(f"✅ Found {len(selected)} optimal markets")
            for i, m in enumerate(selected, 1):
                print(f"   {i}. {m['question'][:55]}...")
            
            result_ids = [(m['token_id'], m['question']) for m in selected]
            self.cache[cache_key] = (time.time(), result_ids)
            
            return result_ids
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return []
    
    def _calculate_score(self, volume, liquidity):
        volume_score = 1 - abs(volume - 200000) / 500000
        liquidity_score = min(liquidity / 50000, 1.0)
        return (volume_score * 0.6) + (liquidity_score * 0.4)

# ============ MARKET TRACKER ============
class MarketTracker:
    def __init__(self, token_id, question="Unknown"):
        self.token_id = token_id
        self.question = question
        self.price_history = deque(maxlen=HISTORY_WINDOW)
        self.volume_history = deque(maxlen=HISTORY_WINDOW)
        self.last_update = None
        self.analyzer = AdvancedPatternAnalyzer()
        self.consecutive_errors = 0
        
    def fetch_market_data(self):
        try:
            url = f"https://clob.polymarket.com/book?token_id={self.token_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('bids') and data.get('asks'):
                best_bid = float(data['bids'][0]['price']) if data['bids'] else 0
                best_ask = float(data['asks'][0]['price']) if data['asks'] else 1
                mid_price = (best_bid + best_ask) / 2
                
                bid_volume = sum(float(b['size']) for b in data['bids'][:15])
                ask_volume = sum(float(a['size']) for a in data['asks'][:15])
                total_volume = bid_volume + ask_volume
                
                self.consecutive_errors = 0  # Reset error counter
                
                return {
                    'price': mid_price,
                    'volume': total_volume,
                    'spread': best_ask - best_bid,
                    'imbalance': (bid_volume - ask_volume) / max(total_volume, 1),
                    'bid_depth': len(data['bids']),
                    'ask_depth': len(data['asks'])
                }
        except Exception as e:
            self.consecutive_errors += 1
            if self.consecutive_errors <= 3:
                print(f"⚠️  Error fetching {self.token_id[:8]}: {e}")
        return None
    
    def update(self):
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
        signals = []
        
        # Volume Spike Detection
        if len(self.volume_history) >= 5:
            recent_avg = statistics.mean(list(self.volume_history)[-5:])
            if recent_avg > 0 and current['volume'] > recent_avg * VOLUME_SPIKE_THRESHOLD:
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'strength': current['volume'] / recent_avg,
                    'msg': f"Volume spike: {current['volume']/recent_avg:.1f}x average"
                })
        
        # Multi-timeframe Momentum
        if len(self.price_history) >= 10:
            short_term = list(self.price_history)[-3:]
            mid_term = list(self.price_history)[-7:]
            
            short_change = (short_term[-1] - short_term[0]) / max(short_term[0], 0.01)
            mid_change = (mid_term[-1] - mid_term[0]) / max(mid_term[0], 0.01)
            
            if abs(short_change) > PRICE_MOMENTUM_THRESHOLD:
                is_accelerating = abs(short_change) > abs(mid_change)
                
                signals.append({
                    'type': 'MOMENTUM',
                    'direction': 'UP' if short_change > 0 else 'DOWN',
                    'strength': abs(short_change),
                    'accelerating': is_accelerating,
                    'msg': f"{'Accelerating' if is_accelerating else 'Steady'} momentum {short_change*100:+.1f}%"
                })
        
        # Order Book Imbalance
        if abs(current['imbalance']) > 0.3:
            side = 'BUY' if current['imbalance'] > 0 else 'SELL'
            signals.append({
                'type': 'IMBALANCE',
                'direction': side,
                'strength': abs(current['imbalance']),
                'msg': f"Order book: {abs(current['imbalance'])*100:.0f}% {side} side"
            })
        
        # Tight Spread
        if current['spread'] < 0.02:
            signals.append({
                'type': 'TIGHT_SPREAD',
                'msg': f"Tight spread: {current['spread']*100:.2f}%"
            })
        
        # Deep Book
        total_depth = current['bid_depth'] + current['ask_depth']
        if total_depth > 30:
            signals.append({
                'type': 'DEEP_BOOK',
                'msg': f"Deep order book: {total_depth} levels"
            })
        
        if signals:
            confidence = self.analyzer.calculate_confidence(
                signals, self.price_history, self.volume_history
            )
            
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

# ============ MAIN BOT ============
class RailwayPolymarketBot:
    def __init__(self):
        self.discovery = MarketDiscovery()
        self.trackers = []
        self.telegram = TelegramNotifier(
            TELEGRAM_BOT_TOKEN, 
            TELEGRAM_CHAT_ID, 
            ENABLE_TELEGRAM
        )
        self.cycle = 0
        self.last_discovery = 0
        self.discovery_interval = 3600
        self.alert_history = set()
        self.start_time = time.time()
        
    def initialize(self):
        print("\n" + "="*60)
        print("🚂 POLYMARKET BOT - RAILWAY EDITION")
        print("="*60)
        print(f"Telegram: {'✅ Connected' if self.telegram.enabled else '❌ Not configured'}")
        print(f"Check Interval: {CHECK_INTERVAL}s")
        print(f"Min Confidence: {MIN_CONFIDENCE_ALERT*100:.0f}%")
        print("="*60 + "\n")
        
        if self.telegram.enabled:
            self.telegram.send_alert(
                "🚂 Bot deployed on Railway and monitoring markets", 
                "info"
            )
        
        self.refresh_markets()
    
    def refresh_markets(self):
        market_data = self.discovery.discover_markets()
        
        if not market_data:
            print("⚠️  No markets found, will retry next cycle")
            return
        
        self.trackers = [
            MarketTracker(token_id, question) 
            for token_id, question in market_data
        ]
        
        self.last_discovery = time.time()
        print(f"\n✅ Now monitoring {len(self.trackers)} markets\n")
    
    def run(self):
        self.initialize()
        
        while True:
            try:
                self.cycle += 1
                now = time.time()
                
                # Refresh market list every hour
                if now - self.last_discovery > self.discovery_interval:
                    print("\n🔄 Refreshing market list...")
                    self.refresh_markets()
                
                # Status update
                uptime = (now - self.start_time) / 3600
                print(f"\n[Cycle {self.cycle}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Uptime: {uptime:.1f}h | Tracking: {len(self.trackers)} markets")
                print("-" * 60)
                
                # Scan all markets
                signals_found = 0
                for tracker in self.trackers:
                    result = tracker.update()
                    if result and result['confidence'] >= MIN_CONFIDENCE_ALERT:
                        self.handle_signal(result)
                        signals_found += 1
                
                if signals_found == 0:
                    print("✓ No high-confidence signals this cycle")
                
                print(f"\n⏳ Next check in {CHECK_INTERVAL}s...")
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                print("Continuing in 30 seconds...")
                time.sleep(30)
    
    def handle_signal(self, result):
        # Create unique alert ID to prevent spam
        alert_id = hashlib.md5(
            f"{result['token_id']}{result['confidence']:.2f}".encode()
        ).hexdigest()[:8]
        
        # Check if we already alerted recently
        if alert_id in self.alert_history:
            return
        
        self.alert_history.add(alert_id)
        
        # Clean old alerts (keep only last 50)
        if len(self.alert_history) > 50:
            self.alert_history = set(list(self.alert_history)[-50:])
        
        # Console output
        print(f"\n{'='*60}")
        print("🚨 HIGH CONFIDENCE SIGNAL DETECTED")
        print(f"{'='*60}")
        print(f"Market: {result['question']}")
        print(f"Price: ${result['price']:.3f}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Predicted Direction: {result['predicted_direction']} "
              f"({result['direction_confidence']*100:.0f}% confidence)")
        print("\nIndicators:")
        for sig in result['signals']:
            print(f"  • {sig['msg']}")
        
        # Make decision
        decision = self.make_decision(result)
        
        if decision:
            print(f"\n💰 RECOMMENDATION: {decision['action']}")
            print(f"   Reason: {decision['reason']}")
            print(f"   Risk Level: {decision['risk']}")
            print(f"{'='*60}\n")
            
            # Send to Telegram
            if self.telegram.enabled:
                self._send_telegram_alert(result, decision)
    
    def _send_telegram_alert(self, result, decision):
        priority = "critical" if result['confidence'] > 0.8 else "high"
        
        msg = f"<b>{decision['action']}</b>\n\n"
        msg += f"<b>Market:</b> {result['question'][:70]}\n"
        msg += f"<b>Price:</b> ${result['price']:.3f}\n"
        msg += f"<b>Confidence:</b> {result['confidence']*100:.0f}%\n"
        msg += f"<b>Direction:</b> {result['predicted_direction']}\n\n"
        msg += f"<b>Reason:</b> {decision['reason']}\n"
        msg += f"<b>Risk:</b> {decision['risk']}"
        
        self.telegram.send_alert(msg, priority)
    
    def make_decision(self, result):
        confidence = result['confidence']
        direction = result['predicted_direction']
        
        if confidence >= 0.80:
            if direction == 'UP':
                return {
                    'action': '🟢 STRONG BUY YES',
                    'reason': 'Multiple strong indicators align',
                    'risk': 'LOW'
                }
            elif direction == 'DOWN':
                return {
                    'action': '🔴 STRONG BUY NO',
                    'reason': 'Multiple strong indicators align',
                    'risk': 'LOW'
                }
        
        elif confidence >= 0.65:
            if direction == 'UP':
                return {
                    'action': '🟡 MODERATE BUY YES',
                    'reason': 'Good signal confluence',
                    'risk': 'MEDIUM'
                }
            elif direction == 'DOWN':
                return {
                    'action': '🟡 MODERATE BUY NO',
                    'reason': 'Good signal confluence',
                    'risk': 'MEDIUM'
                }
        
        return {
            'action': '⚪ WATCH',
            'reason': 'Interesting pattern but wait for stronger confirmation',
            'risk': 'N/A'
        }

# ============ ENTRY POINT ============
if __name__ == "__main__":
    # Check configuration
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n⚠️  WARNING: Telegram not configured")
        print("Set these environment variables in Railway:")
        print("  - TELEGRAM_BOT_TOKEN")
        print("  - TELEGRAM_CHAT_ID")
        print("\nBot will still run but won't send notifications.\n")
        time.sleep(5)
    
    bot = RailwayPolymarketBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped by user")
        if bot.telegram.enabled:
            bot.telegram.send_alert("👋 Bot stopped", "info")
    except Exception as e:
        error_msg = f"❌ Fatal error: {str(e)}"
        print(f"\n{error_msg}")
        if bot.telegram.enabled:
            bot.telegram.send_alert(error_msg, "critical")
        raise
