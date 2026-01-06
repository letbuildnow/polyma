"""
DEBUG VERSION - POLYMARKET BOT
Lower thresholds to catch more signals
Adds detailed logging to see what's happening
"""

import requests
import time
from collections import deque
import statistics
from datetime import datetime
import os
import hashlib

# ============ ADJUSTED CONFIGURATION ============
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# LOWERED THRESHOLDS - Will catch more signals
CHECK_INTERVAL = 90  # Check more frequently (was 120)
VOLUME_SPIKE_THRESHOLD = 1.5  # Was 2.2 - now more sensitive
PRICE_MOMENTUM_THRESHOLD = 0.02  # Was 0.04 - catches smaller moves
HISTORY_WINDOW = 20  # Was 30 - faster signal generation

# More aggressive market selection
MIN_VOLUME = 20000  # Was 50k - includes smaller markets
MAX_VOLUME = 800000  # Was 500k - includes bigger markets too
MIN_LIQUIDITY = 5000  # Was 10k
TARGET_MARKET_COUNT = 10  # Was 7 - monitor more markets

ENABLE_TELEGRAM = True
MIN_CONFIDENCE_ALERT = 0.50  # Was 0.65 - alert on 50%+ confidence

# DEBUG MODE
DEBUG_MODE = True  # Shows detailed analysis every cycle

# ============ TELEGRAM ============
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
                print("✅ Telegram connected")
                return True
            else:
                print("⚠️  Telegram invalid")
                self.enabled = False
        except Exception as e:
            print(f"⚠️  Telegram failed: {e}")
            self.enabled = False
    
    def send_alert(self, message, priority="normal"):
        if not self.enabled:
            return
        
        emoji_map = {"critical": "🔴", "high": "🟠", "normal": "🟢", "info": "ℹ️"}
        formatted_msg = f"{emoji_map.get(priority, '•')} {message}"
        
        try:
            payload = {"chat_id": self.chat_id, "text": formatted_msg, "parse_mode": "HTML"}
            requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=10)
        except:
            pass

# ============ PATTERN ANALYZER ============
class AdvancedPatternAnalyzer:
    def __init__(self):
        self.pattern_weights = {
            'volume_spike': 0.35,  # Increased weight
            'momentum': 0.30,      # Increased weight
            'imbalance': 0.20,
            'spread': 0.05,
            'acceleration': 0.10
        }
    
    def calculate_confidence(self, signals, price_history, volume_history):
        scores = {}
        
        for signal in signals:
            sig_type = signal['type'].lower()
            strength = signal.get('strength', 0.5)
            
            if sig_type == 'volume_spike':
                # More generous scoring
                scores['volume_spike'] = min(1.0, (strength - 1) / 2)
            elif sig_type == 'momentum':
                scores['momentum'] = min(1.0, strength / 0.10)
            elif sig_type == 'imbalance':
                scores['imbalance'] = strength
            elif sig_type == 'tight_spread':
                scores['spread'] = 1.0
        
        if len(price_history) >= 8:
            scores['acceleration'] = self._calculate_acceleration(price_history)
        
        total_score = sum(scores.get(key, 0) * weight for key, weight in self.pattern_weights.items())
        signal_count = len(signals)
        confluence_bonus = min(0.20, (signal_count - 1) * 0.07)
        
        return min(1.0, total_score + confluence_bonus)
    
    def _calculate_acceleration(self, prices):
        recent = list(prices)[-8:]
        velocities = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        if len(velocities) >= 2:
            accelerations = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
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
        
        if len(price_history) >= 4:
            recent = list(price_history)[-4:]
            trend = sum(1 if recent[i] > recent[i-1] else -1 for i in range(1, 4))
            direction_score += trend * 0.4
        
        if direction_score > 0.3:
            return 'UP', min(1.0, abs(direction_score) / 2.5)
        elif direction_score < -0.3:
            return 'DOWN', min(1.0, abs(direction_score) / 2.5)
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
                return cached_data
        
        print("🔍 Discovering markets with LOWER thresholds...")
        
        try:
            url = f"{self.gamma_api}/markets"
            params = {'active': 'true', 'closed': 'false', 'limit': 150}
            
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
                except:
                    continue
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[:TARGET_MARKET_COUNT]
            
            print(f"✅ Found {len(selected)} markets (expanded criteria)")
            for i, m in enumerate(selected, 1):
                print(f"   {i}. Vol:${m['volume']/1000:.0f}k - {m['question'][:45]}...")
            
            result_ids = [(m['token_id'], m['question']) for m in selected]
            self.cache[cache_key] = (time.time(), result_ids)
            
            return result_ids
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return []
    
    def _calculate_score(self, volume, liquidity):
        # Prefer medium-volume markets (more volatile)
        volume_score = 1 - abs(volume - 150000) / 600000
        liquidity_score = min(liquidity / 30000, 1.0)
        return (volume_score * 0.5) + (liquidity_score * 0.5)

# ============ MARKET TRACKER ============
class MarketTracker:
    def __init__(self, token_id, question="Unknown"):
        self.token_id = token_id
        self.question = question
        self.price_history = deque(maxlen=HISTORY_WINDOW)
        self.volume_history = deque(maxlen=HISTORY_WINDOW)
        self.analyzer = AdvancedPatternAnalyzer()
        self.data_points = 0
        
    def fetch_market_data(self):
        try:
            url = f"https://clob.polymarket.com/book?token_id={self.token_id}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('bids') and data.get('asks'):
                best_bid = float(data['bids'][0]['price']) if data['bids'] else 0
                best_ask = float(data['asks'][0]['price']) if data['asks'] else 1
                mid_price = (best_bid + best_ask) / 2
                
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
        except:
            pass
        return None
    
    def update(self):
        data = self.fetch_market_data()
        if not data:
            return None
            
        self.price_history.append(data['price'])
        self.volume_history.append(data['volume'])
        self.data_points += 1
        
        # Need less history now (was 10, now 6)
        if len(self.price_history) < 6:
            return None
            
        return self.analyze_patterns(data)
    
    def analyze_patterns(self, current):
        signals = []
        
        # Volume Spike (more sensitive)
        if len(self.volume_history) >= 4:
            recent_avg = statistics.mean(list(self.volume_history)[-4:])
            if recent_avg > 0 and current['volume'] > recent_avg * VOLUME_SPIKE_THRESHOLD:
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'strength': current['volume'] / recent_avg,
                    'msg': f"Vol spike: {current['volume']/recent_avg:.1f}x"
                })
        
        # Momentum (catches smaller moves)
        if len(self.price_history) >= 6:
            short_term = list(self.price_history)[-3:]
            short_change = (short_term[-1] - short_term[0]) / max(short_term[0], 0.01)
            
            if abs(short_change) > PRICE_MOMENTUM_THRESHOLD:
                signals.append({
                    'type': 'MOMENTUM',
                    'direction': 'UP' if short_change > 0 else 'DOWN',
                    'strength': abs(short_change),
                    'msg': f"Momentum {short_change*100:+.1f}%"
                })
        
        # Order Book Imbalance (lower threshold)
        if abs(current['imbalance']) > 0.25:  # Was 0.3
            side = 'BUY' if current['imbalance'] > 0 else 'SELL'
            signals.append({
                'type': 'IMBALANCE',
                'direction': side,
                'strength': abs(current['imbalance']),
                'msg': f"Book: {abs(current['imbalance'])*100:.0f}% {side}"
            })
        
        # Spread
        if current['spread'] < 0.03:  # Was 0.02 - more lenient
            signals.append({'type': 'TIGHT_SPREAD', 'msg': "Good spread"})
        
        # Return even weak signals in debug mode
        if signals or DEBUG_MODE:
            confidence = self.analyzer.calculate_confidence(signals, self.price_history, self.volume_history)
            direction, dir_conf = self.analyzer.predict_direction(signals, self.price_history)
            
            return {
                'token_id': self.token_id,
                'question': self.question,
                'price': current['price'],
                'signals': signals,
                'confidence': confidence,
                'predicted_direction': direction,
                'direction_confidence': dir_conf,
                'data_points': self.data_points,
                'timestamp': time.time()
            }
        
        return None

# ============ DEBUG BOT ============
class DebugPolymarketBot:
    def __init__(self):
        self.discovery = MarketDiscovery()
        self.trackers = []
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ENABLE_TELEGRAM)
        self.cycle = 0
        self.last_discovery = 0
        self.discovery_interval = 3600
        self.alert_history = set()
        
    def initialize(self):
        print("\n" + "="*60)
        print("🐛 DEBUG MODE - POLYMARKET BOT")
        print("="*60)
        print(f"Volume Threshold: {VOLUME_SPIKE_THRESHOLD}x (lowered from 2.2x)")
        print(f"Momentum Threshold: {PRICE_MOMENTUM_THRESHOLD*100:.1f}% (lowered from 4%)")
        print(f"Min Confidence: {MIN_CONFIDENCE_ALERT*100:.0f}% (lowered from 65%)")
        print(f"Check Interval: {CHECK_INTERVAL}s")
        print(f"Markets: {TARGET_MARKET_COUNT} (increased from 7)")
        print("="*60 + "\n")
        
        if self.telegram.enabled:
            self.telegram.send_alert("🐛 Debug bot started - Lower thresholds active", "info")
        
        self.refresh_markets()
    
    def refresh_markets(self):
        market_data = self.discovery.discover_markets()
        if not market_data:
            return
        
        self.trackers = [MarketTracker(tid, q) for tid, q in market_data]
        self.last_discovery = time.time()
        print(f"\n✅ Tracking {len(self.trackers)} markets\n")
    
    def run(self):
        self.initialize()
        
        while True:
            try:
                self.cycle += 1
                now = time.time()
                
                if now - self.last_discovery > self.discovery_interval:
                    print("\n🔄 Refreshing markets...")
                    self.refresh_markets()
                
                print(f"\n[Cycle {self.cycle}] {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 60)
                
                # Track statistics
                total_signals = 0
                high_conf_signals = 0
                
                for i, tracker in enumerate(self.trackers):
                    result = tracker.update()
                    
                    if result:
                        total_signals += 1
                        
                        # Show ALL signals in debug mode
                        if DEBUG_MODE and result['signals']:
                            print(f"\n📊 Market {i+1}: {result['question'][:40]}...")
                            print(f"   Price: {result['price']:.3f} | Confidence: {result['confidence']*100:.0f}%")
                            print(f"   Data Points: {result['data_points']} | Signals: {len(result['signals'])}")
                            for sig in result['signals']:
                                print(f"   • {sig['msg']}")
                        
                        # Handle high confidence signals
                        if result['confidence'] >= MIN_CONFIDENCE_ALERT:
                            high_conf_signals += 1
                            self.handle_signal(result)
                
                print(f"\n📈 Cycle Summary:")
                print(f"   Signals detected: {total_signals}")
                print(f"   High confidence: {high_conf_signals}")
                print(f"   Next check: {CHECK_INTERVAL}s")
                
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(30)
    
    def handle_signal(self, result):
        alert_id = hashlib.md5(f"{result['token_id']}{result['confidence']:.1f}".encode()).hexdigest()[:8]
        
        if alert_id in self.alert_history:
            return
        
        self.alert_history.add(alert_id)
        
        print(f"\n{'='*60}")
        print("🚨 HIGH CONFIDENCE SIGNAL")
        print(f"{'='*60}")
        print(f"{result['question']}")
        print(f"Price: ${result['price']:.3f}")
        print(f"Confidence: {result['confidence']*100:.0f}%")
        print(f"Direction: {result['predicted_direction']}")
        
        decision = self.make_decision(result)
        
        if decision:
            print(f"\n💰 {decision['action']}")
            print(f"Reason: {decision['reason']}")
            print(f"{'='*60}\n")
            
            if self.telegram.enabled:
                self._send_telegram_alert(result, decision)
    
    def _send_telegram_alert(self, result, decision):
        priority = "critical" if result['confidence'] > 0.75 else "high"
        
        msg = f"<b>{decision['action']}</b>\n\n"
        msg += f"<b>Market:</b> {result['question'][:70]}\n"
        msg += f"<b>Price:</b> ${result['price']:.3f}\n"
        msg += f"<b>Confidence:</b> {result['confidence']*100:.0f}%\n"
        msg += f"<b>Direction:</b> {result['predicted_direction']}\n\n"
        msg += f"<b>Reason:</b> {decision['reason']}"
        
        self.telegram.send_alert(msg, priority)
    
    def make_decision(self, result):
        confidence = result['confidence']
        direction = result['predicted_direction']
        
        if confidence >= 0.75:
            if direction == 'UP':
                return {'action': '🟢 STRONG BUY YES', 'reason': 'Strong indicators'}
            elif direction == 'DOWN':
                return {'action': '🔴 STRONG BUY NO', 'reason': 'Strong indicators'}
        
        elif confidence >= 0.50:
            if direction == 'UP':
                return {'action': '🟡 MODERATE BUY YES', 'reason': 'Good signals'}
            elif direction == 'DOWN':
                return {'action': '🟡 MODERATE BUY NO', 'reason': 'Good signals'}
        
        return None

# ============ RUN ============
if __name__ == "__main__":
    bot = DebugPolymarketBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Stopped")
        if bot.telegram.enabled:
            bot.telegram.send_alert("👋 Debug bot stopped", "info")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
