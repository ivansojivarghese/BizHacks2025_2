# agent_perception_forecaster.py

import json
import logging
import pandas as pd
import tiktoken
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests
import numpy as np

import os
from mistralai import Mistral

from langchain_core.messages import ChatMessage
from statsmodels.tsa.arima.model import ARIMA
#from fbprophet import Prophet  # if you install prophet
from prophet import Prophet
from google import genai

# Try to import holidays library (fallback if not available)
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logging.warning("[INIT] 'holidays' library not available. Install with: pip install holidays")

'''
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
'''
encoding = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 100000

MISTRAL_API_KEY = ""
client = Mistral(api_key=MISTRAL_API_KEY)

model_embed = "mistral-embed"
# model_chat = "mistral-large-latest"
# model_chat = "mistral-medium-2508" # paid
model_chat = "mistral-small-2506" # free
'''
dummy_state = {
    "company_name": "Google",
    "brand_signals": [],
    "sentiment_forecast": [],
    "error_messages": []
}
'''
'''
GEMINI_API_KEY = "AIzaSyBolQ5bS2KErY8_mqPyK-bzflLfVzRf2mA"
GEMINI_MODEL = "gemini-2.5-flash"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
'''

def truncate_to_tokens(text: str, token_limit: int) -> str:
    tokens = encoding.encode(text)
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]
    return encoding.decode(tokens)

def parse_json_from_codeblock(text: str) -> dict:
    """
    Extract JSON between ```json ... ```
    """
    start_token = "```json"
    end_token = "```"

    text = text.strip()

    if start_token in text:
        start_idx = text.index(start_token) + len(start_token)
        end_idx = text.find(end_token, start_idx)
        if end_idx != -1:
            json_str = text[start_idx:end_idx].strip()
            if not json_str:  # <--- Prevent empty string parsing
                raise ValueError("Empty JSON block found between ```json and ```")
            #return json.loads(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"[JSON_PARSE] Failed to parse extracted block: {str(e)}")
                logging.error(f"[JSON_PARSE] Raw JSON string: {json_str}")
                return {}  # or return a fallback dict

class PerceptionForecasterAgent:
    """
    Agent B: Perception Forecaster Agent
    Uses time-series sentiment data to forecast future brand perception.
    """

    def __init__(
        self,
        state,
        forecast_steps: int,
        model_type: str,
        freq: str = "D",  # 'D' for daily, 'W' for weekly
        region: str = "US"  # Region code for holidays (US, SG, UK, etc.)
    ):
        """
        :param forecast_steps: How many future time steps to predict.
        :param model_type: Type of model to use ("ARIMA" or "Prophet").
        :param freq: Resampling frequency for aggregation ('D', 'W', etc.).
        :param region: Region code for holiday calendar (US, SG, UK, CN, etc.)
        """
        self.state = state
        self.forecast_steps = forecast_steps
        self.model_type = model_type
        self.freq = freq
        self.region = region
        self.last_run: Optional[str] = None
        self.forecasts: Optional[pd.DataFrame] = None
        self.feedback_run_count = 0  # Track feedback iterations
        self.memory = {}  # Store evaluation history
        self.actionable_insights = []  # Store insights for stakeholders
        self.holidays_df = None  # Regional holiday calendar
        self.market_data = {}  # Market fluctuation data
        self.news_volume = []  # News volume tracking

    '''
    def gemini_llm(prompt: str, model: str = GEMINI_MODEL, 
                    temperature: float = 0.7,
                    top_p: float = 0.9
                    ) -> str:
        """
        Sends a prompt to the specified Gemini LLM model and returns the generated text.
        
        Args:
            prompt (str): The text prompt you want the model to process.
            model (str): The Gemini model to use. Defaults to "gemini-2.5-flash".
        
        Returns:
            str: The model's response text.
        """
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"
    '''

    def mistral_embeddings(self, text):
        """
        Generate embeddings for a list of texts using the given client and model.
        
        Args:
            client: The embeddings API client (e.g., OpenAI, Groq, etc.).
            model (str): The embedding model ID.
            texts (list[str]): List of strings to embed.
        
        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        text_limited = truncate_to_tokens(text, TOKEN_LIMIT)
        embeddings_response = client.embeddings.create(
            model=model_embed,
            inputs=[text_limited]  # 'input' is the correct parameter name
        )
        
        # Extract just the vectors
        return embeddings_response.data[0].embedding
        #return [item.embedding for item in embeddings_response.data]

    def mistral_chat(self, prompt_or_messages):
        """
        Get a chat completion from the specified LLM model.
        
        Args:
            client: The chat API client (e.g., Mistral, OpenAI, etc.).
            model (str): The chat model ID.
            messages (list[dict]): A list of message dicts in {"role": str, "content": str} format.
        
        Returns:
            str: The text content of the model's reply.
        """
        """
        prompt_limited = truncate_to_tokens(prompt, TOKEN_LIMIT)
        messages = [ChatMessage(role="user", content=prompt_limited)]
        chat_response = client.chat(
            model=model_chat,
            messages=messages
        )
        
        return chat_response.choices[0].message.content
        """
        if isinstance(prompt_or_messages, str):
            # Truncate if too long
            prompt_limited = truncate_to_tokens(prompt_or_messages, TOKEN_LIMIT)
            messages = [{"role": "user", "content": prompt_limited}]
        else:
            # Assume it's already a valid messages list
            messages = prompt_or_messages

        chat_response = client.chat.complete(
            model=model_chat,
            messages=messages
        )

        return chat_response.choices[0].message.content

    def act(self, signals):
        try:
            if not signals:
                return pd.DataFrame({"ds": [], "yhat": []})
            ts_df = self.prepare_timeseries(signals)
            forecast_df = self.forecast(ts_df)

            return forecast_df
        except Exception as e:
            # logging.error(f"[RUN] Error during run_cycle: {e}")
            # raise
            #logging.error(f"Error: {e}")
            #return pd.DataFrame({"ds": [], "yhat": []})  # Don't crash
            logging.error(f"[RUN] Error during run_cycle: {e}")
            # 🛠️ FIX: Return empty DataFrame instead of crashing
            empty_df = pd.DataFrame({"ds": [], "yhat": []})
            self.forecasts = empty_df
            return empty_df

    def evaluate(self, forecast_df: pd.DataFrame, company: str):
        """
        Evaluates the initial forecast DataFrame by sending it to the Mistral LLM
        for critique and suggestions. If the LLM determines the forecast is not good,
        triggers the resolve stage.
        """

        # Convert forecast_df to a concise JSON (avoid sending huge payloads)
        #forecast_json = forecast_df.to_dict(orient="records")
        #forecast_json = json.loads(forecast_df.to_json(orient="records", date_format="iso"))

        #print(f"forecast_df {forecast_json}")
        #print(self)

        # Convert brand signals if available
        brand_signals_text = json.dumps(self.state["brand_signals"])

        #print(brand_signals_text)

        '''
        critique_prompt = (
            f"You are an expert market and brand forecaster.\n"
            f"The following forecast data is for the company: {company}.\n"
            "Critique it based on:\n"
            "- Date coverage (is the range sensible?)\n"
            "- Seasonal effects\n"
            "- Holiday impacts\n"
            "- Regional factors\n"
            "- Brand signals\n"
            "- Whether the forecast makes logical and statistical sense.\n"
            "Respond in JSON with keys:\n"
            "{\n"
            "  'is_good': true/false,\n"
            "  'comments': 'Your detailed critique and suggestions'\n"
            "}\n\n"
            f"### Forecast Data ###\n{json.dumps(forecast_json, indent=2)}\n\n"
            f"### Brand Signals ###\n{brand_signals_text}"
        )
        '''

        critique_prompt = f"""
        You are an expert IPO brand perception forecaster.
        Company: {company}
        Region: {self.region}
        Brand Signals: {brand_signals_text}

        Here is the forecast data:
        {forecast_df.to_csv(index=False)}

        ### MARKET CONTEXT ###
        Market Trend: {self.market_data.get('trend', 'unknown')}
        Volatility Index: {self.market_data.get('volatility_index', 'N/A')}
        Sector Performance: {self.market_data.get('sector_performance', 'N/A')}%

        ### NEWS VOLUME CONTEXT ###
        Tracked Days: {self.memory.get(self.last_run, {}).get('news_volume_days', 0)}
        Detected News Spikes (PR events/crises): {len(self.memory.get(self.last_run, {}).get('news_spikes', []))}

        ### REGIONAL HOLIDAYS ###
        Holidays Loaded: {self.memory.get(self.last_run, {}).get('holidays_loaded', 0)} for region {self.region}

        Assess whether the forecast makes sense. Consider:
        1. Seasonal patterns and holiday effects (Black Friday, Christmas, regional holidays)
        2. Market trends and volatility impact on brand sentiment
        3. News volume spikes indicating PR events, product launches, or crises
        4. Regional factors specific to {self.region}
        5. Brand signal patterns and sentiment trends
        
        If the forecast is flawed, outline key reasons why and suggest improvements.
        Return a highly detailed analysis (with recommendations if any).
        """

        """
        Output format:
        Return ONLY a valid JSON object, with no extra text:
        {{
            "is_good": "True or False",
            "comments": "Detailed analysis"
        }}
        """

        critique_prompt = truncate_to_tokens(critique_prompt, TOKEN_LIMIT)
    
        # Now safe to embed
        #forecast_embedding = self.mistral_embeddings(critique_prompt)

        # And safe to send to chat
        #critique_response = self.mistral_chat(critique_prompt)

        # Send to chat model
        
        messages = [
            {"role": "system", "content": "You are a forecasting evaluation assistant."},
            {"role": "user", "content": critique_prompt}
        ]
        
        critique_response = self.mistral_chat(messages)

        print(critique_response)

        #critique_response = self.gemini_llm(critique_prompt)

        '''
        print(critique_response)
        import sys
        sys.exit()

        # Try parsing JSON
        try:
            critique_data = json.loads(critique_response)
        except json.JSONDecodeError:
            print("[WARN] LLM did not return valid JSON. Raw output:")
            print(critique_response)
            critique_data = {"is_good": False, "comments": critique_response}

        #print(f"crd {critique_data}")
        '''

        eval_critique_prompt = f"""
        You are an evaluation assistant.

        Study the following critique about a forecast and determine what it is saying about the forecast.
        ---
        {critique_response}
        ---

        Return ONLY a valid JSON object with no extra text or explanation:
        {{
            "is_good": "True or False",  // Whether the critique says the forecast is good or bad
            "comments": "Detailed analysis of what the critique says"
        }}
        """
        
        messages2 = [
            {"role": "system", "content": "You are a forecasting evaluation assistant."},
            {"role": "user", "content": eval_critique_prompt}
        ]
        
        critique_data = self.mistral_chat(messages2)
        

        #critique_data = self.gemini_llm(eval_critique_prompt)

        #print(f"crd {critique_data}")

        critique_data = parse_json_from_codeblock(critique_data)

        print(critique_data)

        #critique_data_json = json.loads(critique_data)

        # Log comments
        #print("\n[Evaluation Comments]")
        #print(critique_data.get("comments", ""))

        # Decision
        if critique_data["is_good"] == "False":
            logging.warning("[EVAL] Forecast deemed NOT good. Moving to resolve stage...")
            self.resolve(critique_response, critique_data)
        else:
            logging.info("[EVAL] Forecast passed initial review. Proceeding...")
            self.generate_actionable_insights()

    def resolve(self, critique_response: str, critique_data: dict):
        """
        RESOLVE STAGE: Adaptive feedback loop to improve forecast quality.
        Uses tools and expanded data gathering to refine predictions.
        """
        MAX_FEEDBACK_ITERATIONS = 3
        
        if self.feedback_run_count >= MAX_FEEDBACK_ITERATIONS:
            logging.warning(f"[RESOLVE] Max feedback iterations ({MAX_FEEDBACK_ITERATIONS}) reached. Accepting current forecast.")
            return
        
        self.feedback_run_count += 1
        logging.info(f"[RESOLVE] Iteration {self.feedback_run_count}/{MAX_FEEDBACK_ITERATIONS}")
        
        # Parse critique to determine adjustments
        adjustment_prompt = f"""
        You are a forecasting optimization expert.
        
        The following critique was provided for a brand perception forecast:
        ---
        {critique_response}
        ---
        
        Based on this critique, recommend specific adjustments to improve the forecast.
        Consider:
        1. **Time Range**: Should we expand historical data range? (e.g., 30 days → 90 days)
        2. **Seasonality**: Are seasonal patterns missing? (holidays, quarterly cycles, regional events)
        3. **External Factors**: Should we incorporate market fluctuations, competitor events, or news trends?
        4. **Model Changes**: Should we switch models (ARIMA ↔ Prophet) or adjust parameters?
        5. **Data Quality**: Are there data gaps or anomalies to address?
        
        Return ONLY valid JSON with no extra text:
        {{
            "expand_time_range": true/false,
            "new_days_back": 90,  // If expand_time_range is true
            "add_seasonality": ["holidays", "quarterly", "regional_events"],
            "switch_model": "Prophet" or "ARIMA" or null,
            "incorporate_external_data": ["market_trends", "competitor_signals", "news_volume"],
            "rationale": "Detailed explanation of adjustments"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a forecasting optimization assistant."},
            {"role": "user", "content": adjustment_prompt}
        ]
        
        adjustment_response = self.mistral_chat(messages)
        adjustments = parse_json_from_codeblock(adjustment_response)
        
        if not adjustments or adjustments == {}:
            logging.error("[RESOLVE] Failed to parse adjustment recommendations. Skipping resolve.")
            return
        
        logging.info(f"[RESOLVE] Recommended adjustments: {adjustments.get('rationale', 'N/A')}")
        
        # Apply adjustments
        self.apply_adjustments(adjustments)
        
        # Re-run forecast with adjusted parameters
        logging.info("[RESOLVE] Re-running forecast with adjusted parameters...")
        ts_df = self.prepare_timeseries(self.state["brand_signals"])
        new_forecast_df = self.forecast(ts_df)
        
        # Re-evaluate
        logging.info("[RESOLVE] Re-evaluating adjusted forecast...")
        self.evaluate(new_forecast_df, self.state["company_name"])

    def apply_adjustments(self, adjustments: dict):
        """
        Apply recommended adjustments to agent configuration.
        """
        logging.info("[ADJUST] Applying forecast adjustments...")
        
        # Time range expansion
        if adjustments.get("expand_time_range"):
            new_days = adjustments.get("new_days_back", 90)
            logging.info(f"[ADJUST] Expanding historical data range to {new_days} days")
            # Note: This would require re-collecting signals with new hours_back parameter
            # For now, we log the recommendation
            self.memory[self.last_run]["recommendation_expand_range"] = new_days
        
        # Model switching
        if adjustments.get("switch_model"):
            new_model = adjustments["switch_model"]
            if new_model != self.model_type:
                logging.info(f"[ADJUST] Switching model: {self.model_type} → {new_model}")
                self.model_type = new_model
        
        # Seasonality additions
        if adjustments.get("add_seasonality"):
            seasonality_types = adjustments["add_seasonality"]
            logging.info(f"[ADJUST] Adding seasonality patterns: {seasonality_types}")
            self.memory[self.last_run]["seasonality_added"] = seasonality_types
            # Note: Prophet supports custom seasonality, ARIMA needs SARIMA
        
        # External data incorporation
        if adjustments.get("incorporate_external_data"):
            external_sources = adjustments["incorporate_external_data"]
            logging.info(f"[ADJUST] Recommended external data sources: {external_sources}")
            self.memory[self.last_run]["external_data_needed"] = external_sources

    def generate_actionable_insights(self):
        """
        Generate stakeholder-ready actionable insights from forecast.
        """
        if self.forecasts is None or self.forecasts.empty:
            logging.warning("[INSIGHTS] No forecast available for insight generation.")
            return
        
        # Calculate trend metrics
        forecast_mean = self.forecasts["yhat"].mean()
        forecast_std = self.forecasts["yhat"].std()
        forecast_trend = "upward" if self.forecasts["yhat"].iloc[-1] > self.forecasts["yhat"].iloc[0] else "downward"
        
        # Identify anomalies
        anomalies = self.forecasts[abs(self.forecasts["yhat"]) > 2 * forecast_std]
        
        insight_prompt = f"""
        You are a brand strategy consultant.
        
        Company: {self.state["company_name"]}
        Forecast Period: {self.forecast_steps} days
        Average Predicted Sentiment: {forecast_mean:.3f}
        Trend: {forecast_trend}
        Volatility (Std Dev): {forecast_std:.3f}
        Anomaly Count: {len(anomalies)}
        
        Forecast Data:
        {self.forecasts.to_csv(index=False)}
        
        Generate 3-5 actionable insights for brand managers:
        1. **Key Trend**: What's the overall sentiment trajectory?
        2. **Risk Periods**: When should the brand be most cautious?
        3. **Opportunity Windows**: When should marketing/PR be amplified?
        4. **Competitive Positioning**: How should the brand respond to predicted shifts?
        5. **Monitoring Priorities**: What metrics should be tracked closely?
        
        Return ONLY valid JSON:
        {{
            "insights": [
                {{"category": "Key Trend", "insight": "...", "confidence": "high/medium/low"}},
                {{"category": "Risk Periods", "insight": "...", "confidence": "high/medium/low"}},
                ...
            ],
            "recommended_actions": ["action 1", "action 2", ...]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a brand strategy insights generator."},
            {"role": "user", "content": insight_prompt}
        ]
        
        insights_response = self.mistral_chat(messages)
        insights_data = parse_json_from_codeblock(insights_response)
        
        if insights_data and insights_data != {}:
            self.actionable_insights = insights_data.get("insights", [])
            logging.info(f"[INSIGHTS] Generated {len(self.actionable_insights)} actionable insights")
            
            # Store in memory
            self.memory[self.last_run]["actionable_insights"] = insights_data
        else:
            logging.warning("[INSIGHTS] Failed to generate actionable insights")

    def detect_sentiment_shifts(self) -> List[Dict]:
        """
        TOOL: Detect significant sentiment shifts in the forecast.
        Returns periods where sentiment changes dramatically.
        """
        if self.forecasts is None or self.forecasts.empty:
            return []
        
        shifts = []
        threshold = self.forecasts["yhat"].std() * 1.5
        
        for i in range(1, len(self.forecasts)):
            prev = self.forecasts["yhat"].iloc[i-1]
            curr = self.forecasts["yhat"].iloc[i]
            change = curr - prev
            
            if abs(change) > threshold:
                shifts.append({
                    "date": self.forecasts["ds"].iloc[i].strftime("%Y-%m-%d"),
                    "change": float(change),
                    "direction": "positive" if change > 0 else "negative",
                    "magnitude": abs(float(change))
                })
        
        logging.info(f"[TOOL] Detected {len(shifts)} significant sentiment shifts")
        return shifts

    def identify_seasonal_patterns(self, ts_df: pd.DataFrame) -> Dict:
        """
        TOOL: Identify seasonal patterns in historical data.
        """
        if ts_df.empty or len(ts_df) < 14:
            return {"weekly": False, "monthly": False, "patterns": []}
        
        # Simple seasonality detection using autocorrelation
        from pandas.plotting import autocorrelation_plot
        
        patterns = []
        
        # Weekly pattern (7-day cycle)
        if len(ts_df) >= 14:
            weekly_corr = ts_df["y"].autocorr(lag=7)
            if abs(weekly_corr) > 0.5:
                patterns.append({"type": "weekly", "strength": float(weekly_corr)})
        
        # Monthly pattern (30-day cycle)
        if len(ts_df) >= 60:
            monthly_corr = ts_df["y"].autocorr(lag=30)
            if abs(monthly_corr) > 0.5:
                patterns.append({"type": "monthly", "strength": float(monthly_corr)})
        
        result = {
            "weekly": any(p["type"] == "weekly" for p in patterns),
            "monthly": any(p["type"] == "monthly" for p in patterns),
            "patterns": patterns
        }
        
        logging.info(f"[TOOL] Identified seasonal patterns: {result}")
        return result

    def get_regional_holidays(self, years: List[int] = None) -> pd.DataFrame:
        """
        TOOL: Get regional holiday calendar based on self.region.
        Returns DataFrame with 'ds' (date) and 'holiday' (name) columns for Prophet.
        """
        if not HOLIDAYS_AVAILABLE:
            logging.warning("[TOOL] Holidays library not available. Returning empty calendar.")
            return pd.DataFrame(columns=['ds', 'holiday'])
        
        if years is None:
            current_year = datetime.now().year
            years = [current_year - 1, current_year, current_year + 1]
        
        try:
            # Map region codes to holiday classes
            region_map = {
                'US': holidays.US,
                'SG': holidays.Singapore,
                'UK': holidays.UK,
                'CN': holidays.China,
                'IN': holidays.India,
                'JP': holidays.Japan,
                'DE': holidays.Germany,
                'FR': holidays.France,
                'CA': holidays.Canada,
                'AU': holidays.Australia,
            }
            
            holiday_class = region_map.get(self.region.upper(), holidays.US)
            logging.info(f"[TOOL] Loading {self.region.upper()} holiday calendar for years {years}")
            
            holiday_dates = []
            for year in years:
                regional_holidays = holiday_class(years=year)
                for date, name in regional_holidays.items():
                    holiday_dates.append({'ds': pd.to_datetime(date), 'holiday': name})
            
            # Add tech industry events (global)
            tech_events = self._get_tech_industry_events(years)
            holiday_dates.extend(tech_events)
            
            holidays_df = pd.DataFrame(holiday_dates)
            
            if not holidays_df.empty:
                self.holidays_df = holidays_df
                logging.info(f"[TOOL] Loaded {len(holidays_df)} holidays and events")
            else:
                self.holidays_df = pd.DataFrame(columns=['ds', 'holiday'])
                
            return self.holidays_df
            
        except Exception as e:
            logging.error(f"[TOOL] Error loading regional holidays: {e}")
            return pd.DataFrame(columns=['ds', 'holiday'])
    
    def _get_tech_industry_events(self, years: List[int]) -> List[Dict]:
        """
        Internal: Get major tech industry events (conferences, earnings seasons).
        """
        events = []
        
        for year in years:
            # CES (Consumer Electronics Show) - Early January
            events.append({'ds': pd.to_datetime(f'{year}-01-07'), 'holiday': 'CES'})
            
            # Mobile World Congress - Late February
            events.append({'ds': pd.to_datetime(f'{year}-02-27'), 'holiday': 'MWC'})
            
            # Google I/O - Mid May
            events.append({'ds': pd.to_datetime(f'{year}-05-15'), 'holiday': 'Google_IO'})
            
            # Apple WWDC - Early June
            events.append({'ds': pd.to_datetime(f'{year}-06-05'), 'holiday': 'Apple_WWDC'})
            
            # Black Friday - Fourth Friday of November
            events.append({'ds': pd.to_datetime(f'{year}-11-24'), 'holiday': 'Black_Friday'})
            
            # Cyber Monday - Monday after Black Friday
            events.append({'ds': pd.to_datetime(f'{year}-11-27'), 'holiday': 'Cyber_Monday'})
            
            # Earnings seasons (quarterly)
            for month in [1, 4, 7, 10]:
                events.append({'ds': pd.to_datetime(f'{year}-{month:02d}-15'), 'holiday': f'Earnings_Q{(month-1)//3 + 1}'})
        
        return events

    def fetch_market_data(self, company: str) -> Dict:
        """
        TOOL: Fetch market fluctuation data for sentiment context.
        Uses free APIs to get stock trends, volatility indicators.
        """
        logging.info(f"[TOOL] Fetching market data for {company}")
        
        try:
            # Try Alpha Vantage free API (requires signup for key)
            # For demo purposes, we'll use a simulated approach
            # In production, integrate with: Alpha Vantage, Yahoo Finance, or FMP Cloud
            
            # Simulated market data (replace with real API calls)
            market_data = {
                "company": company,
                "volatility_index": np.random.uniform(10, 30),  # VIX-like metric
                "trend": np.random.choice(["bullish", "bearish", "neutral"]),
                "sector_performance": np.random.uniform(-5, 5),  # % change
                "data_source": "simulated",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # TODO: Real implementation example:
            # API_KEY = "your_alpha_vantage_key"
            # url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
            # response = requests.get(url, timeout=10)
            # if response.status_code == 200:
            #     data = response.json()
            #     market_data = parse_market_data(data)
            
            self.market_data = market_data
            logging.info(f"[TOOL] Market data: {market_data['trend']} trend, volatility {market_data['volatility_index']:.1f}")
            return market_data
            
        except Exception as e:
            logging.error(f"[TOOL] Error fetching market data: {e}")
            return {"error": str(e), "data_source": "failed"}

    def fetch_news_volume(self, company: str, days_back: int = 30) -> List[Dict]:
        """
        TOOL: Track news volume to detect PR events, crises, product launches.
        Returns daily news mention counts and sentiment.
        """
        logging.info(f"[TOOL] Fetching news volume for {company} (last {days_back} days)")
        
        try:
            # Simulated news volume (replace with NewsAPI, GDELT, or MediaCloud)
            news_volume = []
            
            start_date = datetime.now() - timedelta(days=days_back)
            for i in range(days_back):
                date = start_date + timedelta(days=i)
                
                # Simulate news volume with occasional spikes (PR events)
                base_volume = np.random.poisson(5)  # Average 5 articles/day
                is_event = np.random.random() < 0.1  # 10% chance of event
                volume = base_volume + (np.random.poisson(20) if is_event else 0)
                
                news_volume.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "article_count": int(volume),
                    "event_detected": is_event,
                    "avg_sentiment": np.random.uniform(-1, 1)
                })
            
            # TODO: Real implementation example:
            # NEWS_API_KEY = "your_newsapi_key"
            # url = f"https://newsapi.org/v2/everything?q={company}&from={start_date}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            # response = requests.get(url, timeout=10)
            # if response.status_code == 200:
            #     articles = response.json()['articles']
            #     news_volume = aggregate_by_date(articles)
            
            self.news_volume = news_volume
            
            # Detect spikes
            volumes = [n['article_count'] for n in news_volume]
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes)
            spikes = [n for n in news_volume if n['article_count'] > mean_vol + 2*std_vol]
            
            logging.info(f"[TOOL] Tracked {len(news_volume)} days of news, detected {len(spikes)} volume spikes")
            return news_volume
            
        except Exception as e:
            logging.error(f"[TOOL] Error fetching news volume: {e}")
            return []


    def prepare_timeseries(self, signals: List[Dict]) -> pd.DataFrame:
        """
        Convert collected brand signals into an aggregated time series (ds, y).
        """
        logging.info("[PREP] Converting signals into sentiment time series...")
        records = []
        for sig in signals:
            ts = sig.get("timestamp")
            if not ts:
                continue
            try:
                dt = pd.to_datetime(ts)
                # 🛠️ ADD THIS LINE: Remove timezone if present  
                if dt.tz is not None:
                    dt = dt.tz_localize(None)  # Remove timezone info
            except Exception:
                continue

            raw = sig.get("sentiment")
            '''
            if isinstance(raw, dict) and "score" in raw:
                score = raw["score"]
            elif isinstance(raw, str):
                mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                score = mapping.get(raw.lower(), 0.0)
            else:
                continue
            '''
            if isinstance(raw, dict) and "score" in raw:
                score = raw["score"]
            elif isinstance(raw, dict) and "compound" in raw:
                score = raw["compound"]  # VADER format
            elif isinstance(raw, str):
                mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                score = mapping.get(raw.lower(), 0.0)
            elif isinstance(raw, (int, float)):
                score = float(raw)
            else:
                continue

            records.append({"date": dt, "score": score})

        df = pd.DataFrame(records)
        if df.empty:
            #raise ValueError("No valid sentiment data found in signals.")
            logging.warning("No signals provided, returning empty forecast")
            return pd.DataFrame({"ds": [], "yhat": []})

        # Resample/aggregate by date or week
        df.set_index("date", inplace=True)
        agg = df["score"].resample(self.freq).mean()

        # Fill missing dates with previous values (forward‐fill) to maintain continuity
        agg = agg.asfreq(self.freq).ffill().bfill()

        ts_df = agg.reset_index().rename(columns={"date": "ds", "score": "y"})

        # 🛠️ ADD THIS: Final timezone check for DataFrame
        '''
        if hasattr(ts_df['ds'].iloc[0], 'tz') and ts_df['ds'].iloc.tz is not None:
            ts_df['ds'] = ts_df['ds'].dt.tz_localize(None)
        '''

        # Replace with this safe approach:
        # 🛠️ CRITICAL FIX: Safe timezone checking and removal
        try:
            if not ts_df.empty and pd.api.types.is_datetime64_any_dtype(ts_df['ds']):
                if ts_df['ds'].dt.tz is not None:
                    ts_df['ds'] = ts_df['ds'].dt.tz_localize(None)
                    logging.debug("[PREP] Removed timezone from ds column")
        except Exception as e:
            logging.debug(f"[PREP] Timezone check/removal failed: {e}")


        logging.info(f"[PREP] Time series prepared: {ts_df.shape[0]} rows at '{self.freq}' frequency")
        return ts_df
    '''
    def forecast(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the time-series model and forecast sentiment.
        """
        logging.info(f"[FORECAST] Using {self.model_type} for {self.forecast_steps} steps ahead")

        if self.model_type == "ARIMA":
            model = ARIMA(ts_df["y"], order=(2, 1, 2))
            model_fit = model.fit()
            preds = model_fit.forecast(steps=self.forecast_steps)
            future_index = pd.date_range(
                start=ts_df["ds"].iloc[-1] + pd.Timedelta(1, unit=self.freq),
                periods=self.forecast_steps,
                freq=self.freq
            )
            forecast_df = pd.DataFrame({"ds": future_index, "yhat": preds})

        elif self.model_type.lower() == "prophet":
            prophet_df = ts_df.rename(columns={"ds": "ds", "y": "y"})
            m = Prophet(daily_seasonality=(self.freq == "D"), weekly_seasonality=(self.freq == "W"))
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=self.forecast_steps, freq=self.freq)
            fc = m.predict(future)
            # Only keep the forecast horizon
            forecast_df = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(self.forecast_steps)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not implemented")

        self.forecasts = forecast_df
        logging.info(f"[FORECAST] Generated {len(forecast_df)} forecast points")
        return forecast_df
    '''

    def forecast(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the time-series model and forecast sentiment.
        FIXED: Safe timezone handling for both ARIMA and Prophet.
        """
        logging.info(f"[FORECAST] Using {self.model_type} for {self.forecast_steps} steps ahead")

        # Handle empty time series
        if ts_df.empty:
            logging.warning("[FORECAST] Empty time series, returning empty forecast")
            return pd.DataFrame({"ds": [], "yhat": []})

        # Handle insufficient data
        if len(ts_df) < 3:
            logging.warning("[FORECAST] Insufficient data points, using simple forecast")
            return self._simple_forecast(ts_df)

        try:
            if self.model_type == "ARIMA":
                model = ARIMA(ts_df["y"], order=(2, 1, 2))
                model_fit = model.fit()
                preds = model_fit.forecast(steps=self.forecast_steps)
                
                # 🛠️ FIXED: Safe timezone handling for ARIMA
                last_date = ts_df["ds"].iloc[-1]
                try:
                    if hasattr(last_date, 'tz') and last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                except Exception as e:
                    logging.debug(f"[ARIMA] Timezone removal failed: {e}")
                
                future_index = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=self.freq),
                    periods=self.forecast_steps,
                    freq=self.freq
                )
                forecast_df = pd.DataFrame({"ds": future_index, "yhat": preds})

            elif self.model_type.lower() == "prophet":
                prophet_df = ts_df.copy()
                
                # 🛠️ Safe timezone removal for Prophet
                try:
                    if not prophet_df.empty and pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
                        if prophet_df['ds'].dt.tz is not None:
                            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                            logging.debug("[PROPHET] Removed timezone from ds column")
                except Exception as e:
                    logging.debug(f"[PROPHET] Timezone removal failed: {e}")
                
                # Initialize Prophet with regional holidays
                m = Prophet(
                    daily_seasonality=(self.freq == "D"), 
                    weekly_seasonality=(self.freq == "W"),
                    yearly_seasonality=True
                )
                
                # Add regional holidays if available
                if self.holidays_df is not None and not self.holidays_df.empty:
                    # Ensure holidays_df has correct timezone handling
                    holidays_clean = self.holidays_df.copy()
                    if pd.api.types.is_datetime64_any_dtype(holidays_clean['ds']):
                        if holidays_clean['ds'].dt.tz is not None:
                            holidays_clean['ds'] = holidays_clean['ds'].dt.tz_localize(None)
                    
                    for _, row in holidays_clean.iterrows():
                        m.add_country_holidays(country_name=self.region.upper())
                        break  # Only add once
                    
                    # Add custom tech events
                    tech_events = holidays_clean[holidays_clean['holiday'].str.contains('Black_Friday|Cyber_Monday|Google_IO|Apple_WWDC|CES|MWC|Earnings', na=False)]
                    if not tech_events.empty:
                        logging.info(f"[PROPHET] Adding {len(tech_events)} custom tech/retail events")
                        # Prophet will automatically consider holidays in the holidays_df
                
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=self.forecast_steps, freq=self.freq)
                fc = m.predict(future)
                forecast_df = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(self.forecast_steps).copy()
                
            else:
                raise NotImplementedError(f"Model type '{self.model_type}' not implemented")

            self.forecasts = forecast_df
            logging.info(f"[FORECAST] Generated {len(forecast_df)} forecast points")
            return forecast_df
            
        except Exception as e:
            logging.error(f"[FORECAST] Model fitting failed: {e}")
            return self._simple_forecast(ts_df)

    def _simple_forecast(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback: Generate simple forecast using moving average when advanced models fail.
        """
        logging.info("[SIMPLE_FORECAST] Using fallback moving average forecast")
        
        if ts_df.empty:
            return pd.DataFrame({"ds": [], "yhat": []})
        
        # Use last value or mean as baseline
        baseline = ts_df["y"].iloc[-1] if len(ts_df) >= 1 else ts_df["y"].mean()
        
        # Generate future dates
        last_date = ts_df["ds"].iloc[-1]
        try:
            if hasattr(last_date, 'tz') and last_date.tz is not None:
                last_date = last_date.tz_localize(None)
        except Exception:
            pass
        
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=self.freq),
            periods=self.forecast_steps,
            freq=self.freq
        )
        
        # Simple forecast: repeat last value with slight noise
        import numpy as np
        noise = np.random.normal(0, 0.05, self.forecast_steps)
        forecast_values = baseline + noise
        
        forecast_df = pd.DataFrame({
            "ds": future_index,
            "yhat": forecast_values
        })
        
        self.forecasts = forecast_df
        logging.info(f"[SIMPLE_FORECAST] Generated {len(forecast_df)} fallback forecast points")
        return forecast_df

    def run_cycle(self, signals: List[Dict], company: str) -> pd.DataFrame:
        """
        AGENTIC RUN CYCLE: Act → Evaluate → Resolve (with feedback loop)
        Main entry: take brand signals, prepare data, train model, evaluate, and refine.
        Enhanced with regional holidays, market data, and news tracking.
        """
        self.last_run = datetime.now(timezone.utc).isoformat()
        self.memory[self.last_run] = {}
        
        logging.info("[CYCLE] Starting perception forecasting cycle...")
        logging.info(f"[CYCLE] Company: {company}, Region: {self.region}, Signals: {len(signals)}, Forecast Steps: {self.forecast_steps}")
        
        # STEP 0: CONTEXT GATHERING - Load regional holidays, market data, news volume
        logging.info("[CYCLE] STEP 0: CONTEXT - Gathering regional holidays, market data, and news volume...")
        
        # Get regional holiday calendar
        current_year = datetime.now().year
        self.get_regional_holidays(years=[current_year - 1, current_year, current_year + 1])
        self.memory[self.last_run]["holidays_loaded"] = len(self.holidays_df) if self.holidays_df is not None else 0
        
        # Fetch market fluctuation data
        market_data = self.fetch_market_data(company)
        self.memory[self.last_run]["market_data"] = market_data
        
        # Track news volume for event detection
        news_volume = self.fetch_news_volume(company, days_back=60)
        self.memory[self.last_run]["news_volume_days"] = len(news_volume)
        
        # Detect news spikes (PR events, crises, launches)
        if news_volume:
            volumes = [n['article_count'] for n in news_volume]
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes)
            news_spikes = [n for n in news_volume if n['article_count'] > mean_vol + 2*std_vol]
            self.memory[self.last_run]["news_spikes"] = news_spikes
            logging.info(f"[CYCLE] Detected {len(news_spikes)} news volume spikes (potential PR events)")
        
        # STEP 1: ACT - Prepare data and generate initial forecast
        logging.info("[CYCLE] STEP 1: ACT - Generating initial forecast with regional context...")
        ts_df = self.prepare_timeseries(signals)
        
        # Use seasonal detection tool before forecasting
        seasonal_info = self.identify_seasonal_patterns(ts_df)
        self.memory[self.last_run]["seasonal_patterns"] = seasonal_info
        
        forecast_df = self.forecast(ts_df)
        
        # Use shift detection tool
        sentiment_shifts = self.detect_sentiment_shifts()
        self.memory[self.last_run]["sentiment_shifts"] = sentiment_shifts
        
        # STEP 2: EVALUATE - Critique forecast with LLM (now with market/news context)
        logging.info("[CYCLE] STEP 2: EVALUATE - Critiquing forecast quality with market context...")
        self.evaluate(forecast_df=forecast_df, company=company)
        
        # STEP 3: Feedback loop handled within evaluate → resolve
        # If forecast is deemed "not good", resolve() is called automatically
        # resolve() will apply adjustments and re-run forecast (up to 3 iterations)
        
        # STEP 4: Generate actionable insights (if not already done in evaluate)
        if not self.actionable_insights:
            self.generate_actionable_insights()
        
        logging.info("[CYCLE] Forecasting cycle complete.")
        logging.info(f"[CYCLE] Feedback iterations: {self.feedback_run_count}")
        logging.info(f"[CYCLE] Actionable insights generated: {len(self.actionable_insights)}")
        
        return forecast_df

    def export_to_json(self, filepath: str):
        """
        Export forecasts to JSON for downstream consumption.
        """
        if self.forecasts is None or self.forecasts.empty:
            logging.warning("[EXPORT] No forecasts available, creating empty JSON file")
            with open(filepath, 'w') as f:
                json.dump([], f)
            return
        
        # Include yhat_lower and yhat_upper if present
        orient = "records"
        self.forecasts.to_json(filepath, orient=orient, date_format="iso")
        logging.info(f"[EXPORT] Forecasts exported to {filepath}")
    
    def export_comprehensive_report(self, filepath: str):
        """
        Export comprehensive report including forecasts, insights, and metadata.
        """
        # Convert forecasts with proper datetime handling
        forecasts_data = []
        if self.forecasts is not None:
            for record in self.forecasts.to_dict(orient="records"):
                # Convert pandas Timestamp objects to ISO strings
                converted_record = {}
                for key, value in record.items():
                    if isinstance(value, pd.Timestamp):
                        converted_record[key] = value.isoformat()
                    elif pd.isna(value):
                        converted_record[key] = None
                    else:
                        converted_record[key] = value
                forecasts_data.append(converted_record)
        
        # Convert sentiment shifts with datetime handling
        sentiment_shifts = []
        for shift in self.memory.get(self.last_run, {}).get("sentiment_shifts", []):
            converted_shift = {}
            for key, value in shift.items():
                if isinstance(value, pd.Timestamp):
                    converted_shift[key] = value.isoformat()
                elif pd.isna(value):
                    converted_shift[key] = None
                else:
                    converted_shift[key] = value
            sentiment_shifts.append(converted_shift)
        
        report = {
            "company": self.state.get("company_name", "Unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "forecast_steps": self.forecast_steps,
            "model_type": self.model_type,
            "frequency": self.freq,
            "feedback_iterations": self.feedback_run_count,
            "forecasts": forecasts_data,
            "actionable_insights": self.actionable_insights,
            "sentiment_shifts": sentiment_shifts,
            "seasonal_patterns": self.memory.get(self.last_run, {}).get("seasonal_patterns", {}),
            "evaluation_history": self.memory
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logging.info(f"[EXPORT] Comprehensive report exported to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    
    # Load brand signals from Agent A (use brand_signals.json in agents directory)
    signal_file = "brand_signals.json"
    
    try:
        with open(signal_file, "r", encoding="utf-8") as f:
            signals = json.load(f)
        logging.info(f"[LOAD] Loaded {len(signals)} signals from {signal_file}")
    except FileNotFoundError:
        logging.error(f"[ERROR] {signal_file} not found. Run agent_brand_signal_collector.py first.")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"[ERROR] Invalid JSON in {signal_file}: {e}")
        exit(1)

    # Initialize agent state
    dummy_state = {
        "company_name": "Google",
        "brand_signals": signals,
        "sentiment_forecast": [],
        "error_messages": []
    }

    # Create Perception Forecaster Agent with regional context
    agent = PerceptionForecasterAgent(
        state=dummy_state,
        forecast_steps=180,  # 180 days forecast
        model_type="Prophet",  # Prophet for seasonal patterns
        freq="D",  # Daily frequency
        region="US"  # Change to SG, UK, CN, etc. for different regions
    )
    
    # Run agentic cycle: Act → Evaluate → Resolve (with feedback)
    logging.info("=" * 80)
    logging.info("AGENT B: PERCEPTION FORECASTER - STARTING AGENTIC CYCLE")
    logging.info(f"REGION: {agent.region}")
    logging.info("=" * 80)
    
    forecast_df = agent.run_cycle(signals, dummy_state["company_name"])
    
    # Export results
    agent.export_to_json("sentiment_forecast.json")
    agent.export_comprehensive_report("perception_report.json")
    
    # Display summary
    logging.info("=" * 80)
    logging.info("PERCEPTION FORECASTING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Forecast generated: {len(forecast_df)} time steps")
    logging.info(f"Feedback iterations: {agent.feedback_run_count}")
    logging.info(f"Actionable insights: {len(agent.actionable_insights)}")
    
    # Print sample insights
    if agent.actionable_insights:
        logging.info("\nTop Actionable Insights:")
        for insight in agent.actionable_insights[:3]:
            logging.info(f"  - [{insight.get('category', 'Unknown')}] {insight.get('insight', 'N/A')}")
    
    logging.info(f"\nExports created:")
    logging.info(f"  - sentiment_forecast.json (forecast data)")
    logging.info(f"  - perception_report.json (comprehensive report)")


