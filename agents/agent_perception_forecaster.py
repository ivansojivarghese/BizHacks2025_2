# agent_perception_forecaster.py

import json
import logging
import pandas as pd
import tiktoken
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import os
from mistralai import Mistral

from langchain_core.messages import ChatMessage
from statsmodels.tsa.arima.model import ARIMA
#from fbprophet import Prophet  # if you install prophet
from prophet import Prophet
from google import genai

'''
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
'''
encoding = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 100000

MISTRAL_API_KEY = "S3Myzs3gILhBBgYdY2C7p2THb2aCAWad"
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
        freq: str = "D"  # 'D' for daily, 'W' for weekly
    ):
        """
        :param forecast_steps: How many future time steps to predict.
        :param model_type: Type of model to use ("ARIMA" or "Prophet").
        :param freq: Resampling frequency for aggregation ('D', 'W', etc.).
        """
        self.state = state
        self.forecast_steps = forecast_steps
        self.model_type = model_type
        self.freq = freq
        self.last_run: Optional[str] = None
        self.forecasts: Optional[pd.DataFrame] = None

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
        Brand Signals: {brand_signals_text}

        Here is the forecast data:
        {forecast_df.to_csv(index=False)}

        Assess whether the forecast makes sense. Consider seasonal patterns, holiday effects, and brand signal impact relative to today's date.
        If the forecast is flawed, outline key reasons why. Also consider key dates and sentiment trends (based on region) for {company}.
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
            print("[Evaluation] Forecast deemed NOT good. Moving to resolve stage...")
            self.resolve(critique_response, critique_data)
        else:
            print("[Evaluation] Forecast passed initial review. Proceeding...")


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
                
                # 🛠️ YOUR REQUESTED FIX: Safe timezone removal for Prophet
                try:
                    if not prophet_df.empty and pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
                        if prophet_df['ds'].dt.tz is not None:
                            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                            logging.debug("[PROPHET] Removed timezone from ds column")
                except Exception as e:
                    logging.debug(f"[PROPHET] Timezone removal failed: {e}")
                
                m = Prophet(
                    daily_seasonality=(self.freq == "D"), 
                    weekly_seasonality=(self.freq == "W")
                )
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

    def run_cycle(self, signals: List[Dict], company: str) -> pd.DataFrame:
        """
        Main entry: take brand signals, prepare data, train model, return forecast.
        """
        self.last_run = datetime.utcnow().isoformat()
        #forecast_df = self.act(self.state["brand_signals"])
        #self.evaluate(forecast_df=forecast_df, company=self.state["company_name"])
        forecast_df = self.act(signals=signals)
        self.evaluate(forecast_df=forecast_df, company=company)

        # eval - send to LLM for critique and suggestions (date, seasons, holiday, region, etc + brand signals - if forecast make sense?)
            # > if NOT good, then do resolve
        # resolve
            # - USE TOOLS!!!
            # - seasonal adjustments
            # DO THE BOTH BELOW:
            # - IF brand is a consumer brand - fetch the general user sentiment till this point in time
            # - IF brand is enterprise - fetch the finances, etc over past periods - make assumptions
            # - any upcoming news/signals regarding the brand - predict the daily forecast (for next 7 days, then weekly (for 4 weeks), then monthly for 6 months - avg. out)
            # SEND to eval

        return forecast_df


    def export_to_json(self, filepath: str):
        """
        Export forecasts to JSON for downstream consumption.
        """
    
        if self.forecasts is None or self.forecasts.empty:
            #raise ValueError("No forecasts available to export.")
            # 🛠️ FIX: Create empty file instead of raising error
            logging.warning("[EXPORT] No forecasts available, creating empty JSON file")
            with open(filepath, 'w') as f:
                json.dump([], f)
        # Include yhat_lower and yhat_upper if present
        orient = "records"
        self.forecasts.to_json(filepath, orient=orient, date_format="iso")
        logging.info(f"[EXPORT] Forecasts exported to {filepath}")


if __name__ == "__main__":
    with open("Google_brand_signals.json", "r", encoding="utf-8") as f:
        signals = json.load(f)

    # Now if you still want to use self.state, populate it manually:
    dummy_state = {
        "company_name": "Google",
        "brand_signals": signals,
        "sentiment_forecast": [],
        "error_messages": []
    }

    agent = PerceptionForecasterAgent(
        state=dummy_state,
        forecast_steps=180,  # e.g., two weeks
        model_type="Prophet",  # or "ARIMA"
        freq="D"
    )
    forecast_df = agent.run_cycle(signals, dummy_state["company_name"])
    print(forecast_df)
    #agent.export_to_json("sentiment_forecast.json")
