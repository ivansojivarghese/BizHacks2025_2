from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import json
import logging

# Import our agents
from agents.agent_brand_signal_collector import BrandSignalCollectorAgent
from agents.agent_perception_forecaster import PerceptionForecasterAgent

# Configure logging to avoid conflicts
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    force=True  # Override any existing logging config
)

# --- Define shared graph state ---
class GraphState(TypedDict):
    company_name: str
    brand_signals: List[Dict]
    sentiment_forecast: List[Dict]
    error_messages: List[str]

# --- Node A: Brand Signal Collector ---
def node_brand_signal_collector(state: GraphState) -> GraphState:
    """Collect brand signals using Agent A"""
    company = state["company_name"]
    
    try:
        hours_back = 720  # Default to prev. 30 days
        region = "US"
        max_results = 10 # Default max results per source

        logging.info(f"[MAIN] Starting brand signal collection for {company}")
        agent_a = BrandSignalCollectorAgent(company_name=company, hours_back=hours_back, max_results=max_results, country=region)
        signals = agent_a.run_cycle()
        
        # Export signals
        filename = f"{company}_brand_signals.json"
        agent_a.export_to_json(filename)
        
        state["brand_signals"] = signals
        logging.info(f"[MAIN] Collected {len(signals)} brand signals")
        
    except Exception as e:
        error_msg = f"Brand signal collection failed: {str(e)}"
        logging.error(f"[MAIN] {error_msg}")
        state["error_messages"].append(error_msg)
        state["brand_signals"] = []  # Empty list to allow forecaster to handle gracefully
    
    return state

# --- Node B: Perception Forecaster ---
def node_perception_forecaster(state: GraphState) -> GraphState:
    """Generate sentiment forecast using Agent B"""
    
    try:
        forecast_days = 180 # no. of days to forecast ahead

        logging.info("[MAIN] Starting sentiment forecasting")
        agent_b = PerceptionForecasterAgent(state=state, forecast_steps=forecast_days, model_type="Prophet")
        
        # Run forecasting
        #forecast_df = agent_b.run_cycle(state["brand_signals"], state["company_name"])
        forecast_df = agent_b.run_cycle()
        
        # Convert to JSON format
        if not forecast_df.empty:
            forecast_json = json.loads(forecast_df.to_json(orient="records", date_format="iso"))
        else:
            forecast_json = []
            logging.warning("[MAIN] No forecast data generated")
        
        # Export forecast
        filename = f"{state['company_name']}_sentiment_forecast.json"
        agent_b.export_to_json(filename)
        
        state["sentiment_forecast"] = forecast_json
        logging.info(f"[MAIN] Generated {len(forecast_json)} forecast points")
        
    except Exception as e:
        error_msg = f"Sentiment forecasting failed: {str(e)}"
        logging.error(f"[MAIN] {error_msg}")
        state["error_messages"].append(error_msg)
        state["sentiment_forecast"] = []
    
    return state

# --- Build LangGraph ---
def create_workflow():
    """Create and return the compiled workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("BrandSignalCollector", node_brand_signal_collector)
    workflow.add_node("PerceptionForecaster", node_perception_forecaster)
    
    # Set entry point and edges
    workflow.set_entry_point("BrandSignalCollector")
    workflow.add_edge("BrandSignalCollector", "PerceptionForecaster")
    workflow.add_edge("PerceptionForecaster", END)
    
    # Compile and return
    return workflow.compile()

if __name__ == "__main__":
    # Create initial state
    initial_state: GraphState = {
        "company_name": "Google",
        "brand_signals": [],
        "sentiment_forecast": [],
        "error_messages": []
    }
    
    try:
        # Create and run workflow
        app = create_workflow()
        logging.info("[MAIN] Starting workflow execution")
        
        final_state = app.invoke(initial_state)
        
        print("\n=== WORKFLOW EXECUTION COMPLETE ===")
        print(f"Company: {final_state['company_name']}")
        print(f"Brand Signals Collected: {len(final_state['brand_signals'])}")
        print(f"Forecast Points Generated: {len(final_state['sentiment_forecast'])}")
        
        if final_state['error_messages']:
            print(f"\nErrors encountered: {len(final_state['error_messages'])}")
            for error in final_state['error_messages']:
                print(f"  - {error}")
        
        # Show sample data
        if final_state['brand_signals']:
            print(f"\nSample brand signal:")
            sample_signal = final_state['brand_signals'][0]
            for key in ['title', 'text', 'sentiment', 'topic', 'timestamp']:
                if key in sample_signal:
                    value = sample_signal[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {value}")
        
        if final_state['sentiment_forecast']:
            print(f"\nSample forecast points:")
            for i, forecast in enumerate(final_state['sentiment_forecast'][:3]):
                print(f"  {i+1}: {forecast}")
        
        print("\n=== FILES GENERATED ===")
        print(f"- {final_state['company_name']}_brand_signals.json")
        print(f"- {final_state['company_name']}_sentiment_forecast.json")
        
    except Exception as e:
        logging.error(f"[MAIN] Workflow execution failed: {str(e)}")
        print(f"\nWorkflow failed with error: {str(e)}")
        print("Check the logs above for more details.")