# Brand Intelligence System

Problem Statement (from Infosys BizHacks 2025): How might we design a scalable and responsive measurement framework - powered by AI agentic systems - that can track brand performance and connect it to business metrics such as lead quality, sales velocity, and client engagement?

To obtain:
- HF Token (fine-grained): ```signals_collector.py``` line 56
- Groq API Key: ```signals_collector.py``` line 46, ```agent_brand_signal_collector.py``` line 96
- Gemini API Key: ```signals_collector.py``` line 47
- NVIDIA API Key: ```signals_collector.py``` line 40
- Mistral API Key: ```signals_collector.py``` line 56

## Overview

See the [Reference Product](https://app.brand24.com/). 

```main.py``` variables (or parameters) that may be filtered:
- Company name
- Hours back to scrap
- Region
- Max results
- Forecast days (days to forecast)
- Available sources/signals to reference from (more sources = more time)

4 agents. Agents A, B, C & D.
- Agent A: Brand Signals Collector
- Agent B: Perception Forecaster
- Agent C: Scenario Simulator (TBC)
- Agent D: Outcome Mapper (TBC)

### Agent A
- Collect recent public signals about "Company A" from news headlines, Reddit discussions, and (optionally) other sources. 
- Perform initial sentiment analysis on the retrieved content. 
- Normalize the output into a basic format: source, timestamp, title/text, sentiment score.

Tech Stack:
- Reddit API (social media/forums)
- News API (news)
- HackerNews (news)
- Topic trends:
  > GNews (top headlines endpoint) / Groq LLM to summarise headlines into topics/themes - usage limits?
- More diversified sources:
  > Mastodon API (social media/forums);
  > SerpAI (X mentions);
  > APITube (general news);
  > ScrapFly API (CNN / Bloomberg / Forbes);
  > Google Finance API (via SERP API) (finance);
  > Gemini API (LLM) - unpredictable;
  > NVIDIA API (LLM);
  > Together.ai (LLM) - fairly slow responses?;
  > HuggingFace LLM - complicated setup?;
  > Other LLMs if needed? (to manage quotas, future updates may include automatic selection of specific models by LLM depending on context)

### Agent B
- Identifying trends of some sort using time-series data 
- Role: Trains time-series or machine learning models to estimate sentiment and anticipate future shifts.
- How it fits: Connects upstream brand signals to downstream expectations, operationalizing suggestions such as “sentiment analysis” and converting data into actionable insight about brand perception trends—a key request in the scope of work.

Tech Stack:
- Mistral LLM
- Tiktoken to limit prompts as 50000 tokens

How?
- Make it agentic with a feedback loop, etc. 
- run_cycle, divide into components: act, eval, feedback -> repeat
- get first output (time-series forecast, etc.), then run llm, see if good, if not - do bigger time range? 
- consider the seasons, market flunctuations (holidays, etc. - anything relevant to the brand in focus - and keep feedbacking/deciding if the forecast is appropriate) 
- Agent to use this data for/to obtain
- estimate sentiment 
- anticipate future shifts, etc.
- actionable insight into brand perception or trends
- TOOLS for extracting regional (holiday) data info, market data, news volumes, etc.

### Agent C (TBC)
- Role: Models “what-if” scenarios, predicting changes in business metrics based on hypothetical marketing or brand actions.
- How it fits: This simulation layer aligns with the need for an “AI-driven, responsive framework” able to visualize correlations and test different input variables. Useful for dashboards and for prioritizing brand investments by forecasting likely business outcomes of brand perception shifts.

### Agent D (TBC)
- Role: Explicitly maps forecasted changes in perception (from above agents) to business outcomes such as leads generated, quality scores, or deal progression.
- How it fits: Addresses the core challenge articulated—mapping brand KPIs to measurable business outcomes (pipeline velocity, deal progress, engagement, etc.). This final mapping closes the loop, making brand measurement actionable for decision-makers and supporting campaign planning with evidence.

## Usage
- Agent A: ```python agent_brand_signal_collector.py```
- Agent B: ```python agent_perception_forecaster.py```
- Agent C: TBD
- Agent D: TBD
- ```python main.py ``` for full pipeline run.







