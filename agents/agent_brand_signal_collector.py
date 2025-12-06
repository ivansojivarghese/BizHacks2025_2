# agent_brand_signal_collector.py

import os
import json
import logging
import uuid
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import Counter

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import signals_collector

# Import data collection utilities
from util.signals_collector import (
    get_gnews_articles,
    extract_topics_with_groq,
    fetch_reddit,
    fetch_news,
    fetch_hackernews,
    analyze_brand_signals,
    fetch_news_articles,
    scrape_company_tweets,
    fetch_company_news,
    get_combined_finance_analysis,
    get_finance_ticker,
    deduplicate_signals,
    extract_topic_ml,
    parse_json_from_codeblock,
    groq_llm, # low limits
    gemini_llm, # backup for groq
    nvidia_llm,
    hf_llm, # unusable
    together_llm, # very slow
    # New dynamic topic extraction functions
    extract_signal_text,
    extract_topics_from_signals,
    parse_topics_to_labels,
    reclassify_signals,
    update_topics_and_reclassify
)


class BrandSignalCollectorAgent:
    """
    Agentic AI implementation of the Brand Signal Collector.
    Encapsulates reasoning, action, evaluation, and feedback loops for 
    adaptive, multi-source brand signal monitoring.
    """

    def __init__(
        self,
        company_name: str,
        hours_back: int,
        max_results: int,
        country: str,
        topic_extractor: Callable = extract_topics_with_groq,
        config: Optional[Dict] = None,  # Add config param
        taxonomy: Optional[Dict] = None,
        feedback_run_count = 0  # track re-run cycles caused by feedback
    ):
        self.company_name = company_name
        self.hours_back = hours_back
        self.max_results = max_results
        self.country = country
        self.labels: List[str] = []
        self.memory: Dict[str, dict] = {}
        self.collected_signals: List[dict] = []
        self.last_run: Optional[str] = None
        self.topic_extractor = topic_extractor
        self.config = config or {}  # Assign config or empty dict if None
        self.taxonomy = taxonomy or {}
        self.feedback_run_count = feedback_run_count or 0  # track re-run cycles caused by feedback

        # Load API keys from environment
        self.api_keys = {
            '''
            "NEWSAPI": os.getenv("NEWSAPI_API_KEY"),
            "GNEWS": os.getenv("GNEWS_API_KEY"),
            "GROQ": os.getenv("GROQ_API_KEY"),
            "APITUBE": os.getenv("APITUBE_API_KEY"),
            "SCRAPFLY": os.getenv("SCRAPFLY_API_KEY")
            '''
            "NEWSAPI": "44aa2fe007c249b291b84f90ef03c77c",
            "GNEWS": "ef5f8624efbdff7065a8b27b064230d8",
            "GROQ": "",
            "APITUBE": "api_live_4FCFmHUM0FDz9E5wvS2FZYfdiE2Qsf3Air8X8mH0IXFs",
            "SCRAPFLY": "scp-live-1a870f1d40854783b5e42f486702f116"
        }

    # --- Reasoning (PLAN) ---
    def plan(self, feedback_data: Optional[dict] = None) -> List[str]:
        if not feedback_data:
            logging.info(f"[PLAN] Determining brand signal fetching strategy for '{self.company_name}'")

            headlines = get_gnews_articles(
                api_key=self.api_keys["GNEWS"],
                query=self.company_name,
                max_results=10
            )
            logging.info(f"Found {len(headlines)} GNews headlines.")

            topics_raw = self.topic_extractor(
                groq_api_key=self.api_keys["GROQ"],
                headlines=headlines
            )

            import re
            self.labels = [
                re.search(r"\*\*(.*?)\*\*", line).group(1).strip()
                for line in topics_raw.splitlines()
                if re.search(r"\*\*(.*?)\*\*", line)
            ]
            self.labels.append("Other")

            logging.info(f"Extracted topic labels: {self.labels}")
            return headlines

    # --- Acting (EXECUTE) ---
    def act(self, headlines: List[str], feedback_data: Optional[dict] = None):
        if not feedback_data:
            logging.info(f"[EXECUTION] Collecting brand signals for '{self.company_name}'")
            self.collected_signals.clear()

            # Apply config-based adjustments
            fetch_full_text = self.config.get("fetch_full_text", False)
            use_backup_topic_extractor = self.config.get("use_backup_topic_extractor", False)
            dedup_strictness = self.config.get("dedup_strictness", "normal")
            source_filtering = self.config.get("source_filtering", False)
            #quality_filtering = self.config.get("quality_filtering", True)
            self.config["quality_filtering"] = True

            # Optional: filter sources based on config
            allowed_sources = ["Reddit", "News", "HackerNews", "Twitter", "APITube", "Mastodon", "ScrapFly", "Finance"]
            if source_filtering:
                allowed_sources.remove("HackerNews")  # Example: low content quality historically
                allowed_sources.remove("Reddit")
                logging.info(f"[FILTER] Limiting sources to: {allowed_sources}")

            finance_ticker = get_finance_ticker(self.company_name)
            # perform a swap of words between ':'
            finance_ticker = finance_ticker.split(':')[1] + ':' + finance_ticker.split(':')[0]
            #finance_ticker = "GOOGL:NASDAQ"
            print(f"finance_ticker: {finance_ticker}")

            # Core social/media sources
            
            if "Reddit" in allowed_sources:
                reddit_signals = fetch_reddit(self.company_name, self.labels, self.max_results, self.hours_back)
                if reddit_signals:
                    self.collected_signals.extend(reddit_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] Reddit batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            
            if "News" in allowed_sources:
                news_signals = fetch_news(self.company_name, self.labels, self.max_results, self.hours_back)
                if news_signals:
                    self.collected_signals.extend(news_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] News batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            
            if "HackerNews" in allowed_sources:
                hn_signals = fetch_hackernews(self.company_name, self.labels, self.max_results, self.hours_back)
                if hn_signals:
                    self.collected_signals.extend(hn_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] HackerNews batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            
            # SOME ISSUE WITH THIS API BELOW
            #if "Mastodon" in allowed_sources:
                #self.collected_signals.extend(analyze_brand_signals(self.company_name, self.labels, self.max_results, self.hours_back))

            if "APITube" in allowed_sources:
                apitube_signals = fetch_news_articles(self.company_name)
                if apitube_signals:
                    self.collected_signals.extend(apitube_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] APITube batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            '''
            if "Twitter" in allowed_sources:
                twitter_signals = scrape_company_tweets(self.company_name, self.country)
                if twitter_signals:
                    self.collected_signals.extend(twitter_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] Twitter batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            
            if "ScrapFly" in allowed_sources:
                scrapfly_signals = fetch_company_news(self.company_name, self.hours_back)
                if scrapfly_signals:
                    self.collected_signals.extend(scrapfly_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] ScrapFly batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            
            if "Finance" in allowed_sources:
                finance_signals = get_combined_finance_analysis(self.company_name, finance_ticker, self.hours_back)
                if finance_signals:
                    self.collected_signals.extend(finance_signals)
                    # Update topics after each batch
                    self.collected_signals, self.labels = update_topics_and_reclassify(
                        self.company_name, self.collected_signals, self.labels
                    )
                    logging.info(f"[TOPIC_UPDATE] Finance batch: {len(self.collected_signals)} signals, {len(self.labels)} topics")
            '''
            # ===========================================================================================

            #print(f"sig {self.collected_signals}")

            # If enabled, fetch full text for any signal missing content
            if fetch_full_text:
                logging.info("[TEXT] Fetching full text for signals with missing content...")
                for sig in self.collected_signals:
                    if not isinstance(signal, dict):
                        continue
                    if not sig.get("text"):
                        full_text = self.scrape_full_article(sig.get("url", ""))
                        sig["text"] = full_text
                        # Also re-run topic extraction with combined title + text
                        content = f"{sig.get('title', '')} {sig.get('text', '')} {sig.get('full_text', "")} {sig.get('description', "")} {full_text}".strip()
                        sig["topic"] = extract_topic_ml(content, self.labels, use_backup_topic_extractor)

            # Sentiment + Topic Classification Adjustments
            if use_backup_topic_extractor:
                logging.info("[TOPIC] Using backup topic extractor (Together API)")
                for sig in self.collected_signals:
                    if not isinstance(signal, dict):
                        continue
                    content = f"{sig.get('title', '')} {sig.get('text', '')} {sig.get('full_text', "")} {sig.get('description', "")}".strip()
                    sig["topic"] = extract_topic_ml(content, self.labels, use_backup_topic_extractor)  # Fallback ML extractor

            #print(f"sig: {self.collected_signals}")

            # Deduplication with configurable strictness
            #self.collected_signals = deduplicate_signals(self.collected_signals, strictness=dedup_strictness)

            #print(f"sig2 {self.collected_signals}")
            #print(f"sig2: {self.collected_signals}")

            for signal in self.collected_signals:
                if not isinstance(signal, dict):
                    continue
                signal["id"] = str(uuid.uuid4())

    def scrape_full_article(self, url: str) -> str:
        """
        Fetch and return the full article text from a given URL.
        This is a placeholder and should be implemented with requests + BeautifulSoup or newspaper3k.
        """
        if not url:
            return ""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get the main title if available
            title_text = soup.title.get_text(strip=True) if soup.title else ""

            # Extract meaningful elements: paragraphs, headings, blockquotes, list items
            content_parts = []

            for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "blockquote", "li"]):
                text = tag.get_text(strip=True)
                if text and len(text.split()) > 3:  # Avoid super short fragments
                    content_parts.append(text)

            # Combine all with spaces
            full_text = " ".join([title_text] + content_parts).strip()

            return full_text

        except Exception as e:
            logging.warning(f"[SCRAPER] Failed to fetch full article from {url}: {e}")
            return ""

    # --- Evaluation ---
    def evaluate(self):
        perform_feedback = False
        feedback_prompt = ""

        rejected_ids = []

        all_signals = self.collected_signals
        #all_filtered_signals = self.filter_quality_signals(self.collected_signals)

        #print(f"all: {all_signals}")

        # 0. Ensure quality of collected signals
        '''
        if self.config.get("quality_filtering", True):
            quality_signals, rejected_signals = self.filter_quality_signals(self.collected_signals)
        else:
            quality_signals = all_signals
            rejected_signals = []
        '''
        quality_signals = all_signals
        rejected_signals = []
        #quality_signals = all_filtered_signals[0]
        #rejected_signals = all_filtered_signals[1]
        self.collected_signals = quality_signals  # Quality signals

        # Store rejected signals for feedback/re‐eval
        self.rejected_signals = rejected_signals

        # 1. Deduplicate
        self.collected_signals = deduplicate_signals(self.collected_signals, strictness=self.config.get("dedup_strictness", "normal"))
        total_signals = len(self.collected_signals)

        # Example: if many rejects, trigger reeval feedback
        if len(self.rejected_signals) > total_signals * 0.2:
            logging.info(f"[EVAL] High reject rate ({len(self.rejected_signals)}/{total_signals}). Scheduling re‐evaluation.")
            # Optionally call a re‐evaluation routine or flag for human review
            # self.flag_for_reevaluation(self.rejected_signals)
            rejected_ids = [sig["id"] for sig in rejected_signals]
            perform_feedback = True
            feedback_prompt = (
                "The following brand signals were rejected due to quality issues:\n\n"
                + "\n".join(f"- {sig.get('title', '')} {sig.get('text', '')} {sig.get('full_text', "")} {sig.get('description', "")}" for sig in rejected_signals)
                + "\n\nTASK: For each rejected signal, suggest a specific improvement to make it acceptable."
                + "\n- Consider issues like missing text, poor topic classification, duplicate content, or irrelevant source."
                + "\n- Keep each suggestion short (max 20 words)."
                + "\n\nReturn your response as a JSON array of improvement suggestions in the same order as the signals:"
                + "{ 'improvements': [list of improvement suggestions] }"
            )
        # if number of signals too low (< 50)
        elif total_signals < 50: 
            logging.info(f"[EVAL] Number of filtered, quality signals too low ({total_signals}). Scheduling re‐evaluation.")
            # Optionally call a re‐evaluation routine or flag for human review
            # self.flag_for_reevaluation(self.rejected_signals)
            #rejected_ids = [sig["id"] for sig in rejected_signals]
            perform_feedback = True
            self.config["quality_filtering"] = False
            logging.info("  - Set quality_filtering to False.")  
            self.config["dedup_strictness"] = "none"
            logging.info("  - Set deduplication strictness to NONE.")   
        else:
            rejected_ids = []
            
        # 2. Ensure every signal has a topic
        for sig in self.collected_signals:
            if not sig.get("topic"):
                sig["topic"] = "Other"

        # 3. Compute topic distribution and 'Other' ratio
        topic_counts = Counter(sig["topic"] for sig in self.collected_signals)
        topics_found = list({sig.get("topic") for sig in self.collected_signals if sig.get("topic")})
        other_ratio = topic_counts.get("Other", 0) / total_signals if total_signals else 0

        logging.info(f"[RESULT] Found topic(s): {topics_found}")
        logging.info(f"[METRIC] Other ratio: {other_ratio:.2%}")
        logging.info(f"[EVAL] Collected {total_signals} unique signals across {len(topics_found)} topics.")
        logging.info(f"[EVAL] Stored {len(rejected_ids)} rejected signal IDs for feedback.")

        # 4. Adaptive reclassification if needed
        # Criteria: high Other ratio or too few topics relative to signals
        if other_ratio > 0.3 or len(topics_found) < max(3, total_signals // 50):
            logging.info("[ADAPT] High 'Other' ratio or too few topics detected. Re-extracting topics from all signals.")
            
            # Use the new dynamic topic extraction system
            self.collected_signals, self.labels = update_topics_and_reclassify(
                self.company_name, self.collected_signals, self.labels
            )
            
            # Re-compute topic distribution after reclassification
            topic_counts = Counter(sig["topic"] for sig in self.collected_signals)
            topics_found = list({sig.get("topic") for sig in self.collected_signals if sig.get("topic")})
            other_ratio = topic_counts.get("Other", 0) / total_signals if total_signals else 0
            
            logging.info(f"[ADAPT] After reclassification: {len(topics_found)} topics, Other ratio: {other_ratio:.2%}")
            logging.info(f"[ADAPT] Updated topic labels: {self.labels}")
            
            # Generate new topics string for feedback
            new_topics = "\n".join(self.labels)
            perform_feedback = True
            if feedback_prompt:
                feedback_prompt = (
                    "Part 1: The following brand signals were rejected due to quality issues:\n\n"
                    + "\n".join(f"- {sig.get('title', '')} {sig.get('text', '')} {sig.get('full_text', "")} {sig.get('description', "")}" for sig in rejected_signals)
                    + "\n\nTASK: For each rejected signal, suggest a specific improvement to make it acceptable."
                    + "\n- Consider issues like missing text, poor topic classification, duplicate content, or irrelevant source."
                    + "\n- Keep each suggestion short (max 20 words)."
                    + "\n\nPart 2: The following new topics were proposed based on the collected signals:\n"
                    + "\n".join(new_topics)
                    + "\n\nTASK: Evaluate each proposed topic for relevance to the collected brand signals."
                    + "\n- If a topic is relevant, keep it as-is."
                    + "\n- If a topic is irrelevant or unclear, replace it with a more suitable topic based on the collected signals."
                    + "\n\nReturn ONLY a valid JSON object with the format:\n"
                    + "{ 'improvements': [list of improvement suggestions], 'topics': [list of final topics] }"
                )
            else:
                feedback_prompt += (
                    "\n\nThe following new topics were proposed based on the collected signals:\n"
                    + "\n".join(new_topics)
                    + "\n\nTASK: Evaluate each proposed topic for relevance to the collected brand signals."
                    + "\n- If a topic is relevant, keep it as-is."
                    + "\n- If a topic is irrelevant or unclear, replace it with a more suitable topic based on the collected signals."
                    + "\n\nReturn ONLY a valid JSON object with the format:\n"
                    + "{ 'improvements': [], 'topics': [list of final topics] }"
                )
            self.update_taxonomy_and_reclassify(new_topics)

        # 5. Store evaluation in memory
        self.memory[self.last_run] = {
            "count": total_signals,
            "topics": topics_found,
            "other_ratio": other_ratio,
            "labels": self.labels,
            "signals": all_signals,
            "quality_signals": quality_signals,
            "rejected_signals": rejected_signals,
            "rejected_signal_ids": rejected_ids,
            "feedback": perform_feedback,
            "feedback_prompt": feedback_prompt
        }

    # Usage example in signal processing pipeline:
    def filter_quality_signals(self, signals):
        """Filter signals based on quality assessment"""
        quality_signals = []
        rejected_signals = []
        
        #if self.config["quality_filtering"]:
        for signal in signals:
            #quality_result = run_quality_check(signal["text"])
            #signal["quality"] = quality_result

            # Skip deduplication if url contains 'x.com'
            url = signal.get("url", "").lower()
            source = signal.get("source", "").lower()
            if "x.com" in url or "News" in source or "HackerNews" in source or "APITube News" in source or "ScrapFly" in source or "Finance" in source:
                quality_signals.append(signal)
                continue

            quality_result = signal["quality"]

            #print(f"qr: {quality_result}")
            
            if quality_result["is_quality"]:
                quality_signals.append(signal)
            else:
                rejected_signals.append(signal)
                logging.info(f"[FILTER] Rejected signal: {quality_result['quality_issues']}")
        #else:
        '''
            quality_signals = signals # every signal has quality
            rejected_signals = []
        '''
        
        logging.info(f"[FILTER] Quality signals: {len(quality_signals)}, Rejected: {len(rejected_signals)}")
        return quality_signals, rejected_signals

    def update_taxonomy_and_reclassify(self, new_topics: List[str]):
        """
        Update the taxonomy with the new topics and re-run the reclassification pass.
        """
        # Initialize empty keyword lists for new topics
        for topic in new_topics:
            if topic not in self.taxonomy:
                self.taxonomy[topic] = []
        # Ensure 'Other' remains
        if "Other" not in self.taxonomy:
            self.taxonomy["Other"] = []

        # Reclassify all signals with refreshed taxonomy
        self.reclassify()

    # --- Feedback loop ---
    def feedback(self, feedback_data: dict):
        logging.info("[FEEDBACK] Adjusting strategy based on downstream feedback.")

        # Skip if no feedback provided
        if not feedback_data or feedback_data == {} or self.feedback_run_count >= 0:
            logging.info("[FEEDBACK] No feedback provided — skipping adjustments.")
            return None

        # Be careful to avoid infinite loop
        if feedback_data.get("topics") or feedback_data.get("improvements"):
            self.feedback_run_count += 1
            logging.info(f"[FEEDBACK COUNT] {self.feedback_run_count}")
            logging.info("[♻] Re-running cycle with updated strategy...")
            signals = self.run_cycle()  # no feedback this time to prevent loop

            return signals

        return None
            
    def reclassify(self) -> None:
        """
        Re‐assign each signal to a topic using ML-based topic classification.
        Uses the new dynamic topic extraction and reclassification system.
        """
        if not self.collected_signals:
            logging.info("[RECLASSIFY] No signals to reclassify")
            return
        
        # Use the new reclassify_signals function from signals_collector
        self.collected_signals = reclassify_signals(self.collected_signals, self.labels)
        
        # 2. Count signals per topic
        topic_counts = Counter(sig['topic'] for sig in self.collected_signals)
        total = sum(topic_counts.values())

        # 3. Compute 'Other' ratio
        other_count = topic_counts.get('Other', 0)
        other_ratio = other_count / total if total > 0 else 0

        # 4. Store metrics for logging or downstream use
        self.topic_counts = dict(topic_counts)
        self.other_ratio = other_ratio
        
        logging.info(f"[RECLASSIFY] Completed: {total} signals across {len(topic_counts)} topics, Other ratio: {other_ratio:.2%}")

    # --- Run cycle ---
    def run_cycle(self, feedback_data: dict = None) -> List[dict]:
        # reset feedback counter if it's a "fresh" run without feedback
        #if feedback_data is None:
            #self.feedback_run_count = 0

        self.last_run = datetime.utcnow().isoformat()
        # 1. Plan: figure out which headlines to fetch
        headlines = self.plan(feedback_data)
        # 2. Act: fetch and generate raw signals (each with a tentative topic)
        self.act(headlines, feedback_data)
        # 3. Evaluate: perform your normal signal‐level scoring, sentiment, etc.
        self.evaluate()
        # 4. Reclassification pass using refreshed taxonomy
        self.reclassify()

        # 5. If feedback data is provided, adjust strategy
        run_memory = self.memory.get(self.last_run, {})
        if ("feedback" in run_memory and run_memory["feedback"]):
                
                # PROMPT based on feedback_prompt
                #feedback_data = gemini_llm(
                #feedback_data = groq_llm(
                feedback_data = together_llm(
                #feedback_data = nvidia_llm(
                    prompt=run_memory.get("feedback_prompt"),
                    temperature=0.0,       # deterministic, minimal randomness
                    top_p=0.3,              # only consider most probable tokens
                )

                #print(f"feedback_data type: {type(feedback_data)}")
                #print(f"{feedback_data}")
                # Example: parse if string
                feedback_data = parse_json_from_codeblock(feedback_data)

                # Route actions based on what we got
                if feedback_data.get("improvements"):
                    print("[🔁 ROUTER] Applying improvements from feedback...")
                    self.apply_improvements(feedback_data["improvements"])

                if feedback_data.get("topics"):
                    print("[🔁 ROUTER] Updating topic labels from feedback...")

                    # Identify missing topics that were added in feedback
                    original_topics = set(self.labels)
                    updated_topics = set(feedback_data["topics"])
                    missing_topics = list(updated_topics - original_topics)

                    if missing_topics:
                        print(f"[➕] New topics added: {missing_topics}")

                    # Store for memory/debug
                    self.memory[self.last_run]["new_topics_added"] = missing_topics

                    feedback_data["missing_topics"] = missing_topics
                    self.labels = feedback_data["topics"]

                self.feedback(feedback_data)

        # 6. Return the collected signals
        return self.collected_signals

    def apply_improvements(self, improvements):
        """
        Apply improvement suggestions to the agent's configuration for the next run.
        Adjusts internal config flags based on LLM feedback.
        """
        # Ensure config exists
        if not hasattr(self, "config"):
            self.config = {}

        logging.info("[🔧] Applying improvement suggestions...")

        for improvement in improvements:
            text = improvement.lower()

            # Handle missing text/content extraction issues
            if "missing text" in text or "no text" in text or "empty text" in text:
                self.config["fetch_full_text"] = True
                logging.info("  - Enabled full text fetching for all sources.")

            # Handle topic classification improvements
            if "topic classification" in text or "topic extraction" in text or "topic labeling" in text:
                self.config["use_backup_topic_extractor"] = True
                logging.info("  - Enabled backup topic extractor.")

            # Handle irrelevant or low-quality sources
            if "irrelevant source" in text or "filter sources" in text or "source quality" in text:
                self.config["source_filtering"] = True
                logging.info("  - Enabled source filtering.")

            # Handle duplicate content filtering
            if "duplicate" in text or "duplication" in text:
                self.config["dedup_strictness"] = "high"
                logging.info("  - Set deduplication strictness to HIGH.")
            elif "too strict deduplication" in text or "missing coverage" in text or len(self.collected_signals) < 50:
                self.config["quality_filtering"] = False
                logging.info("  - Set quality_filtering to False.") 
                self.config["dedup_strictness"] = "low"
                logging.info("  - Set deduplication strictness to LOW.")    


        logging.info(f"[⚙] Updated config: {self.config}")


    # --- Export ---
    def export_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.collected_signals, f, ensure_ascii=False, indent=2)
        logging.info(f"[EXPORT] Signals exported to {filepath}")


if __name__ == "__main__":
    # Example run (requires API keys in env vars)
    agent = BrandSignalCollectorAgent(
        company_name="Google",
        hours_back=168,
        max_results=50,
        country="US"
    )
    signals = agent.run_cycle()
    agent.export_to_json("brand_signals.json")
