import requests, base64
import praw
import json
import urllib.request
import re
import jmespath
import time
import logging
import numpy as np
from huggingface_hub import InferenceClient
from difflib import SequenceMatcher
from together import Together
from playwright.sync_api import sync_playwright
from serpapi import GoogleSearch
from mastodon import Mastodon
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from scrapfly import ScrapflyClient, ScrapeConfig
from typing import List, Tuple, Dict
from datetime import datetime, timezone, timedelta
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from langdetect import detect, LangDetectException
from google import genai

company_name = "Google"
hours_back = 168  # Default to prev. 7 days
region = "IN"
max_results = 50 # Default max results per source

# KEYS
NEWSAPI_KEY = "44aa2fe007c249b291b84f90ef03c77c"
GNEWSAPI_KEY = "ef5f8624efbdff7065a8b27b064230d8"

#SCRAPFLY_API_KEY = "scp-live-1a870f1d40854783b5e42f486702f116" # Ivan's API
SCRAPFLY_API_KEY = "scp-live-953159fb899846a68a58de25444888d2" # Liam's API

NVIDIA_API_KEY = "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"
SERP_API_KEY = "6ca19dd5cf3bb00dae28957e7c8ee9512aae8e654be50bf22529fadc505ff8e2"
APITUBE_API_KEY = "api_live_4FCFmHUM0FDz9E5wvS2FZYfdiE2Qsf3Air8X8mH0IXFs"
#ABSTRACT_API_KEY = "c15e5c2212c248bbbfb1e827ecd08975"
TOGETHER_API_KEY = "e4e865e00e428d4136bf9b2996d25334496c29bd0a7caf0415ee2f13845fecbd"
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
GROQ_API_KEY = ""
GEMINI_API_KEY = "AIzaSyBolQ5bS2KErY8_mqPyK-bzflLfVzRf2mA"
GEMINI_MODEL = "gemini-2.5-flash"
#GROQ_MODEL = "llama-3.3-70b-versatile"  # Use a versatile model for topic extraction
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
#HF_MODEL = "deepseek-ai/DeepSeek-V3-0324"
HF_MODEL = "HuggingFaceTB/SmolLM3-3B"
HF_TOKEN = "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"

INSTANCE_URL = "https://mastodon.social"

# Global flag to track Groq API exhaustion with time-based cooldown
GROQ_API_EXHAUSTED = False
GROQ_EXHAUSTED_TIMESTAMP = None
GROQ_COOLDOWN_SECONDS = 180  # 3 minutes

vader = SentimentIntensityAnalyzer()

labels = []

def get_user_country():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("country")
    except Exception as e:
        print("Error:", e)
        return None

country = region or get_user_country()
#print(f"Using country: {country or 'Unknown'}")

# load the tokenizer and the model (COMMENTED OUT - NOT CURRENTLY USED)
# device = "cpu"
# tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
# hf_model = AutoModelForCausalLM.from_pretrained(
#     HF_MODEL,
# ).to(device)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and small model

# SETUP SENTIMENT ANALYSIS PIPELINE (HuggingFace)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# SETUP REDDIT
reddit = praw.Reddit(
    client_id='uZdyI6absFFhAJxMYzZxaA',
    client_secret='s51CK4-b6timEvSVxVdWImDkKzDUsQ',
    user_agent='brand_signal_mvp'
)

# === Helper Functions ===

def check_groq_cooldown():
    """
    Check if Groq API cooldown period has expired.
    Returns True if Groq can be used, False if still in cooldown.
    """
    global GROQ_API_EXHAUSTED, GROQ_EXHAUSTED_TIMESTAMP
    
    if not GROQ_API_EXHAUSTED:
        return True
    
    # Check if cooldown period has expired
    if GROQ_EXHAUSTED_TIMESTAMP:
        elapsed = time.time() - GROQ_EXHAUSTED_TIMESTAMP
        if elapsed >= GROQ_COOLDOWN_SECONDS:
            print(f"[COOLDOWN EXPIRED] Groq API cooldown period ({GROQ_COOLDOWN_SECONDS}s) has passed. Re-enabling Groq.")
            GROQ_API_EXHAUSTED = False
            GROQ_EXHAUSTED_TIMESTAMP = None
            return True
        else:
            remaining = GROQ_COOLDOWN_SECONDS - elapsed
            return False
    
    return False

def set_groq_exhausted():
    """
    Mark Groq API as exhausted and start cooldown timer.
    """
    global GROQ_API_EXHAUSTED, GROQ_EXHAUSTED_TIMESTAMP
    GROQ_API_EXHAUSTED = True
    GROQ_EXHAUSTED_TIMESTAMP = time.time()
    print(f"[COOLDOWN STARTED] Groq API exhausted. Switching to Gemini for {GROQ_COOLDOWN_SECONDS}s (3 minutes).")

# === Fetch Company Mentions ===

def fetch_mastodon_posts(company, max_results, hours_back):
    mastodon = Mastodon(
        access_token='UYP4rWPy3ln5BDS4ZJNLmpoyr9MHqTTsZ9FVhb_yJOA',
        api_base_url='https://mastodon.social'
    )

    # Define cutoff time
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    results = mastodon.search(company, result_type="statuses")
    statuses = results["statuses"]

    posts = []
    for status in statuses:
        created_at = status.get("created_at")
        if created_at and created_at < cutoff_time:
            continue

        clean_text = BeautifulSoup(status["content"], "html.parser").get_text(separator=" ", strip=True)
        posts.append({
            "text": clean_text,
            "url": status["url"],
            "hashtags": [tag["name"] for tag in status["tags"]],
        })

        if len(posts) >= max_results:
            break

    return posts


# === Sentiment Analysis ===
def analyze_sentiment(text):
    tb = TextBlob(text)
    polarity = tb.polarity
    subjectivity = tb.subjectivity
    vader_score = vader.polarity_scores(text)["compound"]

    if vader_score >= 0.2:
        sentiment = "positive"
    elif vader_score <= -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, polarity, subjectivity


# === Topic Extraction using Groq ===
def extract_topics_groq(company, posts, hashtags):
    # combined_text = "\n".join(f"- {p['text']}" for p in posts[:10])  # limit input size
    combined_text = "\n".join(f"- {p['text']}" for p in posts)

    # Prepare enriched hashtag context
    hashtag_section = (
        f"\n\nHashtag/Entity Metadata:\n{', '.join(hashtags)}\n\n"
        "Please use the above hashtags to:\n"
        "- Group them by semantic category (e.g. product, location, tech, emotion, religion, etc),\n"
        "- Detect languages present among the hashtags and map cross-lingual variants,\n"
        "- Note which hashtags co-occur in the same posts (implying association),\n"
        "- Highlight any sentiment-laden hashtags (positive or negative),\n"
        "- Normalize/standardize similar hashtags (e.g. 'PRTIMES' vs 'pr_times'),\n"
        "- Provide hashtag frequency and source post examples where relevant,\n"
        "- Infer temporal trend if possible (e.g., are some hashtags trending now?).\n"
    )

    # Final prompt with hashtags and business-oriented topic analysis
    prompt = (
        f"You are a brand analyst building a scalable AI-powered framework to measure how {company}'s brand perception in digital conversations may influence business outcomes "
        "(such as deal velocity, lead quality, or client engagement).\n\n"
        f"Given the following Mastodon posts and hashtag data related to {company}, identify and explain the key discussion topics relevant to {company}'s brand, business performance, or customer trust.\n"
        "Each topic must:\n"
        "- Be clearly tied to public digital conversations around the company,\n"
        "- Offer potential impact on B2B decisions, sales, or client engagement,\n"
        "- Be grounded in specific language used in posts (cite example snippets).\n"
        "Avoid listing topics that are off-topic, generic, or unrelated to business value.\n"
        "Use the hashtag analysis to enrich topic identification.\n"
        f"{hashtag_section}"
        f"Posts:\n{combined_text}\n\n"
        "Return a clear breakdown of:\n"
        "- Topics (with a solid, detailed summary of business relevance),\n"
        "- Supporting post snippets,\n"
        "- Key hashtags related to each topic.\n"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert media analyst and brand perception strategist. Your task is to extract actionable topics from digital conversations that can inform business decisions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.25,
        "top_p": 0.35,
    }

    # Check if Groq cooldown has expired
    can_use_groq = check_groq_cooldown()
    
    if not can_use_groq:
        print("[SKIP] Groq API in cooldown period. Using Gemini API for topic extraction")
        return gemini_llm(prompt=prompt, temperature=0.25, top_p=0.35)

    max_retries = 3
    base_delay = 1.5
    
    for attempt in range(max_retries):
        try:
            # Add delay to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                time.sleep(1.0)
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** (attempt + 1))
                print(f"[RATE LIMIT] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"[ERROR] Topic extraction failed: {e}")
                break
        except Exception as e:
            print(f"[ERROR] Topic extraction failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
    
    # === Fallback: Gemini API ===
    set_groq_exhausted()
    backup_response = gemini_llm(prompt=prompt, temperature=0.25, top_p=0.35)
    return backup_response
    
def together_llm(prompt: str, temperature: float = 0.3, top_p: float = 0.7):
    """
    Call Together AI's chat completion API with error handling.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): Model ID to use. Defaults to TOGETHER_MODEL env var.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        max_tokens (int): Maximum tokens for output.

    Returns:
        str: Model's response text, or error message on failure.
    """
    try:
        together_client = Together(api_key=TOGETHER_API_KEY)  # uses TOGETHER_API_KEY from env
        response = together_client.chat.completions.create(
            #model="openai/gpt-oss-120b", (paid)
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
        )

        content = response.choices[0].message.content.strip()
        print(f"[INFO] Together AI response: {content}")
        return content

    except Exception as e:
        print(f"[ERROR] Together API request failed: {e}")
        return "Failed to get response from Together AI"

# === Hashtag/Entity Co-occurrence ===
def extract_entities(posts):
    hashtags = []
    for post in posts:
        hashtags.extend(post["hashtags"])
    # return Counter(hashtags).most_common(5)
    return hashtags

# === Main Function to Collect Signals ===

def analyze_brand_signals(company, labels, max_results=max_results, hours_back=hours_back):
    posts = fetch_mastodon_posts(company, max_results, hours_back)
    if not posts:
        return []
    
    print(f"📦 Found {len(posts)} posts about '{company}'")

    hashtags = extract_entities(posts)
    topics = extract_topics_groq(company, posts, hashtags)

    print("\n🧠 Topics (Groq):", topics)
    print("\n🔍 Hashtags/Entities:", hashtags)

    # figure out how to use the topics to classify the posts and add info below

    signals = []
    for post in posts:
        source = "Mastodon"
        sentiment_pipeline = run_sentiment(post["text"])
        sentiment, polarity, subjectivity = analyze_sentiment(post["text"])
        analysis = extract_topics_groq(company, [post], post["hashtags"])
        timestamp = post.get("created_at", datetime.now(timezone.utc).isoformat())
        topic = extract_topic_ml(post["text"], labels)  # <-- Use ML to assign topic
        
        quality_context = (
            f"source: {source} "
            f"text: {post["text"]} "
            f"url: {post['url']} "
            f"sentiment_pipeline: {sentiment_pipeline} "
            f"sentiment: {sentiment} "
            f"polarity: {polarity} "
            f"subjectivity: {subjectivity} "
            f"hashtags: {post['hashtags']} "
            f"topic: {topic} "
            f"analysis: {analysis} "
            f"timestamp: {timestamp}"
        )

        signals.append({
            "source": source,
            "text": post["text"],
            "url": post["url"],
            "sentiment_pipeline": sentiment_pipeline,
            "sentiment": sentiment,
            "sentiment_polarity": polarity,
            "sentiment_subjectivity": subjectivity,
            "hashtags": post["hashtags"],
            "topic": topic,
            "analysis": analysis,
            "timestamp": timestamp,
            "quality": run_quality_check(post["text"], quality_context),
        })

    signals.append({
        "source": "mastodon",
        "overall_analysis": topics,
        "overall_hashtags": hashtags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return signals

def fast_topic_classification(text, labels):
    # Precompute embeddings for labels
    label_embeddings = model.encode(labels, convert_to_tensor=True)
    # Encode the input text
    text_embedding = model.encode(text, convert_to_tensor=True)
    # Compute cosine similarity with each label embedding
    cos_scores = util.pytorch_cos_sim(text_embedding, label_embeddings)[0]
    # Find the highest scoring label
    best_label_idx = cos_scores.argmax()
    return labels[best_label_idx]

def get_cutoff_datetime(hours_back):
    return datetime.now(timezone.utc) - timedelta(hours=hours_back)

def normalize_timestamp(ts):
    try:
        if isinstance(ts, (int, float)):  # UNIX timestamp
            # Use new style for Python 3.11+
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        # If string/ISO already, return as is
        return ts
    except Exception as e:
        return None
'''
def run_sentiment(text):
    try:
        result = sentiment_pipe(text[:512])[0]
        return {"label": result['label'], "score": float(result['score'])}
    except:
        return {"label": "NEUTRAL", "score": 0.5}
'''

def parse_quality_response(response: str) -> dict:
    """
    Parse AI response into a dict, stripping triple backticks and 'json'
    from ``` wrappers.
    """
    try:
        cleaned = response.strip()

        # If starts with fenced code block like ```
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop the first line if it starts with ```
            if lines and lines[0].lstrip().startswith("```"):
                # Remove '```json' if present in first line
                lines[0] = lines[0].replace("```json", "").replace("```", "").strip()
                # If after removal first line is empty, drop it
                if lines[0] == "":
                    lines = lines[1:]
            # Drop the last line if it's ```
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        quality = json.loads(cleaned)

        # Validate required keys
        required = {
            "overall_quality_score",
            "is_quality",
            "dimension_scores",
            "quality_issues",
            "recommendations"
        }
        missing = required - set(quality)
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        return quality

    except json.JSONDecodeError as e:
        logging.error(f"[QUALITY_CHECK] JSON parse error: {e}")
    except Exception as e:
        logging.error(f"[QUALITY_CHECK] Unexpected parse error: {e}")

    # Fallback on error
    return {
        "overall_quality_score": 0.0,
        "is_quality": False,
        "dimension_scores": {
            "relevance": 0.0,
            "clarity": 0.0,
            "sentiment_extractable": 0.0,
            "temporal_value": 0.0,
            "actionable_insight": 0.0,
            "noise_level": 0.0
        },
        "quality_issues": ["parsing_error"],
        "recommendations": "Manual review required"
    }

def run_quality_check(text, context):
    """
    Assess signal quality for brand perception forecasting using AI model evaluation.
    
    This function evaluates whether a signal (text content) is of sufficient quality
    for downstream time-series analysis and brand perception trend identification.
    
    Args:
        text (str): The signal text content to evaluate
        
    Returns:
        dict: Quality assessment results including:
            - is_quality (bool): Whether signal meets quality standards
            - quality_score (float): Numerical quality score (0-1)
            - quality_issues (list): Identified quality problems
            - recommendations (str): Improvement suggestions
    """

    # Create comprehensive evaluation prompt for brand perception forecasting context
    evaluation_prompt = f"""
    You are evaluating the quality of a brand signal for time-series analysis and perception forecasting.
    
    CONTEXT: This signal will feed into Agent B (Perception Forecaster) which:
    - Analyzes sentiment trends over time
    - Trains ML models on brand perception data
    - Identifies patterns in consumer sentiment
    - Forecasts future brand perception shifts
    - Requires high-quality, actionable signals for accurate predictions
    
    SIGNAL TEXT TO EVALUATE:
    "{text}"

    ADDITIONAL CONTEXT IN RELATION TO SIGNAL:
    "{context}"
    
    QUALITY CRITERIA - Assess each dimension (0-1 score):
    
    1. RELEVANCE (0-1): Is this directly related to brand perception/sentiment?
    2. CLARITY (0-1): Is the text clear, coherent, and understandable?
    3. SENTIMENT_EXTRACTABLE (0-1): Can meaningful sentiment be extracted?
    4. TEMPORAL_VALUE (0-1): Does this provide time-sensitive brand insight?
    5. ACTIONABLE_INSIGHT (0-1): Can this inform brand perception trends?
    6. NOISE_LEVEL (0-1): Is signal-to-noise ratio acceptable? (1 = low noise)
    
    SCORING CALIBRATION:
    - 0.0-0.3: Poor quality (spam, gibberish, completely off-topic)
    - 0.4-0.6: Below average (tangentially related, unclear, high noise)
    - 0.7-0.8: Good quality (relevant, clear, actionable)
    - 0.9-1.0: Excellent (highly relevant, crystal clear, strong signal)

    Example:
    - Text: "Buy cheap watches now!" → Relevance: 0.1 (spam)
    - Text: "Google stock mentioned in passing during tech discussion" → Relevance: 0.5
    - Text: "Google announces major AI breakthrough - stock surges" → Relevance: 0.95

    QUALITY ISSUES TO DETECT:
    - Spam, gibberish, or incoherent content
    - Duplicate or near-duplicate content
    - Off-topic content unrelated to brand perception
    - Overly generic statements with no specific brand insight
    - Poor sentiment indicators (too neutral/ambiguous)
    - Misleading or false information
    - Content too short to be meaningful
    - Advertisement-only content without perception value

    IMPORTANT:
    - Return only a single valid JSON object with the following keys — no markdown, no extra text.
    - Do NOT use JSON literals or placeholder angle brackets.
    - Your output must parse as JSON without modifications.
    {{
        "overall_quality_score": float <average of 6 criteria scores>,
        "is_quality": boolean <true if overall_score >= 0.6>,
        "dimension_scores": {{
            "relevance": float <score>,
            "clarity": float <score>,
            "sentiment_extractable": float <score>, 
            "temporal_value": float <score>,
            "actionable_insight": float <score>,
            "noise_level": float <score>
        }},
        "quality_issues": [string <list of detected issues>],
        "recommendations": string "<specific improvement suggestions>"
    }}
    """
    try:
        # Call AI model for evaluation (placeholder - replace with actual model call)
        #response = gemini_llm(
        response = groq_llm(
        #response = together_llm(
        #response = nvidia_llm(
            prompt=evaluation_prompt,
            temperature=0.4,  # Low temperature for consistent evaluation
            top_p=0.85
        )
        
        # Parse JSON response
        #quality_assessment = json.loads(response)
        #quality_assessment = parse_quality_response(response)
        if response.strip().startswith("```") and response.strip().endswith("```"):
            # Has triple backtick fencing — parse with cleanup
            quality_assessment = parse_quality_response(response)
        else:
            # Plain JSON string
            quality_assessment = json.loads(response)

        #print(quality_assessment)
        
        # Log quality check results
        logging.info(f"[QUALITY_CHECK] Text: '{text[:100]}...'")
        logging.info(f"[QUALITY_CHECK] Overall Score: {quality_assessment['overall_quality_score']:.2f}")
        logging.info(f"[QUALITY_CHECK] Quality Status: {'PASS' if quality_assessment['is_quality'] else 'FAIL'}")
        
        # Apply quality issue penalty to overall score
        if quality_assessment['quality_issues']:
            num_issues = len(quality_assessment['quality_issues'])
            # Reduce score by 0.1 per issue, minimum score of 0.0
            penalty = min(0.1 * num_issues, quality_assessment['overall_quality_score'])
            original_score = quality_assessment['overall_quality_score']
            quality_assessment['overall_quality_score'] = max(0.0, original_score - penalty)
            
            # Re-evaluate is_quality flag based on penalized score
            quality_assessment['is_quality'] = quality_assessment['overall_quality_score'] >= 0.6
            
            logging.info(f"[QUALITY_CHECK] Issues: {quality_assessment['quality_issues']}")
            logging.info(f"[QUALITY_CHECK] Penalty Applied: -{penalty:.2f} ({num_issues} issues)")
            logging.info(f"[QUALITY_CHECK] Adjusted Score: {quality_assessment['overall_quality_score']:.2f}")
        
        # Add metadata for downstream processing
        quality_assessment.update({
            "evaluation_timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split()),
            "evaluator": "brand_perception_quality_checker_v1"
        })
        
        return quality_assessment
        
    except json.JSONDecodeError:
        logging.error("[QUALITY_CHECK] Failed to parse AI response as JSON")
        return {
            "overall_quality_score": 0.0,
            "is_quality": False,
            "quality_issues": ["evaluation_error"],
            "recommendations": "Manual review required - automated evaluation failed"
        }
    
    except Exception as e:
        logging.error(f"[QUALITY_CHECK] Error during evaluation: {str(e)}")
        return {
            "overall_quality_score": 0.0,
            "is_quality": False,
            "quality_issues": ["system_error"],
            "recommendations": "System error - manual review required"
        }


def run_sentiment(text):
    try:
        result = sentiment_pipe(text[:512])[0]
        
        raw_score = float(result['score'])
        
        # 1️⃣ Round to manageable precision (3 decimal places)
        rounded_score = round(raw_score, 3)
        
        # 2️⃣ Map to confidence buckets for interpretability
        if rounded_score >= 0.9:
            confidence_bucket = "HIGH"
        elif rounded_score >= 0.7:
            confidence_bucket = "MEDIUM"
        else:
            confidence_bucket = "LOW"
        
        # 3️⃣ OPTIONAL: Apply temperature scaling (static 0.9 temp here)
        calibrated_score = round(1 / (1 + np.exp(-((raw_score - 0.5) / 0.9))), 3)

        return {
            "label": result['label'],
            "score": calibrated_score,
            "confidence_bucket": confidence_bucket,
            "raw_score": raw_score
        }
    except Exception as e:
        return {
            "label": "NEUTRAL",
            "score": 0.5,
            "confidence_bucket": "LOW",
            "error": str(e)
        }

def extract_topic_ml(text, labels, backup=None):
    if not text or text.strip() == "":
        return "Other"
    if not backup:
        return fast_topic_classification(text, labels)
    else:
        prompt = (
            "You are an expert brand analyst.\n"
            "Your task is to read the following content and determine the single most relevant topic it belongs to.\n"
            f"Here is the list of existing topics: {', '.join(labels)}\n\n"
            "Rules:\n"
            "- If you believe an existing topic is a perfect fit, choose it.\n"
            "- If the fit is only partial or weak, PROPOSE a new, concise topic instead.\n"
            "- The topic should be 1–4 words only.\n"
            "- Prefer creating a new topic if it captures the content more accurately than any existing one.\n"
            "- Return only the topic string, with no explanations or extra text.\n\n"
            f"Content:\n{text}"
        )
        #backup_response = together_llm(prompt=prompt, temperature=0.35, top_p=0.65)
        #backup_response = groq_llm(prompt=prompt, temperature=0.35, top_p=0.65)
        #backup_response = gemini_llm(prompt=prompt, temperature=0.25, top_p=0.35)
        backup_response = groq_llm(prompt=prompt, temperature=0.25, top_p=0.35)
        return backup_response 

def fetch_reddit(company, labels, max_results=50, hours_back=hours_back):
    cutoff = get_cutoff_datetime(hours_back)
    submissions = reddit.subreddit('all').search(company, limit=max_results)
    posts = []
    for submission in submissions:
        created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created < cutoff:
            continue  # Exclude older posts
        text = (submission.title or '') + " " + (submission.selftext or '')
        sentiment = run_sentiment(text)
        topic = extract_topic_ml(text, labels)

        quality_context = (
            f"source: Reddit "
            f"timestamp: {normalize_timestamp(submission.created_utc)} "
            f"title: {submission.title} "
            f"text: {submission.selftext} "
            f"sentiment: {sentiment} "
            f"url: {submission.url} "
            f"author: u/{submission.author} "
            f"topic: {topic}"
        )

        posts.append({
            "source": "Reddit",
            "timestamp": normalize_timestamp(submission.created_utc),
            "title": submission.title,
            "text": submission.selftext,
            "sentiment": sentiment,
            "url": submission.url,
            "author": f"u/{submission.author}",
            "topic": topic,
            "quality": run_quality_check(text, quality_context)
        })
    return posts

def fetch_news(company, labels, max_results=50, hours_back=hours_back):
    cutoff = get_cutoff_datetime(hours_back).isoformat()
    to_date = datetime.now(timezone.utc).isoformat()
    # url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&pageSize={max_results}"
    url = "https://newsapi.org/v2/everything"  
    params = {
        "q": company,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
        "pageSize": max_results,
        "from": cutoff,
        "to": to_date
    }
    resp = requests.get(url, params=params)
    articles = []
    if resp.status_code == 200:
        news = resp.json()
        for item in news.get('articles', []):
            title = item.get('title', '')
            text = title + ". " + (item.get('description') or '')
            sentiment = run_sentiment(text)
            topic = extract_topic_ml(text, labels)

            quality_context = (
                f"source: News "
                f"timestamp: {item.get('publishedAt')} "
                f"title: {title} "
                f"text: {item.get('description', '')} "
                f"sentiment: {sentiment} "
                f"url: {item.get('url')} "
                f"author: {item.get('author', '')} "
                f"topic: {topic}"
            )

            articles.append({
                "source": "News",
                "timestamp": item.get("publishedAt"),
                "title": title,
                "text": item.get('description', ''),
                "sentiment": sentiment,
                "url": item.get('url'),
                "author": item.get('author', ''),
                "topic": topic,
                "quality": run_quality_check(text, quality_context)
            })
    return articles

def fetch_hackernews(company, labels, max_results=50, hours_back=hours_back):
    cutoff_unix = int(get_cutoff_datetime(hours_back).timestamp())
    # url = f"https://hn.algolia.com/api/v1/search?query={company}&tags=story"
    url = "https://hn.algolia.com/api/v1/search"
    params = {
        "query": company,
        "tags": "story",
        "numericFilters": f"created_at_i>{cutoff_unix}",
        "hitsPerPage": max_results,
        "page": 0
    }
    results = requests.get(url, params=params).json()
    stories = []
    for item in results['hits'][:max_results]:
        text = item.get('title', '') + '. ' + (item.get('story_text', '') or '')
        sentiment = run_sentiment(text)
        topic = extract_topic_ml(text, labels)

        quality_context = (
            f"source: HackerNews "
            f"timestamp: {item.get('created_at')} "
            f"title: {item.get('title', '')} "
            f"text: {item.get('story_text', '')} "
            f"sentiment: {sentiment} "
            f"url: {item.get('url', '')} "
            f"author: {item.get('author', '')} "
            f"topic: {topic}"
        )

        stories.append({
            "source": "HackerNews",
            "timestamp": item.get('created_at'),
            "title": item.get('title', ''),
            "text": item.get('story_text', ''),
            "sentiment": sentiment,
            "url": item.get('url', ''),
            "author": item.get('author', ''),
            "topic": topic,
            "quality": run_quality_check(text, quality_context)
        })
    return stories

def fetch_news_articles(company: str):
    url = "https://api.apitube.io/v1/news/everything"
    params = {
        "title": company,
        "api_key": APITUBE_API_KEY
    }

    '''
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raises an error for bad status codes
    return response.json()
    '''

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Normalize output — expect either {"articles": [...]} or a top-level list
        if isinstance(data, dict) and "articles" in data:
            articles = data["articles"]
        elif isinstance(data, list):
            articles = data
        else:
            articles = []

        normalized = []
        for item in articles:
            if not isinstance(item, dict):
                continue
            sentiment = run_sentiment(f'{item.get('title', '')}. {item.get('description', '')}')
            topic = extract_topic_ml(f'{item.get('title', '')} {item.get('description', '')}', labels if 'labels' in globals() else ['Other'])

            quality_context = (
                f"source: APITube News "
                f"timestamp: {item.get('publishedAt') or item.get('date')} "
                f"title: {item.get('title', '')} "
                f"text: {item.get('description', '')} "
                f"sentiment: {sentiment} "
                f"url: {item.get('url', '')} "
                f"author: {item.get('author', '')} "
                f"topic: {topic}"
            )

            normalized.append({
                "source": "APITube News",
                "timestamp": item.get("publishedAt") or item.get("date"),
                "title": item.get("title", ""),
                "text": item.get("description", ""),
                "sentiment": sentiment,
                "url": item.get("url", ""),
                "author": item.get("author", ""),
                "topic": topic,
                "quality": run_quality_check(
                    f"{item.get('title', '')}. {item.get('description', '')}",
                    quality_context
                ),
            })

        return normalized

    except Exception as e:
        print(f"Error fetching APITube news: {e}")
        return []

def get_gnews_articles(api_key: str, query: str, max_results: int = 10) -> List[str]:
    """
    Fetch GNews article headlines for a given query.
    """
    prompt = (
        f"For the brand '{company_name}', classify it into one of the following categories: "
        "general, world, nation, business, technology, entertainment, sports, science, or health. "
        "Respond with only the most relevant category."
    )
    category = groq_llm(prompt, temperature=0.0, top_p=1.0)
    #category = gemini_llm(prompt, temperature=0.0, top_p=1.0)

    url = (
        f"https://gnews.io/api/v4/top-headlines?"
        f"q={urllib.parse.quote(query)}&lang=en&category={category}&country={country}&max={max_results}&apikey={api_key}"
    )
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])
            headlines = [article['title'] for article in articles if 'title' in article]
            return headlines
    except Exception as e:
        print(f"Error fetching GNews articles: {e}")
        return []
    
# SCRAPFLY API to fetch company news from multiple sources
def fetch_company_news(company_name: str, hours_back : int):
    base_url = "https://api.scrapfly.io/scrape"
    headers = {}

    company_url = get_brand_review_url_llm(company_name, review_platform="Trustpilot")
    print(f"Company URL for reviews: {company_url}")

    ''' # ADD MORE IF NEEDED LATER (PROVIDED QUOTA ALLOWS)
        {
            "source": "Forbes",
            "url": f"https://www.forbes.com/search/?q={company_name}",
            "prompt": "Get all the articles and display as a JSON. Include the publishing dates (in ISO 8601 format [YYYY-MM-DD]) and descriptions."
        },
        {
            "source": "CNN",
            "url": f"https://edition.cnn.com/search?q={company_name}&from=0&size=50&page=1&sort=relevance&types=article&section=",
            "prompt": "Get all the articles and display as a JSON. Include the publishing dates (in ISO 8601 format [YYYY-MM-DD]) and descriptions."
        },
        {
            "source": "Trustpilot",
            "url": f"https://www.trustpilot.com/review/{company_url}",
            # "url": company_url,
            "prompt": "Get all user reviews under 'All reviews' and display as a JSON. Include the publishing dates (in ISO 8601 format [YYYY-MM-DD]), context of review, review rating (out of 5), author of review, and country of reviewer (in ISO 3166-1 alpha-2 format)."
        }
    '''

    urls = [
        
        {
            "source": "Bloomberg",
            "url": f"https://www.bloomberg.com/search?query={company_name}&resource_subtypes=Article&sort=relevance",
            "prompt": "Get all the articles and display as a JSON. Include the publishing dates (in ISO 8601 format [YYYY-MM-DD]) and descriptions."
        }
    ]

    combined_results = []

    for site in urls:
        params = {
            "tags": "player,project:default",
            "extraction_prompt": site["prompt"],
            "asp": "true",
            "render_js": "true",
            "key": SCRAPFLY_API_KEY,
            "url": site["url"]
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            # articles = data.get("result", [])
            articles = data["result"]["extracted_data"]["data"]

            all_articles_data = []

            #print(f"articles: {articles}")
            for article in articles:
                # Skip articles without required fields
                if not isinstance(article, dict):
                    continue
                
                # Extract fields with safe fallbacks
                title = article.get("title", "")
                description = article.get("description", "")
                article_date = article.get("publishing_date") or article.get("date") or article.get("published_at") or datetime.now(timezone.utc).isoformat()
                
                # Skip if no meaningful content
                if not title and not description:
                    continue
                
                article_text = f"{title}. {description}".strip()
                sentiment_result = run_sentiment(article_text)
                article_source = site['source']

                quality_context = (
                    f"date: {article_date} "
                    f"source: {article_source} "
                    f"data: {article_text} "
                    f"sentiment: {sentiment_result}"
                )

                all_articles_data.append({
                    "title": title,
                    "description": description,
                    "date": article_date,
                    "source": f"ScrapFly: {site['source']}",
                    "sentiment": sentiment_result,
                    "quality": run_quality_check(article_text, quality_context),
                })
            
            """
            if isinstance(articles, list):
                for article in articles:
                    article["source"] = site["source"]
                combined_results.extend(articles)
            """

            #combined_results.extend(articles)
            combined_results.extend(all_articles_data)

        except Exception as e:
            print(f"Error fetching from {site['source']}: {e}")

        time.sleep(1)

    """
    # Filter articles based on the cutoff time
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    filtered = []
    articles = combined_results[1:]  # Skip the first 'articles' string
    
    for article in articles:
        date_str = article['date']
        if not date_str:
            continue  # skip articles without date

        try:
            article_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            if article_date >= cutoff_time:
                filtered.append(article)
        except ValueError as e:
            print(f"Error parsing date: {date_str} – {e}")
    """



    return combined_results
    
def get_google_finance_data(ticker: str):
    """
    Fetches Google Finance data for the given stock ticker using SerpApi.
    
    Args:
        ticker (str): Stock ticker symbol, e.g., "GOOGL:NASDAQ".
        api_key (str): Your SerpApi API key.
    
    Returns:
        dict: Parsed JSON response from SerpApi Google Finance engine.
    """
    params = {
        "engine": "google_finance",
        "q": ticker,
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results 

def get_google_finance_data_with_window(ticker: str, hours_back: int):
    """
    Fetches Google Finance data for the given stock ticker using SerpApi,
    selecting the history window based on hours_back.
    """
    window = hours_back_to_window(hours_back)
    #window = "MAX"
    params = {
        "engine": "google_finance",
        "q": ticker,
        "window": window,
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results
'''
def get_finance_ticker(company_name: str) -> str:
    prompt = f"""
    You are a financial data assistant.
    Given the name of a company, respond with its primary stock market ticker symbol only.
    Rules:
    - Return only the ticker symbol (e.g., AAPL, MSFT).
    - Do not include exchange names, explanations, or any other text.
    - If the company is private or no ticker is found, return "None".

    Company name: {company_name}
    """
    response = groq_llm(prompt, temperature=0.0, top_p=1.0)
    # print(f"TICKER {response}")
    return response
'''
'''
def get_finance_ticker(company_name: str) -> str:
    prompt = f"""
    You are a financial data assistant.

    Task:
    - Given a company name, return its primary stock listing in the format: TICKER:EXCHANGE.
      Examples:
      - Apple → AAPL:NASDAQ
      - Google (Alphabet) → GOOGL:NASDAQ
      - Tesla → TSLA:NASDAQ
      - Coca-Cola → KO:NYSE
    - Use the company's main US exchange listing when available.
    - Use uppercase for both exchange and ticker.
    - If the company is private or has no ticker, return exactly: None

    Rules:
    - Output only TICKER:EXCHANGE (no explanations, no extra text).
    - Do not include country codes (no US:, no .AX, no .L).
    - Do not return multiple tickers.

    Company: {company_name}
    Output:
    """
    response = groq_llm(prompt, temperature=0.0, top_p=1.0)
    return response.strip()
'''

def get_finance_ticker(company_name: str) -> str:
    prompt = f"""
    You are a financial data assistant.

    Task:
    - Given a company name, output its *primary stock exchange listing* in the format:
      EXCHANGE:TICKER
    - If the company is based outside the U.S., return the main exchange in its home country.
      Examples:
      - Toyota → TSE:7203
      - Samsung → KRX:005930
      - Nestlé → SIX:NESN
      - HSBC → LSE:HSBA

    Rules:
    - Use the exchange’s standard abbreviation (NASDAQ, NYSE, LSE, TSE, KRX, HKEX, SIX, Euronext, etc.).
    - Ticker must be uppercase (except when the exchange uses numbers).
    - If multiple listings exist, choose the primary home-country listing unless the company is known
      primarily by its U.S. ADR (e.g., BABA → NYSE:BABA).
    - If no ticker exists, return exactly: None
    - Output ONLY EXCHANGE:TICKER — no extra text.

    Company: {company_name}
    Output:
    """
    response = groq_llm(prompt, temperature=0.0, top_p=1.0)
    return response.strip()

def hours_back_to_window(hours_back: int) -> str:
    """
    Maps hours_back value to SerpApi window parameter.
    """
    if hours_back <= 24:
        return "1D"
    elif hours_back <= 5 * 24:
        return "5D"
    elif hours_back <= 30 * 24:
        return "1M"
    elif hours_back <= 6 * 30 * 24:
        return "6M"
    elif hours_back <= 365 * 24:
        return "1Y"
    elif hours_back <= 5 * 365 * 24:
        return "5Y"
    elif hours_back <= 10 * 365 * 24:  # Optionally adjust the boundary if needed.
        return "MAX"
    else:
        return "MAX"

def ai_synthesise_single(data: dict, label: str, hours_back: int = None, is_windowed: bool = False) -> str:
    """
    AI synthesis for financial dataset with context-aware prompting.
    
    Args:
        data: Financial data dictionary
        label: Dataset label (e.g., "Live Data", "Window (168h)")
        hours_back: Time window in hours (for windowed analysis)
        is_windowed: If True, focuses on time-period specific trends
    """
    if is_windowed and hours_back:
        # Time-focused analysis for windowed data
        time_description = f"{hours_back} hours ({hours_back // 24} days)" if hours_back >= 24 else f"{hours_back} hours"
        prompt = (f"""
        You are a senior financial research analyst specializing in short-term market trends.
        Analyze the following {label} finance dataset covering the past {time_description}.
        
        FOCUS AREAS FOR TIME-WINDOWED ANALYSIS:
        1. **Price Movement Over Period**: Identify the trend direction (upward, downward, volatile) during this specific {time_description} window.
        2. **Volatility & Trading Patterns**: Highlight any significant price swings, volume spikes, or unusual trading activity within this timeframe.
        3. **Recent Catalysts**: Identify news, events, or market conditions that drove price changes during this period.
        4. **Momentum Indicators**: Assess whether the stock gained or lost momentum during this window.
        5. **Relative Performance**: Compare performance to broader market indices during this same period (if available).
        6. **Support/Resistance Levels**: Note any technical price levels tested or broken during this timeframe.
        7. **Short-term Sentiment**: Extract sentiment signals from recent price action and volume.
        
        IMPORTANT:
        - Focus exclusively on data from the past {time_description}
        - Emphasize temporal patterns and trend changes
        - Highlight what changed during this specific window
        - Use precise timestamps and date ranges when referencing events
        - Quantify percentage changes over this period
        
        Formatting:
        - Use clear, concise financial language
        - Present findings in bullet points
        - Include specific numbers with context
        - Do not output raw JSON — only the analysis
        
        Dataset ({time_description} window):
        {data}
        """)
    else:
        # Comprehensive analysis for live/current data
        prompt = (f"""
        You are a senior financial research analyst.
        Analyze the following {label} finance dataset and produce a comprehensive, data-rich narrative covering the company's overall financial position and history.

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. **Company Profile**: Identify the entity, sector/industry, market position, and business model.
        2. **Current Valuation**: State latest price, market capitalization, P/E ratio, dividend yield, price-to-book, and other key metrics.
        3. **Historical Performance**: Highlight price movements and percentage changes across ALL available timeframes (1 day, 1 week, 1 month, YTD, 1 year, 5 years, since inception).
        4. **Financial Health**: Summarize revenue, net income, EPS, operating margins, debt levels, cash reserves, and cash flow trends.
        5. **Growth Trajectory**: Analyze revenue growth rates, earnings growth, and expansion patterns over multiple years.
        6. **Market Sentiment**: Mention analyst consensus, target price ranges, recommendation ratings (buy/hold/sell), and institutional ownership.
        7. **News & Events**: Identify major announcements, product launches, regulatory filings, earnings reports, and market-moving events.
        8. **Risk Assessment**: Include risk factors, competitive threats, regulatory challenges, and market vulnerabilities.
        9. **Long-term Outlook**: Provide forward-looking commentary on growth drivers, strategic initiatives, and market opportunities/threats.
        10. **Comparative Analysis**: Position the company relative to peers and industry benchmarks (if data available).

        Formatting rules:
        - Write in clear, professional financial language
        - Use bullet points or numbered lists for clarity
        - Include both absolute numbers and percentage changes
        - Provide historical context for current metrics
        - Cite specific timeframes when discussing trends
        - Avoid adding data not present in the dataset
        - Do not output raw JSON or code — only the narrative

        Dataset:
        {data}
        """)
    
    response = groq_llm(prompt, temperature=0.2, top_p=1.0)
    return response
'''
def ai_synthesise_overall(summary1: str, summary2: str) -> str:
    """
    Placeholder for combining two AI summaries into one.
    Replace with an actual AI API call for a richer synthesis.
    """
    return f"Combined Analysis:\n- {summary1}\n- {summary2}\nOverall: Data compared from current snapshot and historical range."
'''

def ai_synthesise_overall(data_Live: str, data_windowed: str) -> str:
    """
    Uses the NVIDIA LLM to combine two AI-generated summaries:
    - data_Live: current/live financial snapshot
    - data_windowed: historical or windowed dataset summary
    Produces a comparative synthesis highlighting changes, trends, and outlook.
    """

    prompt = f"""
    You are a senior financial strategist.
    You will be given two separate financial summaries for the same entity:

    - data_Live: the most recent, live financial snapshot
    - data_windowed: a historical or windowed dataset capturing past performance

    Task:
    1. Combine these summaries into one coherent, data-rich synthesis.
    2. Identify key similarities and differences between live and historical data.
    3. Highlight changes in valuation, performance metrics, sentiment, and risk factors.
    4. Note any newly emerging trends, events, or risks present in the live data but absent in historical data.
    5. Discuss any sustained trends or consistent signals across both datasets.
    6. Provide an overall forward-looking assessment that accounts for both perspectives.

    Formatting rules:
    - Use bullet points or short paragraphs for clarity.
    - Reference specific metrics or facts from each summary when making comparisons.
    - Keep language professional and precise.
    - Avoid generic filler text — focus on actual differences and insights.
    - End with a concise "Overall Outlook" paragraph summarizing the combined insights.

    Live Data Summary:
    {data_Live}

    Historical/Windowed Data Summary:
    {data_windowed}
    """

    # Call NVIDIA LLM with low temperature for consistency and detailed comparisons
    #response = gemini_llm(prompt, temperature=0.2, top_p=1.0)
    response = groq_llm(prompt, temperature=0.2, top_p=1.0)

    #print(f"ai combined {response}")

    return response


def get_combined_finance_analysis(company: str, ticker: str, hours_back: int):
    """
    Runs both finance data fetches (no window & window-based),
    synthesises each individually, then produces an overall synthesis.
    """
    try:
        # Run both data pulls
        data_live = get_google_finance_data(ticker)
        
        # Extract stock ticker from summary section
        stock_ticker = None
        if "summary" in data_live and "stock" in data_live["summary"]:
            stock_ticker = data_live["summary"]["stock"]
        else:
            # Fallback to original ticker if structure is unexpected
            stock_ticker = ticker
            logging.warning(f"[FINANCE] No summary.stock found, using original ticker: {ticker}")
            logging.warning(f"[FINANCE] Available top-level keys: {list(data_live.keys())}")
       
        data_windowed = get_google_finance_data_with_window(ticker, hours_back)
        
        # Determine which data to analyze - prefer summary, fallback to entire response
        live_data_to_analyze = data_live
        
        # Synthesise each individually with differentiated prompts
        # Live: comprehensive historical analysis from inception
        summary_live = ai_synthesise_single(live_data_to_analyze, "Live Data", hours_back=None, is_windowed=False)
        # Windowed: focused on recent time period trends
        summary_windowed = ai_synthesise_single(data_windowed, f"Window ({hours_back}h)", hours_back=hours_back, is_windowed=True)

        #print(f"SUMMARY LIVE:\n{summary_live}\n")
        #print(f"SUMMARY WINDOWED:\n{summary_windowed}\n")

        #exit()
        
        # Overall synthesis
        combined = ai_synthesise_overall(summary_live, summary_windowed)

        #print(f"combined: {combined}")

        #exit()

        combined_text = (
            (company_name) + " " +
            (ticker) + " " +
            (str(data_live) or "") + " " +
            (str(data_windowed) or "") + " " +
            (summary_live) + " " +
            (summary_windowed) + " " +
            (combined)
        ).strip()
        
        sentiment_result = run_sentiment(combined_text)

        quality_context = (
            f"company: {company_name} "
            f"data: {combined_text} "
            f"sentiment: {sentiment_result}"
        )

        data = [{
            "source": "Finance",
            "company": company_name,
            "ticker": ticker,
            "live_data": data_live,
            "windowed_data": data_windowed,
            "live_summary": summary_live,
            "window_summary": summary_windowed,
            "overall_synthesis": combined,
            "quality": run_quality_check(combined_text, quality_context),
        }]
        
        logging.info(f"[FINANCE] Successfully analyzed {company_name} ({ticker})")
        return data
        
    except KeyError as e:
        logging.error(f"[FINANCE] Missing expected key in API response: {e}")
        logging.error(f"[FINANCE] Available keys in data_live: {list(data_live.keys()) if 'data_live' in locals() else 'N/A'}")
        return [{
            "source": "Finance",
            "company": company_name,
            "ticker": ticker,
            "error": f"API structure mismatch: {str(e)}",
            "quality": {"overall_quality_score": 0.0, "is_quality": False, "quality_issues": ["api_error"]}
        }]
    except Exception as e:
        logging.error(f"[FINANCE] Unexpected error in finance analysis: {e}")
        return [{
            "source": "Finance",
            "company": company_name,
            "ticker": ticker,
            "error": str(e),
            "quality": {"overall_quality_score": 0.0, "is_quality": False, "quality_issues": ["system_error"]}
        }]

# --- Step 1: Google Search for Tweet URLs ---
def get_tweet_urls(company_name, country):
    """
    Fetch tweet URLs from SerpAPI, with fallback to alternate query terms.
    Tries 'x posts' first, then 'twitter posts' if no results found.
    """
    search_queries = [
        company_name + " x posts",
        company_name + " twitter posts"
    ]
    
    for query in search_queries:
        params = {
            "q": query,
            "hl": "en",
            "gl": country,
            "api_key": SERP_API_KEY
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check if 'latest_posts' exists and has content
            if "latest_posts" in results and results["latest_posts"]:
                latest_posts = results["latest_posts"]
                # Extract tweet URLs from latest_posts
                tweet_urls = [post["link"] for post in latest_posts if "link" in post]
                
                if tweet_urls:
                    print(f"Found {len(tweet_urls)} tweet URLs using query: '{query}'")
                    return tweet_urls
            
            # If no results with current query, try next one
            print(f"No results with query '{query}', trying alternate...")
            
        except KeyError as e:
            print(f"KeyError with query '{query}': {e}")
            continue
        except Exception as e:
            print(f"Error fetching tweets with query '{query}': {e}")
            continue
    
    # If all queries fail, return empty list
    print(f"Could not find any tweets for '{company_name}' after trying all query variations")
    return []

# --- Step 2: Scrape Individual Tweet Content ---
def scrape_tweet(url: str) -> dict:
    """
    Scrape a single tweet page for Tweet thread e.g.:
    https://twitter.com/Scrapfly_dev/status/1667013143904567296
    Return parent tweet, reply tweets and recommended tweets
    """
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        # go to url and wait for the page to load
        page.goto(url)
        page.wait_for_selector("[data-testid='tweet']")

        # find all tweet background requests:
        tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
        for xhr in tweet_calls:
            data = xhr.json()
            return data['data']['tweetResult']['result']
        
        return {}
    
# --- Step 3: Run Everything ---
def scrape_company_tweets(company_name, country=country):
    tweet_urls = get_tweet_urls(company_name, country)
    all_tweet_data = []

    for url in tweet_urls:
        print(f"Scraping tweet: {url}")
        tweet_data = scrape_tweet(url)
        if tweet_data:

            #run sentiment
            # Extract tweet text content for sentiment analysis
            # Twitter API response structure: data.tweetResult.result.legacy.full_text
            tweet_text = ""
            if "legacy" in tweet_data:
                tweet_text = tweet_data["legacy"].get("full_text", "")
            elif "tweet" in tweet_data:
                tweet_text = tweet_data["tweet"].get("legacy", {}).get("full_text", "")
            
            # Fallback to top-level fields if nested structure not found
            if not tweet_text:
                tweet_text = (tweet_data.get("full_text") or tweet_data.get("text") or "").strip()
            
            tweet_text = tweet_text.strip()
            
            # Only run sentiment if there is text
            if tweet_text:
                sentiment_result = run_sentiment(tweet_text)
            else:
                sentiment_result = {"label": "NEUTRAL", "score": 0.5, "confidence_bucket": "LOW"}
                print(f"Warning: No text extracted from tweet at {url}")

            quality_context = (
                f"url: {url} "
                f"text: {tweet_text} "
                f"sentiment: {sentiment_result}"
            )

            all_tweet_data.append({
                "url": url,
                "text": tweet_text,
                "data": tweet_data,
                "sentiment": sentiment_result,
                "quality": run_quality_check(tweet_text, quality_context),
            })

    return all_tweet_data

def extract_topics_with_groq(groq_api_key: str, headlines: list, model: str = GROQ_MODEL):
    """
    Use Groq LLM to extract common topics from news headlines.
    """
    if not headlines:
        return "No headlines available."

    prompt = (
        "You are a company-focused news analyst.\n\n"
        "Given a list of recent news headlines, extract only the key topics that are directly relevant to a specific company being monitored. "
        "Ignore headlines that are unrelated to the company, even if they are about similar products or competitors.\n\n"
        "Your output should contain:\n"
        "1. A list of 3–5 core topics or themes that are relevant to the company.\n"
        "2. Each topic should have:\n"
        "   - A short title\n"
        "   - A one-line explanation\n"
        "   - (Optional) Example headlines that support it\n\n"
        "Do NOT include general tech news, competitor product launches, or unrelated stories.\n\n"
        f"Company being analyzed: {company_name}\n\n"
        "Headlines:\n"
        + "\n".join(f"- {h}" for h in headlines)
        + "\n\nNow extract and return only the most relevant company-specific topics."
    )

    # Check if Groq cooldown has expired
    can_use_groq = check_groq_cooldown()
    
    if not can_use_groq:
        print("[SKIP] Groq API in cooldown period. Using Gemini API for headline topic extraction")
        gemini_prompt = (
            "You are a company-focused news analyst.\n\n"
            "Given a list of recent news headlines, extract only the key topics that are directly relevant to a specific company being monitored. "
            "Ignore headlines that are unrelated to the company, even if they are about similar products or competitors.\n\n"
            "Your output should contain:\n"
            "1. A list of 3–5 core topics or themes that are relevant to the company.\n"
            "2. Each topic should have:\n"
            "   - A short title\n"
            "   - A one-line explanation\n"
            "   - (Optional) Example headlines that support it\n\n"
            "Do NOT include general tech news, competitor product launches, or unrelated stories.\n\n"
            f"Company being analyzed: {company_name}\n\n"
            "Headlines:\n"
            + "\n".join(f"- {h}" for h in headlines)
            + "\n\nNow extract and return only the most relevant company-specific topics."
        )
        return gemini_llm(prompt=gemini_prompt, temperature=0.2, top_p=0.4)
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.4,
    }

    max_retries = 3
    base_delay = 1.5
    
    for attempt in range(max_retries):
        try:
            # Add delay to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                time.sleep(1.0)
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429 and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** (attempt + 1))
                print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print("Groq API Error:", response.text)
                break
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
    
    # === Fallback: Gemini API ===
    set_groq_exhausted()
    gemini_prompt = (
        "You are a company-focused news analyst.\n\n"
        "Given a list of recent news headlines, extract only the key topics that are directly relevant to a specific company being monitored. "
        "Ignore headlines that are unrelated to the company, even if they are about similar products or competitors.\n\n"
        "Your output should contain:\n"
        "1. A list of 3–5 core topics or themes that are relevant to the company.\n"
        "2. Each topic should have:\n"
        "   - A short title\n"
        "   - A one-line explanation\n"
        "   - (Optional) Example headlines that support it\n\n"
        "Do NOT include general tech news, competitor product launches, or unrelated stories.\n\n"
        f"Company being analyzed: {company_name}\n\n"
        "Headlines:\n"
        + "\n".join(f"- {h}" for h in headlines)
        + "\n\nNow extract and return only the most relevant company-specific topics."
    )
    return gemini_llm(prompt=gemini_prompt, temperature=0.2, top_p=0.4)

def _text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def deduplicate_signals(signals, strictness=None):
    seen = set()
    deduped = []

    if strictness == "none": # none - no removals
        min_similarity = 1
    elif strictness == "low":
        min_similarity = 0.9  # more     tolerant → fewer removals
    elif strictness == "high":
        min_similarity = 0.6  # more aggressive → removes near duplicates
    else:
        min_similarity = 0.8  # balanced

    if not strictness == "none":
        for sig in signals:
            if not isinstance(sig, dict):
                deduped.append(sig)
                continue
            url = sig.get("url", "").lower()
            # Skip deduplication if url contains 'x.com'
            source = sig.get("source", "").lower()
            if "x.com" in url or "News" in source or "HackerNews" in source or "APITube News" in source or "ScrapFly" in source or "Finance" in source:
                deduped.append(sig)
                continue
            
            text = f"{sig.get('title', '')} {sig.get('text', '')} {sig.get('full_text', "")} {sig.get('description', "")}".strip().lower()
            if not text:
                deduped.append(sig)
                continue

            is_duplicate = any(
                _text_similarity(text, existing) >= min_similarity
                for existing in seen
            )
            if not is_duplicate:
                seen.add(text)
                deduped.append(sig)
    else:
        deduped = signals # return all

    return deduped

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
    '''
    # fallback - try entire text
    text = text.strip()
    if not text:
        raise ValueError("No JSON found in text")
    return json.loads(text)
    '''

    # fallback - try entire text
    if not text:
        raise ValueError("No JSON found in text to parse")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"[JSON_PARSE] Failed to parse entire text: {str(e)}")
        logging.error(f"[JSON_PARSE] Raw text: {text}")
        return {}  # or a default placeholder dict

def nvidia_llm(prompt: str, temperature: float, top_p: float) -> str:
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }

    prompt = prompt

    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "None"

def hf_llm(
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.3
):
    """
    Send a chat completion request to Hugging Face Inference API using the InferenceClient.

    Args:
      prompt (str): The user prompt.
      temperature (float): Sampling temperature.
      top_p (float): Nucleus sampling parameter.
      model (str): The Hugging Face model ID.

    Returns:
      str: The model's response content, or None on error.
    """
    try:
        # Initialize the HF Inference client
        client = InferenceClient(HF_TOKEN)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Call the chat completions endpoint
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return None

def groq_llm(
        prompt: str, 
        temperature: float = 0.0,
        top_p: float = 0.3,
        html_content: str = None,
        max_retries: int = 3,
        base_delay: float = 1.5) -> str:
    """
    Calls Groq chat completion API with the given prompt, returning the assistant's reply.
    Implements exponential backoff retry logic and rate limiting.
    """
    # Check if Groq cooldown has expired
    can_use_groq = check_groq_cooldown()
    
    if not can_use_groq:
        print("[SKIP] Groq API in cooldown period. Using Gemini API")
        return gemini_llm(prompt=prompt, temperature=temperature, top_p=top_p)
    
    # Append HTML content as cleaned text if provided
    if html_content:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            extracted_text = soup.get_text(separator="\n").strip()
            prompt += f"\n\n---\nExtracted HTML Content:\n{extracted_text}"
        except Exception as e:
            print(f"Error parsing HTML: {e}")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert media analyst and brand perception strategist. "
                    "Your task is to extract actionable topics from digital conversations "
                    "that can inform business decisions. If HTML content is provided, "
                    "interpret it accurately as part of your analysis."
                ),
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
    }
    
    for attempt in range(max_retries):
        try:
            # Add delay before each request to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)
            else:
                time.sleep(1.0)  # Small delay for first request
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            if response.status_code == 429:  # Rate limit error
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** (attempt + 1))
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Rate limit exceeded after {max_retries} retries: {e}")
                    break
            else:
                print(f"Request failed: {e}")
                break
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
        except (KeyError, IndexError) as e:
            print(f"Unexpected response format: {e}")
            break
    
    # === Fallback: Gemini API ===
    set_groq_exhausted()
    return gemini_llm(prompt=prompt, temperature=temperature, top_p=top_p)

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

def extract_signal_text(signal: dict) -> str:
    """
    Extract meaningful text from a signal for topic extraction.
    Handles multiple signal formats from different sources.
    """
    text_parts = []
    
    # Try common text fields
    for field in ['title', 'text', 'full_text', 'description', 'selftext']:
        value = signal.get(field, '')
        if value and isinstance(value, str):
            text_parts.append(value)
    
    # For finance signals, extract synthesis text
    if signal.get('source') == 'Finance':
        for field in ['overall_synthesis', 'live_summary', 'window_summary']:
            value = signal.get(field, '')
            if value and isinstance(value, str):
                text_parts.append(value)
    
    return ' '.join(text_parts).strip()

def extract_topics_from_signals(company_name: str, signals: list) -> str:
    """
    Extract common topics from a collection of signals using LLM analysis.
    Returns raw topic extraction response.
    """
    if not signals:
        return "**Other**\n- General uncategorized signals"
    
    # Extract text content from all signals
    signal_texts = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        text = extract_signal_text(signal)
        if text:
            signal_texts.append(text[:200])  # Limit length for API efficiency
    
    if not signal_texts:
        return "**Other**\n- General uncategorized signals"
    
    # Limit to max 50 samples to avoid token limits
    sample_texts = signal_texts[:50]
    
    prompt = (
        "You are a company-focused brand perception analyst.\n\n"
        "Given a collection of signal texts (news, social media, reviews, financial data) about a specific company, "
        "extract 3-7 core topics that emerge from this data. Focus on themes directly relevant to brand perception.\n\n"
        "Your output should:\n"
        "1. List each topic in the format: **Topic Name**\n"
        "2. Each topic should be concise (1-4 words)\n"
        "3. Include a brief one-line explanation under each topic\n"
        "4. Ensure topics are mutually exclusive where possible\n"
        "5. Always include an 'Other' category for miscellaneous signals\n\n"
        "Example output format:\n"
        "**AI Innovation**\n"
        "- Signals related to artificial intelligence developments\n\n"
        "**Financial Performance**\n"
        "- Revenue, earnings, stock price discussions\n\n"
        "**Other**\n"
        "- Miscellaneous brand-related content\n\n"
        f"Company being analyzed: {company_name}\n\n"
        "Signal Texts:\n"
        + "\n".join(f"- {text[:150]}" for text in sample_texts[:30])  # Show max 30 samples
        + "\n\nNow extract the most relevant topics:"
    )
    
    # Use groq_llm directly with custom prompt
    return groq_llm(prompt, temperature=0.2, top_p=0.4)

def parse_topics_to_labels(topics_raw: str) -> list:
    """
    Parse LLM topic extraction response into a list of topic labels.
    Ensures 'Other' is always included.
    """
    import re
    
    labels = [
        re.search(r"\*\*(.*?)\*\*", line).group(1).strip()
        for line in topics_raw.splitlines()
        if re.search(r"\*\*(.*?)\*\*", line)
    ]
    
    # Ensure 'Other' is always present
    if "Other" not in labels:
        labels.append("Other")
    
    return labels

def reclassify_signals(signals: list, labels: list) -> list:
    """
    Reclassify all signals in the list based on updated topic labels.
    Updates the 'topic' field for each signal.
    """
    logging.info(f"[TOPIC_RECLASSIFY] Reclassifying {len(signals)} signals with {len(labels)} topics")
    
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        
        # Extract text for classification
        text = extract_signal_text(signal)
        
        if not text:
            signal['topic'] = 'Other'
            continue
        
        # Use existing topic classification logic
        new_topic = extract_topic_ml(text, labels, backup=None)
        
        # Update topic
        old_topic = signal.get('topic', 'Unknown')
        signal['topic'] = new_topic
        
        # Log reclassifications
        if old_topic != new_topic and old_topic != 'Unknown':
            logging.debug(f"[TOPIC_RECLASSIFY] '{old_topic}' → '{new_topic}': {text[:80]}")
    
    return signals

def update_topics_and_reclassify(company_name: str, signals: list, current_labels: list = None) -> tuple:
    """
    Extract updated topics from current signals and reclassify all signals.
    
    Args:
        company_name: Name of the company being analyzed
        signals: Current list of signals
        current_labels: Optional existing labels to merge with
    
    Returns:
        tuple: (updated_signals, new_labels)
    """
    if not signals:
        return signals, current_labels or ['Other']
    
    logging.info(f"[TOPIC_UPDATE] Extracting topics from {len(signals)} signals")
    
    # Extract topics from current signal set
    topics_raw = extract_topics_from_signals(company_name, signals)
    new_labels = parse_topics_to_labels(topics_raw)
    
    # Merge with existing labels if provided
    if current_labels:
        combined_labels = list(set(current_labels + new_labels))
        if "Other" not in combined_labels:
            combined_labels.append("Other")
        new_labels = combined_labels
    
    logging.info(f"[TOPIC_UPDATE] Updated topic labels: {new_labels}")
    
    # Reclassify all signals with updated labels
    updated_signals = reclassify_signals(signals, new_labels)
    
    return updated_signals, new_labels

def get_brand_review_url_llm(company_name, review_platform="Trustpilot"):
    """
    Get the company's domain name that can be appended to Trustpilot review URLs.
    Returns just the domain (e.g., 'google.com') without protocol or www.
    """
    prompt = (
        f"For the brand '{company_name}', what is its main website domain? "
        "Return ONLY the domain name without http://, https://, or www. prefix. "
        "For example, for Google return 'google.com', for Microsoft return 'microsoft.com'. "
        "If none found, respond with 'None'."
    )
    # Send this prompt to LLM and parse response
    domain = groq_llm(prompt, temperature=0.0, top_p=1.0)
    
    if domain and domain != "None":
        # Clean up the domain in case LLM included protocol or www
        domain = domain.strip().lower()
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.replace("www.", "")
        # Remove trailing slash if present
        domain = domain.rstrip("/")
        return domain
    
    return None

def brand_signal_collector(company_name, hours_back=hours_back):
    """
    Collect brand signals from multiple sources with dynamic topic extraction and reclassification.
    Topics are re-extracted and all signals are reclassified after each batch of signals is added.
    """
    signals = []
    current_labels = ['Other']  # Start with default label
    
    # Phase 1: Fetch signals from various sources
    # Each time signals.extend() is called, we'll update topics and reclassify
    
    # Reddit signals
    logging.info("[SIGNAL_COLLECTION] Fetching Reddit signals...")
    reddit_signals = []  # fetch_reddit(company_name, current_labels, max_results=max_results, hours_back=hours_back)
    if reddit_signals:
        signals.extend(reddit_signals)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # News signals
    logging.info("[SIGNAL_COLLECTION] Fetching News signals...")
    news_signals = []  # fetch_news(company_name, current_labels, max_results=max_results, hours_back=hours_back)
    if news_signals:
        signals.extend(news_signals)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # HackerNews signals
    logging.info("[SIGNAL_COLLECTION] Fetching HackerNews signals...")
    hn_signals = []  # fetch_hackernews(company_name, current_labels, max_results=max_results, hours_back=hours_back)
    if hn_signals:
        signals.extend(hn_signals)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Brand analysis signals
    logging.info("[SIGNAL_COLLECTION] Fetching brand analysis signals...")
    brand_signals = []  # analyze_brand_signals(company_name, current_labels, max_results=max_results, hours_back=hours_back)
    if brand_signals:
        signals.extend(brand_signals)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Phase 2: Additional news sources
    logging.info("[SIGNAL_COLLECTION] Fetching additional news articles...")
    news_articles = []  # fetch_news_articles(company_name)
    if news_articles:
        signals.extend(news_articles)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Twitter/X signals
    logging.info("[SIGNAL_COLLECTION] Fetching Twitter/X signals...")
    tweet_signals = []  # scrape_company_tweets(company_name, country)
    if tweet_signals:
        signals.extend(tweet_signals)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Company news (ScrapFly)
    logging.info("[SIGNAL_COLLECTION] Fetching company news...")
    company_news = []  # fetch_company_news(company_name, hours_back=hours_back)
    if company_news:
        signals.extend(company_news)
        signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
        logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Financial data
    logging.info("[SIGNAL_COLLECTION] Fetching financial data...")
    finance_ticker = get_finance_ticker(company_name)
    if finance_ticker and finance_ticker != "None":
        finance_signals = get_combined_finance_analysis(company_name, finance_ticker, hours_back)
        if finance_signals:
            signals.extend(finance_signals)
            signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
            logging.info(f"[SIGNAL_COLLECTION] Total signals: {len(signals)}, Topics: {current_labels}")
    
    # Final deduplication
    logging.info("[SIGNAL_COLLECTION] Deduplicating signals...")
    signals = deduplicate_signals(signals)
    
    # Final topic update and reclassification
    logging.info("[SIGNAL_COLLECTION] Final topic extraction and reclassification...")
    signals, current_labels = update_topics_and_reclassify(company_name, signals, current_labels)
    
    logging.info(f"[SIGNAL_COLLECTION] Completed. Total signals: {len(signals)}, Final topics: {current_labels}")
    
    return signals

# --- Usage Example ---
'''
signals = brand_signal_collector(company_name)
for signal in signals:
    print(signal)
'''