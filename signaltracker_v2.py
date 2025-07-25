import os
import time
import spacy
import pandas as pd
import requests
from datetime import datetime, timedelta

# === CONSTANTS ===
EXCEL_PATH = "signals.xlsx"
PRIMARY_SHEET = "Primary"
ARCHIVE_SHEET = "AllSignalsArchive"
ENGINE_SHEET = "IndustryEngineInput"
DEDUPLICATION_WINDOW_SECONDS = 600
LOW_REGION_CONFIDENCE_PENALTY = 2
WEAK_CLASSIFICATION_PENALTY = 1
SCHEMA_VERSION = "v1.1"
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "c18ba3121e474c7d837c5415f7fc25e7")  # fallback to hardcoded if env not set

# === NLP SETUP ===
nlp = spacy.load("en_core_web_sm")

# === REGION EXTRACTION ===
def extract_region_from_text(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return "Unknown"

# === REGION NORMALIZATION ===
BROAD_REGION_MAP = {
    "Asia-Pacific": ["China", "Japan", "South Korea", "Australia", "India"],
    "Middle East": ["Saudi Arabia", "Iran", "Israel", "UAE", "Turkey"],
    "Europe": ["Germany", "France", "UK", "Italy", "Spain"],
    "Africa": [],  # too broad → Unknown
    "South America": ["Brazil", "Argentina", "Chile", "Colombia"],
    "North America": ["United States", "Canada", "Mexico"],
    # Extend as needed...
}

def normalize_region(region_text):
    region_text = region_text.strip()
    if region_text in BROAD_REGION_MAP:
        countries = BROAD_REGION_MAP[region_text]
        if countries:
            return countries, 0.5, False
        else:
            return ["Unknown"], 0.3, False
    else:
        return [region_text], 1.0, True

# === LOOKUP TABLES ===
credibility_scores = {
    "Bloomberg": 10,
    "Reuters": 8,
    "Telegram": 5,
    "Darknet": 2
}

urgency_scores = {
    "CDS_SOV_SPIKE": 8,
    "PROTEST_CLUSTERING": 9,
    "NDVI_ANOMALY": 5
}

sensitivity_scores = {
    "CDS_SOV_SPIKE": 9,
    "PROTEST_CLUSTERING": 6,
    "NDVI_ANOMALY": 7
}

# === LOAD EXISTING DATA IF IT EXISTS ===
if os.path.exists(EXCEL_PATH):
    try:
        existing_df = pd.read_excel(EXCEL_PATH, sheet_name=PRIMARY_SHEET, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Could not read {PRIMARY_SHEET} sheet: {e}")
        existing_df = pd.DataFrame()
else:
    existing_df = pd.DataFrame()

if not existing_df.empty and "timestamp" in existing_df.columns:
    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce")

# === SCORING FUNCTION ===
def calculate_signal_score(signal):
    matcher = signal.get("matcher_name", "")
    source = signal.get("source", "")

    credibility_score = credibility_scores.get(source, 5)
    urgency_score = urgency_scores.get(matcher, 5)
    sensitivity_score = sensitivity_scores.get(matcher, 5)

    base_final_score = round(
        (credibility_score * 0.4) +
        (urgency_score * 0.3) +
        (sensitivity_score * 0.3), 2
    )

    penalty = 0
    penalty_reasons = []

    region_confidence = signal.get("region_confidence", 1.0)
    if region_confidence < 0.5:
        penalty += LOW_REGION_CONFIDENCE_PENALTY
        penalty_reasons.append("Low region confidence")

    classified_event = signal.get("classified_event", "").lower()
    if classified_event in ["", "other/miscellaneous"]:
        penalty += WEAK_CLASSIFICATION_PENALTY
        penalty_reasons.append("Weak classification")

    final_score = max(0, round(base_final_score - penalty, 2))

    if final_score >= 8:
        confidence_flag = "High"
    elif final_score >= 5:
        confidence_flag = "Medium"
    else:
        confidence_flag = "Low"

    signal["credibility_score"] = credibility_score
    signal["urgency_score"] = urgency_score
    signal["historical_sensitivity_score"] = sensitivity_score
    signal["base_final_score"] = base_final_score
    signal["score_penalty"] = penalty
    signal["penalty_reason"] = "; ".join(penalty_reasons) if penalty_reasons else "None"
    signal["final_score"] = final_score
    signal["confidence_flag"] = confidence_flag
    signal["forward_to_classifier"] = (confidence_flag == "High")
    signal["schema_version"] = SCHEMA_VERSION
    signal["last_updated"] = datetime.utcnow().isoformat()

    return signal

# === EVENT CLASSIFIER ===
def classify_event(signal):
    text = signal.get("raw_value", "").lower()
    matcher = signal.get("matcher_name", "").upper()

    if any(keyword in text for keyword in ["protest", "demonstration", "rally"]):
        label = "Protest"
    elif any(keyword in text for keyword in ["strike", "walkout", "labor action"]):
        label = "Strike"
    elif any(keyword in text for keyword in ["coup", "overthrow", "regime change"]):
        label = "Coup/Political Unrest"
    elif any(keyword in text for keyword in ["missile", "attack", "invasion", "airstrike"]):
        label = "Military Escalation"
    elif any(keyword in text for keyword in ["earthquake", "tsunami", "volcano", "hurricane", "wildfire", "flood", "drought"]):
        label = "Natural Disaster"
    elif any(keyword in text for keyword in ["ransomware", "cyberattack", "data breach"]):
        label = "Cyberattack"
    elif any(keyword in text for keyword in ["default", "cds", "credit default", "spread widening", "currency collapse", "financial crisis"]):
        label = "Economic Crisis"
    elif any(keyword in text for keyword in ["port strike", "shipping delay", "logistics crisis"]):
        label = "Supply Chain Disruption"
    else:
        if "PROTEST_CLUSTERING" in matcher:
            label = "Protest"
        elif signal.get("category", "").lower().startswith("economic"):
            label = "Economic Crisis"
        else:
            label = "Other/Miscellaneous"

    signal["classified_event"] = label
    return signal

# === NEWSAPI FETCH WITH SIMPLE RETRY ===
def fetch_newsapi_articles(url, retries=3, delay=2):
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        elif response.status_code == 429:
            print("[WARN] Rate limited by NewsAPI, retrying...")
            time.sleep(delay)
        else:
            print(f"[ERROR] Failed to fetch NewsAPI: {response.status_code}, {response.text}")
            break
    return []

if __name__ == "__main__":
    # === TEST SIGNALS SETUP & PROCESSING ===
    test_signals = [
        {
            "matcher_name": "CDS_CDV_SPIKE",
            "source": "Bloomberg",
            "raw_value": "144",
            "indicator_type": "Sovereign CDS",
            "category": "Economic",
            "region": "Asia-Pacific",
            "timestamp": "2025-06-25T12:42:00Z"
        },
        {
            "matcher_name": "PROTEST_CLUSTERING",
            "source": "Reuters",
            "raw_value": "4 cities in 48h",
            "indicator_type": "Civil Unrest",
            "category": "Geopolitical",
            "region": "South America",
            "timestamp": "2025-06-25T13:05:00Z"
        },
        {
            "matcher_name": "MISSILE_WATCH",
            "source": "Al Jazeera",
            "raw_value": "BREAKING: Major missile attack strikes capital city",
            "indicator_type": "News Headline",
            "category": "Geopolitical",
            "region": "Iran",
            "timestamp": "2025-07-07T18:00:00Z"
        }
    ]

    scored_signals = []

    for signal in test_signals:
        normalized_region, region_confidence, precision_flag = normalize_region(signal['region'])
        signal["region"] = ', '.join(normalized_region)
        signal["region_confidence"] = region_confidence
        signal["region_precision_flag"] = precision_flag

        # Classify before deduplication
        signal = classify_event(signal)

        is_duplicate = False
        if not existing_df.empty:
            recent_matches = existing_df[
                (existing_df.get("classified_event", "") == signal.get("classified_event", "")) &
                (existing_df.get("region", "") == signal.get("region", ""))
            ]
            for _, existing_row in recent_matches.iterrows():
                existing_time = existing_row["timestamp"]
                new_time = pd.to_datetime(signal["timestamp"], errors="coerce")
                if abs((new_time - existing_time).total_seconds()) <= DEDUPLICATION_WINDOW_SECONDS:
                    is_duplicate = True
                    break

        if not is_duplicate:
            signal["heartbeat"] = datetime.utcnow().isoformat()
            scored = calculate_signal_score(signal)
            scored_signals.append(scored)
        else:
            print(f"[DEDUPLICATION] Skipped duplicate: {signal.get('classified_event', 'Unknown')} | {signal.get('region', 'Unknown')} at {signal.get('timestamp', 'Unknown')}")

    # Print last 3 scored signals for debug
    for sig in scored_signals[-3:]:
        print(sig)

    # === SYSTEM LOGGING (SIMPLE PRINT) ===
    log = []
    for signal in scored_signals:
        try:
            log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "matcher": signal.get("matcher_name", "UNKNOWN"),
                "status": "Success",
                "final_score": signal.get("final_score", "N/A")
            })
        except Exception as e:
            log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "matcher": signal.get("matcher_name", "UNKNOWN"),
                "status": f"Fail - {str(e)}"
            })

    # === NEWSAPI FETCH ===
    keywords = (
        "protest OR strike OR coup OR riot OR military OR "
        "earthquake OR tsunami OR volcano OR hurricane OR "
        "wildfire OR drought OR flood OR ransomware OR cyberattack OR "
        "data breach OR missile OR invasion OR default OR "
        "currency collapse OR supply chain OR port strike OR logistics crisis"
    )
    newsapi_url = (
        f"https://newsapi.org/v2/everything?q={keywords}"
        f"&language=en&sortBy=publishedAt&pageSize=20&apiKey={NEWSAPI_KEY}"
    )

    articles = fetch_newsapi_articles(newsapi_url)
    print(f"[INFO] Retrieved {len(articles)} articles.")

    news_signals = []
    for article in articles:
        headline = article.get("title", "N/A")
        extracted_region = extract_region_from_text(headline)
        normalized_region, region_confidence, precision_flag = normalize_region(extracted_region)
        signal = {
            "matcher_name": "NEWS_KEYWORD_MATCH",
            "source": "NewsAPI",
            "raw_value": headline,
            "indicator_type": "News Headline",
            "category": "Geopolitical/Economic/Natural",
            "region": ', '.join(normalized_region),
            "timestamp": article.get("publishedAt", datetime.utcnow().isoformat()),
            "region_confidence": region_confidence,
            "region_precision_flag": precision_flag
        }
        signal = classify_event(signal)
        is_duplicate = False
        if not existing_df.empty:
            recent_matches = existing_df[
                (existing_df.get("raw_value", "") == signal.get("raw_value", "")) &
                (existing_df.get("region", "") == signal.get("region", ""))
            ]
            for _, existing_row in recent_matches.iterrows():
                existing_time = existing_row["timestamp"]
                new_time = pd.to_datetime(signal["timestamp"], errors="coerce")
                if abs((new_time - existing_time).total_seconds()) <= DEDUPLICATION_WINDOW_SECONDS:
                    is_duplicate = True
                    break
        if not is_duplicate:
            signal["heartbeat"] = datetime.utcnow().isoformat()
            scored_signal = calculate_signal_score(signal)
            news_signals.append(scored_signal)
        else:
            print(f"[DEDUPLICATION] Skipped duplicate: {signal.get('raw_value', 'UNKNOWN')} | {signal['region']} at {signal['timestamp']}")

    # === CLASSIFICATION AND EXPORT ===
    classified_signals = []
    for signal in scored_signals + news_signals:
        classified_signals.append(signal)
        try:
            if signal.get("final_score", 0) >= 9 and signal.get("region_confidence", 0) >= 0.8:
                print("🚨 MAJOR SIGNAL DETECTED")
                print(f"> Type: {signal.get('classified_event')}")
                print(f"> Region: {signal.get('region')} | Score: {signal.get('final_score')}")
                print(f"> Signal: {signal.get('raw_value')}")
        except Exception as e:
            print(f"[Alert Trigger Error] {e}")

        if signal.get("confidence_flag") == "High" and signal.get("region") != "Unknown":
            signal["forward_to_engine"] = True
        else:
            signal["forward_to_engine"] = False

    # === BATCH EXCEL EXPORT ===
    try:
        archive_df = pd.DataFrame(classified_signals)
        df_classified = pd.DataFrame(classified_signals)
        forwarded_signals = [s for s in classified_signals if s.get("forward_to_engine")]
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a" if os.path.exists(EXCEL_PATH) else "w", if_sheet_exists="replace") as writer:
            archive_df.to_excel(writer, sheet_name=ARCHIVE_SHEET, index=False)
            df_classified.to_excel(writer, sheet_name=PRIMARY_SHEET, index=False)
            if forwarded_signals:
                forward_df = pd.DataFrame(forwarded_signals)
                forward_df.to_excel(writer, sheet_name=ENGINE_SHEET, index=False)
        print("📦 Full signal archive and primary/engine sheets saved to Excel.")
    except Exception as e:
        print(f"⚠️ Excel export error: {e}")