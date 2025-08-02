import os
import time
import spacy
import pandas as pd
import requests
from datetime import datetime, timedelta
from news_fetcher import fetch_lseg_articles

# === CONSTANTS ===
EXCEL_PATH = "signals.xlsx"
PRIMARY_SHEET = "Primary"
ARCHIVE_SHEET = "AllSignalsArchive"
ENGINE_SHEET = "IndustryEngineInput"
DEDUPLICATION_WINDOW_SECONDS = 600
LOW_REGION_CONFIDENCE_PENALTY = 2
WEAK_CLASSIFICATION_PENALTY = 1
SCHEMA_VERSION = "v1.1"

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
    "Darknet": 2,
    "LSEG": 7
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

# === EVENT-BASED SCORING FOR LSEG NEWS ===
event_urgency_scores = {
    "Military Escalation": 9,
    "Protest": 7,
    "Cyberattack": 8,
    "Natural Disaster": 6,
    "Strike": 6,
    "Coup/Political Unrest": 8,
    "Economic Crisis": 7,
    "Supply Chain Disruption": 6,
    "Other/Miscellaneous": 5
}

event_sensitivity_scores = {
    "Military Escalation": 9,
    "Protest": 6,
    "Cyberattack": 8,
    "Natural Disaster": 7,
    "Strike": 5,
    "Coup/Political Unrest": 9,
    "Economic Crisis": 8,
    "Supply Chain Disruption": 6,
    "Other/Miscellaneous": 5
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
    classified_event = signal.get("classified_event", "")

    # Get credibility score
    credibility_score = credibility_scores.get(source, 5)

    # Get urgency and sensitivity scores based on source type
    if source == "LSEG":
        # Use event-based scoring for LSEG news
        urgency_score = event_urgency_scores.get(classified_event, 5)
        sensitivity_score = event_sensitivity_scores.get(classified_event, 5)
    else:
        # Use matcher-based scoring for other sources
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

    if classified_event.lower() in ["", "other/miscellaneous"]:
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
    """
    Advanced event classification based on news headlines and signal metadata.
    Uses comprehensive keyword matching with fallback logic.
    """
    text = signal.get("raw_value", "").lower()
    matcher = signal.get("matcher_name", "").upper()
    category = signal.get("category", "").lower()

    # === PROTEST & CIVIL UNREST ===
    protest_keywords = [
        "protest", "demonstration", "rally", "march", "occupation", "sit-in",
        "civil unrest", "civil disobedience", "mass gathering", "street protest",
        "anti-government", "anti-regime", "people's movement", "popular uprising",
        "citizen protest", "public demonstration", "mass mobilization"
    ]
    if any(keyword in text for keyword in protest_keywords):
        return _set_classification(signal, "Protest")

    # === STRIKES & LABOR ACTIONS ===
    strike_keywords = [
        "strike", "walkout", "labor action", "work stoppage", "industrial action",
        "union strike", "general strike", "wildcat strike", "sympathy strike",
        "labor dispute", "workplace protest", "employee walkout", "job action",
        "industrial dispute", "labor unrest", "workplace shutdown"
    ]
    if any(keyword in text for keyword in strike_keywords):
        return _set_classification(signal, "Strike")

    # === COUPS & POLITICAL UNREST ===
    coup_keywords = [
        "coup", "coup d'état", "overthrow", "regime change", "military takeover",
        "power grab", "government overthrow", "political coup", "military coup",
        "regime overthrow", "government collapse", "political upheaval",
        "military intervention", "constitutional crisis", "political crisis"
    ]
    if any(keyword in text for keyword in coup_keywords):
        return _set_classification(signal, "Coup/Political Unrest")

    # === MILITARY ESCALATION ===
    military_keywords = [
        "missile", "attack", "invasion", "airstrike", "bombing", "shelling",
        "military operation", "armed conflict", "war", "battle", "combat",
        "military strike", "air raid", "artillery", "tank", "troops",
        "military escalation", "armed intervention", "military action",
        "defense", "offensive", "military campaign", "warfare"
    ]
    if any(keyword in text for keyword in military_keywords):
        return _set_classification(signal, "Military Escalation")

    # === NATURAL DISASTERS ===
    natural_disaster_keywords = [
        "earthquake", "tsunami", "volcano", "hurricane", "typhoon", "cyclone",
        "wildfire", "forest fire", "bushfire", "flood", "flooding", "drought",
        "landslide", "avalanche", "tornado", "tropical storm", "storm surge",
        "natural disaster", "seismic", "tectonic", "eruption", "lava flow"
    ]
    if any(keyword in text for keyword in natural_disaster_keywords):
        return _set_classification(signal, "Natural Disaster")

    # === CYBERATTACKS & DIGITAL THREATS ===
    cyberattack_keywords = [
        "ransomware", "cyberattack", "cyber attack", "data breach", "hack",
        "hacking", "malware", "virus", "trojan", "phishing", "ddos",
        "distributed denial of service", "cyber threat", "digital attack",
        "computer virus", "cyber incident", "security breach", "data theft",
        "cybercrime", "digital espionage", "cyber warfare", "cyber security"
    ]
    if any(keyword in text for keyword in cyberattack_keywords):
        return _set_classification(signal, "Cyberattack")

    # === ECONOMIC CRISES ===
    economic_crisis_keywords = [
        "default", "cds", "credit default", "spread widening", "currency collapse",
        "financial crisis", "economic crisis", "recession", "depression",
        "bankruptcy", "insolvency", "debt crisis", "sovereign default",
        "economic collapse", "financial meltdown", "market crash", "stock crash",
        "economic downturn", "financial instability", "currency crisis",
        "debt default", "economic turmoil", "financial panic"
    ]
    if any(keyword in text for keyword in economic_crisis_keywords):
        return _set_classification(signal, "Economic Crisis")

    # === SUPPLY CHAIN DISRUPTIONS ===
    supply_chain_keywords = [
        "port strike", "shipping delay", "logistics crisis", "supply chain",
        "supply disruption", "logistics disruption", "shipping crisis",
        "cargo delay", "freight disruption", "transportation crisis",
        "logistics bottleneck", "supply shortage", "inventory crisis",
        "distribution problem", "shipping backlog", "cargo backlog"
    ]
    if any(keyword in text for keyword in supply_chain_keywords):
        return _set_classification(signal, "Supply Chain Disruption")

    # === FALLBACK LOGIC ===
    # Check matcher name for specific patterns
    if "PROTEST_CLUSTERING" in matcher:
        return _set_classification(signal, "Protest")
    elif "CDS" in matcher or "SOV" in matcher:
        return _set_classification(signal, "Economic Crisis")
    elif "MISSILE" in matcher or "MILITARY" in matcher:
        return _set_classification(signal, "Military Escalation")
    elif "CYBER" in matcher or "HACK" in matcher:
        return _set_classification(signal, "Cyberattack")
    elif "NATURAL" in matcher or "DISASTER" in matcher:
        return _set_classification(signal, "Natural Disaster")

    # Check category field
    if category.startswith("economic") or "financial" in category:
        return _set_classification(signal, "Economic Crisis")
    elif category.startswith("geopolitical") or "political" in category:
        return _set_classification(signal, "Coup/Political Unrest")
    elif category.startswith("military") or "defense" in category:
        return _set_classification(signal, "Military Escalation")
    elif category.startswith("cyber") or "digital" in category:
        return _set_classification(signal, "Cyberattack")
    elif category.startswith("natural") or "environmental" in category:
        return _set_classification(signal, "Natural Disaster")

    # Default fallback
    return _set_classification(signal, "Other/Miscellaneous")

def _set_classification(signal, label):
    """Helper function to set the classified event and return the signal."""
    signal["classified_event"] = label
    return signal

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

    # === LSEG NEWS FETCH ===
    keywords = (
        "protest OR strike OR coup OR riot OR military OR "
        "earthquake OR tsunami OR volcano OR hurricane OR "
        "wildfire OR drought OR flood OR ransomware OR cyberattack OR "
        "data breach OR missile OR invasion OR default OR "
        "currency collapse OR supply chain OR port strike OR logistics crisis"
    )

    articles = fetch_lseg_articles(query=keywords, count=20)
    print(f"[INFO] Retrieved {len(articles)} articles from LSEG.")

    news_signals = []
    for article in articles:
        headline = article.get("headline", "N/A")
        extracted_region = extract_region_from_text(headline)
        normalized_region, region_confidence, precision_flag = normalize_region(extracted_region)
        signal = {
            "matcher_name": "NEWS_KEYWORD_MATCH",
            "source": "LSEG",
            "raw_value": headline,
            "indicator_type": "News Headline",
            "category": "Geopolitical/Economic/Natural",
            "region": ', '.join(normalized_region),
            "timestamp": article.get("timestamp", datetime.utcnow().isoformat()),
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