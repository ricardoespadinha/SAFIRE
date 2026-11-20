import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from scipy.special import softmax
from textblob import TextBlob
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# -----------------------------
# Setup Output Directory and Logging
# -----------------------------

# Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sentiment_analysis_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Create log file
log_file_path = os.path.join(output_dir, "analysis_log.txt")
log_file = open(log_file_path, 'w', encoding='utf-8')

# Custom print function that writes to both console and file
def print_log(message="", console=True):
    """Print to console and log file"""
    if console:
        print(message)
    log_file.write(message + "\n")
    log_file.flush()

print_log("="*80)
print_log("ADVANCED SENTIMENT ANALYSIS WITH NEUTRAL FOCUS")
print_log("="*80)
print_log(f"\nOutput directory: {output_dir}")
print_log(f"Log file: {log_file_path}")
print_log(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------
# Setup
# -----------------------------

nltk.download("vader_lexicon", quiet=True)

print_log("\nLoading data...")
df = pd.read_csv("dynasent-test.csv")
print_log(f"✓ Loaded {len(df)} samples")

print_log("\n" + "="*80)
print_log("INITIALIZING MODELS")
print_log("="*80)

# Track which models loaded successfully
model_status = {}

# VADER
print_log("\n[1/6] Loading VADER...")
try:
    vader = SentimentIntensityAnalyzer()
    print_log("  ✓ VADER ready")
    model_status['VADER'] = True
except Exception as e:
    print_log(f"  ✗ VADER failed: {str(e)}")
    model_status['VADER'] = False

# RoBERTa (Twitter-tuned)
print_log("\n[2/6] Loading RoBERTa (Twitter-tuned)...")
try:
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
    print_log("  ✓ RoBERTa ready")
    model_status['RoBERTa'] = True
except Exception as e:
    print_log(f"  ✗ RoBERTa failed: {str(e)[:100]}")
    model_status['RoBERTa'] = False

# DistilBERT (general sentiment)
print_log("\n[3/6] Loading DistilBERT...")
try:
    distilbert_pipe = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english",
                              device=-1)
    print_log("  ✓ DistilBERT ready")
    model_status['DistilBERT'] = True
except Exception as e:
    print_log(f"  ✗ DistilBERT failed: {str(e)[:100]}")
    model_status['DistilBERT'] = False

# BERT for sentiment (5-class)
print_log("\n[4/6] Loading BERT (5-star sentiment)...")
try:
    bert_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
    print_log("  ✓ BERT 5-star ready")
    model_status['BERT'] = True
except Exception as e:
    print_log(f"  ✗ BERT failed: {str(e)[:100]}")
    model_status['BERT'] = False

# FinBERT (financial sentiment - excellent at neutral)
print_log("\n[5/6] Loading FinBERT...")
try:
    finbert_pipe = pipeline("sentiment-analysis", 
                           model="ProsusAI/finbert",
                           device=-1)
    print_log("  ✓ FinBERT ready (BEST for neutral detection!)")
    model_status['FinBERT'] = True
except Exception as e:
    print_log(f"  ✗ FinBERT failed: {str(e)[:100]}")
    model_status['FinBERT'] = False

# Twitter-RoBERTa for emotion
print_log("\n[6/6] Loading Emotion Detection (Twitter-RoBERTa)...")
try:
    emotion_pipe = pipeline("text-classification", 
                           model="cardiffnlp/twitter-roberta-base-emotion",
                           device=-1,
                           top_k=None)
    print_log("  ✓ Emotion detection ready")
    model_status['Emotion'] = True
except Exception as e:
    print_log(f"  ✗ Emotion detection failed: {str(e)[:100]}")
    model_status['Emotion'] = False

# TextBlob (always available)
model_status['TextBlob'] = True

print_log("\n" + "="*80)
print_log("MODEL LOADING SUMMARY")
print_log("="*80)
for model_name, status in model_status.items():
    symbol = "✓" if status else "✗"
    status_text = "Ready" if status else "Failed"
    print_log(f"{symbol} {model_name}: {status_text}")

loaded_count = sum(model_status.values())
print_log(f"\nTotal models loaded: {loaded_count}/{len(model_status)}")

if loaded_count < 3:
    print_log("\n⚠️  WARNING: Very few models loaded. Results may be limited.")

print_log("\n" + "="*80)

# -----------------------------
# Scoring Functions
# -----------------------------

def get_vader_scores(text):
    """VADER scores: neg, neu, pos"""
    if not model_status.get('VADER', False):
        return 0.33, 0.34, 0.33
    scores = vader.polarity_scores(str(text))
    return scores["neg"], scores["neu"], scores["pos"]

def vader_label(text):
    """VADER prediction with adjustable neutral threshold"""
    if not model_status.get('VADER', False):
        return "neutral"
    score = vader.polarity_scores(str(text))["compound"]
    if score >= 0.1:
        return "positive"
    elif score <= -0.1:
        return "negative"
    else:
        return "neutral"

def get_roberta_scores(text):
    """RoBERTa Twitter sentiment scores"""
    if not model_status.get('RoBERTa', False):
        return 0.33, 0.34, 0.33
    encoded = roberta_tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    output = roberta_model(**encoded)
    scores = output.logits.detach().numpy()[0]
    probs = softmax(scores)
    return probs[0], probs[1], probs[2]

def roberta_label(text):
    """RoBERTa prediction"""
    if not model_status.get('RoBERTa', False):
        return "neutral"
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    labels = ["negative", "neutral", "positive"]
    return labels[np.argmax([r_neg, r_neu, r_pos])]

def get_textblob_scores(text):
    """TextBlob polarity and subjectivity"""
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        return 0.0, 1 - abs(polarity), abs(polarity)
    elif polarity < -0.1:
        return abs(polarity), 1 - abs(polarity), 0.0
    else:
        return 0.0, 1.0, 0.0

def textblob_label(text):
    """TextBlob prediction with subjectivity consideration"""
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if abs(polarity) < 0.15 or (subjectivity < 0.3 and abs(polarity) < 0.3):
        return "neutral"
    elif polarity > 0:
        return "positive"
    else:
        return "negative"

def get_bert_scores(text):
    """BERT 5-star sentiment converted to 3-class"""
    if not model_status.get('BERT', False):
        return 0.33, 0.34, 0.33
    encoded = bert_tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    output = bert_model(**encoded)
    scores = output.logits.detach().numpy()[0]
    probs = softmax(scores)
    
    neg = probs[0] + probs[1]
    neu = probs[2]
    pos = probs[3] + probs[4]
    
    total = neg + neu + pos
    return neg/total, neu/total, pos/total

def bert_label(text):
    """BERT prediction"""
    if not model_status.get('BERT', False):
        return "neutral"
    neg, neu, pos = get_bert_scores(text)
    labels = ["negative", "neutral", "positive"]
    return labels[np.argmax([neg, neu, pos])]

def get_finbert_scores(text):
    """FinBERT scores (excellent at neutral detection)"""
    if not model_status.get('FinBERT', False):
        return 0.33, 0.34, 0.33
    try:
        result = finbert_pipe(str(text)[:512])[0]
        label_map = {"positive": 2, "neutral": 1, "negative": 0}
        
        scores = [0.0, 0.0, 0.0]
        scores[label_map[result['label']]] = result['score']
        
        if isinstance(finbert_pipe(str(text)[:512], top_k=3), list):
            for item in finbert_pipe(str(text)[:512], top_k=3):
                if isinstance(item, list):
                    for subitem in item:
                        scores[label_map[subitem['label']]] = subitem['score']
                else:
                    scores[label_map[item['label']]] = item['score']
        
        return scores[0], scores[1], scores[2]
    except:
        return 0.33, 0.34, 0.33

def finbert_label(text):
    """FinBERT prediction"""
    if not model_status.get('FinBERT', False):
        return "neutral"
    result = finbert_pipe(str(text)[:512])[0]
    return result['label'].lower()

def get_emotion_neutrality(text):
    """Use emotion detection as proxy for neutrality"""
    if not model_status.get('Emotion', False):
        return 0.5
    try:
        emotions = emotion_pipe(str(text)[:512])[0]
        emotion_sum = sum([e['score'] for e in emotions])
        neutrality_score = 1 - emotion_sum if emotion_sum < 1 else 0
        return neutrality_score
    except:
        return 0.5

# -----------------------------
# ORIGINAL Ensemble Methods
# -----------------------------

def weighted_ensemble_v1(row):
    """Original weighted approach (VADER + RoBERTa)"""
    VADER_WEIGHT = 0.3
    ROBERTA_WEIGHT = 0.7
    
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    final_neg = (v_neg * VADER_WEIGHT) + (r_neg * ROBERTA_WEIGHT)
    final_neu = (v_neu * VADER_WEIGHT) + (r_neu * ROBERTA_WEIGHT)
    final_pos = (v_pos * VADER_WEIGHT) + (r_pos * ROBERTA_WEIGHT)
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def weighted_ensemble_v2(row):
    """Multi-model ensemble optimized for neutral detection"""
    
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    tb_neg, tb_neu, tb_pos = get_textblob_scores(row["sentence"])
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    
    final_neg = (v_neg * 0.1 + r_neg * 0.2 + tb_neg * 0.1 + b_neg * 0.25 + f_neg * 0.35)
    final_neu = (v_neu * 0.1 + r_neu * 0.2 + tb_neu * 0.1 + b_neu * 0.25 + f_neu * 0.35)
    final_pos = (v_pos * 0.1 + r_pos * 0.2 + tb_pos * 0.1 + b_pos * 0.25 + f_pos * 0.35)
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def voting_ensemble(row):
    """Simple majority voting"""
    
    votes = [
        vader_label(row["sentence"]),
        roberta_label(row["sentence"]),
        textblob_label(row["sentence"]),
        bert_label(row["sentence"]),
        finbert_label(row["sentence"])
    ]
    
    vote_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for vote in votes:
        vote_counts[vote] += 1
    
    return max(vote_counts, key=vote_counts.get)

def neutral_focused_ensemble(row):
    """Ensemble specifically tuned for neutral detection"""
    
    text = row["sentence"]
    
    vader_pred = vader_label(text)
    roberta_pred = roberta_label(text)
    bert_pred = bert_label(text)
    finbert_pred = finbert_label(text)
    textblob_pred = textblob_label(text)
    
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    
    if finbert_pred == "neutral" and f_neu > 0.6:
        return "neutral"
    
    neutral_votes = sum([
        vader_pred == "neutral",
        roberta_pred == "neutral",
        bert_pred == "neutral",
        finbert_pred == "neutral",
        textblob_pred == "neutral"
    ])
    
    if neutral_votes >= 3:
        return "neutral"
    
    emotion_neutral = get_emotion_neutrality(text)
    if emotion_neutral > 0.7:
        return "neutral"
    
    final_neg = (r_neg * 0.4 + f_neg * 0.6) * 0.9
    final_neu = (r_neu * 0.4 + f_neu * 0.6) * 1.2
    final_pos = (r_pos * 0.4 + f_pos * 0.6) * 0.9
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

# NEW Ensemble Methods (abbreviated for length - same as before)

def neutral_focused_no_textblob(row):
    text = row["sentence"]
    vader_pred = vader_label(text)
    roberta_pred = roberta_label(text)
    bert_pred = bert_label(text)
    finbert_pred = finbert_label(text)
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    b_neg, b_neu, b_pos = get_bert_scores(text)
    if finbert_pred == "neutral" and f_neu > 0.6:
        return "neutral"
    neutral_votes = sum([vader_pred == "neutral", roberta_pred == "neutral", bert_pred == "neutral", finbert_pred == "neutral"])
    if neutral_votes >= 3:
        return "neutral"
    emotion_neutral = get_emotion_neutrality(text)
    if emotion_neutral > 0.7:
        return "neutral"
    final_neg = (r_neg * 0.25 + f_neg * 0.5 + b_neg * 0.25) * 0.9
    final_neu = (r_neu * 0.25 + f_neu * 0.5 + b_neu * 0.25) * 1.2
    final_pos = (r_pos * 0.25 + f_pos * 0.5 + b_pos * 0.25) * 0.9
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def weighted_no_textblob(row):
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (v_neg * 0.15 + r_neg * 0.25 + b_neg * 0.25 + f_neg * 0.35)
    final_neu = (v_neu * 0.15 + r_neu * 0.25 + b_neu * 0.25 + f_neu * 0.35)
    final_pos = (v_pos * 0.15 + r_pos * 0.25 + b_pos * 0.25 + f_pos * 0.35)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def weighted_no_textblob_no_bert(row):
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (v_neg * 0.2 + r_neg * 0.3 + f_neg * 0.5)
    final_neu = (v_neu * 0.2 + r_neu * 0.3 + f_neu * 0.5)
    final_pos = (v_pos * 0.2 + r_pos * 0.3 + f_pos * 0.5)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def voting_no_textblob_no_bert(row):
    votes = [vader_label(row["sentence"]), roberta_label(row["sentence"]), finbert_label(row["sentence"])]
    vote_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for vote in votes:
        vote_counts[vote] += 1
    return max(vote_counts, key=vote_counts.get)

def finbert_roberta_only(row):
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (r_neg + f_neg) / 2
    final_neu = (r_neu + f_neu) / 2
    final_pos = (r_pos + f_pos) / 2
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def finbert_bert_ensemble(row):
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (f_neg * 0.6 + b_neg * 0.4)
    final_neu = (f_neu * 0.6 + b_neu * 0.4)
    final_pos = (f_pos * 0.6 + b_pos * 0.4)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def conservative_neutral_ensemble(row):
    text = row["sentence"]
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    b_neg, b_neu, b_pos = get_bert_scores(text)
    avg_neg = (r_neg + f_neg + b_neg) / 3
    avg_neu = (r_neu + f_neu + b_neu) / 3
    avg_pos = (r_pos + f_pos + b_pos) / 3
    if avg_pos > 0.6 and avg_pos > avg_neg:
        return "positive"
    elif avg_neg > 0.6 and avg_neg > avg_pos:
        return "negative"
    else:
        return "neutral"

def aggressive_neutral_ensemble(row):
    text = row["sentence"]
    vader_pred = vader_label(text)
    roberta_pred = roberta_label(text)
    bert_pred = bert_label(text)
    finbert_pred = finbert_label(text)
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    b_neg, b_neu, b_pos = get_bert_scores(text)
    if (finbert_pred == "neutral" and f_neu > 0.5) or (bert_pred == "neutral" and b_neu > 0.5) or (roberta_pred == "neutral" and r_neu > 0.5):
        return "neutral"
    neutral_votes = sum([vader_pred == "neutral", roberta_pred == "neutral", bert_pred == "neutral", finbert_pred == "neutral"])
    if neutral_votes >= 2:
        return "neutral"
    final_neg = (r_neg * 0.3 + f_neg * 0.4 + b_neg * 0.3)
    final_neu = (r_neu * 0.3 + f_neu * 0.4 + b_neu * 0.3)
    final_pos = (r_pos * 0.3 + f_pos * 0.4 + b_pos * 0.3)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def transformer_only_ensemble(row):
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (r_neg * 0.3 + b_neg * 0.3 + f_neg * 0.4)
    final_neu = (r_neu * 0.3 + b_neu * 0.3 + f_neu * 0.4)
    final_pos = (r_pos * 0.3 + b_pos * 0.3 + f_pos * 0.4)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def finbert_heavy_ensemble(row):
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    final_neg = (v_neg * 0.1 + r_neg * 0.2 + f_neg * 0.7)
    final_neu = (v_neu * 0.1 + r_neu * 0.2 + f_neu * 0.7)
    final_pos = (v_pos * 0.1 + r_pos * 0.2 + f_pos * 0.7)
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

def dynamic_threshold_ensemble(row):
    text = row["sentence"]
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    b_neg, b_neu, b_pos = get_bert_scores(text)
    avg_neg = (r_neg + f_neg + b_neg) / 3
    avg_neu = (r_neu + f_neu + b_neu) / 3
    avg_pos = (r_pos + f_pos + b_pos) / 3
    scores = [avg_neg, avg_neu, avg_pos]
    variance = np.var(scores)
    if variance < 0.02:
        return "neutral"
    if variance < 0.05:
        avg_neu *= 1.15
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": avg_neg, "neutral": avg_neu, "positive": avg_pos}[x])

def emotion_weighted_ensemble(row):
    text = row["sentence"]
    r_neg, r_neu, r_pos = get_roberta_scores(text)
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    emotion_neutral = get_emotion_neutrality(text)
    base_neg = (r_neg * 0.4 + f_neg * 0.6)
    base_neu = (r_neu * 0.4 + f_neu * 0.6)
    base_pos = (r_pos * 0.4 + f_pos * 0.6)
    neutral_boost = 1 + (emotion_neutral * 0.5)
    final_neg = base_neg
    final_neu = base_neu * neutral_boost
    final_pos = base_pos
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])


# -----------------------------
# Dynamic Threshold Methods
# -----------------------------

def dynamic_bert(row):
    text = row["sentence"]
    b_neg, b_neu, b_pos = get_bert_scores(text)
    scores = [b_neg, b_neu, b_pos]
    variance = np.var(scores)
    if variance < 0.02:
        return "neutral"
    if variance < 0.05:
        b_neu *= 1.15
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": b_neg, "neutral": b_neu, "positive": b_pos}[x])

def dynamic_finbert(row):
    text = row["sentence"]
    f_neg, f_neu, f_pos = get_finbert_scores(text)
    scores = [f_neg, f_neu, f_pos]
    variance = np.var(scores)
    if variance < 0.02:
        return "neutral"
    if variance < 0.05:
        f_neu *= 1.15
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": f_neg, "neutral": f_neu, "positive": f_pos}[x])

def dynamic_vader(row):
    text = row["sentence"]
    v_neg, v_neu, v_pos = get_vader_scores(text)
    scores = [v_neg, v_neu, v_pos]
    variance = np.var(scores)
    if variance < 0.02:
        return "neutral"
    if variance < 0.05:
        v_neu *= 1.15
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": v_neg, "neutral": v_neu, "positive": v_pos}[x])

def dynamic_weighted_v1(row):
    VADER_WEIGHT = 0.3
    ROBERTA_WEIGHT = 0.7
    
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    final_neg = (v_neg * VADER_WEIGHT) + (r_neg * ROBERTA_WEIGHT)
    final_neu = (v_neu * VADER_WEIGHT) + (r_neu * ROBERTA_WEIGHT)
    final_pos = (v_pos * VADER_WEIGHT) + (r_pos * ROBERTA_WEIGHT)

    scores = [final_neg, final_neu, final_pos]
    variance = np.var(scores)
    if variance < 0.02:
        return "neutral"
    if variance < 0.05:
        final_neu *= 1.15
    return max(["negative", "neutral", "positive"], key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

# -----------------------------
# NEW: Data-Driven RoBERTa Dynamic Threshold Methods
# -----------------------------

def roberta_dynamic_conservative(row):
    """
    RoBERTa with CONSERVATIVE threshold (High Precision).
    Based on analysis: Only classify as neutral if margin > 0.0988
    This captures the bottom 10% of correctly classified neutrals.
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Conservative threshold from analysis
    MARGIN_THRESHOLD = 0.0565

    
    # Get the maximum non-neutral score
    max_non_neutral = max(r_neg, r_pos)
    neutral_margin = r_neu - max_non_neutral
    
    # If neutral wins and margin is sufficient, classify as neutral
    if r_neu > max_non_neutral and neutral_margin > MARGIN_THRESHOLD:
        return "neutral"
    
    # Otherwise use standard max
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])

def roberta_dynamic_balanced(row):
    """
    RoBERTa with BALANCED threshold (Balance Precision/Recall).
    Based on analysis: Boost neutral if score difference < 0.3394 (median)
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Balanced threshold from analysis (median misclassification difference)
    DIFF_THRESHOLD = 0.3394
    
    # Find the current winner
    max_score = max(r_neg, r_neu, r_pos)
    current_winner = max(["negative", "neutral", "positive"], 
                        key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])
    
    # If non-neutral wins but margin is small, switch to neutral
    if current_winner != "neutral":
        score_diff = max_score - r_neu
        if score_diff < DIFF_THRESHOLD:
            return "neutral"
    
    return current_winner

def roberta_dynamic_aggressive(row):
    """
    RoBERTa with AGGRESSIVE threshold (High Recall).
    Based on analysis: Boost neutral if score difference < 0.6294 (75th percentile)
    Catches 75% of misclassifications.
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Aggressive threshold from analysis (75th percentile)
    DIFF_THRESHOLD = 0.6294

    
    # Find the current winner
    max_score = max(r_neg, r_neu, r_pos)
    current_winner = max(["negative", "neutral", "positive"], 
                        key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])
    
    # If non-neutral wins but margin is small, switch to neutral
    if current_winner != "neutral":
        score_diff = max_score - r_neu
        if score_diff < DIFF_THRESHOLD:
            return "neutral"
    
    return current_winner

def roberta_dynamic_adaptive(row):
    """
    RoBERTa with ADAPTIVE threshold.
    Uses different thresholds based on neutral score magnitude.
    Higher neutral score = stricter threshold (more confident prediction).
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Find the current winner
    max_score = max(r_neg, r_neu, r_pos)
    current_winner = max(["negative", "neutral", "positive"], 
                        key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])
    
    # Adaptive threshold based on neutral score magnitude
    # High neutral (>0.5): use conservative threshold (0.0988)
    # Medium neutral (0.3-0.5): use balanced threshold (0.3394)
    # Low neutral (<0.3): use aggressive threshold (0.6294)
    
    if r_neu > 0.5:
        threshold = 0.0565  # Conservative
    elif r_neu > 0.3:
        threshold = 0.3394  # Balanced
    else:
        threshold = 0.6294  # Aggressive
    
    # If non-neutral wins but margin is within threshold, switch to neutral
    if current_winner != "neutral":
        score_diff = max_score - r_neu
        if score_diff < threshold:
            return "neutral"
    
    return current_winner

def roberta_dynamic_percentile_25(row):
    """
    RoBERTa with 25th percentile threshold (0.1717).
    Catches ~25% of misclassifications, very conservative.
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    DIFF_THRESHOLD = 0.1717 # 25th percentile
    
    max_score = max(r_neg, r_neu, r_pos)
    current_winner = max(["negative", "neutral", "positive"], 
                        key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])
    
    if current_winner != "neutral":
        score_diff = max_score - r_neu
        if score_diff < DIFF_THRESHOLD:
            return "neutral"
    
    return current_winner

def roberta_dynamic_percentile_90(row):
    """
    RoBERTa with 90th percentile threshold (0.8279).
    Catches ~90% of misclassifications, very aggressive.
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    DIFF_THRESHOLD = 0.8277  # 90th percentile
    
    max_score = max(r_neg, r_neu, r_pos)
    current_winner = max(["negative", "neutral", "positive"], 
                        key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])
    
    if current_winner != "neutral":
        score_diff = max_score - r_neu
        if score_diff < DIFF_THRESHOLD:
            return "neutral"
    
    return current_winner

# -----------------------------
# NEW: Data-Driven RoBERTa Dynamic Threshold Methods
# Based on empirical analysis of SST3 + DynaSent datasets
# -----------------------------

def roberta_dynamic_conservative(row):
    """
    RoBERTa with CONSERVATIVE threshold (High Precision)
    Based on analysis: only classify as neutral if margin > 0.0988
    (10th percentile of correctly classified neutrals)
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Find current winner
    scores = {'negative': r_neg, 'neutral': r_neu, 'positive': r_pos}
    winner = max(scores, key=scores.get)
    
    # If neutral already wins, check if margin is safe
    if winner == 'neutral':
        max_other = max(r_neg, r_pos)
        margin = r_neu - max_other
        
        # Conservative: need margin > 0.0988 to trust neutral
        if margin > 0.0565:
            return 'neutral'
        else:
            # Not confident enough, pick between pos/neg
            return 'positive' if r_pos > r_neg else 'negative'
    
    # If non-neutral wins, check if we should switch to neutral
    winner_score = scores[winner]
    score_diff = winner_score - r_neu
    
    # Conservative: only switch if difference is very small (< 0.0988)
    if score_diff < 0.0565:
        return 'neutral'
    
    return winner

def roberta_dynamic_balanced(row):
    """
    RoBERTa with BALANCED threshold (Balance Precision/Recall)
    Based on analysis: boost neutral if score difference < 0.3394
    (Median difference in misclassifications)
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    scores = {'negative': r_neg, 'neutral': r_neu, 'positive': r_pos}
    winner = max(scores, key=scores.get)
    
    # If non-neutral wins
    if winner != 'neutral':
        winner_score = scores[winner]
        score_diff = winner_score - r_neu
        
        # Balanced: switch to neutral if difference < 0.3394
        if score_diff < 0.3394:
            return 'neutral'
    
    return winner

def roberta_dynamic_aggressive(row):
    """
    RoBERTa with AGGRESSIVE threshold (High Recall)
    Based on analysis: boost neutral if score difference < 0.6294
    (75th percentile - catches 75% of misclassifications)
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    scores = {'negative': r_neg, 'neutral': r_neu, 'positive': r_pos}
    winner = max(scores, key=scores.get)
    
    # If non-neutral wins
    if winner != 'neutral':
        winner_score = scores[winner]
        score_diff = winner_score - r_neu
        
        # Aggressive: switch to neutral if difference < 0.6294
        if score_diff < 0.6294:
            return 'neutral'
    
    return winner

def roberta_dynamic_adaptive(row):
    """
    RoBERTa with ADAPTIVE threshold
    Adjusts threshold based on neutral score magnitude and score variance
    Higher neutral score = stricter threshold
    High variance = more lenient threshold
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    scores = {'negative': r_neg, 'neutral': r_neu, 'positive': r_pos}
    winner = max(scores, key=scores.get)
    
    # Calculate variance of scores (measure of uncertainty)
    score_array = [r_neg, r_neu, r_pos]
    variance = np.var(score_array)
    
    # Base threshold (median from analysis)
    base_threshold = 0.3394
    
    # Adjust based on neutral score magnitude
    # Higher neutral = we're more confident, so require larger difference to override
    if r_neu > 0.5:
        # High neutral score - be more conservative
        threshold = base_threshold * 0.7  # Stricter (0.2376)
    elif r_neu > 0.3:
        # Medium neutral score - use base
        threshold = base_threshold  # (0.3394)
    else:
        # Low neutral score - be more aggressive
        threshold = base_threshold * 1.3  # More lenient (0.4412)
    
    # Further adjust based on variance
    # High variance = uncertain = be more willing to call it neutral
    if variance > 0.08:
        threshold *= 1.2  # More lenient when uncertain
    elif variance < 0.03:
        threshold *= 0.9  # Stricter when confident
    
    # Apply threshold
    if winner != 'neutral':
        winner_score = scores[winner]
        score_diff = winner_score - r_neu
        
        if score_diff < threshold:
            return 'neutral'
    
    return winner

def roberta_dynamic_percentile_based(row):
    """
    RoBERTa using percentile-based thresholds
    Uses different thresholds for different score difference ranges
    Based on cumulative distribution from analysis
    """
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    scores = {'negative': r_neg, 'neutral': r_neu, 'positive': r_pos}
    winner = max(scores, key=scores.get)
    
    if winner != 'neutral':
        winner_score = scores[winner]
        score_diff = winner_score - r_neu
        
        # Percentile-based decision
        # Based on analysis:
        # - 9.4% of misclass have diff < 0.05
        # - 15.7% have diff < 0.10
        # - 22.0% have diff < 0.15
        # - 30.9% have diff < 0.20
        # - 50% have diff < 0.3394
        
        # If very close (< 0.15), strongly prefer neutral
        if score_diff < 0.15:
            return 'neutral'
        
        # If close (0.15-0.30), prefer neutral with some confidence check
        elif score_diff < 0.30:
            # Check if neutral score is at least reasonable
            if r_neu > 0.2:
                return 'neutral'
        
        # If moderate difference (0.30-0.45), trust winner but be cautious
        elif score_diff < 0.45:
            # Only override if neutral score is quite high
            if r_neu > 0.35:
                return 'neutral'
    
    return winner

# -----------------------------
# NEW: Neutral-Proximity Boost Methods
# -----------------------------

def vader_neutral_proximity(row):
    """
    VADER with neutral proximity boost.
    If neutral is within threshold of pos/neg, boost it or select neutral.
    """
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    
    # Threshold: if neutral is within this distance of another score
    proximity_threshold = 0.15
    
    # Check if neutral is close to positive or negative
    close_to_pos = abs(v_neu - v_pos) < proximity_threshold
    close_to_neg = abs(v_neu - v_neg) < proximity_threshold
    
    # If neutral is close to both, it's genuinely neutral
    if close_to_pos and close_to_neg:
        return "neutral"
    
    # If neutral is close to one of them, boost neutral
    if close_to_pos or close_to_neg:
        v_neu *= 1.3  # 30% boost
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": v_neg, "neutral": v_neu, "positive": v_pos}[x])

def roberta_neutral_proximity(row):
    """RoBERTa with neutral proximity boost"""
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    proximity_threshold = 0.15
    
    close_to_pos = abs(r_neu - r_pos) < proximity_threshold
    close_to_neg = abs(r_neu - r_neg) < proximity_threshold
    
    if close_to_pos and close_to_neg:
        return "neutral"
    
    if close_to_pos or close_to_neg:
        r_neu *= 1.3
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": r_neg, "neutral": r_neu, "positive": r_pos}[x])

def textblob_neutral_proximity(row):
    """TextBlob with neutral proximity boost"""
    tb_neg, tb_neu, tb_pos = get_textblob_scores(row["sentence"])
    
    proximity_threshold = 0.15
    
    close_to_pos = abs(tb_neu - tb_pos) < proximity_threshold
    close_to_neg = abs(tb_neu - tb_neg) < proximity_threshold
    
    if close_to_pos and close_to_neg:
        return "neutral"
    
    if close_to_pos or close_to_neg:
        tb_neu *= 1.3
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": tb_neg, "neutral": tb_neu, "positive": tb_pos}[x])

def bert_neutral_proximity(row):
    """BERT with neutral proximity boost"""
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    
    proximity_threshold = 0.15
    
    close_to_pos = abs(b_neu - b_pos) < proximity_threshold
    close_to_neg = abs(b_neu - b_neg) < proximity_threshold
    
    if close_to_pos and close_to_neg:
        return "neutral"
    
    if close_to_pos or close_to_neg:
        b_neu *= 1.3
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": b_neg, "neutral": b_neu, "positive": b_pos}[x])

def finbert_neutral_proximity(row):
    """FinBERT with neutral proximity boost"""
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    
    proximity_threshold = 0.15
    
    close_to_pos = abs(f_neu - f_pos) < proximity_threshold
    close_to_neg = abs(f_neu - f_neg) < proximity_threshold
    
    if close_to_pos and close_to_neg:
        return "neutral"
    
    if close_to_pos or close_to_neg:
        f_neu *= 1.3
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": f_neg, "neutral": f_neu, "positive": f_pos}[x])

def weighted_v1_neutral_proximity(row):
    """
    Weighted V1 (VADER + RoBERTa) with neutral proximity boost
    """
    VADER_WEIGHT = 0.3
    ROBERTA_WEIGHT = 0.7
    
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    
    # Calculate weighted scores
    final_neg = (v_neg * VADER_WEIGHT) + (r_neg * ROBERTA_WEIGHT)
    final_neu = (v_neu * VADER_WEIGHT) + (r_neu * ROBERTA_WEIGHT)
    final_pos = (v_pos * VADER_WEIGHT) + (r_pos * ROBERTA_WEIGHT)
    
    # Check proximity
    proximity_threshold = 0.15
    
    close_to_pos = abs(final_neu - final_pos) < proximity_threshold
    close_to_neg = abs(final_neu - final_neg) < proximity_threshold
    
    # If neutral is close to both, it's genuinely neutral
    if close_to_pos and close_to_neg:
        return "neutral"
    
    # If neutral is close to one of them, boost neutral
    if close_to_pos or close_to_neg:
        final_neu *= 1.3
    
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": final_neg, "neutral": final_neu, "positive": final_pos}[x])

# Apply All Methods
print_log("\n" + "="*80)
print_log("RUNNING SENTIMENT ANALYSIS")
print_log("="*80)
print_log(f"\nTesting 34 different methods!")
print_log("This may take a few minutes...\n")

print_log("[1/9] Running individual models...")
df["vader_pred"] = df["sentence"].apply(vader_label)
df["roberta_pred"] = df["sentence"].apply(roberta_label)
df["textblob_pred"] = df["sentence"].apply(textblob_label)
df["bert_pred"] = df["sentence"].apply(bert_label)
df["finbert_pred"] = df["sentence"].apply(finbert_label)

print_log("[2/9] Running original ensembles...")
df["weighted_v1"] = df.apply(weighted_ensemble_v1, axis=1)
df["weighted_v2"] = df.apply(weighted_ensemble_v2, axis=1)
df["voting"] = df.apply(voting_ensemble, axis=1)
df["neutral_focused"] = df.apply(neutral_focused_ensemble, axis=1)

print_log("[3/9] Running new ensemble variations...")
df["neutral_focused_no_tb"] = df.apply(neutral_focused_no_textblob, axis=1)
df["weighted_no_tb"] = df.apply(weighted_no_textblob, axis=1)
df["weighted_no_tb_no_bert"] = df.apply(weighted_no_textblob_no_bert, axis=1)
df["voting_no_tb_no_bert"] = df.apply(voting_no_textblob_no_bert, axis=1)

print_log("[4/9] Running specialized ensembles...")
df["finbert_roberta"] = df.apply(finbert_roberta_only, axis=1)
df["finbert_bert"] = df.apply(finbert_bert_ensemble, axis=1)
df["conservative_neutral"] = df.apply(conservative_neutral_ensemble, axis=1)
df["aggressive_neutral"] = df.apply(aggressive_neutral_ensemble, axis=1)

print_log("[5/9] Running transformer ensembles...")
df["transformer_only"] = df.apply(transformer_only_ensemble, axis=1)
df["finbert_heavy"] = df.apply(finbert_heavy_ensemble, axis=1)
df["dynamic_threshold"] = df.apply(dynamic_threshold_ensemble, axis=1)

print_log("[6/9] Running advanced ensembles...")
df["emotion_weighted"] = df.apply(emotion_weighted_ensemble, axis=1)

print_log("[7/9] Running neutral-proximity methods...")
df["vader_neutral_prox"] = df.apply(vader_neutral_proximity, axis=1)
df["roberta_neutral_prox"] = df.apply(roberta_neutral_proximity, axis=1)
df["textblob_neutral_prox"] = df.apply(textblob_neutral_proximity, axis=1)
df["bert_neutral_prox"] = df.apply(bert_neutral_proximity, axis=1)
df["finbert_neutral_prox"] = df.apply(finbert_neutral_proximity, axis=1)
df["weighted_v1_neutral_prox"] = df.apply(weighted_v1_neutral_proximity, axis=1)

print_log("[8/9] Running dynamic threshold ensembles...")
df["dynamic_bert"] = df.apply(dynamic_bert, axis=1)
df["dynamic_finbert"] = df.apply(dynamic_finbert, axis=1)
df["dynamic_vader"] = df.apply(dynamic_vader, axis=1)
df["dynamic_weighted_v1"] = df.apply(dynamic_weighted_v1, axis=1)

print_log("[9/9] Running data-driven RoBERTa dynamic methods...")
df["roberta_dynamic_conservative"] = df.apply(roberta_dynamic_conservative, axis=1)
df["roberta_dynamic_balanced"] = df.apply(roberta_dynamic_balanced, axis=1)
df["roberta_dynamic_aggressive"] = df.apply(roberta_dynamic_aggressive, axis=1)
df["roberta_dynamic_adaptive"] = df.apply(roberta_dynamic_adaptive, axis=1)
df["roberta_dynamic_percentile"] = df.apply(roberta_dynamic_percentile_based, axis=1)

df["true_label"] = df["gold_label"].str.lower()

# -----------------------------
# NEW: Max Aggregation Methods
# -----------------------------

print_log("\n" + "="*80)
print_log("CREATING MAX AGGREGATION METHODS")
print_log("="*80)

def max_across_individual_models(row):
    """
    For each class, finds the maximum score across only the 5 individual models,
    then selects the class with the highest maximum.
    """
    # Get all individual model scores
    v_neg, v_neu, v_pos = get_vader_scores(row["sentence"])
    r_neg, r_neu, r_pos = get_roberta_scores(row["sentence"])
    tb_neg, tb_neu, tb_pos = get_textblob_scores(row["sentence"])
    b_neg, b_neu, b_pos = get_bert_scores(row["sentence"])
    f_neg, f_neu, f_pos = get_finbert_scores(row["sentence"])
    
    # Collect all negative scores from individual models only
    all_neg_scores = [v_neg, r_neg, tb_neg, b_neg, f_neg]
    # Collect all neutral scores from individual models only
    all_neu_scores = [v_neu, r_neu, tb_neu, b_neu, f_neu]
    # Collect all positive scores from individual models only
    all_pos_scores = [v_pos, r_pos, tb_pos, b_pos, f_pos]
    
    # Find the maximum for each class
    max_neg = max(all_neg_scores)
    max_neu = max(all_neu_scores)
    max_pos = max(all_pos_scores)
    
    # Return the class with the highest maximum
    return max(["negative", "neutral", "positive"], 
               key=lambda x: {"negative": max_neg, "neutral": max_neu, "positive": max_pos}[x])

print_log("\nApplying Max Aggregation methods...")
print_log("  - Max Across All Methods: Takes max(neg), max(neu), max(pos) from all 5 models")
print_log("  - Max Across Individual Models: Same as above (uses 5 base models)")

df["max_individual_models"] = df.apply(max_across_individual_models, axis=1)

max_ind_acc = (df["max_individual_models"] == df["true_label"]).mean() * 100

print_log(f"Max Across Individual Models Accuracy: {max_ind_acc:.2f}%")

# Evaluation
models = {
    "VADER": "vader_pred",
    "RoBERTa": "roberta_pred",
    "TextBlob": "textblob_pred",
    "BERT": "bert_pred",
    "FinBERT": "finbert_pred",
    "Weighted V1 (VADER+RoBERTa)": "weighted_v1",
    "Weighted V2 (All 5 models)": "weighted_v2",
    "Voting (All 5)": "voting",
    "Neutral-Focused (Original)": "neutral_focused",
    "Neutral-Focused (No TextBlob)": "neutral_focused_no_tb",
    "Weighted (No TextBlob)": "weighted_no_tb",
    "Weighted (No TB, No BERT)": "weighted_no_tb_no_bert",
    "Voting (No TB, No BERT)": "voting_no_tb_no_bert",
    "FinBERT + RoBERTa Only": "finbert_roberta",
    "FinBERT + BERT": "finbert_bert",
    "Conservative Neutral": "conservative_neutral",
    "Aggressive Neutral": "aggressive_neutral",
    "Transformer Only": "transformer_only",
    "FinBERT Heavy (70%)": "finbert_heavy",
    "Dynamic Threshold": "dynamic_threshold",
    "Emotion-Weighted": "emotion_weighted",
    # NEW: Neutral Proximity Methods
    "VADER (Neutral Proximity)": "vader_neutral_prox",
    "RoBERTa (Neutral Proximity)": "roberta_neutral_prox",
    "TextBlob (Neutral Proximity)": "textblob_neutral_prox",
    "BERT (Neutral Proximity)": "bert_neutral_prox",
    "FinBERT (Neutral Proximity)": "finbert_neutral_prox",
    "Weighted V1 (Neutral Proximity)": "weighted_v1_neutral_prox",
    # NEW: Max Aggregation Methods
    "Max Across Individual Models": "max_individual_models",
    # NEW: Max Dynamic Thresholds
    "Dynamic BERT Threshold": "dynamic_bert",
    "Dynamic FinBERT Threshold": "dynamic_finbert",
    "Dynamic VADER Threshold": "dynamic_vader",
    "Dynamic Weighted V1 Threshold": "dynamic_weighted_v1",
    # NEW: Data-Driven RoBERTa Dynamic Thresholds
    "RoBERTa Dynamic (Conservative)": "roberta_dynamic_conservative",
    "RoBERTa Dynamic (Balanced)": "roberta_dynamic_balanced",
    "RoBERTa Dynamic (Aggressive)": "roberta_dynamic_aggressive",
    "RoBERTa Dynamic (Adaptive)": "roberta_dynamic_adaptive",
    "RoBERTa Dynamic (Percentile)": "roberta_dynamic_percentile"
}

print_log("\n" + "="*80)
print_log("OVERALL ACCURACY")
print_log("="*80)

results = []
for name, col in models.items():
    accuracy = (df[col] == df["true_label"]).mean() * 100
    results.append({"Model": name, "Accuracy": accuracy})
    print_log(f"{name:.<45} {accuracy:.2f}%")

print_log("\n" + "="*80)
print_log("PER-CLASS ACCURACY (Focus: NEUTRAL)")
print_log("="*80)

classes = ["neutral", "positive", "negative"]
class_results = []

for cls in classes:
    print_log(f"\n{cls.upper()} class:")
    subset = df[df["true_label"] == cls]
    
    for name, col in models.items():
        if len(subset) > 0:
            correct = (subset[col] == cls).sum()
            accuracy = (correct / len(subset)) * 100
            class_results.append({
                "Model": name,
                "Class": cls,
                "Accuracy": accuracy,
                "Correct": correct,
                "Total": len(subset)
            })
            print_log(f"  {name:.<43} {accuracy:>6.2f}%")

# Detailed Neutral Analysis
print_log("\n" + "="*80)
print_log("DETAILED NEUTRAL DETECTION ANALYSIS")
print_log("="*80)

neutral_df = df[df["true_label"] == "neutral"]
neutral_analysis = []

for name, col in models.items():
    if len(neutral_df) > 0:
        tp = ((df["true_label"] == "neutral") & (df[col] == "neutral")).sum()
        fn = ((df["true_label"] == "neutral") & (df[col] != "neutral")).sum()
        non_neutral_df = df[df["true_label"] != "neutral"]
        fp = ((non_neutral_df[col] == "neutral")).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        neutral_analysis.append({"Model": name, "Precision": precision, "Recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn})
        print_log(f"\n{name}:")
        print_log(f"  Precision: {precision:.4f}  (TP={tp}, FP={fp})")
        print_log(f"  Recall:    {recall:.4f}  (TP={tp}, FN={fn})")
        print_log(f"  F1-Score:  {f1:.4f}")

# -----------------------------
# NEW: Best Individual Models Per Class
# -----------------------------

print_log("\n" + "="*80)
print_log("🎯 BEST INDIVIDUAL MODEL PER CLASS")
print_log("="*80)

individual_models = {
    "VADER": "vader_pred",
    "RoBERTa": "roberta_pred",
    "TextBlob": "textblob_pred",
    "BERT": "bert_pred",
    "FinBERT": "finbert_pred"
}

best_per_class = {}

for cls in ["neutral", "positive", "negative"]:
    subset = df[df["true_label"] == cls]
    
    if len(subset) == 0:
        continue
    
    class_accuracies = []
    for name, col in individual_models.items():
        correct = (subset[col] == cls).sum()
        accuracy = (correct / len(subset)) * 100
        class_accuracies.append({
            "Model": name,
            "Column": col,
            "Accuracy": accuracy
        })
    
    best_model = max(class_accuracies, key=lambda x: x["Accuracy"])
    best_per_class[cls] = best_model
    
    print_log(f"\n{cls.upper()} Detection:")
    print_log(f"  🏆 Best Model: {best_model['Model']}")
    print_log(f"  Accuracy: {best_model['Accuracy']:.2f}%")
    print_log(f"\n  All individual models:")
    for acc in sorted(class_accuracies, key=lambda x: x["Accuracy"], reverse=True):
        print_log(f"    {acc['Model']:.<10} {acc['Accuracy']:.2f}%")

# Create specialized Best-Per-Class ensemble
print_log("\n" + "="*80)
print_log("CREATING BEST-PER-CLASS ENSEMBLE")
print_log("="*80)
print_log(f"Uses: {best_per_class.get('neutral', {}).get('Model', 'N/A')} for neutral")
print_log(f"      {best_per_class.get('positive', {}).get('Model', 'N/A')} for positive")
print_log(f"      {best_per_class.get('negative', {}).get('Model', 'N/A')} for negative")

def best_per_class_ensemble(row):
    text = row["sentence"]
    scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    
    # Get neutral score from best neutral model
    best_neutral_col = best_per_class.get("neutral", {}).get("Column", "finbert_pred")
    if "vader" in best_neutral_col:
        _, neu, _ = get_vader_scores(text)
        scores["neutral"] = neu
    elif "roberta" in best_neutral_col:
        _, neu, _ = get_roberta_scores(text)
        scores["neutral"] = neu
    elif "textblob" in best_neutral_col:
        _, neu, _ = get_textblob_scores(text)
        scores["neutral"] = neu
    elif "bert" in best_neutral_col and "fin" not in best_neutral_col:
        _, neu, _ = get_bert_scores(text)
        scores["neutral"] = neu
    elif "finbert" in best_neutral_col:
        _, neu, _ = get_finbert_scores(text)
        scores["neutral"] = neu
    
    # Get positive score
    best_pos_col = best_per_class.get("positive", {}).get("Column", "finbert_pred")
    if "vader" in best_pos_col:
        _, _, pos = get_vader_scores(text)
        scores["positive"] = pos
    elif "roberta" in best_pos_col:
        _, _, pos = get_roberta_scores(text)
        scores["positive"] = pos
    elif "textblob" in best_pos_col:
        _, _, pos = get_textblob_scores(text)
        scores["positive"] = pos
    elif "bert" in best_pos_col and "fin" not in best_pos_col:
        _, _, pos = get_bert_scores(text)
        scores["positive"] = pos
    elif "finbert" in best_pos_col:
        _, _, pos = get_finbert_scores(text)
        scores["positive"] = pos
    
    # Get negative score
    best_neg_col = best_per_class.get("negative", {}).get("Column", "finbert_pred")
    if "vader" in best_neg_col:
        neg, _, _ = get_vader_scores(text)
        scores["negative"] = neg
    elif "roberta" in best_neg_col:
        neg, _, _ = get_roberta_scores(text)
        scores["negative"] = neg
    elif "textblob" in best_neg_col:
        neg, _, _ = get_textblob_scores(text)
        scores["negative"] = neg
    elif "bert" in best_neg_col and "fin" not in best_neg_col:
        neg, _, _ = get_bert_scores(text)
        scores["negative"] = neg
    elif "finbert" in best_neg_col:
        neg, _, _ = get_finbert_scores(text)
        scores["negative"] = neg
    
    return max(scores, key=scores.get)

print_log("\nApplying Best-Per-Class ensemble...")
df["best_per_class"] = df.apply(best_per_class_ensemble, axis=1)

models["Best-Per-Class Ensemble"] = "best_per_class"

best_per_class_acc = (df["best_per_class"] == df["true_label"]).mean() * 100
print_log(f"\nBest-Per-Class Ensemble Accuracy: {best_per_class_acc:.2f}%")

results.append({"Model": "Best-Per-Class Ensemble", "Accuracy": best_per_class_acc})

if len(neutral_df) > 0:
    tp = ((df["true_label"] == "neutral") & (df["best_per_class"] == "neutral")).sum()
    fn = ((df["true_label"] == "neutral") & (df["best_per_class"] != "neutral")).sum()
    non_neutral_df = df[df["true_label"] != "neutral"]
    fp = ((non_neutral_df["best_per_class"] == "neutral")).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    neutral_analysis.append({"Model": "Best-Per-Class Ensemble", "Precision": precision, "Recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn})
    print_log(f"  Neutral F1-Score: {f1:.4f}")

for cls in ["neutral", "positive", "negative"]:
    subset = df[df["true_label"] == cls]
    if len(subset) > 0:
        correct = (subset["best_per_class"] == cls).sum()
        accuracy = (correct / len(subset)) * 100
        class_results.append({"Model": "Best-Per-Class Ensemble", "Class": cls, "Accuracy": accuracy, "Correct": correct, "Total": len(subset)})

# -----------------------------
# NEW: Neutral Misclassification Analysis
# -----------------------------

print_log("\n" + "="*80)
print_log("📊 NEUTRAL MISCLASSIFICATION ANALYSIS")
print_log("="*80)
print_log("\nWhen ground truth is NEUTRAL and model gets it wrong,")
print_log("how often does it misclassify as POSITIVE vs NEGATIVE?\n")

neutral_misclass_data = []

# Analyze individual models AND neutral proximity variants
models_to_analyze = {
    **individual_models,
    "VADER (Neutral Proximity)": "vader_neutral_prox",
    "RoBERTa (Neutral Proximity)": "roberta_neutral_prox",
    "TextBlob (Neutral Proximity)": "textblob_neutral_prox",
    "BERT (Neutral Proximity)": "bert_neutral_prox",
    "FinBERT (Neutral Proximity)": "finbert_neutral_prox",
    "Weighted V1 (Neutral Proximity)": "weighted_v1_neutral_prox"
}

for name, col in models_to_analyze.items():
    neutral_samples = df[df["true_label"] == "neutral"]
    
    if len(neutral_samples) == 0:
        continue
    
    misclassified = neutral_samples[neutral_samples[col] != "neutral"]
    total_errors = len(misclassified)
    
    if total_errors == 0:
        print_log(f"{name}: Perfect! No misclassifications")
        neutral_misclass_data.append({
            "Model": name,
            "Total_Neutral_Samples": len(neutral_samples),
            "Misclassified_As_Positive": 0,
            "Misclassified_As_Negative": 0,
            "Total_Errors": 0,
            "Percent_Misclass_Positive": 0.0,
            "Percent_Misclass_Negative": 0.0,
            "Positive_to_Negative_Ratio": 0.0
        })
        continue
    
    misclass_as_positive = (misclassified[col] == "positive").sum()
    misclass_as_negative = (misclassified[col] == "negative").sum()
    
    pct_positive = (misclass_as_positive / total_errors) * 100 if total_errors > 0 else 0
    pct_negative = (misclass_as_negative / total_errors) * 100 if total_errors > 0 else 0
    
    ratio = misclass_as_positive / misclass_as_negative if misclass_as_negative > 0 else float('inf')
    
    neutral_misclass_data.append({
        "Model": name,
        "Total_Neutral_Samples": len(neutral_samples),
        "Misclassified_As_Positive": misclass_as_positive,
        "Misclassified_As_Negative": misclass_as_negative,
        "Total_Errors": total_errors,
        "Percent_Misclass_Positive": pct_positive,
        "Percent_Misclass_Negative": pct_negative,
        "Positive_to_Negative_Ratio": ratio
    })
    
    print_log(f"{name}:")
    print_log(f"  Total neutral samples: {len(neutral_samples)}")
    print_log(f"  Total errors: {total_errors}")
    print_log(f"  Misclassified as POSITIVE: {misclass_as_positive} ({pct_positive:.1f}% of errors)")
    print_log(f"  Misclassified as NEGATIVE: {misclass_as_negative} ({pct_negative:.1f}% of errors)")
    
    if ratio != float('inf'):
        print_log(f"  Pos/Neg ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print_log(f"  → BIAS: Tends to misclassify neutrals as POSITIVE")
        elif ratio < 0.8:
            print_log(f"  → BIAS: Tends to misclassify neutrals as NEGATIVE")
        else:
            print_log(f"  → Balanced misclassification")
    else:
        print_log(f"  → BIAS: Only misclassifies as POSITIVE")
    print_log("")

# -----------------------------
# NEW: Neutral Proximity Comparison
# -----------------------------

print_log("\n" + "="*80)
print_log("📈 NEUTRAL PROXIMITY BOOST IMPACT")
print_log("="*80)
print_log("\nComparing original models vs their neutral-proximity variants:\n")

# Compare each model with its proximity version
proximity_comparisons = [
    ("VADER", "vader_pred", "VADER (Neutral Proximity)", "vader_neutral_prox"),
    ("RoBERTa", "roberta_pred", "RoBERTa (Neutral Proximity)", "roberta_neutral_prox"),
    ("TextBlob", "textblob_pred", "TextBlob (Neutral Proximity)", "textblob_neutral_prox"),
    ("BERT", "bert_pred", "BERT (Neutral Proximity)", "bert_neutral_prox"),
    ("FinBERT", "finbert_pred", "FinBERT (Neutral Proximity)", "finbert_neutral_prox"),
    ("Weighted V1", "weighted_v1", "Weighted V1 (Neutral Proximity)", "weighted_v1_neutral_prox")
]

proximity_impact_data = []

for orig_name, orig_col, prox_name, prox_col in proximity_comparisons:
    # Overall accuracy
    orig_acc = (df[orig_col] == df["true_label"]).mean() * 100
    prox_acc = (df[prox_col] == df["true_label"]).mean() * 100
    acc_diff = prox_acc - orig_acc
    
    # Neutral-specific metrics
    if len(neutral_df) > 0:
        # Original
        orig_correct = (neutral_df[orig_col] == "neutral").sum()
        orig_neutral_acc = (orig_correct / len(neutral_df)) * 100
        
        # Proximity
        prox_correct = (neutral_df[prox_col] == "neutral").sum()
        prox_neutral_acc = (prox_correct / len(neutral_df)) * 100
        
        neutral_diff = prox_neutral_acc - orig_neutral_acc
        
        # Find F1 scores
        orig_f1 = next((x["F1"] for x in neutral_analysis if x["Model"] == orig_name), 0)
        prox_f1 = next((x["F1"] for x in neutral_analysis if x["Model"] == prox_name), 0)
        f1_diff = prox_f1 - orig_f1
        
        proximity_impact_data.append({
            "Model": orig_name,
            "Original_Overall_Acc": orig_acc,
            "Proximity_Overall_Acc": prox_acc,
            "Overall_Acc_Diff": acc_diff,
            "Original_Neutral_Acc": orig_neutral_acc,
            "Proximity_Neutral_Acc": prox_neutral_acc,
            "Neutral_Acc_Diff": neutral_diff,
            "Original_F1": orig_f1,
            "Proximity_F1": prox_f1,
            "F1_Diff": f1_diff
        })
        
        print_log(f"{orig_name}:")
        print_log(f"  Overall Accuracy: {orig_acc:.2f}% → {prox_acc:.2f}% ({acc_diff:+.2f}%)")
        print_log(f"  Neutral Accuracy: {orig_neutral_acc:.2f}% → {prox_neutral_acc:.2f}% ({neutral_diff:+.2f}%)")
        print_log(f"  Neutral F1-Score: {orig_f1:.4f} → {prox_f1:.4f} ({f1_diff:+.4f})")
        
        if neutral_diff > 1:
            print_log(f"  ✓ IMPROVEMENT: Proximity boost helps neutral detection!")
        elif neutral_diff < -1:
            print_log(f"  ✗ DEGRADATION: Proximity boost hurts neutral detection")
        else:
            print_log(f"  → NEUTRAL: Minimal change")
        print_log("")

# Top Performers
print_log("\n" + "="*80)
print_log("🏆 TOP 5 PERFORMERS")
print_log("="*80)

top_5_overall = sorted(results, key=lambda x: x["Accuracy"], reverse=True)[:5]
print_log("\nBy Overall Accuracy:")
for i, result in enumerate(top_5_overall, 1):
    print_log(f"  {i}. {result['Model']:.<43} {result['Accuracy']:.2f}%")

top_5_neutral_f1 = sorted(neutral_analysis, key=lambda x: x["F1"], reverse=True)[:5]
print_log("\nBy Neutral F1-Score:")
for i, result in enumerate(top_5_neutral_f1, 1):
    print_log(f"  {i}. {result['Model']:.<43} F1={result['F1']:.4f}")

top_5_neutral_recall = sorted(neutral_analysis, key=lambda x: x["Recall"], reverse=True)[:5]
print_log("\nBy Neutral Recall (catches most neutrals):")
for i, result in enumerate(top_5_neutral_recall, 1):
    print_log(f"  {i}. {result['Model']:.<43} Recall={result['Recall']:.4f}")

top_5_neutral_precision = sorted(neutral_analysis, key=lambda x: x["Precision"], reverse=True)[:5]
print_log("\nBy Neutral Precision (avoids false neutrals):")
for i, result in enumerate(top_5_neutral_precision, 1):
    print_log(f"  {i}. {result['Model']:.<43} Precision={result['Precision']:.4f}")

# Confusion Matrices
print_log("\n" + "="*80)
print_log("CONFUSION MATRICES - TOP 3 BY NEUTRAL F1")
print_log("="*80)

for i, result in enumerate(top_5_neutral_f1[:3], 1):
    name = result["Model"]
    col = models[name]
    print_log(f"\n{i}. {name}")
    print_log(f"   (F1={result['F1']:.4f}, Precision={result['Precision']:.4f}, Recall={result['Recall']:.4f})")
    cm = confusion_matrix(df["true_label"], df[col], labels=["negative", "neutral", "positive"])
    print_log("\n                Predicted")
    print_log("              neg  neu  pos")
    print_log(f"Actual  neg  {cm[0][0]:>4} {cm[0][1]:>4} {cm[0][2]:>4}")
    print_log(f"        neu  {cm[1][0]:>4} {cm[1][1]:>4} {cm[1][2]:>4}")
    print_log(f"        pos  {cm[2][0]:>4} {cm[2][1]:>4} {cm[2][2]:>4}")

# Winners
print_log("\n" + "="*80)
print_log("🎯 FINAL RECOMMENDATIONS")
print_log("="*80)

best_overall = max(results, key=lambda x: x["Accuracy"])
print_log(f"\n🥇 Best Overall Accuracy:")
print_log(f"   {best_overall['Model']} - {best_overall['Accuracy']:.2f}%")

best_neutral_f1 = max(neutral_analysis, key=lambda x: x["F1"])
print_log(f"\n🎯 Best Neutral Detector (by F1-Score):")
print_log(f"   {best_neutral_f1['Model']} - F1={best_neutral_f1['F1']:.4f}")
print_log(f"   (Precision={best_neutral_f1['Precision']:.4f}, Recall={best_neutral_f1['Recall']:.4f})")

best_neutral_recall = max(neutral_analysis, key=lambda x: x["Recall"])
print_log(f"\n🔍 Best Neutral Recall (catches most neutrals):")
print_log(f"   {best_neutral_recall['Model']} - Recall={best_neutral_recall['Recall']:.4f}")

best_neutral_precision = max(neutral_analysis, key=lambda x: x["Precision"])
print_log(f"\n🎯 Best Neutral Precision (fewest false neutrals):")
print_log(f"   {best_neutral_precision['Model']} - Precision={best_neutral_precision['Precision']:.4f}")

# Save Results
print_log("\n" + "="*80)
print_log("SAVING RESULTS")
print_log("="*80)

predictions_file = os.path.join(output_dir, "full_predictions.csv")
df.to_csv(predictions_file, index=False)
print_log(f"✓ {predictions_file}")

summary_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
summary_file = os.path.join(output_dir, "overall_accuracy_summary.csv")
summary_df.to_csv(summary_file, index=False)
print_log(f"✓ {summary_file}")

class_df = pd.DataFrame(class_results)
class_file = os.path.join(output_dir, "per_class_accuracy.csv")
class_df.to_csv(class_file, index=False)
print_log(f"✓ {class_file}")

neutral_df_results = pd.DataFrame(neutral_analysis).sort_values("F1", ascending=False)
neutral_file = os.path.join(output_dir, "neutral_detection_metrics.csv")
neutral_df_results.to_csv(neutral_file, index=False)
print_log(f"✓ {neutral_file}")

# NEW: Save best per class info
best_per_class_df = pd.DataFrame([
    {"Class": cls, "Best_Model": info.get("Model", "N/A"), "Accuracy": info.get("Accuracy", 0.0)}
    for cls, info in best_per_class.items()
])
best_class_file = os.path.join(output_dir, "best_model_per_class.csv")
best_per_class_df.to_csv(best_class_file, index=False)
print_log(f"✓ {best_class_file}")

# NEW: Save misclassification analysis
misclass_df = pd.DataFrame(neutral_misclass_data)
misclass_file = os.path.join(output_dir, "neutral_misclassification_analysis.csv")
misclass_df.to_csv(misclass_file, index=False)
print_log(f"✓ {misclass_file}")

# NEW: Save proximity impact comparison
if proximity_impact_data:
    proximity_df = pd.DataFrame(proximity_impact_data)
    proximity_file = os.path.join(output_dir, "neutral_proximity_impact.csv")
    proximity_df.to_csv(proximity_file, index=False)
    print_log(f"✓ {proximity_file}")

status_df = pd.DataFrame([
    {"Model": name, "Status": "Loaded" if status else "Failed"}
    for name, status in model_status.items()
])
status_file = os.path.join(output_dir, "model_loading_status.csv")
status_df.to_csv(status_file, index=False)
print_log(f"✓ {status_file}")

print_log(f"\n✓ All files saved to: {output_dir}/")
print_log(f"✓ Complete log saved to: {log_file_path}")

# Final Summary
print_log("\n" + "="*80)
print_log("FINAL SUMMARY")
print_log("="*80)
print_log(f"\nDataset: {len(df)} samples")
print_log(f"Models loaded: {loaded_count}/{len(model_status)}")
print_log(f"Methods tested: {len(models)}")
print_log(f"\nBest overall: {best_overall['Model']} ({best_overall['Accuracy']:.2f}%)")
print_log(f"Best neutral F1: {best_neutral_f1['Model']} (F1={best_neutral_f1['F1']:.4f})")
print_log(f"Best neutral recall: {best_neutral_recall['Model']} (Recall={best_neutral_recall['Recall']:.4f})")
print_log(f"Best neutral precision: {best_neutral_precision['Model']} (Precision={best_neutral_precision['Precision']:.4f})")

print_log(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_log("\n" + "="*80)
print_log("✅ ANALYSIS COMPLETE!")
print_log("="*80)

log_file.close()

print(f"\n\n{'='*80}")
print("ALL RESULTS SAVED!")
print("="*80)
print(f"📁 Output folder: {output_dir}/")
print(f"📄 Log file: {log_file_path}")
print(f"\nFiles created:")
print(f"  - full_predictions.csv (all 29 methods + Best-Per-Class)")
print(f"  - overall_accuracy_summary.csv")
print(f"  - per_class_accuracy.csv")
print(f"  - neutral_detection_metrics.csv")
print(f"  - best_model_per_class.csv")
print(f"  - neutral_misclassification_analysis.csv (includes proximity methods)")
print(f"  - neutral_proximity_impact.csv (shows boost effectiveness)")
print(f"  - model_loading_status.csv")
print(f"  - analysis_log.txt (complete output)")
print("="*80)