import os
from db_utils import store_feedback
from llm_client import call_llm
import hashlib
from db_utils import store_feedback , init_feedback_db


init_feedback_db()


def analyze_feedback(feedback_text):
    """Analyze user feedback using LLM."""
    prompt = f"""
You are an expert QA reviewer. A user gave the following feedback after receiving AI-generated code recommendations:

--- FEEDBACK START ---
{feedback_text}
--- FEEDBACK END ---

1. Is this feedback helpful and relevant? Answer only YES or NO.
2. What is the sentiment? Choose one: positive / neutral / negative.
3. If the feedback is not useful, say "Reject". If useful, say "Accept".

Format your answer as:
Relevance: <YES/NO>
Sentiment: <positive/neutral/negative>
Decision: <Accept/Reject>
"""
    try:
        result = call_llm(prompt)
        print("[🔍 Feedback Analysis]:", result)

        lines = result.strip().splitlines()
        relevance = sentiment = decision = None
        for line in lines:
            if line.startswith("Relevance:"):
                relevance = line.split(":", 1)[1].strip()
            elif line.startswith("Sentiment:"):
                sentiment = line.split(":", 1)[1].strip()
            elif line.startswith("Decision:"):
                decision = line.split(":", 1)[1].strip()

        return {
            "relevance": relevance,
            "sentiment": sentiment,
            "decision": decision
        }
    except Exception as e:
        import logging
        logging.error(f"Error analyzing feedback: {e}")
        return {
            "relevance": "NO",
            "sentiment": "neutral",
            "decision": "Reject"
        }

def submit_user_feedback(file, code_text, feedback, rating):
    """Process user feedback and store it."""
    if file is not None:
        try:
            # Case 1: file is a file path string
            if isinstance(file, str) and os.path.exists(file):
                filename = os.path.basename(file)
                with open(file, "rb") as f:
                    raw_bytes = f.read()
            # Case 2: file is an UploadedFile object
            elif hasattr(file, "read"):
                filename = getattr(file, "name", "uploaded_file")
                file.seek(0)
                raw_bytes = file.read()
            else:
                return "⚠️ Invalid file object type", "", None

            # Decode content
            try:
                code = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                code = raw_bytes.decode("latin-1")
        except Exception as e:
            return f"⚠️ Error reading uploaded file: {str(e)}", "", None
    elif code_text:
        filename = "manual_input"
        code = code_text
    else:
        return "⚠️ No code to provide feedback on", "", None

    # Create unique key
    file_hash = hashlib.md5(code.encode('utf-8')).hexdigest()

    # Analyze feedback
    analysis = analyze_feedback(feedback)

    if analysis["decision"] == "Accept":
        store_feedback(
            file_hash=file_hash,
            filename=filename,
            feedback=feedback,
            rating=rating,
            relevance=analysis["relevance"],
            sentiment=analysis["sentiment"],
            decision=analysis["decision"],
        )
        return "✅ Thanks! Your feedback was accepted and will help improve future results.", "", None
    else:
        return "⚠️ Thanks! Your feedback was reviewed but marked not useful by our reviewer.", "", None
