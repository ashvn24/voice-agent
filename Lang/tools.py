"""
Vidur AI — Tools Module

All agentic tools that Vidur can use:
  1. web_search        — General web lookup (DuckDuckGo)
  2. calculator         — GPA, CGPA, loan EMI, credit hours, percentages
  3. mood_tracker       — Log & analyze student emotional state across sessions
  4. campus_kb          — RAG over college PDFs (ChromaDB)
  5. job_search         — Search internships & jobs
  6. course_recommender — Find online courses & tutorials
"""

import os
import json
import math
from datetime import datetime
from pathlib import Path

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from config import Config


# ═══════════════════════════════════════════════════
# 1. WEB SEARCH
# ═══════════════════════════════════════════════════

def web_search(query: str) -> str:
    """General web search using DuckDuckGo."""
    try:
        print(f"  🌐 Web search: \"{query}\"")
        results = list(DDGS().text(query, max_results=Config.WEB_SEARCH_MAX_RESULTS))
        if not results:
            return ""
        snippets = [f"{r.get('title','')}: {r.get('body','')}" for r in results]
        print(f"  ✅ Found {len(results)} results")
        return "\n".join(snippets)
    except Exception as e:
        print(f"  ⚠ Web search failed: {e}")
        return ""


# ═══════════════════════════════════════════════════
# 2. CALCULATOR
# ═══════════════════════════════════════════════════

class Calculator:
    """
    Academic & financial calculators:
      - GPA / CGPA from grades
      - Loan EMI
      - Percentage / grade conversion
      - Credit hour totals
      - General math expressions
    """

    # Standard grade → grade point mapping (common Indian/US 10-point scale)
    GRADE_POINTS = {
        "O": 10, "A+": 10, "A": 9, "B+": 8, "B": 7, "C+": 6,
        "C": 5, "D": 4, "F": 0, "P": 5, "AB": 0,
        # 4.0 scale
        "a+": 4.0, "a": 4.0, "a-": 3.7, "b+": 3.3, "b": 3.0,
        "b-": 2.7, "c+": 2.3, "c": 2.0, "c-": 1.7, "d+": 1.3,
        "d": 1.0, "d-": 0.7, "f": 0.0,
    }

    @staticmethod
    def calculate_gpa(grades_and_credits: list[dict]) -> dict:
        """
        Calculate GPA from a list of {grade, credits}.
        Example: [{"grade": "A", "credits": 4}, {"grade": "B+", "credits": 3}]
        """
        total_points = 0
        total_credits = 0

        for item in grades_and_credits:
            grade = item.get("grade", "").strip()
            credits = float(item.get("credits", 0))

            gp = Calculator.GRADE_POINTS.get(grade) or Calculator.GRADE_POINTS.get(grade.lower(), 0)
            total_points += gp * credits
            total_credits += credits

        if total_credits == 0:
            return {"gpa": 0, "total_credits": 0, "error": "No credits provided"}

        gpa = round(total_points / total_credits, 2)
        return {"gpa": gpa, "total_credits": total_credits}

    @staticmethod
    def calculate_cgpa(semester_gpas: list[dict]) -> dict:
        """
        Calculate CGPA from semester GPAs.
        Example: [{"gpa": 8.5, "credits": 24}, {"gpa": 7.8, "credits": 22}]
        """
        total_weighted = 0
        total_credits = 0

        for sem in semester_gpas:
            gpa = float(sem.get("gpa", 0))
            credits = float(sem.get("credits", 0))
            total_weighted += gpa * credits
            total_credits += credits

        if total_credits == 0:
            return {"cgpa": 0, "error": "No data provided"}

        cgpa = round(total_weighted / total_credits, 2)
        return {"cgpa": cgpa, "total_credits": total_credits}

    @staticmethod
    def loan_emi(principal: float, annual_rate: float, years: int) -> dict:
        """Calculate monthly EMI for an education loan."""
        if annual_rate == 0:
            emi = principal / (years * 12)
        else:
            r = annual_rate / (12 * 100)  # monthly rate
            n = years * 12                # total months
            emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)

        total_payment = emi * years * 12
        total_interest = total_payment - principal

        return {
            "monthly_emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "principal": principal,
            "rate": annual_rate,
            "years": years,
        }

    @staticmethod
    def percentage_to_gpa(percentage: float, scale: str = "10") -> dict:
        """Convert percentage to GPA (approximate)."""
        if scale == "10":
            gpa = round(percentage / 9.5, 2)  # Common Indian conversion
            gpa = min(gpa, 10.0)
        elif scale == "4":
            if percentage >= 90: gpa = 4.0
            elif percentage >= 80: gpa = 3.7
            elif percentage >= 70: gpa = 3.3
            elif percentage >= 60: gpa = 3.0
            elif percentage >= 50: gpa = 2.0
            else: gpa = 1.0
        else:
            gpa = round(percentage / 10, 2)

        return {"percentage": percentage, "gpa": gpa, "scale": scale}

    @staticmethod
    def safe_eval(expression: str) -> str:
        """Evaluate a safe math expression."""
        try:
            # Only allow safe math operations
            allowed = set("0123456789+-*/.() %")
            clean = expression.replace("^", "**").replace("x", "*")
            if not all(c in allowed or c.isspace() for c in clean):
                return f"I can help with basic math — could you give me the numbers?"

            result = eval(clean, {"__builtins__": {}}, {"math": math})
            return f"{expression} = {round(result, 4)}"
        except Exception:
            return f"I couldn't calculate that — could you rephrase the numbers?"


def calculator(query: str, data: dict = None) -> str:
    """Route calculator requests to the right function."""
    lower = query.lower()
    data = data or {}

    try:
        if "emi" in lower or "loan" in lower:
            principal = data.get("principal", 500000)
            rate = data.get("rate", 8.5)
            years = data.get("years", 5)
            result = Calculator.loan_emi(principal, rate, years)
            return (
                f"For a loan of ₹{result['principal']:,.0f} at {result['rate']}% for {result['years']} years: "
                f"Monthly EMI is ₹{result['monthly_emi']:,.0f}. "
                f"Total payment: ₹{result['total_payment']:,.0f} "
                f"(interest: ₹{result['total_interest']:,.0f})."
            )

        elif "cgpa" in lower:
            semesters = data.get("semesters", [])
            if semesters:
                result = Calculator.calculate_cgpa(semesters)
                return f"Your CGPA is {result['cgpa']} across {result['total_credits']} total credits."
            return "To calculate your CGPA, I need each semester's GPA and credit hours."

        elif "gpa" in lower:
            grades = data.get("grades", [])
            if grades:
                result = Calculator.calculate_gpa(grades)
                return f"Your GPA is {result['gpa']} across {result['total_credits']} credits."
            return "To calculate your GPA, I need your grades and credit hours for each subject."

        elif "percentage" in lower or "percent" in lower or "convert" in lower:
            pct = data.get("percentage", 0)
            scale = data.get("scale", "10")
            if pct:
                result = Calculator.percentage_to_gpa(pct, scale)
                return f"{result['percentage']}% converts to approximately {result['gpa']} GPA on a {result['scale']}-point scale."
            return "What percentage would you like me to convert?"

        else:
            # Try evaluating as a math expression
            return Calculator.safe_eval(query)

    except Exception as e:
        return f"Could you give me the exact numbers? I want to get this right for you."


# ═══════════════════════════════════════════════════
# 3. MOOD TRACKER
# ═══════════════════════════════════════════════════

class MoodTracker:
    """
    Tracks student emotional state across sessions.
    Stores mood entries in a JSON file.
    Detects patterns (consistent stress, improvement, etc.)
    """

    MOOD_KEYWORDS = {
        "great":     {"score": 5, "label": "great"},
        "good":      {"score": 4, "label": "good"},
        "happy":     {"score": 5, "label": "happy"},
        "excited":   {"score": 5, "label": "excited"},
        "fine":      {"score": 3, "label": "okay"},
        "okay":      {"score": 3, "label": "okay"},
        "alright":   {"score": 3, "label": "okay"},
        "not great": {"score": 2, "label": "not great"},
        "tired":     {"score": 2, "label": "tired"},
        "stressed":  {"score": 1, "label": "stressed"},
        "anxious":   {"score": 1, "label": "anxious"},
        "worried":   {"score": 1, "label": "worried"},
        "overwhelmed": {"score": 1, "label": "overwhelmed"},
        "sad":       {"score": 1, "label": "sad"},
        "depressed": {"score": 0, "label": "depressed"},
        "lonely":    {"score": 1, "label": "lonely"},
        "angry":     {"score": 1, "label": "angry"},
        "frustrated":{"score": 1, "label": "frustrated"},
        "confused":  {"score": 2, "label": "confused"},
        "lost":      {"score": 1, "label": "lost"},
        "scared":    {"score": 1, "label": "scared"},
        "hopeless":  {"score": 0, "label": "hopeless"},
        "burned out": {"score": 0, "label": "burned out"},
        "burnout":   {"score": 0, "label": "burned out"},
    }

    def __init__(self):
        self.mood_file = Config.MOOD_FILE
        self.mood_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load()

    def _load(self) -> list:
        if self.mood_file.exists():
            try:
                with open(self.mood_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self):
        with open(self.mood_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def detect_mood(self, text: str) -> dict | None:
        """Detect mood from student's message."""
        lower = text.lower()
        for keyword, mood_data in self.MOOD_KEYWORDS.items():
            if keyword in lower:
                return mood_data
        return None

    def log_mood(self, student_name: str, text: str) -> dict | None:
        """Log a mood entry if mood is detected in the text."""
        mood = self.detect_mood(text)
        if mood:
            entry = {
                "student": student_name or "unknown",
                "timestamp": datetime.now().isoformat(),
                "mood": mood["label"],
                "score": mood["score"],
                "message_snippet": text[:100],
            }
            self.history.append(entry)
            self._save()
            print(f"  💚 Mood logged: {mood['label']} (score: {mood['score']})")
            return entry
        return None

    def get_pattern(self, student_name: str = None, last_n: int = 5) -> str:
        """Analyze mood patterns for context."""
        entries = self.history
        if student_name:
            entries = [e for e in entries if e.get("student", "").lower() == student_name.lower()]

        if not entries:
            return ""

        recent = entries[-last_n:]
        scores = [e["score"] for e in recent]
        moods = [e["mood"] for e in recent]
        avg_score = sum(scores) / len(scores)

        # Build pattern analysis
        analysis_parts = []

        if avg_score <= 1.5:
            analysis_parts.append(
                f"Student has been consistently struggling emotionally. "
                f"Recent moods: {', '.join(moods)}. Average wellbeing: {avg_score:.1f}/5. "
                f"They may need extra support or a referral to professional counseling."
            )
        elif avg_score <= 2.5:
            analysis_parts.append(
                f"Student seems to be going through a rough patch. "
                f"Recent moods: {', '.join(moods)}. Be extra gentle and supportive."
            )
        elif len(scores) >= 2 and scores[-1] > scores[-2]:
            analysis_parts.append(
                f"Student's mood is improving — they were '{moods[-2]}' before and now '{moods[-1]}'. "
                f"Acknowledge the positive shift."
            )
        elif len(scores) >= 2 and scores[-1] < scores[-2]:
            analysis_parts.append(
                f"Student's mood dropped — was '{moods[-2]}' but now '{moods[-1]}'. "
                f"Something may have happened. Check in gently."
            )

        return " ".join(analysis_parts) if analysis_parts else ""


# ═══════════════════════════════════════════════════
# 4. CAMPUS KNOWLEDGE BASE (RAG)
# ═══════════════════════════════════════════════════

class CampusKnowledgeBase:
    """
    RAG over college documents (PDFs, TXT files).
    
    Setup:
      1. Put college PDFs in ./campus_data/
      2. On first run, docs are chunked and stored in ChromaDB
      3. Student questions are matched against stored chunks
    
    Supports: PDF, TXT, Markdown files
    """

    def __init__(self):
        self.collection = None
        self.client = None
        self.initialized = False

    def initialize(self) -> bool:
        """Load or build the campus vector database."""
        try:
            import chromadb
        except ImportError:
            print("  ⚠ chromadb not installed — campus KB disabled")
            print("    pip install chromadb")
            return False

        docs_dir = Config.CAMPUS_DOCS_DIR
        db_dir = Config.CAMPUS_DB_DIR

        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created {docs_dir}/ — place your college PDFs here")
            return False

        doc_files = list(docs_dir.glob("*.pdf")) + \
                    list(docs_dir.glob("*.txt")) + \
                    list(docs_dir.glob("*.md"))

        if not doc_files:
            print(f"  📁 No documents found in {docs_dir}/ — campus KB disabled")
            print(f"     Add PDFs, TXT, or MD files and restart")
            return False

        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(db_dir))
            self.collection = self.client.get_or_create_collection(
                name="campus_docs",
                metadata={"hnsw:space": "cosine"},
            )

            # Check if we need to (re)index
            existing_count = self.collection.count()
            if existing_count == 0:
                print(f"  📄 Indexing {len(doc_files)} campus documents...")
                self._index_documents(doc_files)
                print(f"  ✅ Campus KB ready: {self.collection.count()} chunks indexed")
            else:
                print(f"  ✅ Campus KB loaded: {existing_count} chunks from {len(doc_files)} docs")

            self.initialized = True
            return True

        except Exception as e:
            print(f"  ⚠ Campus KB error: {e}")
            return False

    def _read_file(self, filepath: Path) -> str:
        """Read content from a document file."""
        ext = filepath.suffix.lower()

        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(filepath))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                # Fallback: try pdfplumber
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(str(filepath)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    return text
                except ImportError:
                    print(f"  ⚠ Install PyMuPDF or pdfplumber to read PDFs: pip install PyMuPDF")
                    return ""

        elif ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        return ""

    def _chunk_text(self, text: str, source: str) -> list[dict]:
        """Split text into overlapping chunks."""
        chunks = []
        size = Config.CAMPUS_CHUNK_SIZE
        overlap = Config.CAMPUS_CHUNK_OVERLAP

        # Clean text
        text = " ".join(text.split())

        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size]
            if len(chunk.strip()) > 50:  # Skip tiny chunks
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "chunk_index": len(chunks),
                })
        return chunks

    def _index_documents(self, doc_files: list[Path]):
        """Read, chunk, and index all documents."""
        all_chunks = []
        for filepath in doc_files:
            print(f"    Reading: {filepath.name}")
            content = self._read_file(filepath)
            if content:
                chunks = self._chunk_text(content, filepath.name)
                all_chunks.extend(chunks)

        if all_chunks:
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                documents=[c["text"] for c in all_chunks],
                metadatas=[{"source": c["source"], "index": c["chunk_index"]} for c in all_chunks],
            )

    def query(self, question: str) -> str:
        """Search campus documents for relevant information."""
        if not self.initialized or not self.collection:
            return ""

        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=Config.CAMPUS_TOP_K,
            )

            if not results["documents"] or not results["documents"][0]:
                return ""

            # Build context from retrieved chunks
            context_parts = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                source = meta.get("source", "unknown")
                context_parts.append(f"[From {source}]: {doc}")

            print(f"  🏫 Campus KB: found {len(context_parts)} relevant chunks")
            return "\n".join(context_parts)

        except Exception as e:
            print(f"  ⚠ Campus KB query error: {e}")
            return ""

    def reindex(self):
        """Force reindex all documents (call after adding new docs)."""
        if self.client and self.collection:
            # Delete existing collection
            self.client.delete_collection("campus_docs")
            self.collection = self.client.get_or_create_collection(
                name="campus_docs",
                metadata={"hnsw:space": "cosine"},
            )
            doc_files = list(Config.CAMPUS_DOCS_DIR.glob("*.pdf")) + \
                        list(Config.CAMPUS_DOCS_DIR.glob("*.txt")) + \
                        list(Config.CAMPUS_DOCS_DIR.glob("*.md"))
            if doc_files:
                self._index_documents(doc_files)
                print(f"  ✅ Reindexed: {self.collection.count()} chunks")


# ═══════════════════════════════════════════════════
# 5. JOB SEARCH
# ═══════════════════════════════════════════════════

def job_search(query: str, location: str = "India") -> str:
    """
    Search for internships and jobs using DuckDuckGo.
    Targets: LinkedIn, Indeed, Internshala, Naukri, Glassdoor
    """
    try:
        search_query = f"{query} internship OR job {location} site:linkedin.com OR site:internshala.com OR site:indeed.com OR site:naukri.com"
        print(f"  💼 Job search: \"{query}\" in {location}")
        results = list(DDGS().text(search_query, max_results=Config.JOB_SEARCH_MAX_RESULTS))

        if not results:
            # Fallback to broader search
            results = list(DDGS().text(f"{query} jobs internships {location}", max_results=Config.JOB_SEARCH_MAX_RESULTS))

        if not results:
            return "No job listings found for that search. Try being more specific about the role or field."

        listings = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("href", "")
            listings.append(f"{title}: {body}")

        print(f"  ✅ Found {len(listings)} job listings")
        return "\n".join(listings)

    except Exception as e:
        print(f"  ⚠ Job search failed: {e}")
        return ""


# ═══════════════════════════════════════════════════
# 6. COURSE RECOMMENDER
# ═══════════════════════════════════════════════════

def course_search(query: str) -> str:
    """
    Search for online courses and tutorials.
    Targets: Coursera, Udemy, NPTEL, edX, freeCodeCamp, YouTube
    """
    try:
        search_query = f"{query} course tutorial site:coursera.org OR site:udemy.com OR site:nptel.ac.in OR site:edx.org OR site:youtube.com"
        print(f"  📚 Course search: \"{query}\"")
        results = list(DDGS().text(search_query, max_results=Config.COURSE_SEARCH_MAX_RESULTS))

        if not results:
            # Fallback
            results = list(DDGS().text(f"best {query} course for beginners", max_results=Config.COURSE_SEARCH_MAX_RESULTS))

        if not results:
            return "Couldn't find specific courses for that. Try a broader topic name."

        courses = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            courses.append(f"{title}: {body}")

        print(f"  ✅ Found {len(courses)} courses")
        return "\n".join(courses)

    except Exception as e:
        print(f"  ⚠ Course search failed: {e}")
        return ""