"""
Vidur AI — LangChain Agent with Full Tool Suite

Tools available:
  1. web_search        — General knowledge lookup
  2. calculator        — GPA, CGPA, EMI, math
  3. mood_tracker      — Emotional pattern tracking
  4. campus_kb         — RAG over college documents
  5. job_search        — Internships & job listings
  6. course_recommender— Online courses & tutorials

Routing: Intent detection decides which tool(s) to use per message.
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import Config, SYSTEM_PROMPT
from tools import (
    web_search, calculator, job_search, course_search,
    MoodTracker, CampusKnowledgeBase,
)
# from learning import LearningKnowledgeBase


# ═════════════════════════════════════════════
# INTENT DETECTION & TOOL ROUTING
# ═════════════════════════════════════════════

# Signals → which tool to use (checked in order, first match wins)
TOOL_ROUTES = [
    # ── Calculator ──
    {
        "tool": "calculator",
        "signals": [
            "gpa", "cgpa", "grade point", "calculate", "emi", "loan",
            "percentage to gpa", "convert percentage", "credit hour",
            "how much", "total cost", "interest rate",
        ],
    },
    # ── Job Search ──
    {
        "tool": "job_search",
        "signals": [
            "internship", "internships", "job search", "job opening",
            "hiring", "vacancy", "vacancies", "placement",
            "work opportunity", "job for", "jobs for", "jobs in",
            "career opportunity", "part time job", "full time job",
            "fresher job", "entry level",
        ],
    },
    # ── Course Recommender ──
    {
        "tool": "course_search",
        "signals": [
            "course", "courses", "tutorial", "tutorials",
            "where to learn", "how to learn", "want to learn",
            "online class", "certification", "certificate",
            "udemy", "coursera", "nptel", "edx", "freecodecamp",
            "youtube tutorial", "best course", "recommend a course",
            "learn python", "learn java", "learn coding",
            "study material", "learning resource",
        ],
    },
    # ── Campus KB ──
    {
        "tool": "campus_kb",
        "signals": [
            "college", "university", "campus", "semester",
            "syllabus", "timetable", "schedule", "exam date",
            "fee structure", "hostel", "library", "department",
            "professor", "faculty", "admission", "enrollment",
            "attendance", "backlog", "revaluation", "grade card",
            "convocation", "placement cell", "student council",
            "registration", "drop course", "add course",
            "academic calendar", "holiday", "policy",
        ],
    },
    # ── Web Search (general fallback for knowledge questions) ──
    {
        "tool": "web_search",
        "signals": [
            "what is", "what are", "what does", "what's",
            "explain", "tell me about", "how does", "how do", "how to",
            "meaning of", "define", "difference between",
            "compare", "vs", "versus", "latest", "new", "recent",
            "architecture", "framework", "library",
            "salary", "scope of", "future of",
            "scholarship", "fafsa", "financial aid",
            "roadmap", "prerequisite", "introduction to",
        ],
    },
]

# Signals that mean "just chat, don't use any tool"
CHAT_ONLY_SIGNALS = [
    "my name is", "i am", "i'm", "hi", "hello", "hey",
    "how are you", "thank", "thanks", "bye", "goodbye",
    "yes", "no", "yeah", "nah", "okay", "ok", "sure",
    "good morning", "good evening", "good night",
    "that's helpful", "got it", "makes sense", "i see",
]

# Signals that indicate emotional content → mood tracker should run
MOOD_SIGNALS = [
    "i feel", "i'm feeling", "feeling", "i am feeling",
    "stressed", "anxious", "worried", "sad", "depressed",
    "lonely", "overwhelmed", "scared", "confused", "lost",
    "angry", "frustrated", "tired", "burned out", "burnout",
    "happy", "excited", "great", "good", "hopeless",
    "not great", "not good", "not okay", "struggling",
    "can't sleep", "can't focus", "don't want to",
    "panic", "breakdown", "crying",
]


# ─────────────────────────────────────────────────────────────
# MALAYALAM SIGNALS  (Unicode script — used when STT returns ml-IN text)
# ─────────────────────────────────────────────────────────────

_MALAYALAM_RE = re.compile(r'[\u0D00-\u0D7F]')

MALAYALAM_TOOL_ROUTES = [
    # ── Calculator ──
    {
        "tool": "calculator",
        "signals": [
            "ജിപിഎ", "സിജിപിഎ", "ഗ്രേഡ്", "ഗ്രേഡ് പോയിന്റ്",
            "ശതമാനം", "കണക്ക്", "കണക്കുകൂട്ടൽ",
            "ലോൺ", "ഇഎംഐ", "മൊത്തം ചെലവ്", "പലിശ",
            "ക്രെഡിറ്റ്", "മൊത്തം",
        ],
    },
    # ── Job Search ──
    {
        "tool": "job_search",
        "signals": [
            "ജോലി", "ഇന്റേൺഷിപ്പ്", "നിയമനം", "കരിയർ",
            "പ്ലേസ്‌മെന്റ്", "ഒഴിവ്", "ജോലി ഒഴിവ്",
            "ഫ്രഷർ ജോബ്", "പാർട്ട് ടൈം", "ഫുൾ ടൈം",
        ],
    },
    # ── Course Recommender ──
    {
        "tool": "course_search",
        "signals": [
            "കോഴ്‌സ്", "ട്യൂട്ടോറിയൽ", "സർട്ടിഫിക്കറ്റ്",
            "ഓൺലൈൻ ക്ലാസ്", "ഓൺലൈൻ പഠനം", "ഓൺലൈൻ കോഴ്‌സ്",
            "എൻപിടിഇഎൽ", "കോഴ്‌സേര", "യൂഡെമി", "എഡ്‌എക്‌സ്",
            "പഠിക്കാൻ", "പ്രോഗ്രാമിംഗ് പഠനം",
        ],
    },
    # ── Campus Knowledge Base ──
    {
        "tool": "campus_kb",
        "signals": [
            "കോളേജ്", "യൂണിവേഴ്‌സിറ്റി", "ക്യാംപസ്",
            "ഫീസ്", "ഹോസ്റ്റൽ", "ലൈബ്രറി",
            "സിലബസ്", "പരീക്ഷ", "ടൈംടേബിൾ",
            "ഡിപ്പാർട്ട്‌മെന്റ്", "ഫാക്കൽറ്റി", "പ്രൊഫസർ",
            "അഡ്മിഷൻ", "ഹാജർ", "ഹോളിഡേ",
            "അക്കാദമിക്", "ബ്ലോക്ക്ലിസ്റ്റ്", "ഗ്രേഡ് കാർഡ്",
        ],
    },
    # ── Web Search (general fallback) ──
    {
        "tool": "web_search",
        "signals": [
            "എന്താണ്", "എന്ത്", "എങ്ങനെ", "വിശദീകരിക്കൂ",
            "അർഥം", "വ്യത്യാസം", "ശമ്പളം", "ഭാവി",
            "സ്കോളർഷിപ്പ്", "ഫിനാൻഷ്യൽ എയ്ഡ്",
            "റോഡ്‌മാപ്പ്", "ആർക്കിടെക്ചർ",
        ],
    },
]

MALAYALAM_MOOD_SIGNALS = [
    "ദുഃഖം", "ദുഖം", "സങ്കടം", "ദേഷ്യം", "ഭയം",
    "ആകുലത", "ഉദ്വേഗം", "ക്ഷീണം", "ക്ഷീണിച്ചു",
    "സ്ട്രെസ്", "സ്ട്രെസ്സ്", "ഒറ്റപ്പെടൽ", "ഒറ്റപ്പെട്ടു",
    "ആകുലപ്പെടുന്നു", "വിഷമം", "നിരാശ", "നിരാശയ്ക്ക്",
    "അസ്വസ്ഥത", "ഉറക്കമില്ല", "ഉറങ്ങാൻ",
    "കരയുന്നു", "കരഞ്ഞു", "ശ്വാസം മുട്ടൽ",
    "ഭ്രാന്ത്", "ഭ്രമം", "ഭ്രമിക്കുന്നു",
    "സന്തോഷം", "ഉത്സാഹം", "കൊള്ളാം", "അടിപൊളി",
    "ശരിയാകില്ല", "മടുത്തു", "ഇനി വേണ്ട",
]

MALAYALAM_CHAT_ONLY_SIGNALS = [
    "ഹലോ", "ഹായ്", "നമസ്കാരം", "ഹേ",
    "ഒക്കേ", "ശരി", "ആണ്", "ഇല്ല", "ഉം", "ഓ",
    "നന്ദി", "ബൈ", "ശുഭ", "ഗുഡ് മോണിംഗ്",
    "ഹ്ഹ", "അതെ", "ഇല്ലേ",
]


def detect_intent(text: str) -> dict:
    """
    Analyze student message and return:
      {
        "tool": "web_search" | "calculator" | "job_search" | "course_search" | "campus_kb" | None,
        "track_mood": True/False,
        "is_chat_only": True/False,
        "search_query": "cleaned query for the tool"
      }
    Supports both English and Malayalam (Unicode) input.
    """
    lower = text.lower().strip()
    result = {"tool": None, "track_mood": False, "is_chat_only": False, "search_query": text}

    # ── Malayalam path: when STT returns ml-IN Unicode text ──
    if _MALAYALAM_RE.search(text):
        # Mood check (parallel — doesn't block tool routing)
        for signal in MALAYALAM_MOOD_SIGNALS:
            if signal in text:
                result["track_mood"] = True
                break

        # Pure-chat check
        for signal in MALAYALAM_CHAT_ONLY_SIGNALS:
            if signal in text:
                result["is_chat_only"] = True
                return result

        # Very short Malayalam messages are conversational
        if len(text.split()) < 3:
            result["is_chat_only"] = True
            return result

        # Tool routing using Malayalam signals
        for route in MALAYALAM_TOOL_ROUTES:
            for signal in route["signals"]:
                if signal in text:
                    result["tool"] = route["tool"]
                    result["search_query"] = text   # pass raw Malayalam to tool
                    return result

        return result

    # ── English path (original logic) ──

    # Check for mood signals (runs alongside other tools)
    for signal in MOOD_SIGNALS:
        if signal in lower:
            result["track_mood"] = True
            break

    # Check if it's pure chat
    for signal in CHAT_ONLY_SIGNALS:
        if lower.startswith(signal) or lower == signal:
            result["is_chat_only"] = True
            return result

    # Very short messages are usually conversational
    if len(lower.split()) < 3:
        result["is_chat_only"] = True
        return result

    # Route to the right tool
    for route in TOOL_ROUTES:
        for signal in route["signals"]:
            if signal in lower:
                result["tool"] = route["tool"]

                # Clean the query for the tool
                query = lower
                for filler in ["can you", "could you", "please", "i want to",
                               "i would like to", "tell me", "i need to",
                               "help me", "i want", "i'd like to",
                               "find me", "search for", "look up",
                               "show me", "give me"]:
                    query = query.replace(filler, "")
                result["search_query"] = query.strip() or text

                return result

    return result


# ═════════════════════════════════════════════
# COUNSELOR AGENT
# ═════════════════════════════════════════════

class CounselorAgent:
    """
    Full-featured counseling agent with:
      - ChatOllama (LLM)
      - Smart intent routing to 6 tools
      - Mood tracking across sessions
      - Campus RAG knowledge base
      - Self-learning from past sessions
    """

    def __init__(self):
        self.llm = None
        self.history: list = []
        self.max_history = Config.MEMORY_WINDOW * 2

        # Tools
        self.mood_tracker = MoodTracker()
        self.campus_kb = CampusKnowledgeBase()
        # self.knowledge = LearningKnowledgeBase()

        # Track student name for mood logging
        self.student_name = ""

    def initialize(self) -> bool:
        """Build LLM and initialize all tools."""
        try:
            print(f"  🔗 Building LangChain agent...")
            print(f"     LLM   : {Config.OLLAMA_MODEL} via Ollama")
            print(f"     Memory: Last {Config.MEMORY_WINDOW} exchanges")

            self.llm = ChatOllama(
                model=Config.OLLAMA_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=Config.TEMPERATURE,
                num_predict=Config.MAX_TOKENS,
            )

            # Warm up LLM
            print(f"  🔥 Warming up LLM...")
            test = self.llm.invoke([HumanMessage(content="hi")])
            if test is None:
                raise ConnectionError("No response from Ollama")
            print(f"  ✅ LLM ready!")

            # Initialize tools
            print(f"\n  🛠  Loading tools...")

            # Campus KB
            print(f"  [1/3] Campus Knowledge Base...")
            if self.campus_kb.initialize():
                print(f"     ✅ Campus KB active")
            else:
                print(f"     ⚠ Campus KB inactive (add PDFs to {Config.CAMPUS_DOCS_DIR}/)")

            # Mood Tracker
            print(f"  [2/3] Mood Tracker...")
            mood_count = len(self.mood_tracker.history)
            print(f"     ✅ Mood tracker active ({mood_count} past entries)")

            # Self-Learning KB
            print(f"  [3/3] Self-Learning Knowledge Base...")
            if Config.LEARNING_ENABLED:
                if self.knowledge.initialize():
                    stats = self.knowledge.get_stats()
                    print(f"     ✅ Learning KB: {stats['qa_pairs']} Q&As, {stats['insights']} insights")
                else:
                    print(f"     ⚠ Learning KB unavailable")

            tools_list = ["web_search", "calculator", "job_search",
                          "course_search", "mood_tracker"]
            if self.campus_kb.initialized:
                tools_list.append("campus_kb")
            print(f"\n     Active tools: {tools_list}")

            return True

        except Exception as e:
            print(f"\n  ❌ Agent build failed: {e}")
            print(f"  💡 1. Is Ollama running?  →  ollama serve")
            print(f"     2. Is the model pulled? →  ollama pull {Config.OLLAMA_MODEL}")
            return False

    def _extract_name(self, text: str):
        """Try to extract student name from message."""
        lower = text.lower()
        for prefix in ["my name is ", "i'm ", "i am ", "this is ", "call me "]:
            if prefix in lower:
                idx = lower.index(prefix) + len(prefix)
                name = text[idx:].split()[0].strip(".,!?")
                if len(name) > 1:
                    self.student_name = name.capitalize()
                    print(f"  👤 Student name: {self.student_name}")
                    break

    def chat(self, user_input: str, lang_instruction: str = "") -> str:
        """
        Process student message through the full pipeline:
          1. Extract name if present
          2. Detect intent → route to tool
          3. Track mood if emotional
          4. Recall past learning
          5. Build context → send to LLM
          6. Return response
        """

        # ── Step 1: Name extraction ──
        self._extract_name(user_input)

        # ── Step 2: Intent detection ──
        intent = detect_intent(user_input)

        # ── Step 3: Mood tracking (runs in parallel with other tools) ──
        mood_context = ""
        if intent["track_mood"]:
            self.mood_tracker.log_mood(self.student_name, user_input)
            pattern = self.mood_tracker.get_pattern(self.student_name)
            if pattern:
                mood_context = pattern

        # ── Step 4: Execute the routed tool ──
        tool_context = ""
        tool_name = intent["tool"]
        query = intent["search_query"]

        if tool_name == "web_search":
            tool_context = web_search(query)
        elif tool_name == "calculator":
            tool_context = calculator(query)
        elif tool_name == "job_search":
            tool_context = job_search(query)
        elif tool_name == "course_search":
            tool_context = course_search(query)
        elif tool_name == "campus_kb":
            tool_context = self.campus_kb.query(query)
            if not tool_context:
                # Fallback to web search if campus KB has no answer
                tool_context = web_search(query)

        # ── Step 5: Recall past learning ──
        past_knowledge = ""
        if Config.LEARNING_ENABLED and self.knowledge.qa_collection:
            past_knowledge = self.knowledge.recall(user_input)

        # ── Step 6: Build augmented message ──
        context_blocks = []

        # Language instruction always first
        if lang_instruction:
            context_blocks.append(lang_instruction)

        if mood_context:
            context_blocks.append(f"[MOOD: {mood_context}]")

        if past_knowledge:
            context_blocks.append(
                f"[PAST KNOWLEDGE — from helping previous students:\n{past_knowledge}]"
            )

        if tool_context:
            label = {
                "web_search": "WEB",
                "calculator": "CALCULATION",
                "job_search": "JOB LISTINGS",
                "course_search": "COURSES",
                "campus_kb": "CAMPUS INFO",
            }.get(tool_name, "TOOL")
            context_blocks.append(
                f"[{label} — use the most useful piece to answer. "
                f"NEVER mention you used a tool:\n{tool_context}]"
            )

        if context_blocks:
            augmented = user_input + "\n\n" + "\n\n".join(context_blocks)
        else:
            augmented = user_input

        # ── Step 7: Build conversation and get LLM response ──
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=augmented))

        ai_response = self.llm.invoke(messages)
        response_text = ai_response.content.strip() if ai_response.content else ""

        if not response_text:
            response_text = "Could you say that differently? I want to make sure I help you right."

        # ── Step 8: Update history (clean input only) ──
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response_text))

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return response_text

    def learn_from_session(self) -> int:
        """Extract knowledge from the conversation at session end."""
        if not Config.LEARNING_ENABLED:
            return 0
        if len(self.history) < Config.LEARNING_MIN_TURNS:
            return 0
        return self.knowledge.learn_from_conversation(self.history, self.llm)

    def get_knowledge_stats(self) -> dict:
        if Config.LEARNING_ENABLED and self.knowledge.qa_collection:
            return self.knowledge.get_stats()
        return {"qa_pairs": 0, "insights": 0, "recalls_this_session": 0}

    def reset(self):
        """Clear conversation memory (keeps knowledge + mood history)."""
        self.history = []
        self.student_name = ""