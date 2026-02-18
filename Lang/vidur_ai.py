"""
╔══════════════════════════════════════════════════════════════════╗
║                        VIDUR AI                                  ║
║          Agentic AI-Powered Student Counseling System             ║
║                                                                  ║
║  Architecture: LangChain Agent + Bilingual (English / Malayalam)     ║
║  LLM  : Llama 3.3 via Ollama                                    ║
║  TTS  : Edge TTS (en-IN-NeerjaNeural / ml-IN-SobhanaNeural)     ║
║  STT  : Google Speech Recognition (en-IN / ml-IN)               ║
║                                                                  ║
║  Say "switch to Malayalam" / "മലയാളത്തിൽ பேசு" to switch languages       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import sys
import asyncio
import tempfile
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


# ── Dependency check ──
def check_deps():
    missing = []
    for pkg, imp in [
        ("langchain-ollama", "langchain_ollama"),
        ("langchain-core", "langchain_core"),
        ("chromadb", "chromadb"),
        ("PyMuPDF", "fitz"),
        ("speech_recognition", "speech_recognition"),
        ("edge_tts", "edge_tts"),
        ("pygame", "pygame"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    # ddgs / duckduckgo_search — either works
    try:
        __import__("duckduckgo_search")
    except ImportError:
        try:
            __import__("ddgs")
        except ImportError:
            missing.append("ddgs")

    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import speech_recognition as sr
import edge_tts
import pygame

from config import Config
from agent import CounselorAgent
import speech_corrector


# ═════════════════════════════════════════════
# LANGUAGE MANAGER (English ↔ Malayalam)
# ═════════════════════════════════════════════
class LanguageManager:
    """
    Auto-detects language from STT results and manages TTS voice + LLM instruction.
    No manual switching needed — just speak in whatever language you want.
    """

    MALAYALAM_RANGE = re.compile(r'[\u0D00-\u0D7F]')

    def __init__(self):
        self.current = Config.DEFAULT_LANGUAGE

    @property
    def lang_config(self) -> dict:
        return Config.LANGUAGES[self.current]

    @property
    def name(self) -> str:
        return self.lang_config["name"]

    @property
    def tts_voice(self) -> str:
        return self.lang_config["tts_voice"]

    @property
    def is_malayalam(self) -> bool:
        return self.current == "ml"

    @property
    def is_english(self) -> bool:
        return self.current == "en"

    @staticmethod
    def has_malayalam(text: str) -> bool:
        """Check if text contains Malayalam Unicode characters."""
        return bool(LanguageManager.MALAYALAM_RANGE.search(text))

    def set_language(self, lang: str):
        """Update active language from auto-detection."""
        if lang != self.current:
            self.current = lang
            print(f"  🌐 Language: {self.name}")

    def get_llm_instruction(self) -> str:
        """Get language instruction to inject into LLM context."""
        if self.current == "ml":
            return (
                "[LANGUAGE: The student is speaking in Malayalam. "
                "Respond ENTIRELY in Malayalam using Malayalam Unicode script (മലയാളം). "
                "Do NOT use transliteration or English. "
                "Keep it to 1-2 sentences. Be warm and natural.]"
            )
        return ""


# ═════════════════════════════════════════════
# CONVERSATION LOGGER
# ═════════════════════════════════════════════
class Logger:
    def __init__(self):
        if Config.ENABLE_LOGGING:
            Config.LOG_DIR.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = Config.LOG_DIR / f"session_{ts}.txt"
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(f"=== Vidur AI Session — {datetime.now():%Y-%m-%d %H:%M} ===\n\n")
        else:
            self.path = None

    def log(self, role: str, text: str):
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] {role}: {text}\n\n")

    def close(self):
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"\n--- Session ended {datetime.now():%H:%M:%S} ---\n")


# ═════════════════════════════════════════════
# TEXT-TO-SPEECH (Edge TTS — Dynamic Voice)
# ═════════════════════════════════════════════
class TTS:
    def __init__(self):
        pygame.mixer.init()
        self.tmp = tempfile.mkdtemp()
        self.n = 0

    async def _generate(self, text: str, path: str, voice: str):
        comm = edge_tts.Communicate(
            text, voice=voice,
            rate=Config.TTS_RATE, volume=Config.TTS_VOLUME,
        )
        await comm.save(path)

    def speak(self, text: str, voice: str = None):
        """Speak text using the given voice (or default English)."""
        voice = voice or Config.TTS_VOICE
        self.n += 1
        path = os.path.join(self.tmp, f"r_{self.n}.mp3")
        asyncio.run(self._generate(text, path, voice))
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"  ⚠ Playback error: {e}")
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(path)
            except OSError:
                pass

    def cleanup(self):
        pygame.mixer.quit()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# SPEECH-TO-TEXT (Google STT — Dynamic Language)
# ═════════════════════════════════════════════
class STT:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = Config.STT_ENERGY_THRESHOLD
        self.rec.pause_threshold = Config.STT_PAUSE_THRESHOLD
        self.rec.dynamic_energy_threshold = True
        self.mic = None

    def init_mic(self) -> bool:
        try:
            self.mic = sr.Microphone()
            with self.mic as src:
                print("  🎤 Calibrating microphone...")
                self.rec.adjust_for_ambient_noise(src, duration=2)
                print("  ✅ Microphone ready!")
            return True
        except Exception as e:
            print(f"  ❌ Mic error: {e}")
            return False

    def listen(self) -> tuple[str, str] | tuple[None, None]:
        """
        Listen and transcribe, auto-detecting language.
        Returns (text, detected_language_code) or (None, None).
        
        Strategy:
          1. Record audio once
          2. Try English FIRST (it's the primary language)
          3. Try Malayalam ONLY if English failed or returned very low confidence
          4. If both succeed, pick Malayalam only when Malayalam script is dominant
        """
        if not self.mic:
            return None, None
        try:
            with self.mic as src:
                print(f"\n  🎤 Listening... (speak in any language)")
                audio = self.rec.listen(src, phrase_time_limit=Config.STT_PHRASE_TIME_LIMIT)
                print("  ⏳ Transcribing...")

            # ── Try English first (primary language) ──
            english_text = None
            try:
                english_text = self.rec.recognize_google(audio, language="en-IN").strip()
            except (sr.UnknownValueError, sr.RequestError):
                pass

            # ── Try Malayalam ──
            malayalam_text = None
            try:
                malayalam_text = self.rec.recognize_google(audio, language="ml-IN").strip()
            except (sr.UnknownValueError, sr.RequestError):
                pass

            # ── Decision logic ──

            # If English worked, check if the student ACTUALLY spoke Malayalam
            if english_text:
                if malayalam_text and self._is_clearly_malayalam(malayalam_text, english_text):
                    # Malayalam script dominant AND English result looks like garbage
                    print(f"  🌐 Detected: Malayalam")
                    return malayalam_text, "ml"
                else:
                    # Default: trust English
                    print(f"  🌐 Detected: English")
                    return english_text, "en"

            # English failed entirely — try Malayalam
            if malayalam_text and LanguageManager.has_malayalam(malayalam_text):
                print(f"  🌐 Detected: Malayalam")
                return malayalam_text, "ml"

            # Both failed
            print("  ⚠ Couldn't catch that. Try again.")
            return None, None

        except sr.WaitTimeoutError:
            return None, None
        except Exception as e:
            print(f"  ❌ STT error: {e}")
            return None, None

    @staticmethod
    def _is_clearly_malayalam(malayalam_text: str, english_text: str) -> bool:
        """
        Determine if the student genuinely spoke Malayalam (not English
        being misinterpreted by the Malayalam recognizer).
        
        Rules:
          - Malayalam text must have a HIGH ratio of Malayalam script characters
          - English text must look like nonsense (garbled)
        """
        if not malayalam_text:
            return False

        # Count Malayalam characters vs total
        malayalam_chars = len(LanguageManager.MALAYALAM_RANGE.findall(malayalam_text))
        total_chars = len(malayalam_text.replace(" ", ""))

        if total_chars == 0:
            return False

        malayalam_ratio = malayalam_chars / total_chars

        # Must be at least 50% Malayalam script to count as Malayalam
        # (English words misrecognized as Malayalam usually have low Malayalam ratio)
        if malayalam_ratio < 0.5:
            return False

        # If English result is clean readable English, trust English
        # Check if English text has mostly ASCII letters
        ascii_chars = sum(1 for c in english_text if c.isascii() and c.isalpha())
        english_alpha_total = sum(1 for c in english_text if c.isalpha())

        if english_alpha_total > 0:
            english_ascii_ratio = ascii_chars / english_alpha_total
            # If English result is >80% clean ASCII, it's probably English
            if english_ascii_ratio > 0.8:
                return False

        return True


# ═════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════
class VidurAI:
    def __init__(self):
        self.tts = TTS()
        self.stt = STT()
        self.agent = CounselorAgent()
        self.logger = Logger()
        self.lang = LanguageManager()
        self.running = False

    def banner(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🎓  VIDUR AI  —  Student Counselor  🎓             ║
║                                                              ║
║     LangChain  •  Llama 3.3  •  Self-Learning  •  RAG       ║
║     Tools: Web, Calculator, Jobs, Courses, Campus KB, Mood   ║
║     🌐 Bilingual: English + Malayalam (auto-detect)              ║
║                                                              ║
║     Speak in English or Malayalam — Vidur adapts automatically   ║
║     Say "bye" to end  •  Ctrl+C quit                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)

    def initialize(self) -> bool:
        self.banner()
        print("━" * 60)
        print("  INITIALIZING")
        print("━" * 60)

        print("\n  [1/2] Microphone...")
        if not self.stt.init_mic():
            print("  ⚠ TEXT-ONLY mode (no mic detected)\n")

        print("\n  [2/2] LangChain Agent + Tools + Knowledge Base...")
        if not self.agent.initialize():
            return False

        print(f"\n  🌐 Languages: English + Malayalam (auto-detected per utterance)")

        print("\n" + "━" * 60)
        print("  ✅ VIDUR AI IS ONLINE")
        print("━" * 60)
        return True

    def say(self, text: str):
        """Print + log + speak in current language voice."""
        print(f"\n  🤖 Vidur: {text}")
        self.logger.log("Vidur", text)
        self.tts.speak(text, voice=self.lang.tts_voice)

    def get_input(self) -> tuple[str, str] | tuple[None, None]:
        """
        Get voice/text input. Returns (text, detected_lang) or (None, None).
        Language is auto-detected — no manual switching needed.
        """
        if self.stt.mic:
            text, lang = self.stt.listen()
            if text:
                print(f"  🗣  Student: {text}")
                return text, lang
            print("  💡 Type instead (or Enter to retry voice):")
        try:
            text = input("  ⌨  You: ").strip()
            if text:
                # Auto-detect language from typed text
                lang = "ml" if LanguageManager.has_malayalam(text) else "en"
                return text, lang
            return None, None
        except EOFError:
            return None, None

    def is_exit(self, text: str) -> bool:
        lower = text.lower().strip()
        exit_words_en = ["goodbye", "bye", "exit", "quit", "stop", "end session"]
        exit_words_ta = ["പോയി വരാം", "ബൈ", "നിർത്തൂ", "കഴിഞ്ഞു"]
        return any(w in lower for w in exit_words_en) or any(w in text for w in exit_words_ta)

    def is_reset(self, text: str) -> bool:
        lower = text.lower().strip()
        return "reset" in lower or "new conversation" in lower or "പുതിയ സംഭാഷണം" in text

    def run(self):
        if not self.initialize():
            print("\n  ❌ Failed to start. Exiting.")
            return

        self.running = True
        self.say("Hey! I'm Vidur, your counseling buddy. What's your name?")
        print("\n" + "─" * 60)

        try:
            while self.running:
                # ── Get input (auto-detects language) ──
                raw, detected_lang = self.get_input()
                if not raw:
                    continue

                # ── Auto-switch language based on what student spoke ──
                if detected_lang:
                    self.lang.set_language(detected_lang)

                # ── Autocorrect (only for English) ──
                if self.lang.is_english:
                    corrected = speech_corrector.auto_correct(raw)
                else:
                    corrected = raw

                self.logger.log("Student", corrected)

                # ── Exit ──
                if self.is_exit(corrected):
                    if self.lang.is_malayalam:
                        self.say("നിങ്ങളോട് സംസാരിച്ചതിൽ വളരെ സന്തോഷം. എപ്പോൾ വേണമെങ്കിലും തിരിച്ചു വരൂ. ബൈ!")
                    else:
                        self.say("I'm really glad we talked. Take care, and come back anytime. Bye!")
                    self.running = False
                    self._learn_and_report()
                    continue

                # ── Reset ──
                if self.is_reset(corrected):
                    self.agent.reset()
                    if self.lang.is_malayalam:
                        self.say("പുതിയ തുടക്കം! നിങ്ങളുടെ മനസ്സിൽ എന്താണ്?")
                    else:
                        self.say("Fresh start! What's on your mind?")
                    print("\n" + "─" * 60)
                    continue

                # ── Agent processes (with language context) ──
                print("\n  💭 Thinking...")
                try:
                    lang_instruction = self.lang.get_llm_instruction()
                    response = self.agent.chat(corrected, lang_instruction=lang_instruction)
                except Exception as e:
                    if self.lang.is_malayalam:
                        response = "ക്ഷമിക്കണം, മനസ്സിലായില്ല. വീണ്ടും പറയാമോ?"
                    else:
                        response = "Sorry, I missed that. Could you say that again?"
                    print(f"  ⚠ Error: {e}")

                self.say(response)
                print("\n" + "─" * 60)

        except KeyboardInterrupt:
            print("\n\n  ⚡ Interrupted.")
            self._learn_and_report()
        finally:
            print("\n  🔄 Shutting down...")
            self.logger.close()
            self.tts.cleanup()
            stats = self.agent.get_knowledge_stats()
            print(f"  🧠 Knowledge base: {stats['qa_pairs']} Q&As, {stats['insights']} insights")
            print(f"  👋 Have a great day!\n")

    def _learn_and_report(self):
        """Extract learnings from the session and report."""
        try:
            count = self.agent.learn_from_session()
            if count > 0:
                print(f"  🧠 Vidur learned {count} new things from this conversation!")
        except Exception as e:
            print(f"  ⚠ Learning error: {e}")


if __name__ == "__main__":
    VidurAI().run()