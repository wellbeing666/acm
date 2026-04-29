from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only before dependencies are installed.
    def load_dotenv() -> bool:
        return False


load_dotenv()

APP_NAME = "ACM RAG"
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_CASES_DIR = DATA_DIR / "generated"
GENERATED_GENERATOR_PATH = GENERATED_CASES_DIR / "generator.py"
TEST_DATA_DIR = BASE_DIR / "test_data"
AGENT_TRACE_DIR = DATA_DIR / "agent_traces"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CHROMA_DB_DIR = DATA_DIR / "chroma"
PROMPTS_DIR = BASE_DIR / "prompts"

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "acm_test_strategy")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled")
DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
DEEPSEEK_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))
TEST_SPEC_MAX_OUTPUT_TOKENS = int(os.getenv("TEST_SPEC_MAX_OUTPUT_TOKENS", "5000"))
GENERATOR_CODE_MAX_OUTPUT_TOKENS = int(os.getenv("GENERATOR_CODE_MAX_OUTPUT_TOKENS", "8000"))
GENERATOR_RUN_TIMEOUT_SECONDS = int(os.getenv("GENERATOR_RUN_TIMEOUT_SECONDS", "20"))
GENERATOR_REPAIR_MAX_ATTEMPTS = int(os.getenv("GENERATOR_REPAIR_MAX_ATTEMPTS", "3"))
