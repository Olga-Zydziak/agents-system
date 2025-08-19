import os
import logging
from enum import Enum
from google.cloud import secretmanager
import langchain
from langchain.cache import SQLiteCache





def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Pobiera wartość sekretu z Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
   
    return response.payload.data.decode("UTF-8")


class ApiType(Enum):
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    def __str__(self):
        return self.value


LOCATION="us-central1"
PROJECT_ID="dark-data-discovery"

#---------AGENTS--------:
MAIN_AGENT="gemini-2.5-pro"
API_TYPE_GEMINI=str(ApiType.GOOGLE)

CRITIC_MODEL="claude-3-7-sonnet-20250219"
ARCHITECT_MODEL ="claude-opus-4-1-20250805"
CODE_MODEL="claude-sonnet-4-20250514"
QUICK_SMART_MODEL="gemini-2.5-flash"

GPT_MODEL = "gpt-4o" # Używamy gpt-4o jako odpowiednika "gpt-5"
API_TYPE_OPENAI = str(ApiType.OPENAI)

API_TYPE_SONNET = str(ApiType.ANTHROPIC)

LANGCHAIN_API_KEY = get_secret(PROJECT_ID,"LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY=get_secret(PROJECT_ID,"ANTHROPIC_API_KEY")
TAVILY_API_KEY = get_secret(PROJECT_ID,"TAVILY_API_KEY")
OPENAI_API_KEY = get_secret(PROJECT_ID, "OPENAI_API_KEY")

MEMORY_ENGINE_DISPLAY_NAME="memory-gamma-way"

INPUT_FILE_PATH = "gs://super_model/data/structural_data/synthetic_fraud_dataset.csv"

MAX_CORRECTION_ATTEMPTS=5



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Projekt Multi-Agent-System Dynamic-graphs"
os.environ["ANTHROPIC_API_KEY"] =ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#---cache-------
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")




#FUNKCJA KONFIGURACYJNA AGENTOW AUTOGEN
def basic_config_agent(agent_name:str, api_type:str, location:str=None, project_id:str=None, api_key:str=None):
    try:
        configuration = {"model": agent_name}
        configuration.update({"api_type": api_type})
        if api_key: configuration["api_key"] = api_key
        if project_id: configuration["project_id"] = project_id
        if location: configuration["location"] = location

        logging.info(f"Model configuration: {configuration}")
        return [configuration]

    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI or configure LLM: {e}")
        print(f"Error: Failed to initialize Vertex AI or configure LLM. Please check your project ID, region, and permissions. Details: {e}")
        exit()
