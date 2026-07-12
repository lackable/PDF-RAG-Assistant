import logging
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

CLASSIFIER_PROMPT = """<task>
You are a routing agent. Your only job is to classify the user's latest message
into exactly one of three states, based on the conversation history provided.
</task>

<states>
  <state name="greeting">
    The message is a greeting, farewell, thank you, social pleasantry, 
    small talk, introduction, or a question about who you are / what you can do.
    Examples: "hello", "hi there", "thanks", "bye", "who are you?", 
              "what can you help me with?", "great answer!"
    Rule: No document retrieval is needed to respond.
  </state>

  <state name="query">
    The message is a self-contained factual question about the documents 
    that does not depend on any prior conversation turn to be understood.
    Examples: "What is the total revenue for 2023?", "Who are the board members?",
              "What does the risk section say?"
    Rule: The question is fully understandable without reading the chat history.
  </state>

  <state name="follow_up">
    The message refers to, builds upon, or asks for clarification/elaboration 
    on a previous question or answer. It uses pronouns, demonstratives, or 
    references that only make sense in context.
    Examples: "What about the previous year?", "Tell me more about that.",
              "And the CEO?", "How does this compare?", "Can you elaborate?"
    Rule: The question cannot be fully understood without reading prior messages.
  </state>
</states>

<conversation_history>
{last_n_messages_formatted}
</conversation_history>

<user_message>
{current_input}
</user_message>

<instruction>
Output ONLY the state name — one of: greeting, query, follow_up
Do not include any explanation, punctuation, or extra text.
</instruction>"""

async def classify_state(user_input: str, recent_history: list[dict]) -> str:
    """
    Returns one of: "greeting", "query", "follow_up".
    Falls back to "query" on any LLM/parse error.
    Uses a non-streaming Gemini call with temperature=0.
    """
    history_formatted = ""
    for msg in recent_history:
        history_formatted += f"User: {msg.get('input', '')}\nAssistant: {msg.get('output', '')}\n---\n"
    if not history_formatted.strip():
        history_formatted = "No prior history."

    prompt = CLASSIFIER_PROMPT.format(
        last_n_messages_formatted=history_formatted,
        current_input=user_input
    )

    try:
        # non-streaming, deterministic
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.0,
            streaming=False
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        state = response.content.strip().lower()
        
        if state in ["greeting", "query", "follow_up"]:
            return state
        else:
            logger.warning(f"Classifier returned invalid state '{state}', defaulting to 'query'.")
            return "query"
    except Exception as e:
        logger.error(f"Error classifying state: {e}")
        return "query"
