import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================
# 1. Load environment variables
# ============================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================
# 2. Streamlit Page Configuration
# ============================================

st.set_page_config(
    page_title="AI Research & Weather Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================
# 3. Check API Keys
# ============================================

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing from your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY is missing from your .env file.")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.error("❌ WEATHERSTACK_API_KEY is missing from your .env file.")
    st.stop()


# ============================================
# 4. Create Groq LLM
# ============================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)


# ============================================
# 5. Create Tavily Search Tool
# ============================================

search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)


# ============================================
# 6. Create Weather Tool
# ============================================

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city."""

    api_key = WEATHERSTACK_API_KEY

    if not api_key:
        return "WeatherStack API key is missing."

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city,
        "units": "m"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return "Sorry, I couldn't fetch the weather data."

        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error']['info']}"

        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        humidity = data["current"]["humidity"]

        return (
            f"The current weather in {city} is "
            f"{temperature}°C, {description}, "
            f"with humidity of {humidity}%."
        )

    except Exception as e:

        return f"Weather request failed: {str(e)}"


# ============================================
# 7. Put Tools Together
# ============================================

tools = [
    search_tool,
    get_weather_data
]


# ============================================
# 8. Load ReAct Prompt
# ============================================

prompt = hub.pull("hwchase17/react")


# ============================================
# 9. Create ReAct Agent
# ============================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ============================================
# 10. Create Agent Executor
# ============================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True
)


# ============================================
# 11. Streamlit UI
# ============================================

st.title("🤖 AI Research & Weather Agent")

st.write(
    "Ask me questions about current information, "
    "news, weather, and more."
)

st.info(
    "🔎 I can search the web using Tavily and "
    "🌤️ check current weather using WeatherStack."
)


# ============================================
# 12. User Input
# ============================================

user_input = st.text_area(
    "Ask your question:",
    placeholder=(
        "Example: Who won the 2026 FIFA World Cup "
        "and what is the current weather in Peshawar?"
    ),
    height=100
)


# ============================================
# 13. Run Agent
# ============================================

if st.button("🚀 Ask Agent"):

    if not user_input.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("🤖 Agent is thinking..."):

            try:

                response = agent_executor.invoke({
                    "input": user_input
                })

                st.success("Agent Response")

                st.write(response["output"])

            except Exception as e:

                st.error(
                    f"Something went wrong:\n\n{str(e)}"
                )


# ============================================
# 14. Example Questions
# ============================================

st.sidebar.title("💡 Example Questions")

st.sidebar.write(
    "Try asking:"
)

st.sidebar.write(
    "🔎 Who won the 2026 FIFA World Cup?"
)

st.sidebar.write(
    "🌤️ What is the current weather in Peshawar?"
)

st.sidebar.write(
    "🌍 What is the latest news about climate change?"
)

st.sidebar.write(
    "🌤️ What is the weather in Islamabad?"
)