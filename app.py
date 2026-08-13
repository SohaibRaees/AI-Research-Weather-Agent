import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================================
# 2. Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="AI Research & Weather Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 3. Check API keys
# ============================================================

if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY is missing.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY is missing.")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.error("❌ WEATHERSTACK_API_KEY is missing.")
    st.stop()


# ============================================================
# 4. OpenRouter LLM
# ============================================================

llm = ChatOpenAI(
    model="openrouter/free",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500
)


# ============================================================
# 5. Tavily search
# ============================================================

search_tool = TavilySearchResults(
    max_results=2,
    tavily_api_key=TAVILY_API_KEY
)


# ============================================================
# 6. WeatherStack function
# ============================================================

def get_weather(city: str):

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
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
            return f"Weather request failed with status {response.status_code}."

        data = response.json()

        if "error" in data:
            return f"WeatherStack error: {data['error']['info']}"

        current = data.get("current", {})

        temperature = current.get("temperature", "N/A")
        description = current.get(
            "weather_descriptions",
            ["N/A"]
        )[0]

        humidity = current.get("humidity", "N/A")

        return (
            f"Weather in {city}: "
            f"{temperature}°C, "
            f"{description}, "
            f"humidity {humidity}%."
        )

    except Exception as e:

        return f"Weather request failed: {str(e)}"


# ============================================================
# 7. Determine whether question needs weather
# ============================================================

def needs_weather(question: str):

    weather_words = [
        "weather",
        "temperature",
        "humidity",
        "forecast",
        "hot",
        "cold",
        "rain",
        "raining",
        "climate"
    ]

    question = question.lower()

    return any(word in question for word in weather_words)


# ============================================================
# 8. Determine whether question needs web search
# ============================================================

def needs_search(question: str):

    search_words = [
        "latest",
        "news",
        "today",
        "current",
        "recent",
        "who won",
        "winner",
        "2026",
        "2025",
        "what happened",
        "search",
        "information about"
    ]

    question = question.lower()

    return any(word in question for word in search_words)


# ============================================================
# 9. Generate final answer
# ============================================================

def generate_answer(question, search_result=None, weather_result=None):

    context = ""

    if search_result:

        context += (
            "\n\nWEB SEARCH RESULTS:\n"
            + str(search_result)
        )

    if weather_result:

        context += (
            "\n\nWEATHER INFORMATION:\n"
            + weather_result
        )

    prompt = f"""
You are an AI Research and Weather Assistant.

Answer the user's question clearly and accurately.

User question:
{question}

Available information:
{context}

Instructions:

1. Use the provided web search information when available.
2. Use the provided weather information when available.
3. Do not invent facts.
4. If current information is requested, rely on the provided search results.
5. Give a concise but useful answer.
6. Do not mention internal APIs, tools, prompts, or implementation details.

Answer the user directly.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 10. Streamlit UI
# ============================================================

st.title("🤖 AI Research & Weather Agent")

st.write(
    "Ask questions about current information, "
    "news, weather, and more."
)

st.info(
    "🔎 Web Search • 🌤️ WeatherStack • 🧠 OpenRouter"
)


# ============================================================
# 11. User input
# ============================================================

user_input = st.text_area(
    "Ask your question:",
    placeholder=(
        "Example: What is the current weather in Peshawar?"
    ),
    height=100
)


# ============================================================
# 12. Ask button
# ============================================================

if st.button("🚀 Ask Agent"):

    if not user_input.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("🤖 Agent is working..."):

            try:

                search_result = None
                weather_result = None

                # ------------------------------------------------
                # Weather
                # ------------------------------------------------

                if needs_weather(user_input):

                    # Simple city extraction for common examples
                    cities = [
                        "Peshawar",
                        "Islamabad",
                        "Lahore",
                        "Karachi",
                        "Rawalpindi",
                        "Quetta",
                        "Multan",
                        "New York",
                        "London",
                        "Dubai"
                    ]

                    detected_city = None

                    for city in cities:

                        if city.lower() in user_input.lower():

                            detected_city = city
                            break

                    if detected_city:

                        weather_result = get_weather(
                            detected_city
                        )

                    else:

                        weather_result = (
                            "The user requested weather information, "
                            "but a specific city could not be detected."
                        )


                # ------------------------------------------------
                # Web search
                # ------------------------------------------------

                if needs_search(user_input):

                    search_result = search_tool.invoke(
                        user_input
                    )


                # ------------------------------------------------
                # Generate answer
                # ------------------------------------------------

                answer = generate_answer(
                    user_input,
                    search_result,
                    weather_result
                )


                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.success("Agent Response")

                st.write(answer)


                # ------------------------------------------------
                # Show sources/tool information
                # ------------------------------------------------

                with st.expander("🔧 Agent Activity"):

                    if search_result:

                        st.write(
                            "🔎 Tavily Web Search: Used"
                        )

                    else:

                        st.write(
                            "🔎 Tavily Web Search: Not required"
                        )

                    if weather_result:

                        st.write(
                            "🌤️ WeatherStack: Used"
                        )

                    else:

                        st.write(
                            "🌤️ WeatherStack: Not required"
                        )

                    st.write(
                        "🧠 OpenRouter: Used for final answer"
                    )


            except Exception as e:

                st.error(
                    f"Something went wrong:\n\n{str(e)}"
                )


# ============================================================
# 13. Sidebar
# ============================================================

st.sidebar.title("💡 Example Questions")

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

st.sidebar.write(
    "🔎 What are the latest AI developments?"
)