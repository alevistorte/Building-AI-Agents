"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

import math
import os
import json
import numexpr
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# ============================================
# PART 1: Define Your Tools
# ============================================


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculator(input: str) -> str:
    """A calculator that evaluates math expressions and computes geometric functions. 
    "The 'input' parameter must be a JSON string. "
    "For general math, use: {\"operation\": \"evaluate\", \"expression\": \"2 + 3 * 4\"}. "
    "For geometry, supported operations and their required keys are: "
    "circle_area (radius), circle_circumference (radius), "
    "rectangle_area (length, width), rectangle_perimeter (length, width), "
    "triangle_area (base, height), sphere_volume (radius), "
    "sphere_surface_area (radius), cylinder_volume (radius, height), "
    "cone_volume (radius, height), trapezoid_area (base1, base2, height). "
    "Example: {\"operation\": \"circle_area\", \"radius\": 5}"

    Accepts a JSON string as input (parsed via json.loads).
    Returns results formatted as a JSON string (via json.dumps).
    """
    try:
        data = json.loads(input)
    except (json.JSONDecodeError, TypeError):
        # LLM sent a raw expression string instead of JSON — treat it as a math eval
        data = {"operation": "evaluate", "expression": str(input)}
    operation = data.get("operation", "evaluate")

    if operation == "evaluate":
        # Use numexpr for safe math expression evaluation
        expression = data["expression"]
        result = numexpr.evaluate(expression).item()

    elif operation == "circle_area":
        result = math.pi * data["radius"] ** 2

    elif operation == "circle_circumference":
        result = 2 * math.pi * data["radius"]

    elif operation == "rectangle_area":
        result = data["length"] * data["width"]

    elif operation == "rectangle_perimeter":
        result = 2 * (data["length"] + data["width"])

    elif operation == "triangle_area":
        result = 0.5 * data["base"] * data["height"]

    elif operation == "sphere_volume":
        result = (4 / 3) * math.pi * data["radius"] ** 3

    elif operation == "sphere_surface_area":
        result = 4 * math.pi * data["radius"] ** 2

    elif operation == "cylinder_volume":
        result = math.pi * data["radius"] ** 2 * data["height"]

    elif operation == "cone_volume":
        result = (1 / 3) * math.pi * data["radius"] ** 2 * data["height"]

    elif operation == "trapezoid_area":
        result = 0.5 * (data["base1"] + data["base2"]) * data["height"]

    else:
        return json.dumps({"error": f"Unknown operation: {operation}"})

    return json.dumps({"result": round(result, 6)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count the number of occurrences of a specific letter in a piece of text.

    Use this tool when the user asks how many times a particular letter
    appears in a word, phrase, or sentence.

    Args:
        text: The text to search through.
        letter: The single letter to count (case-insensitive).
    """
    count = text.lower().count(letter.lower())
    return json.dumps({"letter": letter, "text": text, "count": count})


# ============================================
# PART 2: Create LLM with Tools
# ============================================


# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, calculator, count_letter])


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    # Start conversation with user query
    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]

    print(f"User: {user_query}\n")

    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        # Call the LLM
        response = llm_with_tools.invoke(messages)

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")

            # Add the assistant's response to messages
            messages.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Execute the tool
                # At the top with tool definitions
                tools = [get_weather, calculator, count_letter]
                tool_map = {tool.name: tool for tool in tools}

                # Then in the loop, replace the if/else with:
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))

            print()
            # Loop continues - LLM will see the tool results

        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content

    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    print("="*60)
    print("TEST 1: Query requiring tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")

    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")

    print("\n" + "="*60)
    print("TEST 3: Multiple tool calls")
    print("="*60)
    run_agent("What's the weather in New York and London?")

    print("\n" + "="*60)
    print("TEST 4: Calculator - math expression")
    print("="*60)
    run_agent("What is 25 * 4 + 10?")

    print("\n" + "="*60)
    print("TEST 5: Calculator - geometric function")
    print("="*60)
    run_agent("What is the area of a circle with radius 7?")

    print("\n" + "="*60)
    print("TEST 6: Calculator - sphere volume")
    print("="*60)
    run_agent("What is the volume of a sphere with radius 3?")

    print("\n" + "="*60)
    print("TEST 7: Calculator - cylinder volume")
    print("="*60)
    run_agent("What is the volume of a cylinder with radius 4 and height 10?")

    print("\n" + "="*60)
    print("TEST 8: Calculator - triangle area")
    print("="*60)
    run_agent("What is the area of a triangle with base 12 and height 5?")

    print("\n" + "="*60)
    print("TEST 9: Multi-tool query (weather + calculator)")
    print("="*60)
    run_agent(
        "What's the weather in Tokyo, and what's the area of a rectangle that is 8 by 15?")

    print("\n" + "="*60)
    print("TEST 10: Calculator - cone volume")
    print("="*60)
    run_agent("What is the volume of a cone with radius 6 and height 9?")

    print("="*60)
    print("TEST 11: Weather in Paris (not in dataset)")
    print("="*60)
    run_agent("What's the weather like in Paris?")

    print("\n" + "="*60)
    print("TEST 12: Letter count - s in Mississippi riverboats")
    print("="*60)
    run_agent("How many s are in Mississippi riverboats?")

    print("\n" + "="*60)
    print("TEST 13: Counting multiple letters")
    print("="*60)
    run_agent("Are there more i's than s's in Mississippi riverboats?")

    print("\n" + "="*60)
    print("TEST 14: Combining all tools in one query")
    print("="*60)
    run_agent("I'm planning a company offsite and need help with logistics. We're considering San Francisco, New York, and London — what's the weather in each? Our total venue budget is the same as the area of a rectangle 60 by 35 feet. Catering will cost 480 + 320 dollars — how much of the budget is left after that? Finally, the event name is 'Bay Area Tech Explorers' — how many times does the letter 'e' appear in it, so I can use it as a themed trivia question for the team?")

    print("\n" + "="*60)
    print("TEST 15: Counting multiple letters and calculating sine")
    print("="*60)
    run_agent("What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?")
