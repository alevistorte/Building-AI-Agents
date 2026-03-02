"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.
"""

import json
import math
import os

import numexpr
from openai import OpenAI


# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculator(input: str) -> str:
    """A calculator that evaluates math expressions and geometric functions.

    Accepts a JSON string as input (parsed via json.loads).
    Returns results formatted as a JSON string (via json.dumps).
    """
    data = json.loads(input)
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


# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "A calculator that evaluates math expressions and computes geometric functions. "
                "The 'input' parameter must be a JSON string. "
                "For general math, use: {\"operation\": \"evaluate\", \"expression\": \"2 + 3 * 4\"}. "
                "For geometry, supported operations and their required keys are: "
                "circle_area (radius), circle_circumference (radius), "
                "rectangle_area (length, width), rectangle_perimeter (length, width), "
                "triangle_area (base, height), sphere_volume (radius), "
                "sphere_surface_area (radius), cylinder_volume (radius, height), "
                "cone_volume (radius, height), trapezoid_area (base1, base2, height). "
                "Example: {\"operation\": \"circle_area\", \"radius\": 5}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "A JSON string containing 'operation' and the required numeric parameters."
                    }
                },
                "required": ["input"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

    # Start conversation with user query
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_query}
    ]

    print(f"User: {user_query}\n")

    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,  # ← This tells the LLM what tools are available
            tool_choice="auto"  # Let the model decide whether to use tools
        )

        assistant_message = response.choices[0].message

        # Check if the LLM wants to call a tool
        if assistant_message.tool_calls:
            print(
                f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")

            # Add the assistant's response to messages
            messages.append(assistant_message)

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # THIS IS THE MANUAL DISPATCH
                # In a real system, you'd use a dictionary lookup
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "calculator":
                    result = calculator(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })

            print()
            # Loop continues - LLM will see the tool results

        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content

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
