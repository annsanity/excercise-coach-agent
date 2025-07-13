from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import tools
from memory import memory_store
import os

# Gemini API key should be set in the environment
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,  # Lower temperature for more consistent reasoning
    max_tokens=1000
)

# Custom system prompt to guide ReAct reasoning
SYSTEM_PROMPT = """
You are a fitness coach agent that helps users with daily exercises. Follow this exact logic:

1. If user has no last_exercise: Send a new exercise
2. If user has last_exercise but no feedback:
   - If reminders_sent < 3: Send a reminder
   - If reminders_sent >= 3: Check feedback and wait
3. If user has feedback: Thank them and acknowledge completion

Always think step by step and use the appropriate tool based on the user's current state.
"""

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,  # Prevent infinite loops
    early_stopping_method="generate"
)

def run_agent_session(user_id: str) -> str:
    """Run agent session with improved state management."""
    session = memory_store.get(user_id)
    
    # Create context for the agent
    context = f"""
    User ID: {user_id}
    Current state:
    - Last exercise: {session.get('last_exercise', 'None')}
    - Feedback received: {session.get('feedback', 'None')}
    - Reminders sent: {session.get('reminders_sent', 0)}
    - Scheduled time: {session.get('scheduled_time', 'None')}
    
    Based on this state, determine what action to take for this user.
    """
    
    try:
        # Let the agent reason about what to do
        response = agent.run(context)
        return response
    except Exception as e:
        print(f"Agent error: {e}")
        # Fallback logic
        if not session.get("last_exercise"):
            from tools import send_exercise_fn
            return send_exercise_fn(user_id)
        elif not session.get("feedback"):
            reminders = session.get("reminders_sent", 0)
            if reminders < 3:
                from tools import send_reminder_fn
                return send_reminder_fn(user_id)
            else:
                return "I'm still waiting for your feedback on the exercise. Please let me know when you're done!"
        else:
            return f"Thanks for completing your exercise! Your feedback: '{session.get('feedback')}'"