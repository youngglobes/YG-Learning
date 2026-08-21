"""Module 3 assignment, a deliberately broken agent.

It loops until the call limit on a question that needs one lookup.

Diagnose it from the LangSmith trace ALONE. Do not read past the marker
below until you have written your diagnosis: which step first goes wrong,
what the message contents show, and the root cause.

    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=lsv2_...
    python broken_agent.py
"""

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool

ORDERS = {"A-1001": {"status": "shipped", "carrier": "BlueDart"}}


@tool(parse_docstring=True)
def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order.

    Call this when the user asks about an order.

    Args:
        order_id: The order reference, e.g. A-1001.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return "Not found."
    # ---------------------------------------------------------------
    # Everything above is fine. The bug is on the next line.
    # ---------------------------------------------------------------
    return "Processing, please check again."


agent = create_agent(
    model=MODEL,
    tools=[lookup_order],
    system_prompt=(
        "You are a customer service assistant. Use lookup_order to answer "
        "order questions. Keep checking until you can report a final status."
    ),
    middleware=[ModelCallLimitMiddleware(run_limit=8)],
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the status of order A-1001?"}]},
        config={"recursion_limit": 25},
    )
    print(f"model/tool messages in run: {len(result['messages'])}")
    print(result["messages"][-1].text)

# ======================================================================
# SPOILER, read only after writing your diagnosis.
#
# Two faults compound:
#
#  1. lookup_order ignores the order it just fetched and always returns
#     "Processing, please check again." The result never reads as final,
#     so the model has no way to conclude the task is done.
#
#  2. The system prompt says "Keep checking until you can report a final
#     status", which instructs the model to loop on exactly that signal.
#
# Either alone is survivable. Together they guarantee a loop until the cap.
#
# The fix: return the real status, e.g.
#     return f"Order {order_id}: {order['status']} via {order['carrier']}."
# and soften the prompt so it does not mandate re-checking.
#
# The transferable lesson: a tool result that reads as "not done" is the
# most common cause of a runaway agent, and it is invisible in the source
# until you read what the model actually received.
# ======================================================================
