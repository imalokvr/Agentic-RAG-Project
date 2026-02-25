"""CLI interactive chat entry point.

Run:  python -m app.run_chat
"""

import sys


def main() -> None:
    print("=" * 60)
    print("  Agentic RAG Chat Assistant")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    from orchestrator.orchestrator import ChatOrchestrator
    orch = ChatOrchestrator()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print()
        answer = orch.handle_query(user_input)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
