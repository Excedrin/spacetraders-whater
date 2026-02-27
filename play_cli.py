import inspect
import sys
import shlex
from langchain_core.tools import BaseTool
from tools import ALL_TOOLS

def get_arg_type_hints(func):
    """Extracts type hints from the underlying function of a Tool."""
    # LangChain tools wrap the actual function. We usually access it via .func or ._run
    # depending on how exactly it was created, but inspection is safest on the callable.
    if hasattr(func, 'func'):
        return inspect.signature(func.func)
    return inspect.signature(func)

def parse_and_run(tool, args_list):
    """
    Matches command line string arguments to the tool's function signature,
    converting types (int, float, etc.) where necessary.
    """
    sig = get_arg_type_hints(tool)
    bound_args = None
    
    try:
        # 1. Map string args to function parameters
        # We need to manually convert strings to ints/bools based on type hints
        converted_args = []
        params = list(sig.parameters.values())
        
        for i, arg_str in enumerate(args_list):
            if i >= len(params):
                break
            param = params[i]
            
            # Simple Type Conversion
            if param.annotation == int or param.annotation == "int":
                val = int(arg_str)
            elif param.annotation == float:
                val = float(arg_str)
            elif param.annotation == bool:
                val = arg_str.lower() in ('true', '1', 'yes')
            else:
                val = arg_str # Keep as string
                
            converted_args.append(val)
            
        # 2. Call the tool
        # Note: We are calling the tool directly as a callable, 
        # distinct from tool.run() which expects a single dict/str usually.
        return tool.invoke(input=dict(zip(sig.parameters.keys(), converted_args)))

    except Exception as e:
        return f"Execution Error: {str(e)}"

def start_interactive_session(tools_list):
    """Main Loop"""
    # Create a lookup dictionary for easy access
    tool_map = {t.name: t for t in tools_list}
    
    print("="*60)
    print("🚀 SPACETRADERS MANUAL CONSOLE")
    print("="*60)
    print(f"Loaded {len(tools_list)} tools.")
    print("Type 'help' to see available commands.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👉 ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Closing connection...")
                break

            # Handle Help
            if user_input.lower() == "help":
                print("\nAvailable Tools:")
                for name, t in tool_map.items():
                    # Get argument names for display
                    sig = get_arg_type_hints(t)
                    args_str = ", ".join([f"{p.name}: {p.annotation.__name__ if hasattr(p.annotation, '__name__') else p.annotation}" for p in sig.parameters.values()])
                    print(f"  • {name}({args_str})")
                    print(f"    Desc: {t.description.split('.')[0]}.")
                continue

            # Parse Command
            # shlex.split handles quotes properly, e.g. send_message "Hello World"
            parts = shlex.split(user_input)
            cmd_name = parts[0]
            cmd_args = parts[1:]

            if cmd_name not in tool_map:
                print(f"❌ Unknown tool: '{cmd_name}'. Type 'help' for list.")
                continue

            # Execute
            tool = tool_map[cmd_name]
            print(f"⏳ Running {cmd_name}...")
            result = parse_and_run(tool, cmd_args)
            print(f"✅ Result:\n{result}")

        except KeyboardInterrupt:
            print("\nType 'exit' to quit.")
        except Exception as e:
            print(f"❌ System Error: {e}")

if __name__ == "__main__":
    start_interactive_session(ALL_TOOLS)
