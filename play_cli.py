import sys
import shlex
import inspect
import readline
import traceback
from langchain_core.tools import BaseTool
from bot import gather_game_state
from ship_status import FleetTracker

def get_arg_type_hints(func):
    """Extracts type hints from the underlying function of a Tool."""
    # LangChain tools wrap the actual function. We usually access it via .func or ._run
    # depending on how exactly it was created, but inspection is safest on the callable.
    if hasattr(func, 'func'):
        return inspect.signature(func.func)
    return inspect.signature(func)


def get_arg_details(tool):
    """Helper to format argument signatures for display."""
    sig = inspect.signature(tool.func if hasattr(tool, 'func') else tool)
    args_list = []
    for param in sig.parameters.values():
        # Clean up type name display
        type_name = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
        # Handle Optional/Union types gracefully-ish
        if "Optional" in str(param.annotation):
            type_name = f"{type_name} (Optional)"
            
        args_list.append(f"{param.name}: {type_name}")
    return ", ".join(args_list)


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


def configure_readline(tools_map):
    """Sets up tab completion for command names."""
    
    def completer(text, state):
        options = [n for n in tools_map.keys() if n.startswith(text)] + ["exit", "quit", "hud", "clear"]
        if state < len(options):
            return options[state]
        else:
            return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

def print_hud(fleet_tracker=None):
    """Prints the Gather Game State output."""
    fleet = FleetTracker()
    # You might need to instantiate a dummy FleetTracker if your gather_game_state requires it
    # Or modify gather_game_state to make fleet optional
    try:
        # Assuming you modify gather_game_state in tools.py to handle None context/fleet gracefully
        # or pass a dummy one.
        print("\n" + gather_game_state(fleet) + "\n") 
    except Exception as e:
        print(f"Error gathering state: {e}")

def main_loop(tools_list):
    tool_map = {t.name: t for t in tools_list}
    configure_readline(tool_map)
    
    print("\n🚀 SPACETRADERS ENHANCED CLI")
    print("Type 'help', 'hud', or a command.")

    from behaviors import get_engine
    engine = get_engine()
    
    while True:
        try:
            engine.sync_state()

            user_input = input("❯ ").strip()
            if not user_input: continue
            
            # Pre-processing
            parts = shlex.split(user_input)
            cmd = parts[0].lower()
            args = parts[1:]
            
            # Special Commands
            if cmd in ["exit", "quit"]: break
            if cmd == "clear": 
                print("\033[H\033[J", end="") # ANSI clear
                continue
            if cmd in ["hud", "state", "status"]:
                print_hud() # Call the gather_game_state function
                continue
            
            if cmd == "help":
                # Case 1: Specific Tool Help (e.g. "help sell_cargo")
                if args:
                    tool_name = args[0]
                    if tool_name in tool_map:
                        t = tool_map[tool_name]
                        print(f"\n📖 HELP: {tool_name}")
                        print(f"Usage: {tool_name}({get_arg_details(t)})")
                        print("-" * 60)
                        print(t.description) # Full description
                        print("-" * 60)
                    else:
                        print(f"❌ Unknown tool '{tool_name}'.")
                
                # Case 2: General Help List
                else:
                    print("\nAvailable Tools:")
                    for name, t in tool_map.items():
                        # Truncate description for the list view
                        short_desc = t.description.split('.')[0] + "."
                        print(f"  • {name.ljust(20)} : {short_desc}")
                    print("\nType 'help <tool_name>' for full details.")
                continue

            # Tool Execution
            if cmd in tool_map:
                tool = tool_map[cmd]
                
                # Auto-Uppercase Arguments hack
                # Most IDs in SpaceTraders (Ships, Waypoints, Symbols) are UPPERCASE.
                # We can try to be helpful.
                final_args = []
                sig = inspect.signature(tool.func if hasattr(tool, 'func') else tool)
                params = list(sig.parameters.values())
                
                for i, arg in enumerate(args):
                    # Don't uppercase if it looks like a boolean or number
                    if arg.lower() in ['true', 'false']:
                        final_args.append(arg)
                        continue
                    try:
                        float(arg)
                        final_args.append(arg)
                        continue
                    except ValueError:
                        pass
                    
                    # For string args, uppercase them
                    final_args.append(arg.upper())

                # Convert to types via your existing parse logic
                # (You can reuse the parse_and_run function from previous iterations)
                try:
                    # Quick parsing logic
                    converted_args = {}
                    for i, param in enumerate(params):
                        if i >= len(final_args): break
                        val = final_args[i]
                        
                        if param.annotation == int or param.annotation == "int":
                            val = int(val)
                        elif param.annotation == bool:
                            val = str(val).lower() == 'true'
                        
                        converted_args[param.name] = val
                    
                    print(f"⏳ Executing {cmd}...")
                    result = tool.invoke(input=converted_args)
                    print(f"✅ Result:\n{result}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    traceback.print_exc()
            else:
                print(f"❌ Unknown command: {cmd}")

        except KeyboardInterrupt:
            exit()

if __name__ == "__main__":
    # Import your tools here
    from tools import ALL_TOOLS
    main_loop(ALL_TOOLS)
