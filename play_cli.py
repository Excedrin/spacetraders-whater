import inspect
import readline
import shlex
import traceback

from bot import gather_game_state
from ship_status import FleetTracker
from tools import ALL_TOOLS, get_engine, set_fleet


def get_arg_type_hints(func):
    """Extracts type hints from the underlying function of a Tool."""
    # LangChain tools wrap the actual function. We usually access it via .func or ._run
    # depending on how exactly it was created, but inspection is safest on the callable.
    if hasattr(func, "func"):
        return inspect.signature(func.func)
    return inspect.signature(func)


def get_arg_details(tool):
    """Helper to format argument signatures for display."""
    sig = inspect.signature(tool.func if hasattr(tool, "func") else tool)
    args_list = []
    for param in sig.parameters.values():
        # Clean up type name display
        type_name = (
            param.annotation.__name__
            if hasattr(param.annotation, "__name__")
            else str(param.annotation)
        )
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
                val = arg_str.lower() in ("true", "1", "yes")
            else:
                val = arg_str  # Keep as string

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
        options = [n for n in tools_map.keys() if n.startswith(text)] + [
            "exit",
            "quit",
            "hud",
            "clear",
        ]
        if state < len(options):
            return options[state]
        else:
            return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def print_hud(fleet):
    """Prints the Gather Game State output with behavior status."""
    try:
        output = gather_game_state(fleet)

        # Add behavior status from the behavior engine
        engine = get_engine()
        behavior_status = engine.summary()
        if behavior_status:
            output += f"\n\n[Behavior Status]\n{behavior_status}"

        print("\n" + output + "\n")
    except Exception as e:
        print(f"Error gathering state: {e}")


def main_loop(tools_list):
    tool_map = {t.name: t for t in tools_list}
    configure_readline(tool_map)

    print("\n🚀 SPACETRADERS ENHANCED CLI")
    print("Type 'help', 'hud', or a command.")

    fleet = FleetTracker()
    set_fleet(fleet)
    engine = get_engine()

    while True:
        try:
            engine.sync_state()

            user_input = input("❯ ").strip()
            if not user_input:
                continue

            # Pre-processing
            parts = shlex.split(user_input)
            cmd = parts[0].lower()
            args = parts[1:]

            # Special Commands
            if cmd in ["exit", "quit"]:
                break
            if cmd == "clear":
                print("\033[H\033[J", end="")  # ANSI clear
                continue
            if cmd in ["hud", "state", "status"]:
                print_hud(fleet)
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
                        print(t.description)  # Full description
                        print("-" * 60)
                    else:
                        print(f"❌ Unknown tool '{tool_name}'.")

                # Case 2: General Help List
                else:
                    print("\nAvailable Tools:")
                    for name, t in tool_map.items():
                        # Truncate description for the list view
                        short_desc = t.description.split(".")[0] + "."
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
                sig = inspect.signature(tool.func if hasattr(tool, "func") else tool)
                params = list(sig.parameters.values())

                for i, arg in enumerate(args):
                    # Don't uppercase if it looks like a boolean or number
                    if arg.lower() in ["true", "false"]:
                        final_args.append(arg)
                        continue
                    try:
                        float(arg)
                        final_args.append(arg)
                        continue
                    except ValueError:
                        pass

                    final_args.append(arg)

                # Convert to types via your existing parse logic
                # (You can reuse the parse_and_run function from previous iterations)
                try:
                    # Split into positional args and key=value kwargs
                    positional = []
                    kwargs = {}
                    for arg in final_args:
                        if "=" in arg and not arg.startswith("="):
                            k, _, v = arg.partition("=")
                            kwargs[k] = v
                        else:
                            positional.append(arg)

                    def _coerce(val, annotation):
                        if annotation == int or annotation == "int":
                            return int(val)
                        elif annotation == bool:
                            return str(val).lower() in ("true", "1", "yes")
                        elif annotation == float:
                            return float(val)
                        return val

                    param_map = {p.name: p for p in params}

                    # Map positional args to params (skip params already in kwargs)
                    converted_args = {}
                    pos_idx = 0
                    for param in params:
                        if param.name in kwargs:
                            continue
                        if pos_idx >= len(positional):
                            break
                        converted_args[param.name] = _coerce(
                            positional[pos_idx], param.annotation
                        )
                        pos_idx += 1

                    # Merge and coerce kwargs
                    for k, v in kwargs.items():
                        if k not in param_map:
                            print(f"⚠️  Unknown parameter '{k}' — ignored")
                            continue
                        converted_args[k] = _coerce(v, param_map[k].annotation)

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
    main_loop(ALL_TOOLS)
