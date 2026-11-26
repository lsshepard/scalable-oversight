"""Run every experiment in order and regenerate everything under results/."""
from experiments import (exp1_menu, exp2_equilibrium, exp3_recognizability, exp4_integer,
                         exp5_marker)

if __name__ == "__main__":
    for mod in (exp1_menu, exp2_equilibrium, exp3_recognizability, exp4_integer, exp5_marker):
        print(f"\n===== {mod.__name__} =====", flush=True)
        mod.main()
