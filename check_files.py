import os
print("Current working directory:", os.getcwd())
print("\nChecking if file exists:")
strategy_path = "src/strategy/price_action_ml.py"
print(f"Looking for: {strategy_path}")
print(f"File exists: {os.path.exists(strategy_path)}")

if os.path.exists(strategy_path):
    with open(strategy_path, 'r') as file:
        first_few_lines = file.readlines()[:10]
        print("\nFirst few lines of the file:")
        print(''.join(first_few_lines))