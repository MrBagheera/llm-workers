import sys

from dotenv import load_dotenv

from llm_workers.config import load_config

load_dotenv()

if __name__ == "__main__":
    # check if config file specified as first parameter
    if len(sys.argv) == 2:
        config_filename = sys.argv[1]
    else:
        print("Usage: python3 main.py <config_file>")
        sys.exit(1)
    config = load_config('MetacriticMonkeyTest.yaml')
    print(config)