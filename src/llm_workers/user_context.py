import importlib
import inspect
import logging
import os
import sys
from copy import copy
from pathlib import Path

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from llm_workers.api import WorkerException, UserContext
from llm_workers.config import UserConfig, StandardModelDefinition, ImportModelDefinition, ModelDefinition
from llm_workers.utils import find_and_load_dotenv, ensure_environment_variable

logger = logging.getLogger(__name__)


class StandardUserContext(UserContext):
    CONFIG_DIR_PATH = Path.home() / ".config" / "llm-workers"

    def __init__(self, user_config: UserConfig):
        self._user_config = user_config
        self._models = dict[str, BaseChatModel]()
        self._register_models()

    @property
    def user_config(self) -> UserConfig:
        """Get the user configuration."""
        return self._user_config

    @property
    def models(self) -> list[ModelDefinition]:
        """Get list of available model definitions."""
        return self._user_config.models

    def _register_models(self):
        # register models
        for model_def in self._user_config.models:
            model_params = copy(model_def.config) if model_def.config else model_def.model_extra
            if model_def.rate_limiter:
                model_params['rate_limiter'] = InMemoryRateLimiter(
                    requests_per_second = model_def.rate_limiter.requests_per_second,
                    check_every_n_seconds = model_def.rate_limiter.check_every_n_seconds,
                    max_bucket_size = model_def.rate_limiter.max_bucket_size)
            model: BaseChatModel
            try:
                if isinstance(model_def, StandardModelDefinition):
                    model = init_chat_model(model_def.model, model_provider=model_def.provider,
                                            configurable_fields=None, **model_params)
                elif isinstance(model_def, ImportModelDefinition):
                    # split model.import_from into module_name and symbol
                    segments = model_def.import_from.split('.')
                    module_name = '.'.join(segments[:-1])
                    symbol_name = segments[-1]
                    module = importlib.import_module(module_name)  # Import the module
                    symbol = getattr(module, symbol_name, None)  # Retrieve the symbol
                    if symbol is None:
                        raise ValueError(f"Cannot import model from {model_def.import_from}: symbol {symbol_name} not found")
                    elif isinstance(symbol, BaseChatModel):
                        model = symbol
                    elif inspect.isclass(symbol):
                        model = symbol(**model_params) # use default constructor
                    elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
                        model = symbol(**model_params) # use default constructor
                    else:
                        raise ValueError(f"Invalid symbol type {type(symbol)}")
                    if not isinstance(model, BaseChatModel):
                        raise ValueError(f"Invalid model type {type(model)}")
                else:
                    raise ValueError(f"Invalid config type {type(model_def)}")
            except Exception as e:
                raise WorkerException(f"Failed to create model {model_def.name}: {e}", e)

            self._models[model_def.name] = model
            logger.info(f"Registered model {model_def.name}")

    def get_llm(self, llm_name: str) -> BaseChatModel:
        if llm_name in self._models:
            return self._models[llm_name]
        raise WorkerException(f"LLM {llm_name} not found")

    @classmethod
    def load(cls, config_dir_path: Path = CONFIG_DIR_PATH):
        config_env_path = config_dir_path / ".env"
        config_path = config_dir_path / "config.yaml"
        find_and_load_dotenv(config_env_path)

        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            cls._setup_initial_models(config_path, config_env_path)

        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file) or {}
            user_config = UserConfig(**config_data)
            logger.info(f"Loaded user config from {config_path}")
            return cls(user_config)
        except Exception as e:
            raise WorkerException(f"Failed to load user config from {config_path}: {e}", e)

    @classmethod
    def _setup_initial_models(cls, config_path: Path, config_env_path: Path):
        """Interactive setup for initial model configuration"""
        print("To get started, we need to configure your AI models.\n")

        print("Which AI provider would you like to use?")
        print("1. OpenAI (GPT-5 models, requires verified organization)")
        print("2. OpenAI Old (GPT-4o, o3 models)")
        print("3. Anthropic (Claude models)")
        print("4. Custom configuration")

        while True:
            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                cls._copy_default_models(config_path, "openai")
                cls._setup_api_key(config_env_path, "openai")
                break
            elif choice == "2":
                cls._copy_default_models(config_path, "openai-old")
                cls._setup_api_key(config_env_path, "openai")
                break
            elif choice == "3":
                cls._copy_default_models(config_path, "anthropic")
                cls._setup_api_key(config_env_path, "anthropic")
                break
            elif choice == "4":
                cls._show_custom_example(config_path)
                sys.exit(254)  # Exit to let user edit config
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    @classmethod
    def _copy_default_models(cls, config_path: Path, provider: str):
        """Copy default model configuration for specified provider"""
        import importlib.resources

        default_file = f"default-{provider}-models.yaml"
        with importlib.resources.files("llm_workers").joinpath(default_file).open("r") as f:
            config_content = f.read()
        config_path.write_text(config_content)

        print(f"Config saved to {config_path}")
        print("You can edit the configuration later if needed.")

    @classmethod
    def _show_custom_example(cls, config_path: Path):
        """Show example configuration and create empty config"""
        import importlib.resources

        print("\nPlease define you own model configurations in the file ~/.config/llm-workers/config.yaml.")
        print("\nBelow is an example of Anthropic-based configuration:")
        print("-" * 50)

        with importlib.resources.files("llm_workers").joinpath("default-anthropic-models.yaml").open("r") as f:
            example_content = f.read()
        print(example_content)

        print("-" * 50)

    @classmethod
    def _setup_api_key(cls, config_env_path: Path, provider: str):
        """Setup API key for the specified provider"""
        env_var_name = f"{provider.upper()}_API_KEY"

        # Check if API key is already set in environment
        if os.environ.get(env_var_name):
            print(f"{env_var_name} already configured in environment variables.")
            return

        # Use the new ensure_environment_variable function
        provider_name = "OpenAI" if provider == "openai" else "Anthropic"
        description = f"Your {provider_name} API key for accessing {provider_name} models"
        ensure_environment_variable(env_var_name, description)

