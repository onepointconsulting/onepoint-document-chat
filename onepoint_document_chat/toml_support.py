from pathlib import Path
import tomli
from onepoint_document_chat.config import cfg


def read_toml(file: Path) -> dict:
    with open(file, "rb") as f:
        return tomli.load(f)


def read_prompts_toml() -> dict:
    return read_toml(cfg.project_root / "prompts.toml")


prompts_toml = read_prompts_toml()

if __name__ == "__main__":
    from onepoint_document_chat.log_init import logger

    logger.info(prompts_toml)
