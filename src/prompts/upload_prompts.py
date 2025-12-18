
"""
Upload prompts from src/prompts markdown files into Prompts table
Expected filenames:
  {prompt_name}.system.md
  {prompt_name}.user.md
"""

import os
import sys
import argparse
from typing import Dict, Optional
from sqlalchemy import text

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.database_setup import get_database_manager, Prompts

def scan_prompts(prompts_dir: str, verbose: bool = False):
    """
    Scan through the prompts directory for *.system.md and *.user.md files 
    """

    if not os.path.isdir(prompts_dir):
        raise FileNotFoundError(f"Prompts directory '{prompts_dir}' does not exist")
    
    prompts: Dict[str, Dict[str, str]] = {}

    for entry in os.listdir(prompts_dir):
        path = os.path.join(prompts_dir, entry)
        if not os.path.isfile(path):
            continue
        if entry.endswith('.system.md'):
            prompt_name = entry[:-len('.system.md')]
            key="system"
        elif entry.endswith('.user.md'):
            prompt_name = entry[:-len('.user.md')]
            key="user"
        else:
            continue

        if prompt_name not in prompts:
            prompts[prompt_name] = {"system": None, "user": None}
        
        if verbose:
            print(f"Found {key} prompt: {prompt_name}")
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        prompts[prompt_name][key] = content
    
    return prompts

def upload_prompts(prompts: Dict[str, Dict[str, str]], verbose: bool = False, environment: str = "development"):
    db_manager = get_database_manager(environment=environment)

    session = db_manager.get_session()

    try:
        for prompt_name, content in prompts.items():
            system_prompt = content["system"]
            user_prompt = content["user"]

            if system_prompt is None and user_prompt is None:
                continue

            existing = session.query(Prompts).filter_by(prompt_name=prompt_name).first()

            #If the prompt already exists, update it
            if existing:
                if verbose:
                    print(f"[UPDATE] prompt_name='{prompt_name}'")
                existing.system_prompt = system_prompt
                existing.user_prompt = user_prompt
                session.commit()
            else:
                if verbose:
                    print(f"[INSERT] prompt_name='{prompt_name}'")
                new_prompt = Prompts(
                    prompt_name=prompt_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt)
                session.add(new_prompt)
    
        session.commit()
    
    except Exception as e:
        session.rollback()
        print(f"Error uploading prompts: {e}")
        raise
    finally:
        db_manager.close_session(session)


def main():
    parser = argparse.ArgumentParser(description="Upload prompts to database")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--environment", type=str, default="development", help="Environment to upload prompts to")
    parser.add_argument("--prompts-dir", type=str, default=CURRENT_DIR, help="Directory containing prompt files")

    args = parser.parse_args()

    prompts_data = scan_prompts(args.prompts_dir, verbose=args.verbose)
    
    upload_prompts(
        prompts_data,
        verbose=args.verbose,
        environment=args.environment
    )

if __name__ == "__main__":
    main()

