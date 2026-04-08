#!/usr/bin/env python3
"""
Idempotent tool seeder -- run on every deploy.
Creates default public tools only if they don't already exist.

Usage:
    python src/tools/seed_tools.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.database import get_session, get_public_tool_by_name
from src.tools.create_tool import create_tool_from_file

SEED_USER_ID = 1

SEED_TOOLS = [
    {
        "python_file": "src/tools/math_solver.py",
        "tool_name": "Math Solver",
        "tool_description": "Solve mathematical equations provided as strings",
        "main_function": "solve_equation",
    },
    {
        "python_file": "src/tools/statistics_calculator.py",
        "tool_name": "Statistics Calculator",
        "tool_description": "Calculate mean, median, min, max, and count from a list of numbers",
        "main_function": "calculate_statistics",
    },
    {
        "python_file": "src/tools/pdf_extractor.py",
        "tool_name": "PDF Extractor",
        "tool_description": "Extract text content from all PDF files in a directory using docling",
        "main_function": "extract_text_from_pdfs",
    },
]


def seed_tools():
    session = get_session()
    try:
        for tool_def in SEED_TOOLS:
            existing = get_public_tool_by_name(session, tool_def["tool_name"])
            if existing:
                print(f"  Already exists: '{tool_def['tool_name']}' (id={existing.id}), skipping.")
                continue

            tool_id = create_tool_from_file(
                python_file=tool_def["python_file"],
                tool_name=tool_def["tool_name"],
                tool_description=tool_def["tool_description"],
                main_function=tool_def["main_function"],
                user_id=SEED_USER_ID,
            )
            print(f"  Created: '{tool_def['tool_name']}' (id={tool_id})")
    finally:
        session.close()


if __name__ == "__main__":
    print("Seeding default tools...")
    seed_tools()
    print("Tool seeding complete.")
