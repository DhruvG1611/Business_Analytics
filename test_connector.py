#!/usr/bin/env python
import sys
sys.path.insert(0, 'c:\\Dsg\\acceleronSoln')

from connector import ask_database

print("Testing: 'who are employees'")
try:
    results = ask_database("who are employees")
    print(f"Results: {results}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

