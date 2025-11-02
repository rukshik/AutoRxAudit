"""
Simple runner script for synthetic data generation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from generate_synthetic_data import SyntheticDataGenerator

if __name__ == "__main__":
    print("Starting synthetic data generation...")
    generator = SyntheticDataGenerator()
    generator.generate_all()
    print("Done! Check the synthetic_data folder for results.")