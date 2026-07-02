import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
RANDOM_STATE = 2018
TEST_SIZE = 0.2
TARGET_COLUMN = "Class"
