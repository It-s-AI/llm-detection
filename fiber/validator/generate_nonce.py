import random
import string
import time


def generate_nonce() -> str:
    return f"{time.time_ns()}_{''.join(random.choices(string.ascii_letters + string.digits, k=10))}"
