import time
from datetime import datetime, timedelta

def wait_until(target_time):
    """Wait until the target time (a datetime object)."""
    while datetime.now() < target_time:
        # Sleep for a short period to avoid busy-waiting
        time.sleep(1)
        print(f'wait {datetime.now()}')

def run_at_specific_time(hour, minute):
    """Run the code at the specific hour and minute."""
    now = datetime.now()
    target_time = datetime(now.year, now.month, now.day, hour, minute)

    # Wait until the target time
    wait_until(target_time)

    # Run your code
    print("It's 08:30! Running the code...")

    # Your code goes here
    # For example, print a message
    print("Hello, world!")

# Set the specific time to run the code (08:30)
run_at_specific_time(20, 34)