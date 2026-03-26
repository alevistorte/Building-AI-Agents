import json
import random
import dotenv

# Loading the .env file to load environment variables, if any
dotenv.load_dotenv()

with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nSample example:")
ex = data[1]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")

NUM_TEST_EXAMPLES = 200  # Held-out for evaluation; all remaining data used for training
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples: {len(test_data)}")
