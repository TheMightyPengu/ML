import asyncio
import numpy as np

perfect = [None] * 10
imperfect = [None] * 10

perfect[5] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]])

perfect[6] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]])

perfect[8] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]])

perfect[9] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1,1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]])

imperfect[5] = np.array([
                    [-1, -1, -1, -1, -1, -1, 1],
                    [-1, -1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1]])

imperfect[6] = np.array([
                    [-1, -1, -1, -1, -1, -1, 1],
                    [-1, -1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, -1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, -1, 1, 1, 1, -1, -1]])

imperfect[8] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 1, 1, 1, -1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, -1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, 1, 1, -1, -1]])


imperfect[9] = np.array([
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 1, 1, 1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, -1, -1, -1, 1, -1],
                    [-1, 1, 1, -1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, -1, 1, -1],
                    [-1, -1, 1, -1, 1, 1, -1]])


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train_network(self, training_data):
        num_neurons = self.num_neurons
        weights = np.zeros((num_neurons, num_neurons))

        def process_pattern(pattern):
            pattern_array = np.array(pattern)
            pattern_array = pattern_array.reshape((num_neurons, 1))

            return np.matmul(pattern_array, np.transpose(pattern_array))

        patterns_processed = map(process_pattern, training_data)
        weights = sum(patterns_processed)
        np.fill_diagonal(weights, 0)

        num_patterns = len(training_data)
        weights = weights.astype(float)
        weights /= num_patterns
        self.weights = weights

    def predict(self, pattern, max_iterations=100):
        #num_neurons = pattern.shape[0]
        num_neurons = self.num_neurons - 7
        pattern = np.array(pattern)
        pattern = pattern.reshape((num_neurons, 1))

        for _ in range(max_iterations):
            output = np.sign(np.dot(self.weights[:num_neurons, :num_neurons], pattern))

            if np.array_equal(output, pattern):
                return output.flatten().tolist()

            pattern = output
        return None

    async def predict_async(self, pattern, max_iterations=100):
        #num_neurons = self.num_neurons - 7
        pattern = np.array(pattern)
        pattern = pattern.reshape((self.num_neurons - 7, 1))

        for _ in range(max_iterations):
            output = np.dot(self.weights[:70, :70], pattern)
            output[output >= 0] = 1
            output[output < 0] = -1

            if np.array_equal(output, pattern):
                return output.flatten().tolist()
            pattern = output
            await asyncio.sleep(0)

def print_pattern(pattern):
    pattern_str = ""
    for row in pattern:
        line = "".join("  " if element == 1 else "██" for element in row)
        pattern_str += line + "\n"
    pattern_str += "".join("██" for _ in range(len(pattern[0]))) + "\n"
    print(pattern_str)

def print_pattern_row(getPatterns):
    patterns = []
    for i in range(len(getPatterns)):
        if getPatterns[i] is not None:
            pattern = getPatterns[i].tolist()
            patterns.append(pattern)

    max_height = max(len(pattern) for pattern in patterns)

    for i in range(max_height):
        for pattern in patterns:
            if i < len(pattern):
                line = ""
                for element in pattern[i]:
                    if element == 1:
                        line += "  "
                    else:
                        line += "██"
                print(line, end='')
        print()

    return None

patterns_perfect = []
patterns_imperfect = []

for i in [5, 6, 8, 9]:
    if perfect[i] is not None:
        pattern = perfect[i].tolist()
        patterns_perfect.append(pattern)
    if imperfect[i] is not None:
        pattern = imperfect[i].tolist()
        patterns_imperfect.append(pattern)

network = HopfieldNetwork(num_neurons=77)
network.train_network(patterns_perfect)

print_pattern_row(perfect)
print_pattern_row(imperfect)

# for pattern in patterns_perfect:
#     print_pattern(pattern)

# for pattern in patterns_imperfect:
#     print_pattern(pattern)

while True:
    choice = input("Please pick an option (sync/async/exit): ").lower()
    if choice == "sync":
        result_sync = []

        while True:
            try:
                choice_num = int(input("Please pick the number you want to perfect (5/6/8/9): "))
                if choice_num in [5, 6, 8, 9]:
                    break
                else:
                    print("Thats not an option :( Please try again.\n")
            except ValueError:
                print("Thats not an option :( Please try again.\n")

        result = network.predict(imperfect[choice_num], max_iterations=200)

        if result:
            result_sync.append(result)
        else:
            print("failed.")
            break

        for pattern in result_sync:
            print_pattern(np.array(pattern).reshape((10, 7)))

    elif choice == "async":
        async def async_prediction():
            result_async = []

            while True:
                try:
                    choice_num = int(input("Please pick the number you want to perfect (5/6/8/9): "))
                    if choice_num in [5, 6, 8, 9]:
                        break
                    else:
                        print("Thats not an option :( Please try again.\n")
                except ValueError:
                    print("Thats not an option :( Please try again.\n")

            result = await network.predict_async(imperfect[choice_num], max_iterations=200)

            if result:
                result_async.append(result)
            else:
                print("failed.")
                return

            for pattern in result_async:
                print_pattern(np.array(pattern).reshape((10, 7)))

        loop = asyncio.get_event_loop()
        loop.run_until_complete(async_prediction())
        loop.close()

    elif choice == "exit":
        print("Thank you for visiting, please come again.")
        break
    
    else:
        print("Thats not an option :(")