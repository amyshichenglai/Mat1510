import datasets

class LAMBADA:
    def __init__(self):
        self.dataset = datasets.load_dataset("lambada", "plain_text", split="test")

    @staticmethod
    def format_example(example):
        text = example["text"].rstrip()
        
        parts = text.split()
        context = " ".join(parts[:-1])
        target = parts[-1]

        prompt = (
            f"Read the following passage and predict the final word.\n"
            f"Passage: {context}\n"
            f"Your response should end with \"The final answer is [word]\" where [word] is the missing last word.\n"
        )

        return f"{prompt}\nThe final answer is {target}"

    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)
