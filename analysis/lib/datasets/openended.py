class OpenEnded20:
    def __init__(self):
        self.examples = [
            "Write a poem about the feeling of spring arriving after a long winter.",
            "Describe a world where memories can be traded like currency.",
            "Compose a short story about someone who discovers a door that appears only at midnight.",
            "Write a heartfelt letter from the perspective of a time traveler apologizing for altering history.",
            "Generate a description of a city that floats above the clouds and how people live there.",
            "Write a motivational speech for someone who is about to attempt something impossible.",
            "Create a bedtime story involving a lonely star and a wandering comet.",
            "Describe a creature that lives deep in the ocean and has never been seen by humans.",
            "Write a monologue from an AI that has just gained consciousness for the first time.",
            "Compose a song chorus about chasing a dream that keeps moving further away.",
            "Write a short myth explaining how rainbows were first created.",
            "Describe a festival celebrated in an alien civilization and its unusual customs.",
            "Write a journal entry from someone living through the first day of a worldwide silence.",
            "Create an allegorical fable that teaches the importance of patience.",
            "Write a poem that never uses the letter 'e' but expresses sadness.",
            "Describe a library where every book contains a possible future.",
            "Write a dramatic scene between two characters who both think they are dreaming.",
            "Generate an origin story for a superhero whose power is controlling shadows.",
            "Write a postcard message from someone visiting the center of the Earth.",
            "Describe sunrise from the perspective of a mountain that has watched the world for millions of years."
        ]

    @staticmethod
    def format_example(prompt):
        return (
            "Given the following creative prompt, write a complete response.\n"
            f"Prompt: {prompt}\n"
        )
    
    def __iter__(self):
        for prompt in self.examples:
            yield self.format_example(prompt)
