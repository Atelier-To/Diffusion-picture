from datasets import load_dataset
dataset = load_dataset("huggan/anime-faces",split="train")

dataset["image"]
