import datasets # type: ignore
import custom_console

from llama_index.core.schema import Document # type: ignore

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")


docs = [
    Document(
        text="\n".join([
            f"Name: {guest_dataset['name'][i]}",
            f"Relation: {guest_dataset['relation'][i]}",
            f"Description: {guest_dataset['description'][i]}",
            f"Email: {guest_dataset['email'][i]}"
        ]),
        metadata={"name": guest_dataset['name'][i]}
    )
    for i in range(len(guest_dataset))
]


custom_console.clear_console()
custom_console.simple_spinner(duration=3)
# print(docs)


