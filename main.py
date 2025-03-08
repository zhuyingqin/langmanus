from src.graph import graph


if __name__ == "__main__":
    for s in graph.stream(
        {"messages": [("user", "Give me a report about Manus")]}, subgraphs=True
    ):
        print(s)
        print("----")
