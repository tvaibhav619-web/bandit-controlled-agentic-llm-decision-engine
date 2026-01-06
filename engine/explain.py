def explain(scores, chains):
    return {
        " -> ".join(chains[i]): scores[i]
        for i in range(len(chains))
    }
