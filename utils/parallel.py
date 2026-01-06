from concurrent.futures import ThreadPoolExecutor

def run_parallel(agents, query):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(a.act, query) for a in agents]
        return [f.result() for f in futures]
