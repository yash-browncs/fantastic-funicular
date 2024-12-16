import pickle
import pprint

def inspect_results():
    with open('model_results.pkl', 'rb') as f:
        results = pickle.load(f)

    print("Results structure:")
    pprint.pprint(results)

    print("\nKeys in results dict:")
    print(list(results.keys()))

    if 'results' in results:
        print("\nType of results['results']:")
        print(type(results['results']))
        print("\nFirst item in results['results']:")
        if isinstance(results['results'], list):
            print(results['results'][0])
        elif isinstance(results['results'], dict):
            first_key = next(iter(results['results']))
            print(f"Key: {first_key}")
            print(results['results'][first_key])

if __name__ == "__main__":
    inspect_results()
