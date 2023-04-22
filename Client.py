import requests
import json
import matplotlib.pyplot as plt
import numpy as np
def get_3d_chunk(dataset, x, y, z, width, height, depth, downsample=1, latency=0.0, server_url='http://localhost:8000'):
    try:
        params = {
            'dataset': dataset,
            'x': x,
            'y': y,
            'z': z,
            'width': width,
            'height': height,
            'depth': depth,
            'downsample': downsample,
            'latency': latency,
        }
        response = requests.get(f"{server_url}/get_3d_chunk", params=params)

        if response.status_code != 200:
            raise Exception(response.json()['error'])

        data = json.loads(response.text)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    dataset = 'scroll1'
    x, y, z = 1000, 1000, 50
    width, height, depth = 1000, 1000, 10
    downsample = 1
    latency = 0.

    data = np.array(get_3d_chunk(dataset, x, y, z, width, height, depth, downsample, latency))

    plt.imshow(data[0, :, :], cmap='gray')
    plt.show()
