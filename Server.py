import os
import asyncio
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()

# Define the datasets dictionary
datasets = {
    'scroll1': '../../scroll1-1cm',
    'campfire': '../../campfire/rec',
    # Add more datasets if needed
}

# Load TIFF images into a dictionary of 3D numpy memmaps
image_data = {}

for dataset, path in datasets.items():
    memmap_file = os.path.join(path, f'{dataset}_memmap.dat')

    image_files = sorted(os.listdir(path))
    #check if the file is a tiff file
    image_files = [file for file in image_files if file.endswith('.tif')]
    #sort by number in name
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))

    image_sample = cv2.imread(os.path.join(path, image_files[0]), cv2.IMREAD_GRAYSCALE)
    image_width, image_height = image_sample.shape

    #check if memmap already exists
    if os.path.exists(memmap_file):
        print(f"Loading memmap for {dataset}...")
        image_data[dataset] = np.memmap(memmap_file, dtype=np.uint8, mode='r', shape=(len(image_files), image_height, image_width))
        continue
    
    print(f"Creating memmap for {dataset}...")
    # Create a binary file to store the memmap data
    fp = np.memmap(memmap_file, dtype=np.uint8, mode='w+', shape=(len(image_files), image_width, image_height))

    # Load the image data into the memmap
    for idx, img_file in enumerate(image_files):    
        print(f'Loading {img_file}...')
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        fp[idx, :, :] = np.array(img)
    
    # Add the memmap to the image_data dictionary
    image_data[dataset] = fp

@app.get("/get_3d_chunk")
async def get_3d_chunk(dataset: str, x: int, y: int, z: int, width: int, height: int, depth: int, downsample: int = 1, latency: float = 0.0):
    try:
        # Simulate latency
        await asyncio.sleep(latency)

        chunk = image_data[dataset][z:z+depth:downsample, y:y+height:downsample, x:x+width:downsample]
        return JSONResponse(content=chunk.tolist())
    except KeyError:
        return JSONResponse(status_code=404, content={'error': f'Dataset {dataset} not found'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
