import asyncio
import websockets
import json
import base64
import numpy as np
from PIL import Image

async def main():
    uri = "ws://localhost:8000/ws/generate"
    # <-- max_size=None removes the 1MB limit
    async with websockets.connect(uri, max_size=None, ping_interval=None, close_timeout=None) as ws:

        # Send generation request
        await ws.send(json.dumps({
            "prompt": "a futuristic cityscape at sunset, cinematic lighting",
            "num_samples": 1,
            "blocks_per_chunk": 5
        }))

        frame_count = 0
        while True:
            try:
                msg = await ws.recv()
            except websockets.ConnectionClosedOK:
                print("[Client] Connection closed cleanly")
                break
            except websockets.ConnectionClosedError as e:
                print("[Client] Connection closed with error:", e)
                break

            data = json.loads(msg)

            if data["type"] == "frame":
                frame_count += 1
                shape = data["shape"]  # [H, W, C]
                # Decode frame
                img_bytes = base64.b64decode(data["data"])
                img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
                img = Image.fromarray(img_array)
                img.save(f"frame_{frame_count:03d}.png")
                print(f"Saved frame {frame_count}")

            elif data["type"] == "done":
                print("Generation finished")
                break

            elif data["type"] == "error":
                print("Error:", data["message"])
                break

asyncio.run(main())
