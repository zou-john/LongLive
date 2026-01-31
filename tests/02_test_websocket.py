#!/usr/bin/env python3
"""
Quick WebSocket test script for LongLive streaming server.

Usage:
    python quick_test.py wss://your-url/ws/generate
    python quick_test.py wss://your-url/ws/generate "A cat walking in a garden"
    python quick_test.py wss://your-url/ws/generate "A cat walking" --save-frames
    python quick_test.py wss://your-url/ws/generate "A cat walking" --save-frames --output-dir my_video_frames
"""

import asyncio
import json
import sys
import base64
import io
from pathlib import Path

try:
    import websockets
except ImportError:
    print("❌ Missing websockets library. Install with:")
    print("   pip install websockets Pillow numpy")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("❌ Missing PIL or numpy. Install with:")
    print("   pip install Pillow numpy")
    sys.exit(1)


async def test_websocket(
    ws_url: str, 
    prompt: str = "A beautiful sunset over the ocean",
    save_frames: bool = False,
    output_dir: str = "output_frames"
):
    """Test the WebSocket connection and streaming."""
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing WebSocket Connection")
    print(f"{'='*60}")
    print(f"URL: {ws_url}")
    print(f"Prompt: {prompt}")
    if save_frames:
        print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")
    
    frame_count = 0
    output_path = None
    
    # Create output directory if saving frames
    if save_frames:
        output_path = Path("output") / output_dir

        output_path.mkdir(exist_ok=True, parents=True)
        print(f"💾 Saving frames to: {output_path}/\n")
    
    try:
        print("Connecting to server...")
        # Increase max_size to 10MB to handle large frames
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as websocket:
            print("✅ Connected!")
            
            # Send initial request
            request = {
                "prompt": prompt,
                "num_samples": 1,
                "blocks_per_chunk": 5,
            }
            
            print(f"Sending request: {json.dumps(request, indent=2)}")
            await websocket.send(json.dumps(request))
            print("✅ Request sent\n")
            
            print("Waiting for frames...\n")
            
            # Receive frames
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(message)
                    
                    if data["type"] == "frame":
                        frame_count += 1
                        
                        # Decode the frame
                        frame_data = base64.b64decode(data["data"])
                        
                        # Check if it's JPEG or raw
                        if data.get("format") == "jpeg":
                            # Decode JPEG
                            img = Image.open(io.BytesIO(frame_data))
                            frame_array = np.array(img)
                            format_str = "JPEG"
                        else:
                            shape = data["shape"]
                            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(shape)
                            format_str = "RAW"
                        
                        shape = frame_array.shape
                        data_size = len(frame_data)
                        
                        # Save frame if requested
                        if save_frames and output_path:
                            output_img = Image.fromarray(frame_array)
                            frame_filename = output_path / f"frame_{frame_count:04d}.png"
                            output_img.save(frame_filename)
                            
                            # Show save confirmation for first frame
                            if frame_count == 1:
                                print(f"💾 Saving frames as: frame_XXXX.png\n")
                        
                        # Save first frame for preview even if not saving all
                        elif frame_count == 1:
                            output_img = Image.fromarray(frame_array)
                            output_img.save("test_frame_0.png")
                            print(f"💾 Saved preview to test_frame_0.png\n")
                        
                        # Print progress every 10 frames (or every 5 if not saving)
                        progress_interval = 10 if save_frames else 5
                        if frame_count % progress_interval == 0 or frame_count == 1:
                            print(f"📹 Frame {frame_count:3d} | {format_str:4s} | Shape: {shape[0]}x{shape[1]}x{shape[2]} | Data: {data_size:,} bytes")
                    
                    elif data["type"] == "done":
                        print(f"\n{'='*60}")
                        print(f"✅ SUCCESS! Generation complete")
                        print(f"   Total frames received: {frame_count}")
                        if save_frames and output_path:
                            print(f"   Frames saved to: {output_path}/")
                            print(f"\n💡 Create video with ffmpeg:")
                            print(f"   ffmpeg -framerate 30 -i {output_path}/frame_%04d.png \\")
                            print(f"          -c:v libx264 -pix_fmt yuv420p output.mp4")
                        else:
                            print(f"   Preview saved as: test_frame_0.png")
                        print(f"{'='*60}\n")
                        break
                    
                    elif data["type"] == "error":
                        print(f"\n{'='*60}")
                        print(f"❌ ERROR from server:")
                        print(f"   {data['message']}")
                        print(f"{'='*60}\n")
                        return False
                    
                    else:
                        print(f"⚠️  Unknown message type: {data['type']}")
                
                except asyncio.TimeoutError:
                    print(f"\n❌ Timeout waiting for message (waited 120s)")
                    print(f"   Received {frame_count} frames so far")
                    return False
        
        return True
    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"\n❌ Connection failed with status code: {e.status_code}")
        print(f"   This usually means the server isn't ready or the URL is wrong")
        return False
    
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        return False
    
    except ConnectionRefusedError:
        print(f"\n❌ Connection refused")
        print(f"   Make sure the server is deployed and the URL is correct")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <websocket-url> [prompt] [--save-frames] [--output-dir DIR]")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} wss://your-url/ws/generate")
        print(f"  {sys.argv[0]} wss://your-url/ws/generate \"A cat walking\"")
        print(f"  {sys.argv[0]} wss://your-url/ws/generate \"A cat walking\" --save-frames")
        print(f"  {sys.argv[0]} wss://your-url/ws/generate \"A cat\" --save-frames --output-dir my_frames")
        sys.exit(1)
    
    # Parse arguments
    ws_url = sys.argv[1]
    
    # Default values
    prompt = "A beautiful sunset over the ocean"
    save_frames = False
    output_dir = "output_frames"
    
    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--save-frames":
            save_frames = True
            i += 1
        elif arg == "--output-dir":
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                print("❌ --output-dir requires a directory name")
                sys.exit(1)
        elif not arg.startswith("--"):
            # It's the prompt
            prompt = arg
            i += 1
        else:
            print(f"❌ Unknown argument: {arg}")
            sys.exit(1)
    
    # Run the async test
    success = asyncio.run(test_websocket(ws_url, prompt, save_frames, output_dir))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
