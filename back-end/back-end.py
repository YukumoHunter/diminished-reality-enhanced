import asyncio
import websockets
import numpy as np
from PIL import Image, ImageOps
from time import perf_counter
from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
import base64
import json
import ssl
import torch
import pathlib
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
import os
from trackers import ByteTrackTracker
import supervision as sv

BASE_DIR = pathlib.Path(__file__).parent.parent / "cert"
SSL_CERT_PATH = BASE_DIR / "cert.pem"
SSL_KEY_PATH = BASE_DIR / "key.pem"

# Initialize Model
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": ".",
            "trt_fp16_enable": True,
        },
    ),
    "CUDAExecutionProvider",
]

print("Loading inference_model.onnx...")
session = ort.InferenceSession("model/model.onnx", providers=providers)

# Analyze Inputs
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_h, model_w = input_shape[2], input_shape[3]

# Analyze Outputs
output_map = {}
for i, node in enumerate(session.get_outputs()):
    output_map[node.name] = i

if os.name == "nt":
    jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
else:
    jpeg = TurboJPEG()

# Create a ThreadPool for running inference without blocking the asyncio event loop
# This prevents the websocket from timing out during heavy computation
executor = ThreadPoolExecutor(max_workers=1)

# Warm up model (Optional but good practice)
# We need to construct a dummy input matching the ONNX expectation
dummy_input = np.zeros((1, 3, model_h, model_w), dtype=np.float32)
try:
    session.run(None, {input_name: dummy_input})
except Exception as e:
    print(f"Warmup failed: {e}")

CLASS_MAPPING = {
    1: "AH Rode linzen penne",
    2: "Jumbo Sojadrink Naturel",
    3: "Jumbo Haverdrink Naturel",
    4: "AH Cranberry notenmix ongezouten",
    5: "La bioidea Penne",
    6: "Heinz Beanz",
    7: "Ekoplaza Volkoren penne",
    8: "Bonne Maman Pindapasta romig",
    9: "Smaakt Bio Zwarte Bonen",
    10: "Jumbo Organic Whole Wheat Rigatoni",
    11: "AH Dunne crackers kaas",
    12: "Jumbo Hollandse Bruine Bonen",
    13: "AH Biologisch Smeuïge pindakaas met stukjes pinda",
    14: "Hak Bonen Bruine",
    15: "Freshona Linzen Lentils lidl",
    16: "Oatly! Organic oat drink",
    17: "Natural Happiness nut mix Raw",
    18: "Grand' Italia Penne tradizionali",
    19: "Alpro Sojadrink Houdbaar",
    20: "Whole earth Drizzler golden roasted peanut butter",
    21: "Bolletje Low carb groentecrackers paprika",
    22: "Duyvis Oven baked peanuts honey salt",
    23: "AH Terra Plantaardige haverdrink ongezoet",
    24: "AH Terra Jonge kapucijners",
    25: "Rummo Volkoren Biologische Penne Rigate",
    26: "AH Dunne crackers volkoren",
    27: "Rude Health Barista soja",
    28: "Jumbo's 100% Biologische Pindakaas Naturel",
    29: "Jumbo Groentecrackers Wortel & Zoete Aardappel",
    30: "AH Terra Plantaardige sojadrink",
    31: "BioToday Peanutbutter crunchy bio",
    32: "Luna E Terra Pindakaas Smooth Bio",
    33: "AH Terra Plantaardige erwtendrink ongezoet",
    34: "Bolletje Oerknack Waldkorn",
    35: "AH Terra Plantaardig biologisch 100% pindakaas",
    36: "AH Biologisch Stevige crackers kaas",
    37: "Alpro Haverdrink zonder suikers",
    38: "Jumbo Penne Rigate Whole Wheat",
    39: "Jumbo's 100% Natural Peanut Butter",
    40: "Hak Kapucijners",
    41: "Calvé Pindakaas Original",
    42: "La Molisana Penne rigate",
    43: "Natural Happiness proteïnemix",
    44: "Duyvis Oven roasted pinda's original",
    45: "Bonduelle Zwarte bonen",
    46: "Jumbo Stevige Crackers Spelt 8",
    47: "AH Terra Elitehaver ongebrand",
    48: "Bolletje Ontbijt Crackers Spelt Volkoren",
    49: "Bonduelle Chilibonen in saus",
    50: "Jumbo Studentenhaver Ongezouten",
    51: "Skippy Creamy peanut butter",
    52: "Alpro Barista soja",
    53: "Lidl cashew cranberrymix mix alesto",
    54: "De Cecco Penne Rigate Nr. 41",
    55: "AH Terra Noten cranberry rozijn mix ongebrand",
    56: "AH Biologisch Penne",
    57: "Alesto Noten mix",
    58: "TastyBasics Low carb-high protein cracker meerzaden",
    59: "AH Stevige crackers waldkorn",
    60: "Hak witte bonen in tomatensaus",
}

SCHIJF_VAN_VIJF = {
    # Peanut Butter
    "Calvé Pindakaas Original": False,
    "Skippy Creamy peanut butter": False,
    "Bonne Maman Pindapasta romig": False,
    "AH Biologisch Smeuïge pindakaas met stukjes pinda": False,
    "BioToday Peanutbutter crunchy bio": False,
    "Jumbo's 100% Natural Peanut Butter": True,
    "Whole earth Drizzler golden roasted peanut butter": True,
    "AH Terra Plantaardig biologisch 100% pindakaas": True,
    "Jumbo's 100% Biologische Pindakaas Naturel": True,
    "Luna E Terra Pindakaas Smooth Bio": True,
    # Pasta
    "De Cecco Penne Rigate Nr. 41": False,
    "La Molisana Penne rigate": False,
    "AH Biologisch Penne": False,
    "Grand' Italia Penne tradizionali": False,
    "La bioidea Penne": False,
    "Jumbo Penne Rigate Whole Wheat": True,
    "Jumbo Organic Whole Wheat Rigatoni": True,
    "Rummo Volkoren Biologische Penne Rigate": True,
    "Ekoplaza Volkoren penne": True,
    "AH Rode linzen penne": True,
    # Crackers
    "AH Biologisch Stevige crackers kaas": False,
    "AH Dunne crackers volkoren": False,
    "Jumbo Stevige Crackers Spelt 8": False,
    "Jumbo Groentecrackers Wortel & Zoete Aardappel": False,
    "Bolletje Oerknack Waldkorn": False,
    "AH Dunne crackers kaas": True,
    "AH Stevige crackers waldkorn": True,
    "Bolletje Low carb groentecrackers paprika": True,
    "Bolletje Ontbijt Crackers Spelt Volkoren": True,
    "TastyBasics Low carb-high protein cracker meerzaden": True,
    # Nuts
    "AH Cranberry notenmix ongezouten": False,
    "AH Terra Noten cranberry rozijn mix ongebrand": False,
    "Lidl cashew cranberrymix mix alesto": False,
    "Duyvis Oven baked peanuts honey salt": False,
    "Duyvis Oven roasted pinda's original": False,
    "Natural Happiness proteïnemix": True,
    "AH Terra Elitehaver ongebrand": True,
    "Alesto Noten mix": True,
    "Jumbo Studentenhaver Ongezouten": True,
    "Natural Happiness nut mix Raw": True,
    # Beans
    "Bonduelle Chilibonen in saus": False,
    "Hak  witten bonen in tomatensaus": False,
    "Heinz Beanz": False,
    "Smaakt Bio Zwarte Bonen": False,
    "Hak Bonen Bruine": False,
    "Jumbo Hollandse Bruine Bonen": True,
    "Bonduelle Zwarte bonen": True,
    "Freshona Linzen Lentils lidl": True,
    "Hak Kapucijners": True,
    "AH Terra Jonge kapucijners": True,
    # Plant-based Drinks
    "AH Terra Plantaardige haverdrink ongezoet": False,
    "Alpro Haverdrink zonder suikers": False,
    "Jumbo Haverdrink Naturel": False,
    "Oatly! Organic oat drink": False,
    "Rude Health Barista soja": False,
    "AH Terra Plantaardige erwtendrink ongezoet": True,
    "Alpro Sojadrink Houdbaar": True,
    "Alpro Barista soja": True,
    "AH Terra Plantaardige sojadrink": True,
    "Jumbo Sojadrink Naturel": True,
    # Pasta Sauce
    "Spagheroni Tradizionale": False,
    "Heinz Traditional pasta sauce": False,
    "La Dolce Vita Traditional Pastasaus": False,
    "AH Biologisch Pastasaus tradizionale": False,
    "Fertilia Pastasaus traditionale": False,
    "Ecoplaza Pastasaus sugo tradizionale": False,
}

# TESTING: set all to False
for k in SCHIJF_VAN_VIJF:
    SCHIJF_VAN_VIJF[k] = False


def get_real_coordinates(boxes_norm, orig_w, orig_h, model_dim=560):
    """
    boxes_norm: np.array of shape (N, 4) -> [cx, cy, w, h] (0-1 normalized)
    """
    # 1. Denormalize to the model's 560x560 frame
    # We multiply by 560, NOT original_w/h yet
    boxes_560 = boxes_norm * model_dim  # [cx, cy, w, h] in 560 pixels

    # 2. Calculate the Padding and Scale used by ImageOps.pad
    scale = min(model_dim / orig_w, model_dim / orig_h)
    pad_x = (model_dim - orig_w * scale) / 2
    pad_y = (model_dim - orig_h * scale) / 2

    # 3. Remove Padding (Shift) and Scale Back (Resize)
    # Apply to Center X and Center Y
    cx_real = (boxes_560[:, 0] - pad_x) / scale
    cy_real = (boxes_560[:, 1] - pad_y) / scale

    # Apply to Width and Height (only Scale, no shift)
    w_real = boxes_560[:, 2] / scale
    h_real = boxes_560[:, 3] / scale

    # 4. Convert Center-Format [cx, cy, w, h] -> Corner-Format [x1, y1, x2, y2]
    # Supervision expects [x1, y1, x2, y2]
    x1 = cx_real - (w_real / 2)
    y1 = cy_real - (h_real / 2)
    x2 = cx_real + (w_real / 2)
    y2 = cy_real + (h_real / 2)

    return np.stack([x1, y1, x2, y2], axis=1)


def run_inference_sync(image_bytes, request_id, tracker):
    """
    Synchronous function to handle image decoding and inference.
    This runs in a separate thread to keep the Websocket heartbeat alive.
    """

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    try:
        # 1. Decode Image
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = jpeg.decode(npimg, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)

        if img is None:
            return {"error": "Failed to decode image"}

        original_h, original_w = img.shape[:2]

        # 2. Preprocess
        img = Image.fromarray(img)
        img = ImageOps.pad(img, (model_w, model_h))

        # (H,W,C) -> (C,H,W) -> (1,C,H,W)
        img = torch.from_numpy(np.asarray(img)).float() / 255.0
        img = img.permute((2, 0, 1))
        input_tensor = img.unsqueeze(0)

        # 3. Inference
        start = perf_counter()
        raw_results = session.run(None, {input_name: input_tensor})
        end = perf_counter()

        inference_time = (end - start) * 1000
        print(f"inference took {inference_time:.2f}ms")

        # 4. Format Results (vectorized)
        box_data = raw_results[output_map["dets"]]  # Shape: [1, N, 4]
        score_data = raw_results[output_map["labels"]]  # Shape: [1, N, 61]

        boxes = box_data[0]
        logits = score_data[0]
        probs = softmax(logits)

        num_queries = boxes.shape[0]
        threshold = 0.5

        # Vectorized filtering
        class_ids = np.argmax(probs, axis=1)
        scores = probs[np.arange(num_queries), class_ids]

        valid_mask = (scores > threshold) & (class_ids < 60)
        mapped_ids = class_ids + 1
        has_mapping = np.array([int(mid) in CLASS_MAPPING for mid in mapped_ids])
        valid_mask = valid_mask & has_mapping

        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]

        if len(valid_boxes) == 0:
            return {"detections": [], "requestId": request_id}

        # Convert to xyxy coordinates
        xyxy = get_real_coordinates(valid_boxes, original_w, original_h, model_dim=model_w)

        # Run ByteTrack tracker
        sv_detections = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=valid_scores.astype(np.float32),
            class_id=valid_class_ids.astype(int),
        )
        sv_detections = tracker.update(sv_detections)

        # Build response
        detections = []
        for i in range(len(sv_detections)):
            mapped_id = int(sv_detections.class_id[i]) + 1
            class_name = CLASS_MAPPING[mapped_id]

            x1, y1, x2, y2 = sv_detections.xyxy[i]
            width_px = x2 - x1
            height_px = y2 - y1

            det = {
                "class": class_name,
                "confidence": float(sv_detections.confidence[i]),
                "in_schijf_van_vijf": SCHIJF_VAN_VIJF.get(class_name, True),
                "bbox": [float(x1), float(y1), float(width_px), float(height_px)],
            }

            if sv_detections.tracker_id is not None:
                det["tracker_id"] = int(sv_detections.tracker_id[i])

            detections.append(det)

        return {"detections": detections, "requestId": request_id}

    except Exception as e:
        print(f"Inference Error: {e}")
        return {"error": str(e)}


async def inference_worker(websocket, queue, tracker):
    """
    Consumer: Pulls frames from queue and runs inference.
    """
    while True:
        # Get the next frame from the queue
        message = await queue.get()

        try:
            if isinstance(message, bytes):
                # Binary protocol: [4 bytes uint32 requestId BE] + [JPEG bytes]
                request_id = int.from_bytes(message[:4], byteorder="big")
                image_bytes = message[4:]
            else:
                # Legacy JSON/base64 fallback
                data = json.loads(message)
                request_id = data.get("requestId")
                image_data = data["image"].split(",")[1]
                image_bytes = base64.b64decode(image_data)

            # Run inference in a separate thread so we don't block the event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                executor, run_inference_sync, image_bytes, request_id, tracker
            )

            # Send result back
            await websocket.send(json.dumps(response))

        except Exception as e:
            print(f"Worker Error: {e}")
        finally:
            queue.task_done()


async def detect_frame_handler(websocket):
    """
    Producer: Receives frames and puts them in a size-limited queue.
    """
    # Maxsize=1 ensures we only keep the LATEST frame.
    # If the GPU is busy, we drop incoming frames.
    queue = asyncio.Queue(maxsize=1)

    # Per-connection tracker for persistent object IDs
    tracker = ByteTrackTracker()

    # Start the consumer task
    worker_task = asyncio.create_task(inference_worker(websocket, queue, tracker))

    try:
        async for message in websocket:
            if queue.full():
                # BUFFER STRATEGY: Drop the oldest frame (LIFO)
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # Put the new frame in the queue
            await queue.put(message)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        # Clean up the worker when connection closes
        worker_task.cancel()


async def main():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    try:
        ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_KEY_PATH)
    except Exception as e:
        print(f"Error loading certificates: {e}")
        return

    print(f"Starting secure server (WSS) on 0.0.0.0:5174...")

    async with websockets.serve(detect_frame_handler, "0.0.0.0", 5174, ssl=ssl_context):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
