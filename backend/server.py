import asyncio
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import onnxruntime as ort
import supervision as sv
import websockets
from PIL import Image, ImageOps
from trackers import ByteTrackTracker
from turbojpeg import TurboJPEG, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE

# --- ONNX Model ---

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

print("Loading model/model.onnx...")
session = ort.InferenceSession("model/model.onnx", providers=providers)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_h, model_w = input_shape[2], input_shape[3]

output_map = {}
for i, node in enumerate(session.get_outputs()):
    output_map[node.name] = i

# --- TurboJPEG ---

if os.name == "nt":
    jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
else:
    jpeg = TurboJPEG()

# --- Thread pool ---

executor = ThreadPoolExecutor(max_workers=1)

# --- Warmup ---

dummy_input = np.zeros((1, 3, model_h, model_w), dtype=np.float32)
try:
    session.run(None, {input_name: dummy_input})
    print("Model warmup complete.")
except Exception as e:
    print(f"Warmup failed: {e}")

# --- Class & health data ---

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
    "Hak witte bonen in tomatensaus": False,
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

NUM_PRODUCT_CLASSES = len(CLASS_MAPPING)
MIN_TRACKING_SCORE = 0.2
TRACK_ACTIVATION_SCORE = 0.5
TRACK_HIGH_CONF_SCORE = 0.5
TRACK_HOLD_FRAMES = 3
TRACK_MEMORY_TTL_FRAMES = 30
CLASS_VOTE_DECAY = 0.8

# --- Coordinate math ---


def get_real_coordinates(boxes_norm, orig_w, orig_h, model_dim=560):
    """
    boxes_norm: np.array of shape (N, 4) -> [cx, cy, w, h] (0-1 normalized)
    Returns: np.array of shape (N, 4) -> [x1, y1, x2, y2] in original image coords
    """
    boxes_px = boxes_norm * model_dim

    scale = min(model_dim / orig_w, model_dim / orig_h)
    pad_x = (model_dim - orig_w * scale) / 2
    pad_y = (model_dim - orig_h * scale) / 2

    cx_real = (boxes_px[:, 0] - pad_x) / scale
    cy_real = (boxes_px[:, 1] - pad_y) / scale
    w_real = boxes_px[:, 2] / scale
    h_real = boxes_px[:, 3] / scale

    x1 = cx_real - (w_real / 2)
    y1 = cy_real - (h_real / 2)
    x2 = cx_real + (w_real / 2)
    y2 = cy_real + (h_real / 2)

    return np.stack([x1, y1, x2, y2], axis=1)


# --- Inference ---


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def build_detection(xyxy, class_id, confidence, tracker_id=None):
    mapped_id = int(class_id) + 1
    class_name = CLASS_MAPPING[mapped_id]
    x1, y1, x2, y2 = xyxy

    det = {
        "class": class_name,
        "confidence": float(confidence),
        "in_schijf_van_vijf": SCHIJF_VAN_VIJF.get(class_name, True),
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
    }
    if tracker_id is not None:
        det["tracker_id"] = int(tracker_id)
    return det


def get_track_state(track_memory, track_id):
    state = track_memory.get(track_id)
    if state is None:
        state = {
            "bbox": None,
            "confidence": 0.0,
            "miss_count": 0,
            "stable_class_id": None,
            "class_votes": np.zeros(NUM_PRODUCT_CLASSES, dtype=np.float32),
        }
        track_memory[track_id] = state
    return state


def build_stable_detections(sv_dets, track_memory):
    detections = []
    seen_track_ids = set()

    for i in range(len(sv_dets)):
        track_id = None
        if sv_dets.tracker_id is not None:
            track_id = int(sv_dets.tracker_id[i])

        # Ignore tentative tracks until ByteTrack confirms them with a real ID.
        if track_id is None or track_id < 0:
            continue

        seen_track_ids.add(track_id)
        state = get_track_state(track_memory, track_id)
        class_id = int(sv_dets.class_id[i])
        confidence = float(sv_dets.confidence[i])
        bbox = sv_dets.xyxy[i].astype(np.float32).copy()

        state["class_votes"] *= CLASS_VOTE_DECAY
        state["class_votes"][class_id] += confidence
        state["stable_class_id"] = int(np.argmax(state["class_votes"]))
        state["bbox"] = bbox
        state["confidence"] = confidence
        state["miss_count"] = 0

        detections.append(
            build_detection(
                bbox,
                state["stable_class_id"],
                confidence,
                tracker_id=track_id,
            )
        )

    stale_track_ids = []
    for track_id, state in list(track_memory.items()):
        if track_id in seen_track_ids:
            continue

        state["miss_count"] += 1
        if (
            state["miss_count"] <= TRACK_HOLD_FRAMES
            and state["bbox"] is not None
            and state["stable_class_id"] is not None
        ):
            detections.append(
                build_detection(
                    state["bbox"],
                    state["stable_class_id"],
                    state["confidence"],
                    tracker_id=track_id,
                )
            )

        if state["miss_count"] > TRACK_MEMORY_TTL_FRAMES:
            stale_track_ids.append(track_id)

    for track_id in stale_track_ids:
        del track_memory[track_id]

    return detections


def run_inference_sync(image_bytes, request_id, tracker, track_memory):
    try:
        t0 = perf_counter()

        # Decode
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = jpeg.decode(npimg, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
        if img is None:
            return {"error": "Failed to decode image"}

        t_decode = perf_counter()

        original_h, original_w = img.shape[:2]

        # Preprocess (pure numpy, no torch)
        img = Image.fromarray(img)
        img = ImageOps.pad(img, (model_w, model_h))
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        input_tensor = np.ascontiguousarray(
            np.transpose(img_arr, (2, 0, 1))[np.newaxis, ...]
        )

        t_preprocess = perf_counter()

        # Inference
        raw_results = session.run(None, {input_name: input_tensor})
        t_inference = perf_counter()

        # Post-process (vectorized)
        boxes = raw_results[output_map["dets"]][0]  # (N, 4)
        logits = raw_results[output_map["labels"]][0]  # (N, 61)
        probs = softmax(logits)

        num_queries = boxes.shape[0]
        class_ids = np.argmax(probs, axis=1)
        scores = probs[np.arange(num_queries), class_ids]

        valid_mask = (scores > MIN_TRACKING_SCORE) & (class_ids < NUM_PRODUCT_CLASSES)
        mapped_ids = class_ids + 1
        has_mapping = np.array([int(mid) in CLASS_MAPPING for mid in mapped_ids])
        valid_mask = valid_mask & has_mapping

        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]

        if len(valid_boxes) == 0:
            sv_dets = tracker.update(sv.Detections.empty())
        else:
            xyxy = get_real_coordinates(
                valid_boxes, original_w, original_h, model_dim=model_w
            )

            # ByteTrack
            sv_dets = sv.Detections(
                xyxy=xyxy.astype(np.float32),
                confidence=valid_scores.astype(np.float32),
                class_id=valid_class_ids.astype(int),
            )
            sv_dets = tracker.update(sv_dets)

        detections = build_stable_detections(sv_dets, track_memory)
        t_postprocess = perf_counter()

        decode_ms = (t_decode - t0) * 1000
        preprocess_ms = (t_preprocess - t_decode) * 1000
        inference_ms = (t_inference - t_preprocess) * 1000
        postprocess_ms = (t_postprocess - t_inference) * 1000
        total_ms = (t_postprocess - t0) * 1000
        print(f"[req {request_id}] decode={decode_ms:.1f}ms preprocess={preprocess_ms:.1f}ms inference={inference_ms:.1f}ms postprocess={postprocess_ms:.1f}ms total={total_ms:.1f}ms")

        return {"detections": detections, "requestId": request_id}

    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}


# --- WebSocket server ---


async def inference_worker(websocket, queue, tracker, track_memory):
    last_done = perf_counter()
    while True:
        t_wait_start = perf_counter()
        message = await queue.get()
        t_got = perf_counter()
        try:
            # Binary protocol: [4 bytes uint32 requestId BE] + [JPEG bytes]
            request_id = int.from_bytes(message[:4], byteorder="big")
            image_bytes = message[4:]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                executor,
                run_inference_sync,
                image_bytes,
                request_id,
                tracker,
                track_memory,
            )
            t_inferred = perf_counter()
            await websocket.send(json.dumps(response))
            t_sent = perf_counter()

            idle_ms = (t_got - last_done) * 1000
            queue_ms = (t_got - t_wait_start) * 1000
            send_ms = (t_sent - t_inferred) * 1000
            round_ms = (t_sent - t_got) * 1000
            print(f"[req {request_id}] idle={idle_ms:.1f}ms queue_wait={queue_ms:.1f}ms send={send_ms:.1f}ms round_trip={round_ms:.1f}ms")
            last_done = t_sent
        except Exception as e:
            print(f"Worker error: {e}")
            last_done = perf_counter()
        finally:
            queue.task_done()


async def handler(websocket):
    queue = asyncio.Queue(maxsize=1)
    tracker = ByteTrackTracker(
        track_activation_threshold=TRACK_ACTIVATION_SCORE,
        high_conf_det_threshold=TRACK_HIGH_CONF_SCORE,
        minimum_consecutive_frames=1,
    )
    track_memory = {}
    worker_task = asyncio.create_task(
        inference_worker(websocket, queue, tracker, track_memory)
    )
    client = getattr(websocket, "remote_address", None)
    print(f"Client connected: {client}")

    try:
        async for message in websocket:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(message)
    except websockets.exceptions.ConnectionClosed as e:
        print(
            f"Client disconnected: {client}, code={e.code}, reason={e.reason or 'none'}"
        )
    finally:
        worker_task.cancel()


async def main():
    print("Starting WebSocket server on 0.0.0.0:5174...")
    async with websockets.serve(handler, "0.0.0.0", 5174):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
