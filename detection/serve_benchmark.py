import sys
import time
import json
from typing import List

import asyncio
import aiohttp
import requests
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.utils import logger

from io import BytesIO
from PIL import Image
import numpy as np

import torch
import torchvision
import transforms as T

NUM_CLIENTS = 10
CALLS_PER_BATCH = 10

config = {"num_gpus": 0.1}


def dump_json(data, fpath):
    with open(fpath, "w") as write_json:
        json.dump(data, write_json)


async def timeit(name, fn):
    start = time.time()
    while time.time() - start < 1:
        await fn()
    # real run
    stats = []
    for _ in range(4):
        start = time.time()
        await fn()
        end = time.time()
        stats.append((end - start))

    logger.info("\t{} {} +- {} sec".format(name, round(np.mean(stats), 2),
                                           round(np.std(stats), 2)))
    return {"mean": round(np.mean(stats), 2), "std": round(np.std(stats), 2)}


async def fetch(session, data):
    async with session.post("http://127.0.0.1:8000/api",
                            data=data) as response:
        response = await response.json(content_type=None)
        #assert response == "ok", response


@ray.remote
class InferenceClient():
    def __init__(self):
        self.session = aiohttp.ClientSession()

    async def do_queries(self, num, data):
        # Query a batch of images
        for _ in range(num):
            await fetch(self.session, data)


async def trial(num_replicas, max_batch_size, max_concurrent_queries,
                result_json):

    test_image_bytes = requests.get(
        "http://farm8.staticflickr.com/7353/9879082044_66c4f5a6fb_z.jpg"
    ).content
    test_image_size = round(sys.getsizeof(test_image_bytes) / 1000., 2)

    trial_key_base = (
        f"calls_per_batch:{CALLS_PER_BATCH}/replica:{num_replicas}/batch_size:{max_batch_size}/"
        f"concurrent_queries:{max_concurrent_queries}/"
        f"input_data_size(KB):{test_image_size}")

    logger.info(f"num_replicas={num_replicas},"
                f"max_batch_size={max_batch_size},"
                f"max_concurrent_queries={max_concurrent_queries},"
                f"input_data_size(KB)={test_image_size}")

    @serve.deployment(name="api",
                      max_concurrent_queries=max_concurrent_queries,
                      ray_actor_options={"num_gpus": config["num_gpus"]},
                      num_replicas=num_replicas)
    class DetectionModelEndpoint:
        def __init__(self):
            self.model = torchvision.models.detection.__dict__[
                "maskrcnn_resnet50_fpn"](
                    pretrained=True).cuda().eval()  # CUDA/CPU
            self.preprocessor = T.ToTensor()

        @serve.batch(max_batch_size=max_batch_size)
        async def batch_handler(self, reqs: List):
            results = []
            for req in reqs:
                # Image Loading
                image_payload_bytes = await req.body()

                pil_image = Image.open(BytesIO(image_payload_bytes))
                pil_images = [pil_image]

                # Image Preprocessing
                input_tensor = torch.cat([
                    self.preprocessor(i)[0] for i in pil_images
                ]).cuda()  # CUDA/CPU

                # Inference
                with torch.no_grad():
                    output_tensor = self.model([input_tensor])

                # Prediction Postprocessing
                result = {}
                for k in output_tensor[0]:
                    result[k] = output_tensor[0][k].cpu().detach().numpy(
                    )  # CUDA/CPU
                # With score threshold
                #for idx, score in enumerate(output_tensor[0]["scores"]):
                #    if score > 0.9:
                #        idxs.append(idx)
                #for k in output_tensor[0]:
                #    result[k] = output_tensor[0][k][idxs].cpu().numpy()

                results.append(result)

            return results

        async def __call__(self, req: Request):
            return await self.batch_handler(req)

    DetectionModelEndpoint.deploy()
    routes = requests.get("http://127.0.0.1:8000/-/routes").json()
    assert "/api" in routes, routes

    async with aiohttp.ClientSession() as session:

        async def single_client():
            for _ in range(CALLS_PER_BATCH):
                await fetch(session, test_image_bytes)

        single_client_avg_tps = await timeit("single client", single_client)
        key = "num_client:1/" + trial_key_base
        result_json.update({key: single_client_avg_tps})

    clients = [InferenceClient.remote() for _ in range(NUM_CLIENTS)]

    async def many_clients():
        ray.get([
            c.do_queries.remote(CALLS_PER_BATCH, test_image_bytes)
            for c in clients
        ])

    multi_client_avg_tps = await timeit("{} clients".format(len(clients)),
                                        many_clients)
    key = f"num_client:{len(clients)}/" + trial_key_base
    result_json.update({key: multi_client_avg_tps})

    logger.info(result_json)


async def main():
    result_json = {}
    for num_replicas, max_batch_size, max_concurrent_queries in [[5, 1, 5],
                                                                 [5, 5, 5],
                                                                 [10, 1, 10],
                                                                 [10, 10, 10]]:
        await trial(num_replicas, max_batch_size, max_concurrent_queries,
                    result_json)
    dump_json(result_json, "./serve_benchmark_gpu2.json")


if __name__ == "__main__":
    ray.init(address="auto")
    serve.start()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
