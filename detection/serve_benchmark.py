import time
import numpy as np

from typing import List
from starlette.requests import Request
from io import BytesIO
from PIL import Image
import requests
import asyncio
import httpx

import ray
from ray import serve
from ray.serve.utils import logger

import torch
import torchvision
import transforms as T

config = {"num_gpus": 0.5}
MAX_CONCURRENT_QUERIES = 1000


def deploy(num_replicas=1, max_batch_size=1):
    @serve.deployment(route_prefix="/detection",
                      ray_actor_options={"num_gpus": 0.5},
                      num_replicas=num_replicas,
                      max_concurrent_queries=MAX_CONCURRENT_QUERIES)
    class DetectionModelEndpoint:
        def __init__(self):
            self.model = torchvision.models.detection.__dict__[
                "maskrcnn_resnet50_fpn"](pretrained=True).eval().cuda()
            self.preprocessor = T.ToTensor()

        def __del__(self):
            # Release GPU memory
            del self.model

        @serve.batch(max_batch_size=max_batch_size)
        async def __call__(self, starlette_requests):
            batch_size = len(starlette_requests)
            pil_images = []
            for request in starlette_requests:
                image_payload_bytes = await request.body()
                pil_image = Image.open(BytesIO(image_payload_bytes))
                pil_images.append(pil_image)

            input_tensors = torch.cat(
                [self.preprocessor(i)[0].unsqueeze(0) for i in pil_images])
            input_tensors = input_tensors.cuda()
            with torch.no_grad():
                outputs = self.model(input_tensors)
                torch.cuda.synchronize()

            results = []
            # Prediction Postprocessing
            for i in range(batch_size):
                result = {}
                for k in outputs[i]:
                    result[k] = outputs[i][k].cpu().detach().numpy()
                results.append(result)

            return results

    DetectionModelEndpoint.deploy()


test_image_bytes = requests.get(
    "http://farm8.staticflickr.com/7353/9879082044_66c4f5a6fb_z.jpg").content


async def benchmark(num_iters_per_client, num_clients):
    async def client():
        client_timing = []
        async with httpx.AsyncClient(timeout=None) as client:
            for _ in range(num_iters_per_client):
                start = time.time()
                resp = await client.post("http://localhost:8000/detection",
                                         data=test_image_bytes)
                end = time.time()
                client_timing.append(end - start)
        return client_timing

    all_timings = await asyncio.gather(*[client() for _ in range(num_clients)])
    return np.array(all_timings).flatten()


async def main():
    # Scalingout Becnhmark
    result = []
    for num_replicas in [1, 2, 4]:
        deploy(num_replicas)
        for num_clients in [1, 16]:
            num_iters_per_client = 20
            try:
                timings = await benchmark(num_iters_per_client, num_clients)
                percentiles = np.percentile(
                    np.array(timings).flatten(), [50, 90])
            except httpx.ReadTimeout:
                percentiles = [None, None, None]
            result.append({
                "num_replicas": num_replicas,
                "num_clients": num_clients,
                "p50_latency": percentiles[0],
                "p90_latency": percentiles[1],
            })
            print(result[-1])

    # Batching Benchmark
    result = []
    for max_batch_size in [1, 4, 8]:
        deploy(num_replicas=1, max_batch_size=max_batch_size)
        try:
            timings = await benchmark(num_iters_per_client=20, num_clients=16)
            percentiles = np.percentile(np.array(timings).flatten(), [50, 90])
        except httpx.ReadTimeout:
            percentiles = [None, None, None]
        result.append({
            "num_replicas": num_replicas,
            "max_batch_size": max_batch_size,
            "p50_latency": percentiles[0],
            "p90_latency": percentiles[1],
        })
        print(result[-1])


if __name__ == "__main__":
    ray.init(address="auto")
    serve.start()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
