from typing import List
from starlette.requests import Request
import time

import ray
from ray import serve

from io import BytesIO
from PIL import Image

import torch
import torchvision
import transforms as T

config = {"num_gpus": 0.1, "num_replicas": 50, "max_batch_size": 16}


@serve.deployment(name="maskrcnn_resnet50",
                  route_prefix="/instance_segmentation",
                  ray_actor_options={"num_gpus": config["num_gpus"]},
                  num_replicas=config["num_replicas"])
class DetectionModel:
    def __init__(self):
        self.model = torchvision.models.detection.__dict__[
            "maskrcnn_resnet50_fpn"](pretrained=True).to("cuda").eval()
        self.preprocessor = T.ToTensor()

    @serve.batch(max_batch_size=config["max_batch_size"])
    async def batch_handler(self, requests: List):
        results = []
        for request in requests:
            image_payload_bytes = await request.body()
            pil_image = Image.open(BytesIO(image_payload_bytes))
            print("[1/3] Parsed image data: {}".format(pil_image))
            pil_images = [pil_image]

            input_tensor = torch.cat(
                [self.preprocessor(i)[0] for i in pil_images]).to("cuda")
            print("[2/3] Images transformed, tensor shape {}".format(
                input_tensor.shape))

            with torch.no_grad():
                output_tensor = self.model([input_tensor])
            print("[3/3] Inference done!")

            result = {}
            for k in output_tensor[0]:
                result[k] = output_tensor[0][k].cpu().numpy().tolist()

            results.append(result)

        return results

    async def __call__(self, request: Request):
        return await self.batch_handler(request)


if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    serve.start()
    DetectionModel.deploy()
    while True:
        time.sleep(5)
        print(serve.list_deployments())