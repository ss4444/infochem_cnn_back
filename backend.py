import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
from schemas import Predict
from starlette.middleware.cors import CORSMiddleware

model = torch.load('gg_softmax2.pth')

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


preprocess = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    return image


@app.post('/upload_file')
async def predict(file: UploadFile = File(...)):
    file_read = await file.read()
    img = read_imagefile(file_read)
    img_preproc = preprocess(img)
    batch_img = torch.unsqueeze(img_preproc, 0)
    model.eval()
    out = model(batch_img)
    out = out.tolist()
    return Predict(
        benz_92=round(out[0][0] * 100, 3),
        benz_95=round(out[0][1] * 100, 3),
        benz_98=round(out[0][2] * 100, 3),
    )


if __name__ == "__main__":
    uvicorn.run(app)
