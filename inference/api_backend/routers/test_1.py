import uvicorn
from fastapi import FastAPI, Query, Form, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time
import os


app = FastAPI()
router = APIRouter()


@router.get('/fetch_project')
async def fetch_project():
    start = time.time()
    return {'time': time.time() - start, 'data': ''}



@router.post('/add')
async def add(
        name: str = Form(..., description='name', example='Name'),
        color: str = Form(..., description='color', example='#00CCFF')
):
    start = time.time()
    print(name, color)
    return {'time': time.time() - start}


@router.put('/change')
async def change(
        f_id: str = Form(..., description='file id', example='Name'),
):
    start = time.time()
    print(f_id)
    return {'time': time.time() - start}


@router.delete('/delete')
async def delete(
        c_id: str = Query(..., description='label/class id', example='4beb867cdeba4f259d9202f5bc58a47c')
):
    start = time.time()
    print(c_id)
    return {'time': time.time() - start}



app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8003, workers=1)
