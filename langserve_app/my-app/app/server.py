from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pirate_speak.chain import chain as pirate_speak_chain


app = FastAPI()



@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")



# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
add_routes(app, pirate_speak_chain, path="/pirate-speak")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


'''
LANGSERVE: ⚠️ Using pydantic 2.5.2. OpenAPI docs for invoke, batch, stream, stream_log endpoints will not be generated. 
API endpoints and playground should work as expected. 
If you need to see the docs, you can downgrade to pydantic 1. 
For example, `pip install pydantic==1.10.13`. See https://github.com/tiangolo/fastapi/issues/10360 for details.
'''