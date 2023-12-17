# method 1:
#---------------------------------------------------------------------------------
# import requests

# query = {"input": {"question": "what do you know about alzheimer", "chat_history": [("hi", "I want to know about medical diseases")]}}
# response = requests.get("http://0.0.0.0:8000/search", json=query)

# response.json()


# # method 2:
# #---------------------------------------------------------------------------------
# from langserve import RemoteRunnable

# # SyntaxError: 'await' outside function

# remote_runnable = RemoteRunnable("http://localhost:8000/search")
# print( remote_runnable.invoke({"question": "what do you know about alzheimer", "chat_history": [("hi", "I want to know about medical diseases")]}))

# # for chunk in remote_runnable.stream({"question": "what do you know about harrison", "chat_history": [("hi", "hi")]}):
# #     print(chunk)


# # method 3:
# #---------------------------------------------------------------------------------
# import httpx

# async def make_request():
#     query = {
#         "input": {
#             "question": "what do you know about alzheimer",
#             "chat_history": [("hi", "I want to know about medical diseases")]
#         }
#     }
#     query = {"input": "what do you know about alzheimer"}

#     async with httpx.AsyncClient() as client:
#         response = await client.get("http://localhost:8000/search", params=query)
    
#     if response.status_code == 200:
#         result = response.json()
#         print("Server response:", result)
#     else:
#         print("Error:", response.status_code, response.text)

# # Run the event loop to make the asynchronous request
# import asyncio
# asyncio.run(make_request())



import requests

inputs = {"input": "tree"}
response = requests.post("http://localhost:8000/invoke", json=inputs)

print(response.json())
