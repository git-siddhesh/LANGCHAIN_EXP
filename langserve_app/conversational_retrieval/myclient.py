# method 1:
#---------------------------------------------------------------------------------
import requests

inputs = {"input": {"question": "what do you know about harrison", "chat_history": []}}
response = requests.post("http://localhost:8000/invoke", json=inputs)

response.json()


# method 2:
#---------------------------------------------------------------------------------
from langserve import RemoteRunnable

# SyntaxError: 'await' outside function

remote_runnable = RemoteRunnable("http://localhost:8000/")
# print(await remote_runnable.ainvoke({"question": "what do you know about harrison", "chat_history": []}))
print( remote_runnable.invoke({"question": "what do you know about harrison", "chat_history": [("hi", "hi")]}))

# async for chunk in remote_runnable.astream({"question": "what do you know about harrison", "chat_history": [("hi", "hi")]}):
for chunk in remote_runnable.stream({"question": "what do you know about harrison", "chat_history": [("hi", "hi")]}):
    print(chunk)

#stream log shows all intermediate steps as well!
# for chunk in remote_runnable.astream_log({"question": "what do you know about harrison", "chat_history": [("hi", "hi")]}):
#     print(chunk)
