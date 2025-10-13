from bitsec.utils.chutes_llm import chutes_client, GPTMessage
import numpy as np

code = ""
with open("code1.sol", "r") as f:
    code = f.read()

print(f"code: {code[:100]}...")

embedding = chutes_client.embed(code)
print(f"embedding: {embedding}...")
print(f"embedding length: {len(embedding)}")

# with open("/home/60/bitsec_subnet/samples/clean-codebases/1.sol", "r") as f:
#     code2 = f.read()

with open("/home/60/bitsec_subnet/samples/clean-codebases/9061.sol", "r") as f:
    code2 = f.read()

print(f"code2: {code2[:100]}...")

embedding2 = chutes_client.embed(code2)
print(f"embedding2 length: {len(embedding2)}")

similarity = np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
print(f"similarity: {similarity}")